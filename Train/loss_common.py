"""New, vectorised composite loss that parses <bbob> snippets once per batch.

This drops the per-snippet Python loops of the old implementation while
keeping the public factory name ``create_compute_loss_func`` unchanged so
`train_vision.py` continues to work.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F
from torchvision.ops import complete_box_iou_loss
import warnings

from .loss_helpers import (
    TAG_OPEN,
    TAG_CLOSE,
    EPSILON,
    hungarian_match,
    iou_matrix_xywh,
    xywh_to_xyxy,
    decode_pred_gt,
    NUM_DIGITS,
    COORD_SCALE,
)

try:
    from torch_linear_assignment import batch_linear_assignment, assignment_to_indices  # type: ignore
    _use_tla = True
except ImportError:  # pragma: no cover
    _use_tla = False

# -----------------------------------------------------------------------------
# Composite loss – vectorised implementation
# -----------------------------------------------------------------------------

class CompositeLoss:
    def __init__(
        self,
        tokenizer,
        *,
        lambda_iou: float = 1.5,
        lambda_detection: float = 0.2,
        lambda_match_penalty: float = 0.5,
        lm_target: float = 2,
        smoothing_factor: float = 0.95,
        logger=None,
        log_interval: int = 100,
    ):
        self.tok = tokenizer
        self.lambda_iou = lambda_iou
        self.lambda_det = lambda_detection
        self.lambda_match = lambda_match_penalty
        self.lm_target = lm_target
        self.smoothing = smoothing_factor
        self.logger = logger
        self.log_interval = max(1, log_interval)
        self.step = 0
        self.lm_loss_ema: float | None = None
        self.is_eval = False
        self._dangling_total = 0  # aggregate counter for malformed coord digits
        self.last_pred_boxes: int = 0  # number of predicted boxes after filtering per step

        # digit/ST cache
        digit_ids = [tokenizer.convert_tokens_to_ids(str(i)) for i in range(10)]
        self.digit_ids = torch.tensor(digit_ids, dtype=torch.long)
        if any(i == tokenizer.unk_token_id or i == -1 for i in digit_ids):
            raise ValueError("Tokenizer must contain explicit '0'..'9' tokens")
        self.digit_set = set(digit_ids)
        self._values10: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._pow10: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self.tau_start, self.tau_end, self.tau_steps = 5.0, 0.1, 50_000

    # ------------- straight-through digit expectation ----------------
    def _st_expect(self, logits_slice: torch.Tensor) -> torch.Tensor:
        device, dtype = logits_slice.device, logits_slice.dtype
        ids = self.digit_ids.to(device)
        if ids.max().item() >= logits_slice.size(-1):
            warnings.warn("Digit token IDs exceed vocab range; falling back to uniform random coords and disabling ST path.")
            return torch.rand(logits_slice.shape[:-1], device=device, dtype=dtype)

        dl = logits_slice.index_select(-1, ids)
        prog = min(1.0, self.step / max(1, self.tau_steps))
        tau = max(self.tau_end, self.tau_start * (1 - prog))
        y_soft = F.gumbel_softmax(dl.clamp(-10, 10), tau=tau, hard=False, dim=-1)
        if not torch.isfinite(y_soft).all():
            y_soft = torch.softmax(dl / max(tau, 0.1), dim=-1)

        idx = y_soft.argmax(-1)
        y_hard = F.one_hot(idx, num_classes=10).type_as(y_soft)
        y_st = y_hard + (y_soft - y_soft.detach())
        key = (device, dtype)
        if key not in self._values10:
            self._values10[key] = torch.arange(10, device=device, dtype=dtype)
        return (y_st * self._values10[key]).sum(-1)

    # ---------------- helper: cached [10^(n-1) … 10^0] tensor ----------------
    def _pow10_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        if key not in self._pow10:
            self._pow10[key] = torch.tensor([10 ** (NUM_DIGITS - 1 - k) for k in range(NUM_DIGITS)], device=device, dtype=dtype)
        return self._pow10[key]

    # ---------------- helper: parse coordinates from one sample ----------------
    def _parse_coords_sample(self, token_row: torch.Tensor, logits_row: torch.Tensor) -> torch.Tensor:
        """Vectorised extraction of `(N, 4)` *xywh* boxes from one sample.

        Compared to the previous Python-loop implementation this version:
        1. Uses tensor operations to identify regions between `<bbob>` tags.
        2. Collects all digit-token logits in a single `index_select` call.
        3. Calls the straight-through expectation once on the **entire** stack
           and reconstructs the coordinates with a few vector ops.
        The resulting code is drastically faster when hundreds of boxes are
        emitted because most of the work happens on the GPU.
        """

        # Ensure we have cached tag IDs only once
        if not hasattr(self, "_id_open"):
            self._id_open = self.tok.convert_tokens_to_ids(TAG_OPEN)
            self._id_close = self.tok.convert_tokens_to_ids(TAG_CLOSE)
        id_open: int = self._id_open  # type: ignore[attr-defined]
        id_close: int = self._id_close  # type: ignore[attr-defined]

        # ------------------------------------------------------------------
        # 1. Build a boolean mask that is True for tokens *inside* a pair of
        #    <bbob> … </bbob> tags (exclusive).
        # ------------------------------------------------------------------
        open_mask = token_row == id_open
        close_mask = token_row == id_close

        # Cumulative depth: #opens seen minus #closes seen so far
        depth = open_mask.cumsum(0) - close_mask.cumsum(0)
        inside_mask = depth > 0  # True between the tags

        # Exclude the tag tokens themselves
        inside_mask = inside_mask & ~(open_mask | close_mask)

        # ------------------------------------------------------------------
        # 2. Keep only digit tokens that are inside a <bbob> region.
        # ------------------------------------------------------------------
        digit_ids_device = self.digit_ids.to(token_row.device)
        digit_mask = (token_row.unsqueeze(-1) == digit_ids_device).any(-1)

        candidate_mask = inside_mask & digit_mask
        positions = candidate_mask.nonzero(as_tuple=False).flatten()  # (D,)

        total_digits = positions.numel()
        if total_digits < NUM_DIGITS:
            # Not enough digits for a single coordinate – return empty tensor
            if total_digits > 0:
                self._dangling_total += total_digits
            return logits_row.new_zeros((0, 4))

        # Discard dangling digits that don't fit into full NUM_DIGITS groups
        num_complete = total_digits // NUM_DIGITS
        if total_digits % NUM_DIGITS:
            self._dangling_total += total_digits % NUM_DIGITS
            positions = positions[: num_complete * NUM_DIGITS]

        # ------------------------------------------------------------------
        # 3. Compute straight-through expectations for *all* digit logits in
        #    one shot and reshape them into (num_coords, NUM_DIGITS).
        # ------------------------------------------------------------------
        dig_logits = logits_row.index_select(0, positions)  # (D, V)
        dig_vals = self._st_expect(dig_logits)             # (D,)

        dig_vals = dig_vals.view(num_complete, NUM_DIGITS)
        coord_vals = (dig_vals * self._pow10_tensor(logits_row.device, dig_vals.dtype)).sum(-1) / COORD_SCALE
        coord_vals = coord_vals.clamp_(0.0, 1.0)

        # ------------------------------------------------------------------
        # 4. Reshape to (N, 4) box tensor.  Any leftover <4 coords were already
        #    accounted for in the dangling counter above.
        # ------------------------------------------------------------------
        if coord_vals.numel() >= 4:
            n_boxes = coord_vals.numel() // 4
            return coord_vals[: n_boxes * 4].view(-1, 4)

        return logits_row.new_zeros((0, 4))

    # ---------------- helper: parse coords per sample ----------------

   # ------------- detection loss (vectorised coords) ----------------
    def _detection_loss(self, logits: torch.Tensor, gt_boxes):
        """Compute IoU loss & match-rate in a batched, GPU-friendly way."""

        B = logits.size(0)
        token_ids = logits.argmax(dim=-1)

        # Reset counter for this step
        self.last_pred_boxes = 0

        diff_preds: List[torch.Tensor] = [
            self._parse_coords_sample(token_ids[b], logits[b]) for b in range(B)
        ]

        # --------------------------------------------------------------
        # Fast path – use torch_linear_assignment in batch on GPU
        # --------------------------------------------------------------
        if _use_tla:
            pred_lens = [p.size(0) for p in diff_preds]
            gt_lens   = [len(g) if isinstance(g, list) else g.size(0) for g in gt_boxes]

            max_pred = max(pred_lens) if pred_lens else 0
            max_gt   = max(gt_lens)   if gt_lens else 0

            if max_pred == 0 or max_gt == 0:
                return logits.new_tensor(0.0), 0.0

            cost_batch = logits.new_ones((B, max_pred, max_gt))  # init cost=1 (no match)

            for b in range(B):
                pred = diff_preds[b]
                gt = gt_boxes[b]
                if isinstance(gt, list):
                    gt = torch.tensor(gt, device=logits.device, dtype=logits.dtype)

                # Filter non-degenerate boxes and count them for logging
                pred_f = pred[(pred[:, 2] > EPSILON) & (pred[:, 3] > EPSILON)]
                self.last_pred_boxes += int(pred_f.size(0))
                gt_f   = gt[(gt[:, 2] > EPSILON) & (gt[:, 3] > EPSILON)]

                if pred_f.numel() == 0 or gt_f.numel() == 0:
                    continue

                # Build cost matrix using FILTERED boxes
                cost_batch[b, : pred_f.size(0), : gt_f.size(0)] = 1.0 - iou_matrix_xywh(pred_f, gt_f)

            assignment = batch_linear_assignment(cost_batch)
            rows, cols = assignment_to_indices(assignment)  # shape (B, K)

            pm_list, gm_list = [], []
            matched, total = 0, 0

            for b in range(B):
                # Retrieve the original (unfiltered) predictions and GT for this sample
                pred = diff_preds[b]
                gt   = gt_boxes[b]
                if isinstance(gt, list):
                    gt = torch.tensor(gt, device=logits.device, dtype=logits.dtype)

                # Re-apply the same filter so that the row/col indices from assignment
                # correctly refer to the filtered tensors used in cost matrix construction
                pred_f = pred[(pred[:, 2] > EPSILON) & (pred[:, 3] > EPSILON)]
                gt_f   = gt  [(gt  [:, 2] > EPSILON) & (gt  [:, 3] > EPSILON)]

                P, G = pred_f.size(0), gt_f.size(0)
                if P == 0 or G == 0:
                    total += G  # still count GT objects even if no prediction passes
                    continue

                valid = (rows[b] >= 0) & (rows[b] < P) & (cols[b] >= 0) & (cols[b] < G)
                r_sel = rows[b][valid]
                c_sel = cols[b][valid]

                if r_sel.numel() > 0:
                    pm_list.append(pred_f[r_sel])
                    gm_list.append(gt_f[c_sel])
                    matched += r_sel.numel()
                total += G

            if pm_list:
                pm_all = torch.cat(pm_list, dim=0)
                gm_all = torch.cat(gm_list, dim=0)
                iou_loss = complete_box_iou_loss(
                    xywh_to_xyxy(pm_all), xywh_to_xyxy(gm_all), reduction="mean"
                ).mean()
            else:
                iou_loss = logits.new_tensor(0.0)

            match_rate = matched / total if total else 0.0
            return iou_loss, match_rate

        # --------------------------------------------------------------
        # Slow fallback – per-sample SciPy/Torch helper
        # --------------------------------------------------------------

        pm_list: list[torch.Tensor] = []
        gm_list: list[torch.Tensor] = []
        matched, total = 0, 0

        for b in range(B):
            pred = diff_preds[b]
            gt = gt_boxes[b]
            if isinstance(gt, list):
                gt = torch.tensor(gt, device=logits.device, dtype=logits.dtype)

            # keep only non-degenerate boxes and count them for logging
            pred = pred[(pred[:, 2] > EPSILON) & (pred[:, 3] > EPSILON)]
            self.last_pred_boxes += int(pred.size(0))
            gt   = gt[(gt[:, 2] > EPSILON) & (gt[:, 3] > EPSILON)]
            if pred.numel() == 0 or gt.numel() == 0:
                total += gt.size(0)
                continue

            pairs = hungarian_match(pred, gt)
            if pairs:
                pm_list.append(torch.stack([pred[i] for i, _ in pairs]))
                gm_list.append(torch.stack([gt[j] for _, j in pairs]))
                matched += len(pairs)
            total += gt.size(0)

        if pm_list:
            pm_all = torch.cat(pm_list, dim=0)
            gm_all = torch.cat(gm_list, dim=0)
            iou_loss = complete_box_iou_loss(
                xywh_to_xyxy(pm_all), xywh_to_xyxy(gm_all), reduction="none"
            ).mean()
        else:
            iou_loss = logits.new_tensor(0.0)

        match_rate = matched / total if total else 0.0
        return iou_loss, match_rate

    # -------------------- callable -----------------------------
    def __call__(self, outputs, labels, **kw):
        logits = outputs.logits
        lm_labels = labels
        gt_boxes = kw.get("target_boxes", getattr(outputs, "target_boxes", None))
        if gt_boxes is None:
            gt_boxes = [torch.zeros((0, 4), device=logits.device, dtype=logits.dtype) for _ in range(logits.size(0))]
        vocab = logits.size(-1)

        lm_loss = F.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, vocab),
            lm_labels[..., 1:].contiguous().view(-1),
            ignore_index=-100,
        )
        if not torch.isfinite(lm_loss):
            raise RuntimeError("NaN/Inf in LM loss")

        # ---------------- adaptive detection weighting -----------------
        # Maintain EMA of LM loss to avoid jitter
        if self.lm_loss_ema is None:
            self.lm_loss_ema = lm_loss.detach().item()
        else:
            self.lm_loss_ema = (
                self.smoothing * self.lm_loss_ema
                + (1.0 - self.smoothing) * lm_loss.detach().item()
            )

        # When the EMA of LM loss falls to the `lm_target` value we want the
        # detection branch to carry *15 ×* its base weight (so that
        #   effective_det_weight = lambda_det * det_scale ≈ 0.15 × 15 = 2.25).
        # Earlier in training (lm_loss_ema > lm_target) the scale grows
        # linearly from 1 → 15 and saturates there.
        ratio = self.lm_target / max(self.lm_loss_ema, 1e-4)
        det_scale = max(1.0, min(10.0, 10.0 * ratio))

        iou_loss, match_rate = self._detection_loss(logits, gt_boxes)

        match_pen = self.lambda_match * (1.0 - match_rate)
        det_loss = self.lambda_iou * iou_loss + match_pen
        total_loss = lm_loss + (self.lambda_det * det_scale) * det_loss

        if self.logger and self.is_eval and self.step % self.log_interval == 0:
            sample_pred_ids = logits.argmax(dim=-1)[0].detach().cpu()
            sample_gt_ids = lm_labels[0].detach().cpu()
            pred_str, gt_str = decode_pred_gt(sample_pred_ids, sample_gt_ids, self.tok)
            self.logger.info({"sample_pred": pred_str, "sample_gt": gt_str})
        if self.logger and self.step % self.log_interval == 0:
            log_dict = {
                "step": self.step,
                "loss": self._val(total_loss),
                "lm loss": self._val(lm_loss),
                "iou loss": self._val(iou_loss),
                "pred boxes": self.last_pred_boxes,
                "match rate": round(match_rate, 4),
                "match penalty": round(match_pen.item() if torch.is_tensor(match_pen) else match_pen, 6),
                "det_scale": round(det_scale, 3),
            }
            if self._dangling_total > 0:
                log_dict["dangling_coords"] = int(self._dangling_total)
                self._dangling_total = 0
            self.logger.info(log_dict)
        self.step += 1
        return total_loss

    # ------------------- utils -------------------------
    @staticmethod
    def _val(x):
        return round(x.item(), 6) if torch.is_tensor(x) else x

# -----------------------------------------------------------------------------
# factory ----------------------------------------------------------------------

def create_compute_loss_func(tokenizer, **kw):
    return CompositeLoss(tokenizer, **kw)
