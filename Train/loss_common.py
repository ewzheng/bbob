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
from torchvision.ops import complete_box_iou_loss as _complete_box_iou_loss  # type: ignore
import warnings

from .loss_helpers import (
    TAG_OPEN,
    TAG_CLOSE,
    EPSILON,
    hungarian_match,
    xywh_to_xyxy,
    decode_pred_gt,
    NUM_DIGITS,
    COORD_SCALE,
)

# -----------------------------------------------------------------------------
# Composite loss – vectorised implementation
# -----------------------------------------------------------------------------

class CompositeLoss:
    def __init__(
        self,
        tokenizer,
        *,
        lambda_iou: float = 1.5,
        lambda_detection: float = 0.15,
        lm_target: float = 1.5,
        smoothing_factor: float = 0.95,
        logger=None,
        log_interval: int = 100,
    ):
        self.tok = tokenizer
        self.lambda_iou = lambda_iou
        self.lambda_det = lambda_detection
        self.lm_target = lm_target
        self.smoothing = smoothing_factor
        self.logger = logger
        self.log_interval = max(1, log_interval)
        self.step = 0
        self.lm_loss_ema: float | None = None
        self.is_eval = False

        # digit/ST cache
        digit_ids = [tokenizer.convert_tokens_to_ids(str(i)) for i in range(10)]
        self.digit_ids = torch.tensor(digit_ids, dtype=torch.long)
        if any(i == tokenizer.unk_token_id or i == -1 for i in digit_ids):
            raise ValueError("Tokenizer must contain explicit '0'..'9' tokens")
        self.digit_set = set(digit_ids)
        self._values10: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._pow10: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self.tau_start, self.tau_end, self.tau_steps = 2.0, 0.1, 10_000

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
    def _parse_coords_sample(self, token_row: list[int], logits_row: torch.Tensor) -> torch.Tensor:
        """Extract `(N,4)` xywh tensor for a single sample using ST estimator.

        Clamps each coordinate to [0,1] and guarantees proper shape even when
        no valid boxes were decoded.
        """
        # Cache tag IDs for speed
        if not hasattr(self, "_id_open"):
            self._id_open = self.tok.convert_tokens_to_ids(TAG_OPEN)
            self._id_close = self.tok.convert_tokens_to_ids(TAG_CLOSE)
        id_open: int = self._id_open  # type: ignore[attr-defined]
        id_close: int = self._id_close  # type: ignore[attr-defined]

        coords: list[float] = []
        buf: list[int] = []
        i = 0
        while True:
            try:
                st = token_row.index(id_open, i)
                ed = token_row.index(id_close, st + 1)
            except ValueError:
                break

            for pos in range(st + 1, ed):
                if token_row[pos] in self.digit_set:
                    buf.append(pos)
                    if len(buf) == NUM_DIGITS:
                        slice_logits = logits_row[buf]  # fancy index, no new tensor
                        dig_vals = self._st_expect(slice_logits)
                        coord_val = (dig_vals * self._pow10_tensor(logits_row.device, dig_vals.dtype)).sum() / COORD_SCALE
                        coords.append(float(torch.clamp(coord_val, 0.0, 1.0).item()))
                        buf.clear()
            i = ed + 1

        # ensure multiple of 4 values
        if len(coords) % 4:
            if self.logger:
                self.logger.warning("Dropped %d dangling coord values", len(coords) % 4)
            coords = coords[: len(coords) // 4 * 4]

        if coords:
            return torch.tensor(coords, device=logits_row.device, dtype=logits_row.dtype).view(-1, 4)
        return logits_row.new_zeros((0, 4))

    # ---------------- helper: parse coords per sample ----------------

    # ------------- detection loss (vectorised coords) ----------------
    def _detection_loss(self, logits: torch.Tensor, gt_boxes):
        B = logits.size(0)

        token_ids = logits.argmax(dim=-1)
        diff_preds: List[torch.Tensor] = [
            self._parse_coords_sample(token_ids[b].tolist(), logits[b]) for b in range(B)
        ]

        iou_vals = []
        matched, total = 0, 0
        for b in range(B):
            pred = diff_preds[b]
            gt = gt_boxes[b]
            if isinstance(gt, list):
                gt = torch.tensor(gt, device=logits.device, dtype=logits.dtype)
            pred = pred[(pred[:, 2] > EPSILON) & (pred[:, 3] > EPSILON)]
            gt = gt[(gt[:, 2] > EPSILON) & (gt[:, 3] > EPSILON)]
            if pred.numel() == 0 or gt.numel() == 0:
                continue
            pairs = hungarian_match(pred, gt)
            if not pairs:
                continue
            pm = torch.stack([pred[i] for i, _ in pairs])
            gm = torch.stack([gt[j] for _, j in pairs])
            ciou = _complete_box_iou_loss(xywh_to_xyxy(pm), xywh_to_xyxy(gm), reduction="none").mean()
            iou_vals.append(ciou)
            matched += len(pairs)
            total += gt.size(0)

        iou_loss = torch.stack(iou_vals).mean() if iou_vals else logits.new_tensor(0.0)
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

        iou_loss, match_rate = self._detection_loss(logits, gt_boxes)
        det_loss = self.lambda_iou * iou_loss
        total_loss = lm_loss + self.lambda_det * det_loss

        if self.is_eval and self.logger is not None:
            sample_pred_ids = logits.argmax(dim=-1)[0].detach().cpu()
            sample_gt_ids = lm_labels[0].detach().cpu()
            pred_str, gt_str = decode_pred_gt(sample_pred_ids, sample_gt_ids, self.tok)
            self.logger.info({"sample_pred": pred_str, "sample_gt": gt_str})

        if self.logger and self.step % self.log_interval == 0:
            self.logger.info({
                "step": self.step,
                "loss": self._val(total_loss),
                "lm": self._val(lm_loss),
                "iou": self._val(iou_loss),
                "match": round(match_rate, 4),
            })
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
