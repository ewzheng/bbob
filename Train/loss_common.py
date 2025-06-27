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


from torch_linear_assignment import batch_linear_assignment, assignment_to_indices  # type: ignore


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
        lambda_cls: float = 1.0,
        lambda_match_penalty: float = 0.5,
        lm_target: float = 1.5,
        smoothing_factor: float = 0.95,
        logger=None,
        log_interval: int = 100,
    ):
        self.tok = tokenizer
        self.lambda_iou = lambda_iou
        self.lambda_det = lambda_detection
        self.lambda_match = lambda_match_penalty
        self.lambda_cls = lambda_cls
        self.lm_target = lm_target
        self.smoothing = smoothing_factor
        self.logger = logger
        self.log_interval = max(1, log_interval)
        self.step = 0
        self.lm_loss_ema: float | None = None
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

        # Maximum multiplier applied to the detection loss once the language
        # modelling branch has converged.  15 keeps previous behaviour
        #   (λ_det≈0.15  × 15 ≈ 2.25 effective weight).
        self.det_scale_max: float = 15.0

        # ------------------------------------------------------------------
        # Internal buffers that are re-used across forward passes to avoid
        # frequent cudaMallocs / cpu->gpu allocations.
        # ------------------------------------------------------------------
        self._cost_buf: torch.Tensor | None = None  # (B, P_max, G_max)

        # Cache digit-id tensor per device to avoid `.to(device)` every step
        self._digit_ids_cache: dict[torch.device, torch.Tensor] = {}

        self._id_colon = self._get_id(":")
        self._id_comma = self._get_id(",")
        self._id_dot   = self._get_id(".")
        self._id_lbr = self._get_id("[")
        self._id_rbr = self._get_id("]")
        self._id_space = self._get_id(" ")

        # Tag token IDs – needed by __call__ masking before any forward pass
        self._id_open = tokenizer.convert_tokens_to_ids(TAG_OPEN)
        self._id_close = tokenizer.convert_tokens_to_ids(TAG_CLOSE)

    # Punctuation tokens used inside <bbob> snippets.  If the tokenizer
    # does not contain a dedicated token we store -1 and simply skip the
    # check when scanning.
    def _get_id(self, tok: str) -> int:
        tid = self.tok.convert_tokens_to_ids(tok)
        return tid if tid is not None else -1
    
    # ------------- straight-through digit expectation ----------------
    def _st_expect(self, logits_slice: torch.Tensor) -> torch.Tensor:
        device, dtype = logits_slice.device, logits_slice.dtype
        ids = self.digit_ids.to(device)
        if (ids >= logits_slice.size(-1)).any() or (ids < 0).any():
            warnings.warn("Digit token IDs exceed vocab range; falling back to uniform random coords and disabling ST path.")
            return torch.rand(logits_slice.shape[:-1], device=device, dtype=dtype)

        dl = logits_slice.index_select(-1, ids)
        prog = min(1.0, self.step / max(1, self.tau_steps))
        tau = max(self.tau_end, self.tau_start * (1 - prog))
        y_soft = F.gumbel_softmax(dl.clamp(-10, 10), tau=tau, hard=False, dim=-1)
        if not torch.isfinite(y_soft).all():
            y_soft = torch.softmax(dl / max(tau, 0.1), dim=-1)

        # Faster straight-through path that avoids host <-> device sync by
        # staying entirely on GPU and using `scatter_` instead of argmax →
        # item() extraction.
        idx = y_soft.argmax(-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, idx, 1.0)
        y_st = y_hard - y_soft.detach() + y_soft
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

    # ---------------- helper: lazy device copy of digit ids -----------------
    def _digit_ids_on(self, device: torch.device) -> torch.Tensor:
        if device not in self._digit_ids_cache:
            self._digit_ids_cache[device] = self.digit_ids.to(device)
        return self._digit_ids_cache[device]

    # ---------------- helper: mask tokens inside <bbob> tags ----------------
    def _inside_mask(self, row: torch.Tensor) -> torch.Tensor:
        """Return boolean mask for positions between <bbob> tags (exclusive)."""
        open_mask = row == self._id_open
        close_mask = row == self._id_close
        depth = torch.cumsum(open_mask.int() - close_mask.int(), dim=0)
        inside = depth > 0
        # exclude the tag tokens themselves
        return inside & ~(open_mask | close_mask)

    # ---------------- helper: parse coordinates from one sample ----------------
    def _parse_coords_sample(self, token_row: torch.Tensor, logits_row: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]]]:
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
            return logits_row.new_zeros((0, 4)), logits_row.new_zeros((0,), dtype=torch.long), logits_row.new_zeros((0,), dtype=torch.long), []

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
        # Keep computations in the same precision as the logits (fp16 / bf16 / fp32)
        scale_tensor = self._pow10_tensor(logits_row.device, dig_vals.dtype)
        coord_vals = (dig_vals * scale_tensor).sum(-1) / scale_tensor.new_tensor(COORD_SCALE)
        coord_vals = coord_vals.clamp_(0.0, 1.0)

        # ------------------------------------------------------------------
        # 4. Reshape to (N, 4) box tensor.  Any leftover <4 coords were already
        #    accounted for in the dangling counter above.
        # ------------------------------------------------------------------
        if coord_vals.numel() >= 4:
            n_boxes = coord_vals.numel() // 4

            # ---------------- label extraction -----------------------
            # Heuristic: take the first *non-punctuation, non-digit* token
            # that appears to the *left* of the first digit of each box.
            # This matches the typical   "<bbob> cat : 0 0 1 1 ..."  layout.

            lbl_ids: list[int] = []
            lbl_pos: list[int] = []          # last token pos per class (for scalar id)
            lbl_pos_seq: list[list[int]] = []  # full token-position list per box
            digits_per_box = NUM_DIGITS * 4
            for i in range(n_boxes):
                first_digit_pos = positions[i * digits_per_box].item()
                j = first_digit_pos - 1
                pos_seq: list[int] = []
                tok_seq: list[int] = []
                while j >= 0:
                    tid = token_row[j].item()
                    # Stop only on *hard* delimiters (digit, ':' or tag).
                    # Soft punctuation such as space/comma/dot are skipped
                    # but do *not* terminate the scan so we can capture
                    # full phrases like "traffic light" when the tokenizer
                    # emits an explicit space token between the sub-tokens.
                    if tid in self.digit_set or tid == self._id_colon or tid in {id_open, id_close}:
                        break  # reached the class/coord separator or tag

                    # Ignore soft punctuation but keep scanning.
                    if tid in {self._id_dot, self._id_comma, self._id_space}:
                        j -= 1
                        continue

                    pos_seq.append(j)
                    tok_seq.append(tid)
                    j -= 1

                pos_seq.reverse()  # maintain natural L→R order
                tok_seq.reverse()

                from Utils.class_id_map import get_id  # local import to avoid circular deps
                label_id = get_id(tok_seq) if tok_seq else -1

                lbl_ids.append(label_id)
                lbl_pos.append(pos_seq[-1] if pos_seq else -1)
                lbl_pos_seq.append(pos_seq)

            labels = logits_row.new_tensor(lbl_ids, dtype=torch.long)
            positions_cls = logits_row.new_tensor(lbl_pos, dtype=torch.long)

            # Drop boxes where we could not identify a class label (label == -1)
            keep = labels >= 0
            kept_positions_seq = [p for k, p in zip(keep.tolist(), lbl_pos_seq) if k]

            return (
                coord_vals[: n_boxes * 4].view(-1, 4)[keep],
                labels[keep],
                positions_cls[keep],
                kept_positions_seq,
            )

        return (
            logits_row.new_zeros((0, 4)),
            logits_row.new_zeros((0,), dtype=torch.long),
            logits_row.new_zeros((0,), dtype=torch.long),
            [],
        )

    # ---------------- helper: parse coords per sample ----------------

   # ------------- detection loss (vectorised coords) ----------------
    def _detection_loss(self, logits: torch.Tensor, gt_boxes):
        """Compute IoU loss & match-rate in a batched, GPU-friendly way."""

        # ------------------------------------------------------------------
        # Full-vocabulary straight-through Gumbel-Softmax
        # ------------------------------------------------------------------
        # We replace the hard arg-max with an ST Gumbel sample so that *all*
        # vocabulary logits inside <bbob> spans receive gradients.  The forward
        # pass still feeds discrete token IDs to the downstream masking and
        # coordinate-parsing logic, therefore Hungarian matching & IoU code
        # remain unchanged.

        B = logits.size(0)

        # Anneal the temperature with the same schedule used by _st_expect
        prog = min(1.0, self.step / max(1, self.tau_steps))
        tau  = max(self.tau_end, self.tau_start * (1 - prog))

        # (B,S,V) → soft probabilities (back-prop) + hard one-hot (forward)
        y_soft = F.gumbel_softmax(logits.clamp(-10, 10), tau=tau, hard=False, dim=-1)
        idx    = y_soft.argmax(-1, keepdim=True)                 # (B,S,1)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, idx, 1.0) # hard sample

        token_ids = idx.squeeze(-1)                              # (B,S) hard IDs

        # --------------------------------------------------------------
        # Restrict gradient-carrying path to tokens *inside* <bbob> spans
        # --------------------------------------------------------------
        inside_masks = [self._inside_mask(token_ids[b]) for b in range(B)]
        inside_mask  = torch.stack(inside_masks, dim=0)          # (B,S) bool
        mask_f       = inside_mask.unsqueeze(-1).type_as(logits) # broadcast to vocab dim

        # ST trick limited to inside tokens; outside we keep pure hard one-hot
        logits = y_hard - mask_f * y_soft.detach() + mask_f * y_soft

        # Reset counter for this step
        self.last_pred_boxes = 0

   
        diff_preds: List[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]]]] = [
            self._parse_coords_sample(token_ids[b], logits[b]) for b in range(B)
        ]

        # --------------------------------------------------------------
        # Fast path – use torch_linear_assignment in batch on GPU
        # --------------------------------------------------------------
        pred_lens = [p[0].size(0) for p in diff_preds]
        gt_lens   = [len(g) if isinstance(g, list) else g.size(0) for g in gt_boxes]

        max_pred = max(pred_lens) if pred_lens else 0
        max_gt   = max(gt_lens)   if gt_lens else 0

        if max_pred == 0 or max_gt == 0:
            return logits.new_tensor(0.0), 0.0, logits.new_tensor(0.0)

        # --------------------------------------------------------------
        # Allocate / grow a persistent cost buffer once to reduce
        # allocation overhead.  Keep dtype identical to `logits` so that
        # mixed-precision runs avoid implicit casting.
        # --------------------------------------------------------------
        if (
            self._cost_buf is None
            or self._cost_buf.size(0) < B
            or self._cost_buf.size(1) < max_pred
            or self._cost_buf.size(2) < max_gt
            or self._cost_buf.dtype != logits.dtype
            or self._cost_buf.device != logits.device
        ):
            # Grow dimensions by 1.5× to cut realloc frequency
            new_B = max(B, int((self._cost_buf.size(0) if self._cost_buf is not None else 0) * 1.5))
            new_P = max(max_pred, int((self._cost_buf.size(1) if self._cost_buf is not None else 0) * 1.5))
            new_G = max(max_gt, int((self._cost_buf.size(2) if self._cost_buf is not None else 0) * 1.5))
            self._cost_buf = logits.new_empty((new_B, new_P, new_G))

        cost_batch = self._cost_buf[:B, :max_pred, :max_gt]
        cost_batch.fill_(1e6)  # prohibitively high cost for cross-class pairs

        # Caches for filtered boxes so we don't recompute in the second
        # loop.
        cached_pred: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]], torch.Tensor | None]] = []
        cached_gt:   list[tuple[torch.Tensor, torch.Tensor]] = []

        for b in range(B):
            pred, pred_lbl_full, pred_pos_full, pred_pos_seq = diff_preds[b]

            # Ground-truth may come as (boxes, labels) tuple or just boxes
            gt, gt_lbl_full = gt_boxes[b]  # collator guarantees tuple of tensors
            gt = gt.to(device=logits.device, dtype=logits.dtype)
            gt_lbl_full = gt_lbl_full.to(device=logits.device, dtype=torch.long)

            # Filter non-degenerate boxes and propagate class labels
            valid_pred = (pred[:, 2] > EPSILON) & (pred[:, 3] > EPSILON)
            pred_f = pred[valid_pred]
            pred_lbl = pred_lbl_full[valid_pred] if pred_lbl_full.numel() else pred_lbl_full.new_zeros((0,))

            valid_gt = (gt[:, 2] > EPSILON) & (gt[:, 3] > EPSILON)
            gt_f   = gt[valid_gt]
            gt_lbl = gt_lbl_full[valid_gt] if gt_lbl_full.numel() else gt_lbl_full.new_zeros((0,))

            pred_pos = pred_pos_full[valid_pred]

            if pred_pos_seq:
                valid_idx = valid_pred.nonzero(as_tuple=False).squeeze(1).tolist()
                pred_pos_seq = [pred_pos_seq[i] for i in valid_idx]
            # Preconvert to tensors once to cut kernel launches later
            pred_pos_seq_t = [torch.tensor(p, device=logits.device, dtype=torch.long) for p in pred_pos_seq]
            cached_pred.append((pred_f, pred_lbl, pred_pos, pred_pos_seq_t, iou_matrix_xywh(pred_f, gt_f) if pred_f.numel() and gt_f.numel() else None))
            cached_gt.append((gt_f, gt_lbl))

            self.last_pred_boxes += int(pred_f.size(0))

            if pred_f.numel() and gt_f.numel():
                # After all filtering, compute IoU & distance matrices once
                iou_mat = iou_matrix_xywh(pred_f, gt_f)              # (P,G)
                # distance matrix will be computed per-class on demand;
                # avoid allocating a large (P×G) tensor here only to discard it.

                # Fill cost matrix per class to forbid cross-class matches
                # Iterate only over GT classes to avoid scanning the whole
                # vocabulary when the model emits spurious class IDs that do
                # not appear in the ground truth.
                unique_classes = torch.unique(gt_lbl)
                for cls_id in unique_classes.tolist():
                    if cls_id < 0:
                        continue  # skip placeholder / invalid class ids
                    p_idx = (pred_lbl == cls_id).nonzero(as_tuple=False).squeeze(1)
                    g_idx = (gt_lbl == cls_id).nonzero(as_tuple=False).squeeze(1)
                    if p_idx.numel() == 0 or g_idx.numel() == 0:
                        continue  # nothing to match for this class

                    # Compute IoU/dist only for the selected rows/columns to
                    # guarantee shape alignment regardless of earlier
                    # filtering.
                    sub_pred = pred_f[p_idx]
                    sub_gt   = gt_f[g_idx]

                    sub_iou  = iou_matrix_xywh(sub_pred, sub_gt)               # (P',G')
                    sub_dist = torch.cdist(sub_pred[:, :2], sub_gt[:, :2])      # (P',G')
                    sub_dist = (sub_dist / math.sqrt(2.0)).clamp_(max=1.0)

                    sub_cost = torch.where(
                        sub_iou > 0,
                        1.0 - sub_iou,
                        0.99 + sub_dist * 0.009,
                    )

                    cost_batch[b][p_idx[:, None], g_idx[None, :]] = sub_cost

        assignment = batch_linear_assignment(cost_batch)  # (B, max_pred)

        pm_list, gm_list = [], []
        matched, total = 0, 0
        # Accumulate class-token positions & targets for *all* samples so we
        # can run one large gather/CE call instead of many small ones.
        cls_logits_all: list[torch.Tensor] = []
        cls_tgt_all: list[torch.Tensor] = []

        for b in range(B):
            # Retrieve cached filtered tensors (computed in the first loop)
            pred_f, pred_lbl, pred_pos, pred_pos_seq, iou_mat = cached_pred[b]
            gt_f, gt_lbl = cached_gt[b]

            P, G = pred_f.size(0), gt_f.size(0)
            if P == 0 or G == 0:
                total += G  # still count GT objects even if no prediction passes
                continue

            # (P,G) centre-distance matrix (re-use later for fallback)
            dist = torch.cdist(pred_f[:, :2], gt_f[:, :2])  # (P,G)

            assign_vec = assignment[b, :P].clone()  # (P,) each ∈ [0,G-1] or −1

            # -------- nearest fallback restricted to same-class ----------
            for pi in (assign_vec == -1).nonzero(as_tuple=False).flatten():
                cls_id = pred_lbl[pi]
                mask_same = gt_lbl == cls_id
                if mask_same.any():
                    g_candidates = torch.nonzero(mask_same, as_tuple=False).squeeze(1)
                    nearest_idx = g_candidates[dist[pi, g_candidates].argmin()]
                    assign_vec[pi] = nearest_idx
                else:
                    # No GT of same class → drop this prediction from loss
                    assign_vec[pi] = -1

            keep_pred_mask = assign_vec >= 0

            # ----------------------------------------------------------------
            # Guard against rare cases where the cached `pred_pos` indices are
            # out of the valid \[0, seq_len) range due to upstream token
            # truncation or mis-alignment.  Using such indices with advanced
            # indexing on the logits tensor would trigger a CUDA assert
            # ("index out of bounds") which is hard to debug once it bubbles
            # up from inside the ATen kernels.  We proactively mask them out
            # so training can continue seamlessly while still keeping IoU /
            # matching statistics intact.
            # ----------------------------------------------------------------
            seq_len = logits.size(1)
            valid_pos_mask = (pred_pos >= 0) & (pred_pos < seq_len)

            # Guard against assignment indices outside the GT range
            valid_assign_mask = (assign_vec >= 0) & (assign_vec < G)

            # Combine all validity checks
            keep_pred_mask = keep_pred_mask & valid_pos_mask & valid_assign_mask

            # If *any* invalid combination existed we silently skip them. This
            # is strictly defensive and should be extremely rare once the
            # model starts producing sensible outputs.

            # Nothing left once we removed the invalid positions
            if keep_pred_mask.any():
                pm_list.append(pred_f[keep_pred_mask])            # (K,4)
                gm_list.append(gt_f[assign_vec[keep_pred_mask]])   # (K,4)
                # -------- gather logits & targets for *all* class tokens ----
                sel = torch.arange(P, device=pred_f.device)[keep_pred_mask]
                if sel.numel():
                    pos_tensors = [pred_pos_seq[i] for i in sel if pred_pos_seq[i]]
                    # Remove out-of-range values from each tensor *before* we concatenate –
                    # this is cheaper than building one big mask afterwards and guards
                    # against empty tensors producing shape mismatches.
                    if pos_tensors:
                        seq_len_b = logits.size(1)
                        pos_tensors = [pt[(pt >= 0) & (pt < seq_len_b)] for pt in pos_tensors if pt.numel()]
                        pos_tensors = [pt for pt in pos_tensors if pt.numel()]
                    if pos_tensors:
                        pos_tensor = torch.cat(pos_tensors)  # (M,)
                        logits_seq = logits[b].index_select(0, pos_tensor)  # (M, V)
                        tgt_tensor = self._lm_labels_batch[b].index_select(0, pos_tensor)  # (M,)

                        # Filter valid vocabulary targets
                        valid_mask = (tgt_tensor >= 0) & (tgt_tensor < logits.size(-1))
                        if valid_mask.any():
                            cls_logits_all.append(logits_seq[valid_mask])
                            cls_tgt_all.append(tgt_tensor[valid_mask])
                # If *no* valid class targets remain we skip CE for this sample
                # but still keep IoU / match stats (they are unaffected).

                if iou_mat is None:
                    iou_mat = iou_matrix_xywh(pred_f, gt_f)
                ious_sel = iou_mat[torch.arange(P, device=pred_f.device)[keep_pred_mask], assign_vec[keep_pred_mask]]
                cls_match = pred_lbl[keep_pred_mask] == gt_lbl[assign_vec[keep_pred_mask]]
                matched += int(((ious_sel > 0) & cls_match).sum().item())
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
        # ---------------- compute class CE once for the whole batch -------------
        if cls_logits_all:
            logits_cat = torch.cat(cls_logits_all, dim=0)
            tgt_cat    = torch.cat(cls_tgt_all,  dim=0)

            cls_loss = F.cross_entropy(logits_cat, tgt_cat, reduction="mean")
        else:
            cls_loss = logits.new_tensor(0.0)

        # (Unreachable) – code continues beyond this point

    # ---------------- helper: optionally shrink cost buffer --------------
        needed_P, needed_G = max_pred, max_gt
        if (
            self._cost_buf is not None
            and self._cost_buf.size(1) > 2 * needed_P
            and self._cost_buf.size(2) > 2 * needed_G
        ):
            # Reclaim VRAM: drop the oversized buffer – it will be
            # reallocated lazily on the next forward pass.
            self._cost_buf = None

        return iou_loss, match_rate, cls_loss

    # -------------------- callable -----------------------------
    def __call__(self, outputs, labels, **kw):
        logits = outputs.logits
        lm_labels = labels
        # Store a detached copy of labels so _detection_loss can access
        # ground-truth class tokens without worrying about masking.
        self._lm_labels_batch = lm_labels.detach()
        gt_boxes = kw.get("target_boxes", getattr(outputs, "target_boxes", None))
        if gt_boxes is None:
            self.logger.info("No ground truth boxes found, creating empty placeholder")
            # Create geometry-only placeholder tuples so downstream code can
            # treat every entry uniformly.
            empty_box = torch.zeros((0, 4), device=logits.device, dtype=logits.dtype)
            empty_lbl = torch.zeros((0,), device=logits.device, dtype=torch.long)
            gt_boxes = [(empty_box, empty_lbl) for _ in range(logits.size(0))]
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

        # ---------------- logarithmic growth of detection scale ----------------
        # Let   ratio = lm_target / lm_loss_ema   ∈ (0, +∞)
        # Clamp to 0‥1 so that the scale saturates once the CE loss reaches
        # the target.  A logarithmic mapping makes the curve flatter early on
        # (CE dominates) and steeper close to convergence where detection
        # feedback is most useful.

        ratio = self.lm_target / max(self.lm_loss_ema, 1e-6)
        progress = min(1.0, ratio)  # 0 → far from target, 1 → reached target

        # Map progress ∈ [0,1] ↦ det_scale ∈ [1, det_scale_max] using
        #   f(x) = 1 + (M-1) * log10(1 + 9x)
        # so that f(0)=1, f(1)=M, and growth accelerates as x→1.
        det_scale = 1.0 + (self.det_scale_max - 1.0) * math.log10(1.0 + 9.0 * progress)

        iou_loss, match_rate, cls_loss = self._detection_loss(logits, gt_boxes)

        # Make sure all loss components share the same dtype to avoid implicit
        # casts when mixed-precision / autocast is enabled (saves memory and
        # keeps gradient scaling straightforward).
        if torch.is_tensor(iou_loss) and iou_loss.dtype != lm_loss.dtype:
            iou_loss = iou_loss.to(lm_loss.dtype)
        if torch.is_tensor(cls_loss) and cls_loss.dtype != lm_loss.dtype:
            cls_loss = cls_loss.to(lm_loss.dtype)

        match_pen = self.lambda_match * (1.0 - match_rate)
        det_loss = self.lambda_iou * iou_loss + match_pen

        # ------------------------------------------------------------------
        # Dynamically balance LM ↔ detection:  lm_loss weight = 1/det_scale
        # – Early training:   det_scale≈1  ⇒ lm_loss keeps full strength.
        # – Late training:    det_scale→det_scale_max  ⇒ lm_loss fades while
        #                      detection gets amplified (λ_det·det_scale).
        # ------------------------------------------------------------------
        lm_weight = 1.0 / det_scale
        total_loss = lm_weight * lm_loss + self.lambda_cls * cls_loss + (self.lambda_det * det_scale) * det_loss

        if self.logger and self.step % (self.log_interval * 4) == 0:
            sample_pred_ids = logits.argmax(dim=-1)[0].detach().cpu()
            sample_gt_ids = lm_labels[0].detach().cpu()
            pred_str, gt_str = decode_pred_gt(sample_pred_ids, sample_gt_ids, self.tok)
            self.logger.info({"sample_pred": pred_str, "sample_gt": gt_str})
        if self.logger and self.step % self.log_interval == 0:
            log_dict = {
                "step": self.step,
                "loss": self._val(total_loss),
                "lm loss": self._val(lm_loss),
                "cls loss": self._val(cls_loss),
                "iou loss": self._val(iou_loss),
                "pred boxes": self.last_pred_boxes,
                "match rate": round(match_rate, 4),
                "match penalty": round(match_pen.item() if torch.is_tensor(match_pen) else match_pen, 6),
                "det_scale": round(det_scale * self.lambda_det, 3),
                "lm_scale": round(lm_weight, 3),
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
