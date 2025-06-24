import torch
import torch.nn.functional as F
import re
import math  # for isnan / isfinite checks

from scipy.optimize import linear_sum_assignment  # type: ignore
from torchvision.ops import box_iou as _box_iou  # type: ignore
from torchvision.ops import complete_box_iou_loss as _complete_box_iou_loss  # type: ignore

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

EPSILON: float = 1e-6       # small value to avoid division-by-zero
FMT_OK_THRESHOLD: float = 0.25  # consider a box "well-formed" when fmt_err < 0.25

# ----------------------------------------------------------------------
# Tag tokens
# ----------------------------------------------------------------------

TAG_OPEN = "<|bbob|>"
TAG_CLOSE = "</|bbob|>"
TAG_PATTERN = re.compile(re.escape(TAG_OPEN) + r"(.*?)" + re.escape(TAG_CLOSE))

def _val(x: torch.Tensor | float | int):
    """Utility to convert tensors to Python scalars for logging."""
    if isinstance(x, torch.Tensor):
        return round(x.item(), 6)
    return x

def _parse_boxes(logits, tokenizer):
    """Extract detection strings without decoding whole sequences.

    Strategy
    --------
    1. Arg-max over vocab → token-id matrix (B, S).
    2. Locate indices of the special tokens "<bbob>" and "</bbob>".
    3. Slice *only* the sub-sequences between those tag IDs.
    4. Batch-decode that much shorter list of snippets.

    The rest of the sequence (instruction, filler tokens) is never
    converted back to text, eliminating most of the Python/string cost.
    """

    if logits.dim() != 3:
        raise ValueError(
            f"Expected logits with shape (batch, seq_len, vocab) but got {logits.shape}"
        )

    token_ids = logits.argmax(dim=-1)  # (B, S)

    # Get tag IDs once
    id_open, id_close = tokenizer.convert_tokens_to_ids([TAG_OPEN, TAG_CLOSE])

    all_snippets: list[list[int]] = []
    snippet_owner: list[int] = []  # which batch index produced each snippet

    ids_cpu = token_ids.cpu()  # single transfer

    for b, row in enumerate(ids_cpu.tolist()):
        i = 0
        while True:
            try:
                start = row.index(id_open, i)
            except ValueError:
                break  # no more opening tags in this sequence

            # Look for the corresponding closing tag. If not found,
            # add an *empty* snippet so that downstream format_loss = 1.0,
            # then stop scanning this sequence (no more tags can be valid).
            try:
                end = row.index(id_close, start + 1)
            except ValueError:
                all_snippets.append([])  # unmatched opening tag → fmt_err = 1
                snippet_owner.append(b)
                break

            # When both tags are present collect the snippet if non-empty
            if end - start > 1:
                snippet = row[start + 1 : end]
            else:
                snippet = []  # empty snippet, still yields fmt_err = 1
            all_snippets.append(snippet)
            snippet_owner.append(b)
            i = end + 1

    # Decode only the snippets (they are short → negligible cost)
    if all_snippets:
        decoded = tokenizer.batch_decode(all_snippets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    else:
        decoded = []

    detections_batch: list[list[str]] = [[] for _ in range(token_ids.size(0))]
    for owner, txt in zip(snippet_owner, decoded):
        detections_batch[owner].append(txt.strip())

    return detections_batch

def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert `[..., x,y,w,h]` to `[..., x1,y1,x2,y2]`."""
    x, y, w, h = boxes.unbind(-1)
    return torch.stack((x, y, x + w, y + h), -1)

# ----------------------------------------------------------------------
# Vectorised IoU helpers (Torchvision fallback to our previous impl)
# ----------------------------------------------------------------------

def _iou_matrix_xywh(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pair-wise IoU matrix using Torchvision for speed if available."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    if _box_iou is not None:
        return _box_iou(_xywh_to_xyxy(boxes1), _xywh_to_xyxy(boxes2))

    # fallback to manual implementation (previous version)
    b1_x1 = boxes1[:, 0].unsqueeze(1)
    b1_y1 = boxes1[:, 1].unsqueeze(1)
    b1_x2 = (boxes1[:, 0] + boxes1[:, 2]).unsqueeze(1)
    b1_y2 = (boxes1[:, 1] + boxes1[:, 3]).unsqueeze(1)

    b2_x1 = boxes2[:, 0].unsqueeze(0)
    b2_y1 = boxes2[:, 1].unsqueeze(0)
    b2_x2 = (boxes2[:, 0] + boxes2[:, 2]).unsqueeze(0)
    b2_y2 = (boxes2[:, 1] + boxes2[:, 3]).unsqueeze(0)

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, 2] * boxes1[:, 3]).unsqueeze(1)
    area2 = (boxes2[:, 2] * boxes2[:, 3]).unsqueeze(0)

    union = area1 + area2 - inter_area + EPSILON
    return inter_area / union

def _match_boxes_hungarian(pred, gt):
    """Return index pairs using Hungarian algorithm on IoU cost."""
    if pred.numel() == 0 or gt.numel() == 0:
        return []

    # Filter out degenerate (zero-area) boxes before matching
    pred_mask = (pred[:, 2] > EPSILON) & (pred[:, 3] > EPSILON)
    gt_mask   = (gt[:, 2] > EPSILON) & (gt[:, 3] > EPSILON)

    if not pred_mask.any() or not gt_mask.any():
        return []

    pred = pred[pred_mask]
    gt   = gt[gt_mask]

    iou_mat = _iou_matrix_xywh(pred, gt)
    # Replace possible NaNs (can arise from zero-area boxes) with zeros
    iou_mat = torch.nan_to_num(iou_mat, nan=0.0, posinf=0.0, neginf=0.0)

    # Convert to cost matrix suitable for SciPy (detach from graph, move to CPU, NumPy array).
    # Note: Detaching is essential because SciPy will internally call .numpy();
    # calling that on a tensor that requires gradients raises an error.
    cost = (1.0 - iou_mat + EPSILON)

    # SciPy operates on NumPy arrays; ensure the tensor is on CPU, detached, and uses a supported dtype.
    cost_np = cost.detach().cpu().float().numpy()

    if cost_np.size == 0:
        return []

    # Perform Hungarian matching on the NumPy cost matrix.
    row_ind, col_ind = linear_sum_assignment(cost_np)
    return [(int(r), int(c)) for r, c in zip(row_ind.tolist(), col_ind.tolist())]

class CompositeLoss:
    """Compute combined language-model and detection losses.

    The total loss is:
        L = w_lm · L_lm + λ_det · (λ_iou · (1 − CIoU) + λ_fmt · format_error)

    where the auxiliary terms are only added when ground-truth boxes are
    provided. Hungarian matching is used to assign predictions to GT boxes.
    """

    def __init__(
        self,
        tokenizer,
        *,
        lambda_iou: float = 1.5,
        lambda_detection: float = 0.15,
        lambda_format: float = 0.5,
        lm_target: float = 1.5,
        smoothing_factor: float = 0.95,
        logger=None,
        log_interval: int = 100,
    ):
        self.tokenizer = tokenizer
        self.lambda_iou = lambda_iou
        self.lambda_detection = lambda_detection
        self.lambda_format = lambda_format
        
        # Curriculum parameters
        self.lm_target = lm_target
        # scalars for adaptive detection loss weight
        self.min_detection_weight = 0.1
        self.max_detection_weight = 15
        self.smoothing_factor = smoothing_factor
        
        # Tracking variables
        self.lm_loss_ema = None  # Exponential moving average of LM loss
        self.step_count = 0

        # Optional external logger
        self.logger = logger
        self._log_interval = max(1, log_interval)

        # Store latest curriculum progress for logging
        self.progress_lm: float | None = None
        self.progress_fmt: float | None = None

        # --------------------------------------------------------------
        # Optional straight-through numeric gradient path.
        # Only enable it when the tokenizer contains *distinct* tokens
        # for every integer 0‥999 (i.e. when those tokens were added).
        # When they are missing – typical when we rely on the model to
        # compose numbers from sub-tokens – we fall back to the old
        # non-differentiable detection loss.
        # --------------------------------------------------------------

        # Require digit tokens 0-9; raise if missing
        digit_ids = [tokenizer.convert_tokens_to_ids(str(i)) for i in range(10)]
        # ------------------------------------------------------------------
        # Validate that tokenizer contains all required special tokens and
        # numeric tokens.  Fail fast so training doesn't proceed with an
        # invalid vocabulary that would crash later during indexing.
        # ------------------------------------------------------------------

        required_tokens = [TAG_OPEN, TAG_CLOSE] + [str(i) for i in range(10)]
        missing = [t for t in required_tokens if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
        if missing:
            raise ValueError(f"Tokenizer missing required tokens: {missing}")

        if len(set(digit_ids)) != 10 or -1 in digit_ids or tokenizer.unk_token_id in digit_ids:
            raise RuntimeError("Tokenizer lacks distinct tokens '0'–'9' required for digit-wise straight-through")

        # Store digit token ids on CPU; we create a device-local copy lazily
        self._digit_token_ids = torch.tensor(digit_ids, dtype=torch.long)
        self._digit_token_id_set = set(digit_ids)
        self._enable_digit_st = True

        # Gumbel-Softmax temperature schedule
        self.tau_start = 2.0
        self.tau_end = 0.1
        self.tau_steps = 10_000  # linear decay of Gumbel-Softmax temperature

    def _parse_detection_string(self, det_str):
        """Robustly parse a detection string in the canonical form

            ``<label>: [x, y, w, h]``

        Brackets, commas, and variable whitespace are tolerated; the parser
        extracts up to four floating-point numbers *in order*.  Any missing
        coordinates are imputed with ``0.5`` so that downstream code always
        receives a complete box.

        Returns
        -------
        coords : list[float]
            Always four values, each clamped to the [0,1] range so the rest
            of the loss pipeline can operate without crashing.
        fmt_err : float
            *Format error* in [0,1] – 0 for a perfectly formatted box, 1 for a
            completely malformed one.  We combine two signals:

            1. **Missing components** – +¼ for each of the four numbers that
               could not be parsed;
            2. **Out-of-range values** – the absolute amount of clipping    
               required, averaged over the four coordinates.
        """

        parts = det_str.split(":", 1)
        if len(parts) != 2:
            # Cannot even find the ':' separator → full error, default box.
            return [0, 0, 0, 0], 1.0

        # ------------------------------------------------------------------
        # 0) blank-label penalty – if the substring before ':' has no
        #    non-whitespace characters we add a fixed surcharge.  This
        #    encourages the model to always output a class name.
        # ------------------------------------------------------------------

        BLANK_LABEL_PENALTY = 0.25
        label_str = parts[0].strip()
        blank_label_pen = BLANK_LABEL_PENALTY if len(label_str) == 0 else 0.0

        # ------------------------------------------------------------------
        # 1) extract numeric substrings (robust to commas, multiple spaces…)
        # ------------------------------------------------------------------
        num_strs = re.findall(r"[-+]?\d*\.?\d+", parts[1])

        nums = []
        for s in num_strs[:4]:  # take at most 4 numbers
            try:
                val = float(s)
                # Clamp to [0,1] regardless of how the number was written
                nums.append(max(0.0, min(1.0, val)))
            except ValueError:
                continue

        missing = 4 - len(nums)
        # Fill the rest with a neutral prior (centre-ish square)
        nums.extend([0.5] * missing)

        # ------------------------------------------------------------------
        # 2) clamp to [0,1] and accumulate formatting error
        # ------------------------------------------------------------------
        clip_diffs = []
        clamped = []
        for v in nums:
            v_clamped = max(0.0, min(1.0, v))
            clamped.append(v_clamped)
            clip_diffs.append(abs(v - v_clamped))

        # Missing components add fixed penalty; out-of-range adds proportional
        fmt_err = (missing / 4.0) + (sum(clip_diffs) / 4.0) + blank_label_pen

        # ------------------------------------------------------------------
        # 3) penalise degenerate (zero-area) boxes
        # ------------------------------------------------------------------
        # When either width or height is exactly zero after clamping we add a
        # fixed surcharge to the format error so the model receives an
        # explicit gradient to predict *positive* extents.  The surcharge is
        # chosen so that two consecutive infractions push fmt_err above the
        # FMT_OK_THRESHOLD used elsewhere in the curriculum.

        ZERO_AREA_PENALTY = 0.25  # tuned so one degenerate side = +0.25 fmt
        w, h = clamped[2], clamped[3]
        if w == 0.0 or h == 0.0:
            fmt_err += ZERO_AREA_PENALTY

        # Leave the raw error (can exceed 1 slightly) for analysis; clamp only
        # when computing the loss to keep gradients bounded.

        return clamped, fmt_err

    def _update_lm_loss_tracking(self, current_lm_loss):
        """Update the EMA of LM loss while guarding against NaN/Inf values."""
        lm_loss_value = current_lm_loss.detach().item()

        # --------------------------------------------------------------
        # Skip the update when the loss is not finite to avoid
        # contaminating the EMA with NaN/Inf and breaking the curriculum.
        # --------------------------------------------------------------
        if not math.isfinite(lm_loss_value):
            # Still advance global step to keep schedules in sync
            self.step_count += 1
            return

        if self.lm_loss_ema is None:
            self.lm_loss_ema = lm_loss_value
        else:
            self.lm_loss_ema = (
                self.smoothing_factor * self.lm_loss_ema
                + (1 - self.smoothing_factor) * lm_loss_value
            )

        self.step_count += 1

    def _compute_adaptive_weights(self, parsed_predictions=None):
        """Compute adaptive detection loss weights based on LM performance and format compliance."""
        # Guard against uninitialised or invalid EMA values
        if self.lm_loss_ema is None or not math.isfinite(self.lm_loss_ema):
            return self.min_detection_weight
        
        # ------------------------------------------------------------------
        # Continuous curriculum:
        #   • `progress_lm`   ∈ [0,1]   – how close the EMA loss is to the target.
        #     0 when loss ≥ 2× target,   1 when loss ≤ target.
        #   • `progress_fmt`  ∈ [0,1]   – fraction of boxes that can be parsed,
        #     scaled so that 0.8 compliance → 1.0 (cap at 1).
        #   • Overall progress = progress_lm × progress_fmt.
        #   • Weight  = min + progress × (max − min).   
        # ------------------------------------------------------------------

        # --- language-model progress ---------------------------------------
        lm_ratio = self.lm_target / (self.lm_loss_ema + EPSILON)
        progress_lm = max(0.0, min(1.0, 2.0 * lm_ratio - 1.0))  # linear ramp: 0 when ratio≤0.5, 1 when ratio≥1

        # --- format progress ----------------------------------------------
        progress_fmt = 1.0  # default when we have no predictions info
        if parsed_predictions is not None:
            total_preds = sum(len(p) for p in parsed_predictions)
            if total_preds > 0:
                valid_preds = 0
                for plist in parsed_predictions:
                    for p in plist:
                        coords, fmt_err = self._parse_detection_string(p)
                        if fmt_err < FMT_OK_THRESHOLD:
                            valid_preds += 1
                fmt_rate = valid_preds / total_preds
                progress_fmt = max(0.0, min(1.0, fmt_rate / 0.8))  # 0.8 compliance ⇒ 1.0
            else:
                progress_fmt = 0.0  # no predictions yet

        # Use *product* so detection weight stays low until BOTH
        # language modelling and format compliance have made progress.
        progress = progress_lm * progress_fmt

        # Store for later logging
        self.progress_lm  = progress_lm
        self.progress_fmt = progress_fmt

        return self.min_detection_weight + progress * (self.max_detection_weight - self.min_detection_weight)

    # ------------------------------------------------------------------
    # Differentiable coordinate helper using straight-through Gumbel-Softmax
    # ------------------------------------------------------------------

    def _st_expect_coord(self, logits_slice: torch.Tensor) -> torch.Tensor:
        """Return differentiable expected coordinate in [0,1] for a slice
        of logits corresponding to numeric token ids.

        Uses the straight-through Gumbel-Softmax estimator so the forward
        pass behaves like an arg-max (integer token) while the backward
        pass uses soft probabilities.
        """
        if self._enable_digit_st:
            # ----------------------------------------------------------
            # Digit-wise expectation: soft arg-max over {0..9}
            # ----------------------------------------------------------
            device = logits_slice.device
            # Ensure token-id tensor is on the same device *without* mutating the
            # class attribute (safe for multi-device setups).
            digit_token_ids = self._digit_token_ids.to(device)

            # Guard: when tokenizer size has shrunk or vocab changed such that
            # a digit id exceeds the vocabulary dimension we return a neutral
            # 0.5 expectation to avoid indexing errors.
            if digit_token_ids.max().item() >= logits_slice.size(-1):
                return torch.full(logits_slice.shape[:-1], 0.5, device=device, dtype=logits_slice.dtype)

            digit_logits = logits_slice.index_select(-1, digit_token_ids)  # (...,10)

            # Temperature schedule shared with full mode
            # Guard against tau_steps = 0
            tau_progress = min(1.0, self.step_count / max(1.0, float(self.tau_steps)))
            tau = max(self.tau_end, self.tau_start * (1.0 - tau_progress))

            digit_logits = digit_logits.clamp(min=-10.0, max=10.0)
            y_soft = torch.nn.functional.gumbel_softmax(digit_logits, tau=tau, hard=False, dim=-1)

            # ------------------------------------------------------------------
            # Robustness: if Gumbel-Softmax produced non-finite values fall back
            # to a plain softmax at a small temperature so gradients survive.
            # ------------------------------------------------------------------
            if not torch.isfinite(y_soft).all():
                y_soft = torch.softmax(digit_logits / max(tau, 0.1), dim=-1)

                # Recompute hard assignments after fallback
                idx_hard = y_soft.argmax(dim=-1)
                y_hard = torch.nn.functional.one_hot(idx_hard, num_classes=10).type_as(y_soft)
            else:
                idx_hard = y_soft.argmax(dim=-1)
                y_hard = torch.nn.functional.one_hot(idx_hard, num_classes=10).type_as(y_soft)

            # Straight-through: propagate gradients via y_soft (safe path)
            y_st = y_hard + (y_soft - y_soft.detach())

            values = torch.arange(10, device=device, dtype=logits_slice.dtype)
            digit_val = (y_st * values).sum(dim=-1)  # (...,)

            # Return raw digit value 0-9; caller will compose 3 digits
            return digit_val
        else:
            # ST disabled – return centre (0.5)
            return torch.full(logits_slice.shape[:-1], 0.5, device=logits_slice.device, dtype=logits_slice.dtype)

    def compute_detection_loss(self, lm_logits, target_boxes, parsed_strs=None):
        """Compute detection losses and match statistics.

        Integrates straight-through Gumbel-Softmax so that coordinate
        values carry gradients.
        """
        if target_boxes is None:
            zeros = torch.tensor(0.0, device=lm_logits.device)
            return zeros, zeros, zeros, zeros, 0.0

        batch_size, seq_len, _ = lm_logits.shape

        # ------------------------------------------------------------------
        # Pass 1: non-differentiable parsing for curriculum / progress stats
        # ------------------------------------------------------------------
        if parsed_strs is None:
            parsed_strs = _parse_boxes(lm_logits, self.tokenizer)

        # Build discrete prediction tensors (no gradients) from parsed_strs
        disc_preds_batch: list[torch.Tensor] = []
        for det_list in parsed_strs:
            coords_list = []
            for det in det_list:
                coords, _ = self._parse_detection_string(det)
                coords_list.append(coords)
            if coords_list:
                disc_preds_batch.append(torch.tensor(coords_list, device=lm_logits.device, dtype=lm_logits.dtype))
            else:
                disc_preds_batch.append(torch.zeros((0, 4), device=lm_logits.device, dtype=lm_logits.dtype))

        # ------------------------------------------------------------------
        # Pass 2: differentiable extraction of coordinates using ST-Gumbel.
        # We scan the arg-max ids to identify numeric-token positions, but we
        # compute the coordinate values with _st_expect_coord, which keeps
        # gradients w.r.t. the logits.  The positional scan is hard, so the
        # graph is still cut if a numeric token is mis-typed, but correct
        # numeric tokens now receive meaningful gradients.
        # ------------------------------------------------------------------

        token_ids = lm_logits.argmax(dim=-1)  # (B, S) – used only for pattern detection
        id_open, id_close = self.tokenizer.convert_tokens_to_ids([TAG_OPEN, TAG_CLOSE])

        batch_preds: list[torch.Tensor] = []  # each [K,4] differentiable

        for b in range(batch_size):
            row = token_ids[b].tolist()
            logits_row = lm_logits[b]  # (S, V)

            i = 0
            coords_list = []
            digit_buffer: list[torch.Tensor] = []  # collect 3 digits per coordinate
            coord_buffer: list[torch.Tensor] = []  # collect 4 coords per box
            while True:
                try:
                    start = row.index(id_open, i)
                except ValueError:
                    break
                try:
                    end = row.index(id_close, start + 1)
                except ValueError:
                    break  # unmatched → ignore rest

                # Iterate positions between start and end (exclusive)
                for pos in range(start + 1, end):
                    tok_id = row[pos]
                    if tok_id in self._digit_token_id_set:
                        digit_val = self._st_expect_coord(logits_row[pos : pos + 1, :])[0]
                        digit_buffer.append(digit_val)
                        if len(digit_buffer) == 3:
                            # Compose 3 digits into a single coordinate value in [0,1]
                            coord_int = (digit_buffer[0] * 100 + digit_buffer[1] * 10 + digit_buffer[2])
                            coord_val = coord_int / 1000.0  # 000-999 → 0-0.999
                            coord_buffer.append(coord_val)
                            digit_buffer = []

                            if len(coord_buffer) == 4:
                                coords_list.append(torch.stack(coord_buffer))
                                coord_buffer = []
                i = end + 1

            if coords_list:
                preds_b = torch.stack(coords_list)  # (K,4)
            else:
                preds_b = torch.zeros((0, 4), device=lm_logits.device, dtype=lm_logits.dtype)
            batch_preds.append(preds_b)

        # ------------------------------------------------------------------
        # Aggregate losses per sample (CIoU and count) using differentiable
        # predictions.  This mirrors the old logic but retains gradients.
        # ------------------------------------------------------------------

        iou_vals: list[torch.Tensor] = []
        soft_iou_vals: list[torch.Tensor] = []

        matched_gt, total_gt = 0, 0
        disc_iou_vals: list[torch.Tensor] = []

        for b, (pred, gt) in enumerate(zip(batch_preds, target_boxes)):
            # Convert GT to tensor on the correct device / dtype
            if isinstance(gt, list):
                gt = torch.tensor(gt, device=lm_logits.device, dtype=lm_logits.dtype)
            elif gt is None:
                gt = torch.zeros((0, 4), device=lm_logits.device, dtype=lm_logits.dtype)

            gt_count = gt.shape[0]
            pred_count = pred.shape[0]

            # When ST is disabled pred may be zeros; skip in that case too
            if pred.numel() == 0 or gt.numel() == 0 or not self._enable_digit_st:
                continue

            # Pre-filter zero-area boxes (already done inside matcher but we need
            # the mapping to original indices for discrete branch)
            pred_valid = (pred[:, 2] > EPSILON) & (pred[:, 3] > EPSILON)
            gt_valid   = (gt[:, 2] > EPSILON) & (gt[:, 3] > EPSILON)

            if not pred_valid.any() or not gt_valid.any():
                continue

            pred_f = pred[pred_valid]
            gt_f   = gt[gt_valid]

            # Keep mapping to original indices for discrete IoU later
            pred_idx_map = torch.nonzero(pred_valid, as_tuple=False).flatten()
            gt_idx_map   = torch.nonzero(gt_valid, as_tuple=False).flatten()

            pairs = _match_boxes_hungarian(pred_f, gt_f)
            if not pairs:
                continue

            p_matched = torch.stack([pred_f[i] for i, _ in pairs])
            g_matched = torch.stack([gt_f[j] for _, j in pairs])

            # ------------------------------------------------------
            # Filter out degenerate boxes (zero width/height) to
            # avoid NaNs in IoU/CIoU.  If all pairs are degenerate we
            # skip this sample entirely.
            # ------------------------------------------------------
            valid_mask = (
                (p_matched[:, 2] > EPSILON)
                & (p_matched[:, 3] > EPSILON)
                & (g_matched[:, 2] > EPSILON)
                & (g_matched[:, 3] > EPSILON)
            )
            if not valid_mask.any():
                continue

            p_matched = p_matched[valid_mask]
            g_matched = g_matched[valid_mask]

            ciou = _complete_box_iou_loss(
                _xywh_to_xyxy(p_matched), _xywh_to_xyxy(g_matched), reduction="none"
            ).mean()
            ciou = torch.nan_to_num(ciou, nan=2.0, posinf=2.0, neginf=0.0)
            iou_vals.append(ciou)

            # Convert differentiable CIoU to IoU (approx) for correlation
            soft_iou_vals.append((1.0 - ciou.detach()))

            # -------------------------------------------
            # Discrete IoU using arg-max parsed boxes
            # -------------------------------------------
            disc_pred_sample = disc_preds_batch[b]
            if disc_pred_sample.shape[0] > 0:
                # Map pair indices back to original pred / gt indices
                valid_pairs = []
                for idx_pair, (i_f, j_f) in enumerate(pairs):
                    if not valid_mask[idx_pair]:
                        continue  # pair involves degenerate box filtered earlier

                    i_orig = pred_idx_map[i_f].item()
                    j_orig = gt_idx_map[j_f].item()

                    if i_orig < disc_pred_sample.size(0) and j_orig < gt.shape[0]:
                        valid_pairs.append((i_orig, j_orig))
                if valid_pairs:
                    pred_idx = torch.tensor([i for i, _ in valid_pairs], device=lm_logits.device)
                    gt_idx   = torch.tensor([j for _, j in valid_pairs], device=lm_logits.device)

                    d_matched = disc_pred_sample.index_select(0, pred_idx)
                    g_sub      = gt.index_select(0, gt_idx)

                    with torch.no_grad():
                        disc_pair_iou = _iou_matrix_xywh(d_matched, g_sub).diag().mean()
                    disc_iou_vals.append(disc_pair_iou)

            matched_gt += len({j for _, j in pairs})
            total_gt += gt_count

        iou_loss = torch.stack(iou_vals).mean() if iou_vals else torch.tensor(0.0, device=lm_logits.device)

        # ------------------------------------------------------------------
        # Keep format / count evaluation via existing parser
        # ------------------------------------------------------------------
        fmt_vals: list[float] = []

        for det_list in parsed_strs:
            for det in det_list:
                _, fmt_err = self._parse_detection_string(det)
                fmt_vals.append(fmt_err)

        if fmt_vals:
            fmt_tensor = torch.tensor([min(1.0, max(0.0, v)) for v in fmt_vals], device=lm_logits.device, dtype=lm_logits.dtype)
            format_loss = fmt_tensor.mean()
        else:
            format_loss = torch.tensor(0.0, device=lm_logits.device)
        
        match_rate = (matched_gt / total_gt) if total_gt else 0.0

        # --------------------------------------------------------------
        # Pearson correlation between discrete IoU and differentiable IoU
        # --------------------------------------------------------------
        if len(disc_iou_vals) > 1 and len(disc_iou_vals) == len(soft_iou_vals):
            dx = torch.stack(disc_iou_vals)
            sx = torch.stack(soft_iou_vals)
            mean_d = dx.mean()
            mean_s = sx.mean()
            cov = ((dx - mean_d) * (sx - mean_s)).mean()
            var_d = dx.var(unbiased=False)
            var_s = sx.var(unbiased=False)
            if var_d < EPSILON or var_s < EPSILON:
                corr = torch.tensor(0.0, device=lm_logits.device)
            else:
                corr = cov / (torch.sqrt(var_d * var_s) + EPSILON)
        else:
            corr = torch.tensor(float('nan'), device=lm_logits.device)

        return iou_loss, format_loss, match_rate, corr

    def __call__(self, outputs, labels, **kwargs):
        """
        Loss function with adaptive curriculum learning.
        
        Args:
            outputs: The model outputs
            labels: The ground-truth labels
            **kwargs: Additional keyword arguments
            
        Returns:
            loss (torch.Tensor)
        """
        # Optional: accept ground-truth boxes via keyword or as attribute on outputs
        target_boxes = kwargs.get("target_boxes", getattr(outputs, "target_boxes", None))

        # Extract logits and labels
        lm_logits = outputs.logits
        lm_labels = labels
        
        if lm_labels is None:
            raise ValueError("Labels must be provided in inputs for loss computation")
        
        # --------------------------------------------------------------
        # Align tokens for causal language-model training.
        #   At position *t* the model must predict token *t*+1, therefore
        #   we drop the last time-step from the logits and the first one
        #   from the labels before computing the CE loss.  This matches
        #   the internal behaviour of transformersʼ `AutoModelForCausalLM`.
        # --------------------------------------------------------------

        vocab_size = lm_logits.size(-1)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = lm_labels[..., 1:].contiguous()

        lm_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )
        
        # ----------------------------------------------------------
        # Numerical safety: abort early if the LM loss is NaN/Inf so
        # we do not corrupt the model with undefined gradients.
        # ----------------------------------------------------------
        if not torch.isfinite(lm_loss):
            raise RuntimeError("NaN or Inf detected in LM loss – aborting training step.")
        
        # Update curriculum tracking
        self._update_lm_loss_tracking(lm_loss)
        
        # Parse predicted boxes once for curriculum assessment and detection loss
        parsed_strs = _parse_boxes(lm_logits, self.tokenizer)
        
        # Compute adaptive weights based on LM performance AND format compliance
        weight_multiplier = self._compute_adaptive_weights(parsed_strs)

        # ------------------------------------------------------------------
        # Derive ground-truth boxes from labels when they were not provided
        # separately.  We decode the label IDs (ignoring -100 paddings) back to
        # text and extract <bbob> … </bbob> fragments using the same regex that
        # is applied to the predictions.
        # ------------------------------------------------------------------
        if target_boxes is None and lm_labels is not None:
            gt_texts = []
            for row in lm_labels:
                # Remove ignore_index tokens to avoid bogus ids in decoding
                valid_ids = row[row != -100].tolist()
                gt_texts.append(self.tokenizer.decode(valid_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True))

            target_boxes = []
            for txt in gt_texts:
                boxes = []
                # Match using the exact special tokens so we stay in sync
                for det in TAG_PATTERN.findall(txt):
                    coords, fmt_err = self._parse_detection_string(det)
                    boxes.append(coords)
                target_boxes.append(boxes)

        # Compute detection losses
        iou_loss, format_loss, match_rate, corr = self.compute_detection_loss(
            lm_logits, target_boxes, parsed_strs
        )
        
        # Apply adaptive weights to detection losses
        adaptive_lambda_detection = self.lambda_detection * weight_multiplier
        detection_loss = (
            self.lambda_iou * iou_loss +
            self.lambda_format * format_loss
        )

        # Inverse scaling: smoothly decrease LM weight as detection importance rises.
        # Linear ramp: weight_multiplier ≈1 → lm_weight≈1; ≥15 → lm_weight→0.3.
        lm_weight = max(0.3, min(1.0, 1.0 - 0.05 * (weight_multiplier - 1.0)))

        total_loss = lm_weight * lm_loss + adaptive_lambda_detection * detection_loss

        # ------------------------------------------------------------------
        # Optional inline logging every `log_interval` calls
        # ------------------------------------------------------------------

        if self.logger is not None and (self.step_count % self._log_interval == 0):
            # Average number of GT boxes in this batch
            if target_boxes is not None:
                avg_gt = round(sum(len(tb) if isinstance(tb, list) else (tb.shape[0] if tb is not None else 0) for tb in target_boxes) / max(1, len(target_boxes)), 3)
            else:
                avg_gt = 0.0

            loss_dict = {
                "loss_total": _val(total_loss),
                "loss_lm": _val(lm_loss),
                "loss_iou": _val(iou_loss),
                "loss_fmt": _val(format_loss),
                "det_weight": _val(adaptive_lambda_detection),
                "gt_match_rate": match_rate,
                "parsed_boxes_avg": round(sum(len(p) for p in parsed_strs) / max(1, len(parsed_strs)), 3),
                "gt_boxes_avg": avg_gt,
                "progress_lm": _val(self.progress_lm) if self.progress_lm is not None else None,
                "progress_fmt": _val(self.progress_fmt) if self.progress_fmt is not None else None,
                'lm_loss_ema': self.lm_loss_ema,
                'lm_target': self.lm_target,
                'current_weight_multiplier': weight_multiplier,
                'step_count': self.step_count,
                'corr': _val(corr)
            }
            
            self.logger.info(f"LOSS STATUS: {loss_dict}")

            # ----------------------------------------------------------
            # Log one raw model output occasionally for qualitative check
            # ----------------------------------------------------------
            if self.step_count % (self._log_interval * 5) == 0:
                sample_pred_ids = lm_logits[0].argmax(dim=-1)
                sample_pred_text = self.tokenizer.decode(sample_pred_ids.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True)

                # decode ground-truth of the same sample (filter ignore_index)
                gt_ids = lm_labels[0][lm_labels[0] != -100].tolist()
                sample_gt_text = self.tokenizer.decode(gt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)

                self.logger.info(
                    f"[sample] pred: {sample_pred_text} || gt: {sample_gt_text}"
                )

        return total_loss

    
def create_compute_loss_func(
    tokenizer,
    *,
    lambda_iou=1.5,
    lambda_detection: float = 0.15,
    lambda_format: float = 0.5,
    lm_target: float = 2.0,
    logger=None,
    log_interval: int = 100,
):
    """Return a CompositeLoss instance with the given weight settings.
    """
    loss_computer = CompositeLoss(
        tokenizer=tokenizer,
        lambda_iou=lambda_iou,
        lambda_detection=lambda_detection,
        lambda_format=lambda_format,
        lm_target=lm_target,
        logger=logger,
        log_interval=log_interval,
    )
    return loss_computer