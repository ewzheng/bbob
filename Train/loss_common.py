import torch
import torch.nn.functional as F
import re

from scipy.optimize import linear_sum_assignment  # type: ignore
from torchvision.ops import box_iou as _box_iou  # type: ignore
from torchvision.ops import complete_box_iou_loss as _complete_box_iou_loss  # type: ignore

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

EPSILON: float = 1e-8       # small value to avoid division-by-zero
BBOX_SCALE: float = 999.0    # integer scale for coordinate tokens 0–999
FMT_OK_THRESHOLD: float = 0.25  # consider a box "well-formed" when fmt_err < 0.25

# ----------------------------------------------------------------------
# Tag tokens
# ----------------------------------------------------------------------

TAG_OPEN = "<|bbob|>"
TAG_CLOSE = "</|bbob|>"

def _val(x):
    return x.item() if isinstance(x, torch.Tensor) else float(x)

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
    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    x2 = boxes[..., 0] + boxes[..., 2]
    y2 = boxes[..., 1] + boxes[..., 3]
    return torch.stack((x1, y1, x2, y2), dim=-1)

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

    iou_mat = _iou_matrix_xywh(pred, gt)
    # Replace possible NaNs (can arise from zero-area boxes) with zeros
    iou_mat = torch.nan_to_num(iou_mat, nan=0.0, posinf=0.0, neginf=0.0)

    # numpy does not understand torch.bfloat16.  Cast to float32 on CPU first.
    cost = (1.0 - iou_mat + EPSILON)

    # Torch's linear_sum_assignment currently expects a CPU tensor.
    if cost.is_cuda:
        cost_cpu = cost.cpu()
    else:
        cost_cpu = cost

    row_ind, col_ind = linear_sum_assignment(cost_cpu)
    return [(int(r), int(c)) for r, c in zip(row_ind.tolist(), col_ind.tolist())]

class CompositeLoss:
    """Compute combined language-model and detection losses.

    The total loss is:
        L = L_lm + λd (λ1 * L1(box) + λ2 * (1 − IoU) + λ3 * count_penalty)

    where the auxiliary terms are only added when ground-truth boxes are
    provided. Simple 1-to-1 matching is assumed (first K predictions matched
    to first K GT boxes).
    """

    def __init__(
        self,
        tokenizer,
        *,
        lambda_iou=1.5,
        lambda_detection=0.15,
        lambda_format=0.5,
        lm_target=1.5,
        smoothing_factor=0.95,
        logger=None,
        log_interval: int = 100,
        gumbel_tau: float = 1.0,
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

        self._numeric_token_ids = torch.tensor([
            tokenizer.convert_tokens_to_ids(str(i)) for i in range(1000)
        ], dtype=torch.long)
        # Pre-compute value vector 0..999 / scale for expectation trick
        self._numeric_values = torch.arange(1000, dtype=torch.float32) / BBOX_SCALE
        # Gumbel-Softmax temperature schedule
        self.tau_start = 2.0
        self.tau_end = 0.1
        self.tau_steps = 10_000  # linear decay over this many optimisation steps
        self.gumbel_tau = gumbel_tau  # kept for backward compatibility; may be overridden by schedule

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
                nums.append(float(s))
            except ValueError:
                continue

        missing = 4 - len(nums)
        # Fill the rest with a neutral prior (centre-ish square)
        nums.extend([0.5 * BBOX_SCALE] * missing)

        # ------------------------------------------------------------------
        # 2) clamp to [0,1] and accumulate formatting error
        # ------------------------------------------------------------------
        clip_diffs = []
        clamped = []
        for v in nums:
            # Convert integer-ish token back to 0–1 float
            v_float = v / BBOX_SCALE
            v_clamped = max(0.0, min(1.0, v_float))
            clamped.append(v_clamped)
            clip_diffs.append(abs(v_float - v_clamped))

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
        """Update the exponential moving average of LM loss."""
        lm_loss_value = current_lm_loss.detach().item()
        
        if self.lm_loss_ema is None:
            self.lm_loss_ema = lm_loss_value
        else:
            self.lm_loss_ema = (self.smoothing_factor * self.lm_loss_ema + 
                               (1 - self.smoothing_factor) * lm_loss_value)
        
        self.step_count += 1

    def _compute_adaptive_weights(self, parsed_predictions=None):
        """Compute adaptive detection loss weights based on LM performance and format compliance."""
        if self.lm_loss_ema is None:
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

        progress = progress_lm + progress_fmt

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
        # logits_slice: (..., vocab)
        device = logits_slice.device
        # select numeric logits
        numeric_logits = logits_slice.index_select(-1, self._numeric_token_ids.to(device))  # (..., 1000)

        # If gradients are disabled (e.g., evaluation), skip soft path entirely
        if not torch.is_grad_enabled():
            idx_hard = numeric_logits.argmax(dim=-1)
            values = self._numeric_values.to(device, dtype=logits_slice.dtype)
            coord = values[idx_hard]
            return coord

        # --------------------------------------------------------------
        # Training: temperature-annealed Gumbel-Softmax
        # --------------------------------------------------------------

        # Linear decay of τ from start → end
        tau_progress = min(1.0, self.step_count / float(self.tau_steps))
        tau = max(self.tau_end, self.tau_start * (1.0 - tau_progress))

        # Clamp logits for numerical stability
        numeric_logits = numeric_logits.clamp(min=-10.0, max=10.0)

        y_soft = torch.nn.functional.gumbel_softmax(
            numeric_logits, tau=tau, hard=False, dim=-1
        )

        idx_hard = y_soft.argmax(dim=-1)
        y_hard = torch.nn.functional.one_hot(idx_hard, num_classes=1000).type_as(y_soft)

        # Straight-through substitute; NaN guard on soft output
        if torch.isnan(y_soft).any() or torch.isinf(y_soft).any():
            y_st = y_hard  # fall back to hard path if numerical issues
        else:
            y_st = y_hard + (y_soft - y_soft.detach())

        values = self._numeric_values.to(device, dtype=logits_slice.dtype)
        coord = (y_st * values).sum(dim=-1)
        return coord

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
            numeric_buffer = []  # store differentiable coords
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
                    if tok_id in self._numeric_token_ids:
                        # Differentiable coordinate value
                        coord_val = self._st_expect_coord(logits_row[pos : pos + 1, :])[0]
                        numeric_buffer.append(coord_val)
                        if len(numeric_buffer) == 4:
                            coords_list.append(torch.stack(numeric_buffer))
                            numeric_buffer = []
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

        for pred, gt in zip(batch_preds, target_boxes):
            # Convert GT to tensor on the correct device / dtype
            if isinstance(gt, list):
                gt = torch.tensor(gt, device=lm_logits.device, dtype=lm_logits.dtype)
            elif gt is None:
                gt = torch.zeros((0, 4), device=lm_logits.device, dtype=lm_logits.dtype)

            gt_count = gt.shape[0]
            pred_count = pred.shape[0]

            if pred.numel() == 0 or gt.numel() == 0:
                continue

            pairs = _match_boxes_hungarian(pred, gt)
            if not pairs:
                continue

            p_matched = torch.stack([pred[i] for i, _ in pairs])
            g_matched = torch.stack([gt[j] for _, j in pairs])

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
                d_matched = torch.stack([disc_pred_sample[i] for i, _ in pairs])
                with torch.no_grad():
                    disc_pair_iou = _iou_matrix_xywh(d_matched, g_matched).diag().mean()
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
            reduction="mean"
        )
        
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
                pattern = re.escape(TAG_OPEN) + r"(.*?)" + re.escape(TAG_CLOSE)
                for det in re.findall(pattern, txt):
                    coords, fmt_err = self._parse_detection_string(det)
                    if coords is not None:
                        boxes.append(coords)
                target_boxes.append(boxes)

        # Compute detection losses
        iou_loss, format_loss, match_rate, corr = self.compute_detection_loss(
            lm_logits, target_boxes, parsed_strs
        )
        
        # Apply adaptive weights to detection losses
        adaptive_lambda_detection = self.lambda_detection * weight_multiplier
        detection_loss = self.lambda_iou * iou_loss + self.lambda_format * format_loss
        
        # Update curriculum tracking
        self._update_lm_loss_tracking(lm_loss)
        
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
                "loss_total": _val(iou_loss + detection_loss),
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

        return iou_loss + detection_loss

    
def create_compute_loss_func(
    tokenizer,
    *,
    lambda_iou=1.5,
    lambda_detection=0.15,
    lambda_format=0.5,
    lm_target=2.0,
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