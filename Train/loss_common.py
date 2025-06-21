import torch
import torch.nn.functional as F
import re

from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou as _box_iou  

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

EPSILON: float = 1e-8       # small value to avoid division-by-zero
FMT_OK_THRESHOLD: float = 0.25  # consider a box "well-formed" when fmt_err < 0.25

def _val(x):
    return x.item() if isinstance(x, torch.Tensor) else float(x)

def _parse_boxes(logits, tokenizer):
    """Decode logits and extract detection strings.

    Args:
        logits (torch.Tensor): Model output logits with shape (batch_size, seq_len, vocab_size).
        tokenizer: Tokenizer that provides a `decode(ids, **kwargs)` method.

    Returns:
        List[List[str]]: A list with one entry per batch element, containing all detection strings
        (the raw content found between <bbob> and </bbob> tokens).
    """

    # Safety check: ensure we have the expected tensor dimensionality
    if logits.dim() != 3:
        raise ValueError(
            f"Expected `logits` to have shape (batch, seq_len, vocab_size) but got tensor with shape {logits.shape}"
        )

    # Greedy decoding – take the most probable token at each position
    token_ids = logits.argmax(dim=-1)  # (batch_size, seq_len)

    detections_batch = []

    for ids in token_ids:
        # Convert tensor row to python list for the tokenizer
        ids_list = ids.tolist()

        # Decode the sequence into text. We skip special tokens to avoid extraneous artifacts.
        try:
            decoded_text = tokenizer.decode(ids_list, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except Exception as e:
            # Handle potential tokenization errors gracefully
            print(f"Warning: Tokenization error: {e}")
            detections_batch.append([])
            continue

        # Use regex to find every substring wrapped by <bbob> ... </bbob>
        # The non-greedy qualifier (.*?) ensures we catch individual detections.
        matches = re.findall(r"<bbob>(.*?)</bbob>", decoded_text)

        # Strip surrounding whitespace from each detection
        detections = [m.strip() for m in matches]

        detections_batch.append(detections)

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

    # numpy does not understand torch.bfloat16.  Cast to float32 on CPU first.
    cost = (1.0 - iou_mat + EPSILON)

    # Torch's linear_sum_assignment works on both CPU & CUDA tensors.
    row_ind, col_ind = linear_sum_assignment(cost)
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
        lambda_l1=0.3,
        lambda_iou=0.5,
        lambda_count=0.25,
        lambda_detection=0.15,
        lambda_format=0.1,
        lm_target=1.5,
        smoothing_factor=0.95,
        logger=None,
        log_interval: int = 100,
    ):
        self.tokenizer = tokenizer
        self.lambda_l1 = lambda_l1
        self.lambda_iou = lambda_iou
        self.lambda_count = lambda_count
        self.lambda_detection = lambda_detection
        self.lambda_format = lambda_format
        
        # Curriculum parameters
        self.lm_target = lm_target
        # scalars for adaptive detection loss weight
        self.min_detection_weight = 0.002
        self.max_detection_weight = 256
        self.smoothing_factor = smoothing_factor
        
        # Tracking variables
        self.lm_loss_ema = None  # Exponential moving average of LM loss
        self.step_count = 0

        # Optional external logger
        self.logger = logger
        self._log_interval = max(1, log_interval)

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
        fmt_err = (missing / 4.0) + (sum(clip_diffs) / 4.0)
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

        progress = progress_lm * progress_fmt

        return self.min_detection_weight + progress * (self.max_detection_weight - self.min_detection_weight)

    def compute_detection_loss(self, lm_logits, target_boxes, parsed_strs=None):
        """Compute detection losses and match statistics.

        Parameters
        ----------
        lm_logits : Tensor
            Language-model logits (needed for dtype / device).
        target_boxes : list | Tensor | None
            Ground-truth boxes, optional.
        parsed_strs : list[list[str]] | None
            Output of `_parse_boxes` if already available.  When *None* the
            method will call `_parse_boxes` internally.  Supplying it avoids
            double work in the main forward pass.
        """
        if target_boxes is None:
            zeros = torch.tensor(0.0, device=lm_logits.device)
            return zeros, zeros, zeros, zeros, 0.0

        # Parse predicted boxes from language-model output once if not given
        if parsed_strs is None:
            parsed_strs = _parse_boxes(lm_logits, self.tokenizer)

        # ------------------------------------------------------------------
        # Vectorised parsing: build a single flat list of coords, then split
        # back into per-sample tensors to avoid thousands of small tensor
        # allocations on GPU each step.
        # ------------------------------------------------------------------

        flat_coords: list[list[float]] = []
        fmt_vals: list[float] = []
        counts: list[int] = []

        for det_list in parsed_strs:
            counts.append(len(det_list))
            for det in det_list:
                coords, fmt_err = self._parse_detection_string(det)
                flat_coords.append(coords)
                fmt_vals.append(fmt_err)

        if flat_coords:
            all_preds = torch.tensor(flat_coords, device=lm_logits.device, dtype=lm_logits.dtype)
            split_preds = list(all_preds.split(counts))
        else:
            # Handle edge-case when no predictions at all in batch
            split_preds = [torch.zeros((0, 4), device=lm_logits.device, dtype=lm_logits.dtype) for _ in counts]

        batch_preds = split_preds

        # Compute loss per sample with simple first-K matching
        l1_vals = []
        iou_vals = []
        count_vals = []
        
        matched_gt, total_gt = 0, 0
        
        for pred, gt in zip(batch_preds, target_boxes):
            # Convert gt to tensor if it's a list
            if isinstance(gt, list):
                if not gt:  # Empty list
                    gt_count = 0
                    gt = torch.zeros((0, 4), device=lm_logits.device, dtype=lm_logits.dtype)
                else:
                    gt_count = len(gt)
                    gt = torch.tensor(gt, device=lm_logits.device, dtype=lm_logits.dtype)
            elif gt is None:
                continue
            else:
                gt_count = gt.shape[0]
            
            pred_count = pred.shape[0]
            
            # Count penalty: penalize difference in number of predictions
            count_diff = abs(pred_count - gt_count)
            # When gt_count == 0, bound the penalty growth to avoid exploding losses
            if gt_count == 0:
                count_penalty = min(count_diff ** 2, 16.0)  # cap at 16
            else:
                count_penalty = (count_diff ** 2) / gt_count
            count_vals.append(count_penalty)
            
            if pred.numel() == 0 or gt.numel() == 0:
                continue

            pairs = _match_boxes_hungarian(pred, gt)

            if not pairs:
                continue

            p_matched = torch.stack([pred[i] for i, _ in pairs])
            g_matched = torch.stack([gt[j] for _, j in pairs])

            l1_vals.append(F.l1_loss(p_matched, g_matched, reduction="mean"))
            iou_vals.append(1.0 - _iou_matrix_xywh(p_matched, g_matched).mean())

            # -------------------------------------------------------------
            # GT match rate statistics
            # -------------------------------------------------------------
            matched_gt += len({j for _, j in pairs})
            total_gt   += gt_count

        # Aggregate losses
        l1_loss = torch.stack(l1_vals).mean() if l1_vals else torch.tensor(0.0, device=lm_logits.device)
        iou_loss = torch.stack(iou_vals).mean() if iou_vals else torch.tensor(0.0, device=lm_logits.device)
        count_loss = torch.tensor(count_vals, device=lm_logits.device, dtype=lm_logits.dtype).mean() if count_vals else torch.tensor(0.0, device=lm_logits.device)
        if fmt_vals:
            # Clamp to [0,1] to keep the scale bounded before averaging.
            fmt_tensor = torch.tensor([min(1.0, max(0.0, v)) for v in fmt_vals], device=lm_logits.device, dtype=lm_logits.dtype)
            format_loss = fmt_tensor.mean()
        else:
            format_loss = torch.tensor(0.0, device=lm_logits.device)
        
        match_rate = (matched_gt / total_gt) if total_gt else 0.0
        return l1_loss, iou_loss, count_loss, format_loss, match_rate

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
                gt_texts.append(self.tokenizer.decode(valid_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))

            target_boxes = []
            for txt in gt_texts:
                boxes = []
                for det in re.findall(r"<bbob>(.*?)</bbob>", txt):
                    coords, fmt_err = self._parse_detection_string(det)
                    if coords is not None:
                        boxes.append(coords)
                target_boxes.append(boxes)

        # Compute detection losses
        l1_loss, iou_loss, count_loss, format_loss, match_rate = self.compute_detection_loss(
            lm_logits, target_boxes, parsed_strs
        )
        
        # Apply adaptive weights to detection losses
        adaptive_lambda_detection = self.lambda_detection * weight_multiplier
        detection_loss = (
            self.lambda_l1 * l1_loss
            + self.lambda_iou * iou_loss
            + self.lambda_count * count_loss
            + self.lambda_format * format_loss
        )
        
        # Total loss with adaptive weighting
        total_loss = lm_loss + adaptive_lambda_detection * detection_loss
        
        # ------------------------------------------------------------------
        # Optional inline logging every `log_interval` calls
        # ------------------------------------------------------------------

        if self.logger is not None and (self.step_count % self._log_interval == 0):
            loss_dict = {
                "loss_total": _val(total_loss),
                "loss_lm": _val(lm_loss),
                "loss_l1": _val(l1_loss),
                "loss_iou": _val(iou_loss),
                "mean_iou": 1.0 - _val(iou_loss) if _val(iou_loss) > 0 else 0.0,
                "loss_count": _val(count_loss),
                "loss_fmt": _val(format_loss),
                "det_weight": _val(adaptive_lambda_detection),
                "gt_match_rate": match_rate,
                "parsed_boxes_avg": round(sum(len(p) for p in parsed_strs) / max(1, len(parsed_strs)), 3),
            }
            cur_dict = self._get_curriculum_status(weight_multiplier)
            self.logger.info(f"LOSS STATUS: {loss_dict} | CURRICULUM STATUS: {cur_dict}")

        return total_loss

    def _get_curriculum_status(self, weight_multiplier):
        """Get current curriculum learning status for logging/debugging."""
        return {
            'lm_loss_ema': self.lm_loss_ema,
            'lm_target': self.lm_target,
            'current_weight_multiplier': weight_multiplier,
            'step_count': self.step_count
        }


def create_compute_loss_func(
    tokenizer,
    *,
    lambda_l1=0.35,
    lambda_iou=0.5,
    lambda_count=0.2,
    lambda_detection=0.15,
    lambda_format=0.1,
    lm_target=2.0,
    logger=None,
    log_interval: int = 100,
):
    """Return a CompositeLoss instance with the given weight settings.
    """
    loss_computer = CompositeLoss(
        tokenizer=tokenizer,
        lambda_l1=lambda_l1,
        lambda_iou=lambda_iou,
        lambda_count=lambda_count,
        lambda_detection=lambda_detection,
        lambda_format=lambda_format,
        lm_target=lm_target,
        logger=logger,
        log_interval=log_interval,
    )
    return loss_computer