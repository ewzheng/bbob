import torch
import torch.nn.functional as F
import re
from scipy.optimize import linear_sum_assignment

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

def _iou_xywh(boxes1, boxes2):
    """Compute IoU between two (N,4) tensors of [x,y,w,h] normalised coords."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(min(boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    b1_x1 = boxes1[:, 0]
    b1_y1 = boxes1[:, 1]
    b1_x2 = boxes1[:, 0] + boxes1[:, 2]
    b1_y2 = boxes1[:, 1] + boxes1[:, 3]

    b2_x1 = boxes2[:, 0]
    b2_y1 = boxes2[:, 1]
    b2_x2 = boxes2[:, 0] + boxes2[:, 2]
    b2_y2 = boxes2[:, 1] + boxes2[:, 3]

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]

    union = area1 + area2 - inter_area + 1e-7
    return inter_area / union   

def _match_boxes_hungarian(pred, gt):
    """Return index pairs using Hungarian algorithm on IoU cost."""
    if pred.numel() == 0 or gt.numel() == 0:
        return []

    iou_mat = _iou_xywh(
        pred.unsqueeze(1).expand(-1, gt.shape[0], -1).reshape(-1, 4),
        gt.unsqueeze(0).expand(pred.shape[0], -1, 4).reshape(-1, 4),
    ).view(pred.shape[0], gt.shape[0])

    # numpy does not understand torch.bfloat16.  Cast to float32 on CPU first.
    cost = (1.0 - iou_mat).float().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if iou_mat[r, c] > 0]

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
        lambda_l1=0.35,
        lambda_iou=0.5,
        lambda_count=0.1,
        lambda_detection=0.15,
        lm_target=1.5,
        smoothing_factor=0.9,
        logger=None,
        log_interval: int = 100,
    ):
        self.tokenizer = tokenizer
        self.lambda_l1 = lambda_l1
        self.lambda_iou = lambda_iou
        self.lambda_count = lambda_count
        self.lambda_detection = lambda_detection
        
        # Curriculum parameters
        self.lm_target = lm_target
        self.min_detection_weight = lambda_detection/64
        self.max_detection_weight = lambda_detection*64
        self.smoothing_factor = smoothing_factor
        
        # Tracking variables
        self.lm_loss_ema = None  # Exponential moving average of LM loss
        self.step_count = 0

        # Optional external logger
        self.logger = logger
        self._log_interval = max(1, log_interval)

    def _parse_detection_string(self, det_str):
        """Parse a detection string like 'cat:0.1 0.2 0.3 0.4' into box coordinates."""
        parts = det_str.split(":")
        if len(parts) != 2:
            return None
        
        try:
            # Extract numerical values and take first 4
            nums = [float(x) for x in parts[1].strip().split()][:4]
            if len(nums) == 4:
                # Clamp values to [0, 1] range for safety
                return [max(0.0, min(1.0, n)) for n in nums]
        except (ValueError, IndexError):
            pass
        
        return None

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
        lm_ratio = self.lm_target / (self.lm_loss_ema + 1e-8)
        progress_lm = max(0.0, min(1.0, 2.0 * lm_ratio - 1.0))  # linear ramp: 0 when ratio≤0.5, 1 when ratio≥1

        # --- format progress ----------------------------------------------
        progress_fmt = 1.0  # default when we have no predictions info
        if parsed_predictions is not None:
            total_preds = sum(len(p) for p in parsed_predictions)
            if total_preds > 0:
                valid_preds = sum(
                    len([p for p in plist if self._parse_detection_string(p) is not None])
                    for plist in parsed_predictions
                )
                fmt_rate = valid_preds / total_preds
                progress_fmt = max(0.0, min(1.0, fmt_rate / 0.8))  # 0.8 compliance ⇒ 1.0
            else:
                progress_fmt = 0.0  # no predictions yet

        progress = progress_lm * progress_fmt

        return self.min_detection_weight + progress * (self.max_detection_weight - self.min_detection_weight)

    def compute_detection_loss(self, lm_logits, target_boxes):
        """Compute detection losses (L1 + IoU + count penalty) and GT-match rate."""
        if target_boxes is None:
            zeros = torch.tensor(0.0, device=lm_logits.device)
            return zeros, zeros, zeros, 0.0

        # Parse predicted boxes from language model output
        parsed_strs = _parse_boxes(lm_logits, self.tokenizer)

        batch_preds = []
        for det_list in parsed_strs:
            boxes = []
            for det in det_list:
                parsed_box = self._parse_detection_string(det)
                if parsed_box is not None:
                    boxes.append(parsed_box)
            
            if boxes:
                batch_preds.append(torch.tensor(boxes, device=lm_logits.device, dtype=lm_logits.dtype))
            else:
                batch_preds.append(torch.zeros((0, 4), device=lm_logits.device, dtype=lm_logits.dtype))

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
            count_penalty = (count_diff ** 2) / max(1, gt_count)
            count_vals.append(count_penalty)
            
            if pred.numel() == 0 or gt.numel() == 0:
                continue

            pairs = _match_boxes_hungarian(pred, gt)

            if not pairs:
                continue

            p_matched = torch.stack([pred[i] for i, _ in pairs])
            g_matched = torch.stack([gt[j] for _, j in pairs])

            l1_vals.append(F.l1_loss(p_matched, g_matched, reduction="mean"))
            iou_vals.append(1.0 - _iou_xywh(p_matched, g_matched).mean())

            # -------------------------------------------------------------
            # GT match rate statistics
            # -------------------------------------------------------------
            matched_gt += len({j for _, j in pairs})
            total_gt   += gt_count

        # Aggregate losses
        l1_loss = torch.stack(l1_vals).mean() if l1_vals else torch.tensor(0.0, device=lm_logits.device)
        iou_loss = torch.stack(iou_vals).mean() if iou_vals else torch.tensor(0.0, device=lm_logits.device)
        count_loss = torch.tensor(count_vals, device=lm_logits.device, dtype=lm_logits.dtype).mean() if count_vals else torch.tensor(0.0, device=lm_logits.device)
        
        match_rate = (matched_gt / total_gt) if total_gt else 0.0
        return l1_loss, iou_loss, count_loss, match_rate

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
        
        # Compute language modeling loss
        vocab_size = lm_logits.size(-1)
        lm_loss = F.cross_entropy(
            lm_logits.view(-1, vocab_size), 
            lm_labels.view(-1), 
            ignore_index=-100
        )
        
        # Update curriculum tracking
        self._update_lm_loss_tracking(lm_loss)
        
        # Parse predicted boxes for both detection loss computation and curriculum assessment
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
                    parsed = self._parse_detection_string(det)
                    if parsed is not None:
                        boxes.append(parsed)
                target_boxes.append(boxes)

        # Compute detection losses
        l1_loss, iou_loss, count_loss, match_rate = self.compute_detection_loss(lm_logits, target_boxes)
        
        # Apply adaptive weights to detection losses
        adaptive_lambda_detection = self.lambda_detection * weight_multiplier
        detection_loss = (self.lambda_l1 * l1_loss + 
                            self.lambda_iou * iou_loss + 
                            self.lambda_count * count_loss)
        
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
                "det_weight": _val(adaptive_lambda_detection),
                "gt_match_rate": match_rate,
            }
            cur_dict = self.get_curriculum_status()
            self.logger.info(f"LOSS STATUS: {loss_dict} | CURRICULUM STATUS: {cur_dict}")

        return total_loss

    def get_curriculum_status(self):
        """Get current curriculum learning status for logging/debugging."""
        return {
            'lm_loss_ema': self.lm_loss_ema,
            'lm_target': self.lm_target,
            'current_weight_multiplier': self._compute_adaptive_weights() if self.lm_loss_ema else None,
            'step_count': self.step_count
        }


def create_compute_loss_func(
    tokenizer,
    *,
    lambda_l1=0.35,
    lambda_iou=0.5,
    lambda_count=0.1,
    lambda_detection=0.15,
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
        lm_target=lm_target,
        logger=logger,
        log_interval=log_interval,
    )
    return loss_computer