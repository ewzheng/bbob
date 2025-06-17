import torch
import torch.nn.functional as F
import re
from scipy.optimize import linear_sum_assignment

def _parse_boxes(logits, tokenizer):
    """Decode logits and extract detection strings.

    Args:
        logits (torch.Tensor): Model output logits with shape (batch_size, seq_len, vocab_size).
        tokenizer: Tokenizer that provides a `decode(ids, **kwargs)` method.

    Returns:
        List[List[str]]: A list with one entry per batch element, containing all detection strings
        (the raw content found between <detect> and </detect> tokens).
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

        # Use regex to find every substring wrapped by <detect> ... </detect>
        # The non-greedy qualifier (.*?) ensures we catch individual detections.
        matches = re.findall(r"<detect>(.*?)</detect>", decoded_text)

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

    cost = (1.0 - iou_mat).cpu().numpy()
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

    def __init__(self, tokenizer, lambda_l1=0.35, lambda_iou=0.5, lambda_count=0.1, lambda_detection=0.3,
                 lm_target=1.5,
                 smoothing_factor=0.9):
        self.tokenizer = tokenizer
        self.lambda_l1 = lambda_l1
        self.lambda_iou = lambda_iou
        self.lambda_count = lambda_count
        self.lambda_detection = lambda_detection
        
        # Curriculum parameters
        self.lm_target = lm_target
        self.min_detection_weight = lambda_detection
        self.max_detection_weight = lambda_detection*3
        self.smoothing_factor = smoothing_factor
        
        # Tracking variables
        self.lm_loss_ema = None  # Exponential moving average of LM loss
        self.step_count = 0

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
        
        # Check language modeling readiness
        lm_ready = self.lm_loss_ema <= self.lm_target
        
        # Check format compliance if predictions are provided
        format_ready = True  # Default to True if no predictions to check
        if parsed_predictions is not None:
            total_predictions = sum(len(pred_list) for pred_list in parsed_predictions)
            if total_predictions > 0:
                valid_predictions = sum(
                    len([p for p in pred_list if self._parse_detection_string(p) is not None]) 
                    for pred_list in parsed_predictions
                )
                format_success_rate = valid_predictions / total_predictions
                format_ready = format_success_rate >= 0.8  # 80% of boxes must be parseable
            else:
                format_ready = False  # No predictions at all
        
        # Two-stage curriculum logic
        if lm_ready and format_ready:
            # Stage 2: Both LM and format are good - focus on detection accuracy
            return self.max_detection_weight
        elif lm_ready and not format_ready:
            # Stage 1.5: LM is good but format still learning - minimal detection weight
            return self.min_detection_weight
        else:
            # Stage 1: Still learning basic language - very low detection weight
            return self.min_detection_weight

    def compute_detection_loss(self, lm_logits, target_boxes):
        """Compute detection losses (L1 + IoU + count penalty) from parsed model outputs."""
        if target_boxes is None:
            return torch.tensor(0.0, device=lm_logits.device), torch.tensor(0.0, device=lm_logits.device), torch.tensor(0.0, device=lm_logits.device)

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

        # Aggregate losses
        l1_loss = torch.stack(l1_vals).mean() if l1_vals else torch.tensor(0.0, device=lm_logits.device)
        iou_loss = torch.stack(iou_vals).mean() if iou_vals else torch.tensor(0.0, device=lm_logits.device)
        count_loss = torch.tensor(count_vals, device=lm_logits.device, dtype=lm_logits.dtype).mean() if count_vals else torch.tensor(0.0, device=lm_logits.device)
        
        return l1_loss, iou_loss, count_loss

    def __call__(self, model, inputs, return_outputs=False):
        """
        Loss function with adaptive curriculum learning.
        
        Args:
            model: The model being trained
            inputs: Dictionary containing input_ids, attention_mask, labels, and optionally target_boxes
            return_outputs: Whether to return model outputs along with loss
            
        Returns:
            loss (torch.Tensor) or tuple of (loss, outputs) if return_outputs=True
        """
        # Extract target boxes from inputs if available
        target_boxes = inputs.pop("target_boxes", None)
        
        # Forward pass
        outputs = model(**inputs)
        
        # Get logits and labels
        lm_logits = outputs.logits
        lm_labels = inputs.get("labels")
        
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
        
        # Compute detection losses
        l1_loss, iou_loss, count_loss = self.compute_detection_loss(lm_logits, target_boxes)
        
        # Apply adaptive weights to detection losses
        adaptive_lambda_detection = self.lambda_detection * weight_multiplier
        detection_loss = (self.lambda_l1 * l1_loss + 
                            self.lambda_iou * iou_loss + 
                            self.lambda_count * count_loss)
        
        # Total loss with adaptive weighting
        total_loss = lm_loss + adaptive_lambda_detection * detection_loss
        
        # Store loss components and curriculum info in outputs for logging
        outputs.lm_loss = lm_loss.detach()
        outputs.l1_loss = l1_loss.detach()
        outputs.iou_loss = iou_loss.detach()
        outputs.count_loss = count_loss.detach()
        outputs.loss = total_loss
        
        # Store curriculum information for monitoring
        outputs.lm_loss_ema = self.lm_loss_ema
        outputs.detection_weight_multiplier = weight_multiplier
        outputs.adaptive_lambda_detection = adaptive_lambda_detection
        
        if return_outputs:
            return total_loss, outputs
        return total_loss

    def get_curriculum_status(self):
        """Get current curriculum learning status for logging/debugging."""
        return {
            'lm_loss_ema': self.lm_loss_ema,
            'lm_target': self.lm_target,
            'current_weight_multiplier': self._compute_adaptive_weights() if self.lm_loss_ema else None,
            'step_count': self.step_count
        }


def create_compute_loss_func(tokenizer, lambda_l1=0.35, lambda_iou=0.5, lambda_count=0.1, lambda_detection=0.3,
                           lm_target=2.0, min_detection_weight=0.1, max_detection_weight=2.0):
    """Factory function to create a loss computer"""
    loss_computer = CompositeLoss(
        tokenizer=tokenizer,
        lambda_l1=lambda_l1, 
        lambda_iou=lambda_iou, 
        lambda_count=lambda_count,
        lambda_detection=lambda_detection,
        lm_target=lm_target,
        min_detection_weight=min_detection_weight,
        max_detection_weight=max_detection_weight
    )
    return loss_computer