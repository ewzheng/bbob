import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import re
import numpy as np
from scipy.optimize import linear_sum_assignment

class CompositeLoss(nn.Module):
    """
    Composite loss for multimodal object detection and language modeling.
    Combines language modeling loss, classification loss, and bounding box regression loss.
    Supports Hungarian matching for object assignment and applies penalties for unmatched predictions/ground truths.
    """
    def __init__(self, classification_weight=0.4, coordinate_weight=0.6, iou_weight=0.75, l1_weight=0.25, lm_weight=0.2, detection_weight=0.8):
        """
        Initialize the composite loss module.
        Args:
            classification_weight (float): Weight for classification loss.
            coordinate_weight (float): Weight for bounding box regression loss.
            iou_weight (float): Weight for IoU loss (relative to l1_weight).
            l1_weight (float): Weight for L1 loss (relative to iou_weight).
            lm_weight (float): Weight for language modeling loss.
            detection_weight (float): Weight for detection/classification loss.
        """
        super().__init__()
        self.classification_weight = classification_weight
        self.coordinate_weight = coordinate_weight
        # Normalize IoU and L1 weights to sum to 1.0
        total_coord_weight = iou_weight + l1_weight
        self.iou_weight = iou_weight / total_coord_weight
        self.l1_weight = l1_weight / total_coord_weight
        self.lm_weight = lm_weight
        self.detection_weight = detection_weight

    def parse_multi_object_from_text(self, text, class_map=None, reference_tensor=None):
        """
        Parse multiple objects (class and bbox) from a prediction string.
        Args:
            text (str): String containing zero or more 'Class: [x1, y1, x2, y2]' patterns.
            class_map (dict, optional): Mapping from class names to integer labels.
            reference_tensor (Tensor, optional): Tensor to match dtype and device.
        Returns:
            classes (list): List of class indices (or names if class_map is None).
            boxes (Tensor): Tensor of shape [num_objects, 4] (or empty tensor).
        """
        if not text:
            return [], torch.empty((0, 4), dtype=torch.float32)
        # Regex pattern for 'Class: [x1, y1, x2, y2]'
        pattern = r'([\w\- ]+):\s*[\[(](-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?),?\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?),?\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?),?\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)[\])]'  # noqa
        matches = re.findall(pattern, str(text))
        classes = []
        boxes = []
        for match in matches:
            class_name = match[0].strip()
            coords = [float(match[i]) for i in range(1, 5)]
            # Only keep boxes with positive area
            if coords[2] > coords[0] and coords[3] > coords[1]:
                if class_map is not None:
                    classes.append(class_map.get(class_name, -1))
                else:
                    classes.append(class_name)
                boxes.append(coords)
        dtype = reference_tensor.dtype if reference_tensor is not None else torch.float32
        if boxes:
            return classes, torch.tensor(boxes, dtype=dtype)
        else:
            return [], torch.empty((0, 4), dtype=dtype)

    def match_predictions_to_targets(self, pred_boxes, target_boxes):
        """
        Greedy IoU matching between predicted and target boxes.
        Args:
            pred_boxes (Tensor): Predicted boxes [N, 4].
            target_boxes (Tensor): Ground truth boxes [M, 4].
        Returns:
            matches (list): List of (pred_idx, target_idx) pairs.
        """
        if len(pred_boxes) == 0 or len(target_boxes) == 0:
            return []
        ious = ops.box_iou(pred_boxes, target_boxes)  # [num_pred, num_target]
        matches = []
        used_pred = set()
        used_target = set()
        # Greedy matching: repeatedly pick the highest IoU pair
        while True:
            max_iou = torch.max(ious)
            if max_iou <= 0:
                break
            idx = torch.argmax(ious)
            pred_idx, target_idx = divmod(idx.item(), ious.shape[1])
            if pred_idx in used_pred or target_idx in used_target:
                ious[pred_idx, target_idx] = -1
                continue
            matches.append((pred_idx, target_idx))
            used_pred.add(pred_idx)
            used_target.add(target_idx)
            ious[pred_idx, :] = -1
            ious[:, target_idx] = -1
        return matches

    def coco_to_corners(self, boxes):
        """
        Convert bounding boxes from COCO format [x, y, w, h] to [x1, y1, x2, y2] format.
        Args:
            boxes (Tensor): [N, 4] boxes in COCO format.
        Returns:
            Tensor: [N, 4] boxes in corner format.
        """
        if boxes.numel() == 0:
            return boxes
        x1 = boxes[..., 0]
        y1 = boxes[..., 1]
        x2 = x1 + boxes[..., 2]
        y2 = y1 + boxes[..., 3]
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def forward(self, lm_logits, lm_labels, class_logits, box_preds, target_labels, target_boxes=None, target_text=None, class_map=None, return_components=False):
        """
        Compute the total loss as a weighted sum of language modeling, classification, and bounding box losses.
        Uses Hungarian matching for assignment and applies penalties for unmatched predictions/ground truths.
        Args:
            lm_logits (Tensor): Language model logits [B, seq_len, vocab_size].
            lm_labels (Tensor): Ground truth token ids [B, seq_len].
            class_logits (Tensor): Detection/classification logits [B, num_visual_tokens, num_classes].
            box_preds (Tensor): Predicted bounding boxes [B, num_visual_tokens, 4].
            target_labels (Tensor): Ground truth class indices [B, num_objects].
            target_boxes (Tensor, optional): Ground truth boxes [B, num_objects, 4].
            target_text (str, optional): Ground truth text for parsing (not used in this loss).
            class_map (dict, optional): Class name to index mapping.
            return_components (bool): Whether to return granular loss components for logging.
        Returns:
            total_loss (Tensor): Weighted sum of all losses.
            (If return_components=True, also returns metrics for logging.)
        """
        device = lm_logits.device
        # Language modeling loss (cross-entropy, ignoring -100 labels)
        lm_loss = nn.functional.cross_entropy(
            lm_logits.view(-1, lm_logits.size(-1)),
            lm_labels.view(-1),
            ignore_index=-100
        )
        batch_size = class_logits.shape[0]
        total_loss = 0.0
        total_correct_classes = 0
        total_gt_objects = 0
        total_iou_sum = 0.0
        total_iou_matches = 0
        total_l1_sum = 0.0
        for i in range(batch_size):
            pred_boxes = box_preds[i]
            pred_logits = class_logits[i]
            tgt_boxes = target_boxes[i] if target_boxes is not None else torch.empty((0, 4), dtype=box_preds.dtype, device=box_preds.device)
            tgt_classes = target_labels[i] if isinstance(target_labels[i], torch.Tensor) else torch.tensor(target_labels[i], device=pred_logits.device)
            # Filter out padded labels and degenerate boxes
            if tgt_boxes.numel() > 0 and tgt_classes.numel() > 0:
                valid_mask = (tgt_classes != -100) & (tgt_boxes[:, 2] > 0) & (tgt_boxes[:, 3] > 0)
                tgt_boxes = tgt_boxes[valid_mask]
                tgt_classes = tgt_classes[valid_mask]
            elif tgt_boxes.numel() > 0:
                keep = (tgt_boxes[:, 2] > 0) & (tgt_boxes[:, 3] > 0)
                tgt_boxes = tgt_boxes[keep]
            elif tgt_classes.numel() > 0:
                keep = (tgt_classes != -100)
                tgt_classes = tgt_classes[keep]
            num_classes = pred_logits.shape[-1]
            # Sanity checks for labels and boxes
            if tgt_classes.numel() > 0:
                assert not torch.isnan(tgt_classes).any(), f"NaN in target labels: {tgt_classes}"
                assert not torch.isinf(tgt_classes).any(), f"Inf in target labels: {tgt_classes}"
                assert ((tgt_classes == -100) | ((tgt_classes >= 0) & (tgt_classes < num_classes))).all(), f"Invalid target label: {tgt_classes}, num_classes: {num_classes}"
            # Convert boxes to [x1, y1, x2, y2] for IoU/loss
            pred_boxes_corners = self.coco_to_corners(pred_boxes)
            tgt_boxes_corners = self.coco_to_corners(tgt_boxes)
            # Sanity checks for boxes
            for box_tensor, name in [(pred_boxes_corners, 'pred_boxes'), (tgt_boxes_corners, 'tgt_boxes')]:
                if box_tensor.numel() > 0:
                    assert not torch.isnan(box_tensor).any(), f"NaN in {name}: {box_tensor}"
                    assert not torch.isinf(box_tensor).any(), f"Inf in {name}: {box_tensor}"
            total_gt_objects += len(tgt_classes)
            if len(tgt_boxes) > 0 and len(pred_boxes) > 0 and len(tgt_classes) > 0:
                ious = torchvision.ops.box_iou(pred_boxes_corners, tgt_boxes_corners)
                # Hungarian matching for optimal assignment
                cost_matrix = -ious.detach().cpu().numpy()  # maximize IoU = minimize -IoU
                pred_indices, target_indices = linear_sum_assignment(cost_matrix)
                matches = []
                used_pred = set()
                used_target = set()
                for p, t in zip(pred_indices, target_indices):
                    if ious[p, t] >= 0.5:
                        matches.append((p, t))
                        used_pred.add(p)
                        used_target.add(t)
                for pred_idx, tgt_idx in matches:
                    # Classification loss for matched pairs
                    classification_loss = nn.functional.cross_entropy(
                        pred_logits[pred_idx].unsqueeze(0),
                        tgt_classes[tgt_idx].unsqueeze(0),
                        reduction='mean'
                    )
                    # Bounding box regression losses (IoU and L1)
                    iou_loss = torchvision.ops.generalized_box_iou_loss(
                        pred_boxes_corners[pred_idx].unsqueeze(0),
                        tgt_boxes_corners[tgt_idx].unsqueeze(0),
                        reduction='mean'
                    )
                    l1_loss = nn.functional.smooth_l1_loss(
                        pred_boxes_corners[pred_idx].unsqueeze(0),
                        tgt_boxes_corners[tgt_idx].unsqueeze(0),
                        reduction='mean'
                    )
                    bbox_loss = self.coordinate_weight * (self.iou_weight * iou_loss + self.l1_weight * l1_loss)
                    total_loss += self.classification_weight * classification_loss + bbox_loss
                    # Metrics for logging
                    pred_class = pred_logits[pred_idx].argmax().item()
                    true_class = tgt_classes[tgt_idx].item() if isinstance(tgt_classes, torch.Tensor) else tgt_classes[tgt_idx]
                    if pred_class == true_class:
                        total_correct_classes += 1
                    pred_box = pred_boxes_corners[pred_idx].unsqueeze(0)
                    tgt_box = tgt_boxes_corners[tgt_idx].unsqueeze(0)
                    iou = torchvision.ops.box_iou(pred_box, tgt_box)[0, 0].item()
                    total_iou_sum += iou
                    total_iou_matches += 1
                    total_l1_sum += l1_loss.item()
                # Penalties for unmatched predictions (false positives)
                unmatched_preds = set(range(len(pred_boxes))) - used_pred
                for pred_idx in unmatched_preds:
                    unmatched_penalty = 0.1 * self.coordinate_weight * self.l1_weight  # Penalty for unmatched box
                    unmatched_cls_penalty = 0.075 * self.classification_weight  # Penalty for unmatched class
                    total_loss += unmatched_penalty
                    total_loss += unmatched_cls_penalty
                # Penalties for unmatched ground truths (false negatives)
                unmatched_targets = set(range(len(tgt_boxes))) - used_target
                for tgt_idx in unmatched_targets:
                    unmatched_penalty = 0.2 * self.coordinate_weight * self.l1_weight
                    unmatched_cls_penalty = 0.1 * self.classification_weight
                    total_loss += unmatched_penalty
                    total_loss += unmatched_cls_penalty
            elif len(pred_boxes) > 0 and len(tgt_boxes) > 0 and len(tgt_classes) == 0:
                # No valid targets, skip loss computation for this sample
                continue
            elif len(pred_boxes) > 0 and len(tgt_boxes) == 0:
                # All predictions are unmatched (false positives)
                for _ in range(len(pred_boxes)):
                    unmatched_penalty = 0.15 * self.coordinate_weight * self.l1_weight
                    unmatched_cls_penalty = 0.075 * self.classification_weight
                    total_loss += unmatched_penalty
                    total_loss += unmatched_cls_penalty
            elif len(pred_boxes) == 0 and len(tgt_boxes) > 0:
                # All ground truths are unmatched (false negatives)
                for tgt_idx in range(len(tgt_boxes)):
                    unmatched_penalty = 0.15 * self.coordinate_weight * self.l1_weight
                    unmatched_cls_penalty = 0.075 * self.classification_weight
                    total_loss += unmatched_penalty
                    total_loss += unmatched_cls_penalty
        # Normalize detection loss by number of matches (or batch size if no matches)
        if total_iou_matches > 0:
            detection_loss = total_loss / total_iou_matches
        else:
            detection_loss = total_loss / batch_size if batch_size > 0 else torch.tensor(1.0, device=device, requires_grad=True)
        # Weighted sum of language modeling and detection/classification loss
        total_loss = self.lm_weight * lm_loss + self.detection_weight * detection_loss
        if return_components:
            return (total_loss, total_correct_classes, total_gt_objects, total_iou_sum, total_iou_matches, total_l1_sum)
        else:
            return total_loss

    @staticmethod
    def compute_iou(pred_text, target_text, class_map=None):
        """
        Compute mean IoU between predicted and target boxes for validation.
        Parses all objects from both texts and matches them by greedy IoU.
        Args:
            pred_text (str): Predicted text string (multi-object).
            target_text (str): Ground truth text string (multi-object).
            class_map (dict, optional): Class name to index mapping.
        Returns:
            mean_iou (float): Mean IoU over matched pairs.
        """
        def parse(text):
            pattern = r'([\w\- ]+):\s*[\[(](-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?),?\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?),?\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?),?\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)[\])]'  # noqa
            matches = re.findall(pattern, str(text))
            boxes = []
            for match in matches:
                coords = [float(match[i]) for i in range(1, 5)]
                if coords[2] > coords[0] and coords[3] > coords[1]:
                    boxes.append(coords)
            if boxes:
                return torch.tensor(boxes, dtype=torch.float32)
            else:
                return torch.empty((0, 4), dtype=torch.float32)
        pred_boxes = parse(pred_text)
        target_boxes = parse(target_text)
        if len(pred_boxes) == 0 or len(target_boxes) == 0:
            return 0.0
        ious = ops.box_iou(pred_boxes, target_boxes)
        if ious.numel() == 0:
            return 0.0
        max_ious, _ = ious.max(dim=1)
        return max_ious.mean().item()
        