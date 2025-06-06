import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import re
import logging

class CompositeLoss(nn.Module):
    """
    Composite loss combining language modeling, classification, and bounding box losses for multi-object detection.
    Supports joint supervision of output format (text) and detection/classification accuracy.
    """
    def __init__(self, classification_weight=0.2, coordinate_weight=0.8, iou_weight=0.7, l1_weight=0.3, lm_weight=0.1, detection_weight=1.0):
        super().__init__()
        self.classification_weight = classification_weight  # Classification loss weight (default: 0.2)
        self.coordinate_weight = coordinate_weight          # Bounding box loss weight (default: 0.8)
        # normalize iou and l1 weights to sum to 1.0
        total_coord_weight = iou_weight + l1_weight
        self.iou_weight = iou_weight / total_coord_weight  # IoU loss weight (default: 0.7)
        self.l1_weight = l1_weight / total_coord_weight    # L1 loss weight (default: 0.3)
        self.lm_weight = lm_weight  # Language modeling loss weight (default: 0.1)
        self.detection_weight = detection_weight

    def parse_multi_object_from_text(self, text, class_map=None, reference_tensor=None):
        """
        Parse multiple objects (class and bbox) from a prediction string.
        
        Parameters:
            - text: string containing zero or more 'Class: [x1, y1, x2, y2]' patterns
            - class_map: dict mapping class names to integer labels (optional)
            - reference_tensor: tensor to match dtype and device from (optional)
            
        Returns:
            - classes: list of class indices (or names if class_map is None)
            - boxes: tensor of shape [num_objects, 4] (or empty tensor)
        """
        if not text:
            logging.debug(f"[CompositeLoss] Empty text input to parse_multi_object_from_text.")
            return [], torch.empty((0, 4), dtype=torch.float32)
        # Pattern: Class: [x1, y1, x2, y2] (allow for multiple, separated by ; or newlines)
        pattern = r'([\w\- ]+):\s*[\[(](-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?),?\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?),?\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?),?\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)[\])]'  # noqa
        matches = re.findall(pattern, str(text))
        logging.debug(f"[CompositeLoss] Parsing text: {text}")
        logging.debug(f"[CompositeLoss] Found matches: {matches}")
        classes = []
        boxes = []
        for match in matches:
            class_name = match[0].strip()
            coords = [float(match[i]) for i in range(1, 5)]
            # basic sanity check
            if coords[2] > coords[0] and coords[3] > coords[1]:
                if class_map is not None:
                    classes.append(class_map.get(class_name, -1))
                else:
                    classes.append(class_name)
                boxes.append(coords)
        logging.debug(f"[CompositeLoss] Parsed classes: {classes}")
        logging.debug(f"[CompositeLoss] Parsed boxes: {boxes}")
        # dtype
        dtype = reference_tensor.dtype if reference_tensor is not None else torch.float32
        if boxes:
            return classes, torch.tensor(boxes, dtype=dtype)
        else:
            return [], torch.empty((0, 4), dtype=dtype)

    def match_predictions_to_targets(self, pred_boxes, target_boxes):
        """
        Greedy IoU matching between predicted and target boxes.
        Returns list of (pred_idx, target_idx) pairs.
        """
        if len(pred_boxes) == 0 or len(target_boxes) == 0:
            return []
        ious = ops.box_iou(pred_boxes, target_boxes)  # [num_pred, num_target]
        matches = []
        used_pred = set()
        used_target = set()
        # Greedy matching
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

    def forward(self, lm_logits, lm_labels, class_logits, box_preds, target_labels, target_boxes=None, target_text=None, class_map=None, return_components=False):
        """
        Compute weighted combination of language modeling, classification, and bounding box losses.
        Parameters:
            - lm_logits: language model logits [B, seq_len, vocab_size]
            - lm_labels: ground truth token ids [B, seq_len]
            - class_logits: detection/classification logits [B, num_visual_tokens, num_classes]
            - box_preds: predicted bounding boxes [B, num_visual_tokens, 4]
            - target_labels: ground truth class indices [B, num_objects]
            - target_boxes: ground truth boxes [B, num_objects, 4]
            - target_text: (optional) ground truth text for parsing
            - class_map: (optional) class name to index mapping
            - return_components: whether to return granular loss components
        Returns:
            - total_loss: weighted sum of all losses
        """
        device = lm_logits.device
        logging.debug(f"[CompositeLoss] lm_logits shape: {lm_logits.shape}, lm_labels shape: {lm_labels.shape}")
        logging.debug(f"[CompositeLoss] class_logits shape: {class_logits.shape}, box_preds shape: {box_preds.shape}")
        if target_labels is not None:
            logging.debug(f"[CompositeLoss] target_labels: {target_labels}")
        if target_boxes is not None:
            logging.debug(f"[CompositeLoss] target_boxes: {target_boxes}")
        # Language modeling loss
        lm_loss = nn.functional.cross_entropy(
            lm_logits.view(-1, lm_logits.size(-1)),
            lm_labels.view(-1),
            ignore_index=-100
        )
        # Detection/classification loss (as before)
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
            logging.debug(f"[CompositeLoss] Batch {i}: pred_boxes shape: {pred_boxes.shape}, pred_logits shape: {pred_logits.shape}")
            logging.debug(f"[CompositeLoss] Batch {i}: tgt_boxes shape: {tgt_boxes.shape}, tgt_classes: {tgt_classes}")
            num_classes = pred_logits.shape[-1]
            total_gt_objects += len(tgt_classes)
            if len(tgt_boxes) > 0 and len(pred_boxes) > 0:
                ious = torchvision.ops.box_iou(pred_boxes, tgt_boxes)
                logging.debug(f"[CompositeLoss] Batch {i}: IoU matrix: {ious}")
                matches = []
                used_pred = set()
                used_target = set()
                while True:
                    max_iou = torch.max(ious)
                    if max_iou < 0.5:
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
                for pred_idx, tgt_idx in matches:
                    classification_loss = nn.functional.cross_entropy(
                        pred_logits[pred_idx].unsqueeze(0),
                        tgt_classes[tgt_idx].unsqueeze(0),
                        reduction='mean'
                    )
                    iou_loss = torchvision.ops.generalized_box_iou_loss(
                        pred_boxes[pred_idx].unsqueeze(0),
                        tgt_boxes[tgt_idx].unsqueeze(0),
                        reduction='mean'
                    )
                    l1_loss = nn.functional.smooth_l1_loss(
                        pred_boxes[pred_idx].unsqueeze(0),
                        tgt_boxes[tgt_idx].unsqueeze(0),
                        reduction='mean'
                    )
                    bbox_loss = self.coordinate_weight * (self.iou_weight * iou_loss + self.l1_weight * l1_loss)
                    logging.debug(f"[CompositeLoss] Batch {i}: pred_idx={pred_idx}, tgt_idx={tgt_idx}, class_loss={classification_loss.item()}, iou_loss={iou_loss.item()}, l1_loss={l1_loss.item()}, bbox_loss={bbox_loss.item()}")
                    total_loss += self.classification_weight * classification_loss + bbox_loss
                    # Metrics
                    pred_class = pred_logits[pred_idx].argmax().item()
                    true_class = tgt_classes[tgt_idx].item() if isinstance(tgt_classes, torch.Tensor) else tgt_classes[tgt_idx]
                    if pred_class == true_class:
                        total_correct_classes += 1
                    pred_box = pred_boxes[pred_idx].unsqueeze(0)
                    tgt_box = tgt_boxes[tgt_idx].unsqueeze(0)
                    iou = torchvision.ops.box_iou(pred_box, tgt_box)[0, 0].item()
                    total_iou_sum += iou
                    total_iou_matches += 1
                    total_l1_sum += l1_loss.item()
                # Penalize unmatched predictions (false positives)
                unmatched_preds = set(range(len(pred_boxes))) - used_pred
                for pred_idx in unmatched_preds:
                    unmatched_penalty = 0.5 * self.coordinate_weight * self.l1_weight
                    unmatched_cls_penalty = 0.5 * self.classification_weight
                    total_loss += unmatched_penalty
                    total_loss += unmatched_cls_penalty
                # Penalize unmatched ground truths (false negatives)
                unmatched_targets = set(range(len(tgt_boxes))) - used_target
                for tgt_idx in unmatched_targets:
                    unmatched_penalty = 0.5 * self.coordinate_weight * self.l1_weight
                    unmatched_cls_penalty = 0.5 * self.classification_weight
                    total_loss += unmatched_penalty
                    total_loss += unmatched_cls_penalty
            elif len(pred_boxes) > 0 and len(tgt_boxes) > 0:
                # Mean L1 distance between all pairs
                l1_matrix = torch.cdist(pred_boxes, tgt_boxes, p=1)
                l1_loss = l1_matrix.mean()
                total_loss += self.coordinate_weight * self.l1_weight * l1_loss
                total_l1_sum += l1_loss.item()
                # Penalize all predictions and ground truths as unmatched
                for _ in range(len(pred_boxes)):
                    unmatched_penalty = 0.5 * self.coordinate_weight * self.l1_weight
                    unmatched_cls_penalty = 0.5 * self.classification_weight
                    total_loss += unmatched_penalty
                    total_loss += unmatched_cls_penalty
                for tgt_idx in range(len(tgt_boxes)):
                    unmatched_penalty = 0.5 * self.coordinate_weight * self.l1_weight
                    unmatched_cls_penalty = 0.5 * self.classification_weight
                    total_loss += unmatched_penalty
                    total_loss += unmatched_cls_penalty
            elif len(pred_boxes) == 0 and len(tgt_boxes) > 0:
                for tgt_idx in range(len(tgt_boxes)):
                    unmatched_penalty = 0.5 * self.coordinate_weight * self.l1_weight
                    unmatched_cls_penalty = 0.5 * self.classification_weight
                    total_loss += unmatched_penalty
                    total_loss += unmatched_cls_penalty
            elif len(pred_boxes) > 0 and len(tgt_boxes) == 0:
                for _ in range(len(pred_boxes)):
                    unmatched_penalty = 0.5 * self.coordinate_weight * self.l1_weight
                    unmatched_cls_penalty = 0.5 * self.classification_weight
                    total_loss += unmatched_penalty
                    total_loss += unmatched_cls_penalty
        if total_iou_matches > 0:
            detection_loss = total_loss / total_iou_matches
        else:
            detection_loss = total_loss / batch_size if batch_size > 0 else torch.tensor(1.0, device=device, requires_grad=True)
        total_loss = self.lm_weight * lm_loss + self.detection_weight * detection_loss
        logging.debug(f"[CompositeLoss] total_loss: {total_loss}, total_correct_classes: {total_correct_classes}, total_gt_objects: {total_gt_objects}, total_iou_sum: {total_iou_sum}, total_iou_matches: {total_iou_matches}, total_l1_sum: {total_l1_sum}")
        if return_components:
            return (total_loss, total_correct_classes, total_gt_objects, total_iou_sum, total_iou_matches, total_l1_sum)
        else:
            return total_loss

    @staticmethod
    def compute_iou(pred_text, target_text, class_map=None):
        """
        Compute mean IoU between predicted and target boxes for validation.
        Parses all objects from both texts and matches them by greedy IoU.
        
        Parameters:
            - pred_text: predicted text string (multi-object)
            - target_text: ground truth text string (multi-object)
            - class_map: dict mapping class names to integer labels (optional)
            
        Returns:
            - mean_iou: mean IoU over matched pairs (float)
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
        