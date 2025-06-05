import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import re

class CompositeLoss(nn.Module):
    """
    Composite loss combining classification and bounding box losses
    
    Parameters:
        - classification_weight: weight for classification loss component
        - coordinate_weight: weight for coordinate loss component
        - iou_weight: weight for GIoU loss component (within coordinate loss)
        - l1_weight: weight for L1 loss component (within coordinate loss)
    """
    def __init__(self, classification_weight=0.9, coordinate_weight=0.1, iou_weight=0.67, l1_weight=0.33):
        super().__init__()
        self.classification_weight = classification_weight
        self.coordinate_weight = coordinate_weight
        
        # normalize iou and l1 weights to sum to 1.0
        total_coord_weight = iou_weight + l1_weight
        self.iou_weight = iou_weight / total_coord_weight
        self.l1_weight = l1_weight / total_coord_weight
        

    def parse_bbox_from_text(self, text_list, reference_tensor=None):
        """
        Parse bounding boxes from text predictions
        
        Parameters:
            - text_list: list of text strings containing potential bbox coordinates
            - reference_tensor: tensor to match dtype and device from
            
        Returns:
            - parsed_boxes: tensor of parsed boxes [N, 4] or None if no valid boxes
            - valid_mask: boolean mask indicating which samples have valid boxes
        """
        if not text_list:  # handle empty list
            return None, torch.tensor([], dtype=torch.bool)
            
        parsed_boxes = []
        valid_mask = []
        
        # improved regex - handles negative numbers and scientific notation
        bbox_pattern = r'[\[\(](-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?),?\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?),?\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?),?\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)[\]\)]'
        
        for text in text_list:
            match = re.search(bbox_pattern, str(text))
            if match:
                try:
                    coords = [float(match.group(i)) for i in range(1, 5)]
                    # basic sanity check - ensure x2 > x1 and y2 > y1
                    if coords[2] > coords[0] and coords[3] > coords[1]:
                        parsed_boxes.append(coords)
                        valid_mask.append(True)
                    else:
                        parsed_boxes.append([0.0, 0.0, 0.0, 0.0])  # dummy box
                        valid_mask.append(False)
                except (ValueError, IndexError):
                    parsed_boxes.append([0.0, 0.0, 0.0, 0.0])  # dummy box
                    valid_mask.append(False)
            else:
                parsed_boxes.append([0.0, 0.0, 0.0, 0.0])  # dummy box
                valid_mask.append(False)
        
        # determine dtype from reference tensor or default to float32
        if reference_tensor is not None:
            dtype = reference_tensor.dtype
        else:
            dtype = torch.float32
            
        if any(valid_mask):
            return torch.tensor(parsed_boxes, dtype=dtype), torch.tensor(valid_mask)
        else:
            return None, torch.tensor(valid_mask)

    def forward(self, pred_logits, pred_text, target_labels, target_boxes=None):
        """
        Compute weighted combination of classification and bounding box losses
        
        Parameters:
            - pred_logits: classification logits [N, num_classes]
            - pred_text: list of predicted text strings containing potential bbox coordinates
            - target_labels: ground truth classification labels [N]
            - target_boxes: ground truth bounding boxes [N, 4] format (x1, y1, x2, y2)
            
        Returns:
            - total_loss: weighted sum of classification and bbox losses
        """
        
        # get device from pred_logits for consistency
        device = pred_logits.device
        
        # classification loss - always computed
        classification_loss = F.cross_entropy(pred_logits, target_labels, reduction='mean')
        total_loss = self.classification_weight * classification_loss
        
        # bounding box loss - only if valid boxes are parsed and targets provided
        if target_boxes is not None and pred_text:
            parsed_boxes, valid_mask = self.parse_bbox_from_text(pred_text, reference_tensor=target_boxes)
            
            if parsed_boxes is not None and valid_mask.any():
                # ensure all tensors are on same device
                parsed_boxes = parsed_boxes.to(device)
                valid_mask = valid_mask.to(device)
                target_boxes = target_boxes.to(device)
                
                # only compute bbox loss for samples with valid parsed boxes
                valid_pred_boxes = parsed_boxes[valid_mask]
                valid_target_boxes = target_boxes[valid_mask]
                
                if len(valid_pred_boxes) > 0:
                    iou_loss = ops.generalized_box_iou_loss(valid_pred_boxes, valid_target_boxes, reduction='mean')
                    l1_loss = F.smooth_l1_loss(valid_pred_boxes, valid_target_boxes, reduction='mean')
                    
                    bbox_loss = self.coordinate_weight * (self.iou_weight * iou_loss + self.l1_weight * l1_loss)
                    total_loss += bbox_loss
        
        return total_loss
        