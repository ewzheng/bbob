import torch
import torch.nn.functional as F
from .loss_helpers import (
    decode_pred_gt,
    TAG_OPEN,
    TAG_CLOSE,
    hungarian_match,
    xywh_to_xyxy,
    ids_to_boxes_labels,
)

# Torchvision losses – fall back gracefully if not available
try:
    from torchvision.ops import complete_box_iou_loss as _ciou_loss_fn  # type: ignore
except ImportError:  # pragma: no cover
    _ciou_loss_fn = None
from torchvision.ops import generalized_box_iou as _g_box_iou  # type: ignore

class BBOBLoss:
    """Minimal language-model cross-entropy loss for BBOB.

    This treats the entire detection string as ordinary text; supervision is
    plain autoregressive CE with an ignore-index mask supplied by the collator.
    The class still exposes a ``digit_ids`` attribute so that BBOBTrainer's
    guided-sampling helper continues to work unchanged.
    """

    def __init__(self, tokenizer, *, ignore_index: int = -100, logger=None, log_interval: int = 100,
                 lambda_digit: float = 0.25,  # weight for numeric-token aux CE
                 lambda_class: float = 0.25,  # weight for object-class token aux CE
                 **kwargs):
        """
        Initialise the loss function.

        Parameters:
            tokenizer: HuggingFace tokenizer instance
            ignore_index: Index to ignore in loss calculation (default: -100)
            lambda_digit: Weight for numeric token auxiliary loss
            lambda_punct: Weight for punctuation auxiliary loss
            lambda_class: Weight for class token auxiliary loss
            logger: Logger instance for warnings
            log_interval: How often to log metrics
        """
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.lambda_digit = lambda_digit
        self.lambda_class = lambda_class
        self.logger = logger
        self.log_interval = log_interval
        self._logged_samples = 0

        # CRITICAL: Initialize device-specific tensors as None - they'll be created on first use
        self.numeric_ids = None
        self.punct_ids = None
        self.digit_ids = None  # For backward compatibility
        self._device_cache = None  # Cache the device these tensors are on
        self._digit_punct_cache = None  # Cache concatenated digit+punct tensor for performance

        # Store the raw token lists for device-specific tensor creation
        self._raw_numeric_ids = []
        self._raw_punct_ids = []

        # ------------------------------------------------------------------
        # Only cache numeric token IDs if digit loss is enabled
        # ------------------------------------------------------------------
        if self.lambda_digit > 0:
            num_tokens = [f"{i/1000:.3f}" for i in range(1001)]  # 1 001 tokens
            num_ids = []
            for tok in num_tokens:
                tid = tokenizer.convert_tokens_to_ids(tok)
                if tid is not None and tid not in (tokenizer.unk_token_id, -1):
                    num_ids.append(tid)

            if len(num_ids) < 1001:
                missing = 1001 - len(num_ids)
                if logger is not None:
                    logger.warning(f"{missing} numeric tokens were not found in the tokenizer vocabulary; aux loss will ignore them")

            self._raw_numeric_ids = num_ids



        # ------------------------------------------------------------------
        # Only cache class token IDs if class loss is enabled
        # ------------------------------------------------------------------
        if self.lambda_class > 0:
            # class token IDs remain empty for now (as in original implementation)
            pass
        
        # CRITICAL: Initialize token IDs for class-token auxiliary loss
        # These are needed even if lambda_class is 0 to avoid AttributeError
        self._tok_open = tokenizer.convert_tokens_to_ids(TAG_OPEN)
        self._tok_colon = tokenizer.convert_tokens_to_ids(":")
        
        # Initialize step counter for logging
        self.step = 0

    def _ensure_device_tensors(self, device):
        """
        CRITICAL: Ensure cached tensors are on the correct device.
        This method creates device-specific tensors only once per device.
        """
        if self._device_cache == device:
            return  # Already on correct device
        
        # Invalidate cached concatenated tensor when device changes
        self._digit_punct_cache = None
            
        # Create device-specific tensors
        if self._raw_numeric_ids and self.lambda_digit > 0:
            self.numeric_ids = torch.tensor(self._raw_numeric_ids, dtype=torch.long, device=device)
            self.digit_ids = self.numeric_ids  # For backward compatibility
            
        self._device_cache = device

    # callable ----------------------------------------------------------
    def __call__(self, outputs, labels, **kwargs):
        """Return autoregressive CE loss.

        Parameters
        ----------
        outputs : transformers.modeling_outputs.BaseModelOutputWithPast
            Must expose ``logits`` of shape (B, S, V).
        labels : LongTensor (B, S)
            Target token IDs with masking (ignored positions == ``ignore_index``).
        """
        logits = outputs.logits  # (B, S, V)
        vocab = logits.size(-1)
        
        # CRITICAL: Ensure cached tensors are on the correct device
        self._ensure_device_tensors(logits.device)
        
        # OPTIMIZED: Pre-compute shifted tensors once
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        # Use reshape to handle possibly non-contiguous tensors from slicing
        flat_logits = shift_logits.reshape(-1, vocab)
        flat_labels = shift_labels.reshape(-1)
        
        # Main cross-entropy loss
        loss = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=self.ignore_index,
            reduction="mean",
        )

        # Initialize auxiliary losses
        aux_loss_digit = torch.tensor(0.0, device=logits.device)
        aux_loss_class = torch.tensor(0.0, device=logits.device)

        # Only compute auxiliary losses if their lambda values are non-zero
        # Use pre-computed flat tensors to avoid redundant computations
        if self.lambda_digit > 0 or self.lambda_class > 0:
            # Auxiliary loss focused on numeric tokens
            if self.lambda_digit > 0 and self.numeric_ids is not None:
                # OPTIMIZED: No repeated .to(device) calls - tensors already on correct device
                numeric_mask = (flat_labels >= 0) & torch.isin(flat_labels, self.numeric_ids)
                # CRITICAL: Use .sum() > 0 instead of .any() to avoid GPU-CPU sync
                if numeric_mask.sum() > 0:
                    aux_loss_digit = F.cross_entropy(
                        flat_logits[numeric_mask], flat_labels[numeric_mask], reduction="mean"
                    )



            # Class-token auxiliary CE – combats wrong / hallucinated labels
            if self.lambda_class > 0:
                B = labels.size(0)
                seq_len = shift_labels.size(1)  # Use pre-computed shift_labels
                class_mask_list = []
                for b in range(B):
                    row = shift_labels[b]  # Use pre-computed shift_labels
                    mask_row = torch.zeros_like(row, dtype=torch.bool)
                    inside = False
                    for idx in range(seq_len):
                        tok = row[idx]
                        if tok == self._tok_open:
                            inside = True
                            continue
                        if tok == self._tok_colon:
                            inside = False
                            continue
                        if inside:
                            mask_row[idx] = True
                    class_mask_list.append(mask_row)

                class_mask = torch.cat(class_mask_list, dim=0)
                # CRITICAL: Use .sum() > 0 instead of .any() to avoid GPU-CPU sync
                if class_mask.sum() > 0:
                    aux_loss_class = F.cross_entropy(
                        flat_logits[class_mask], flat_labels[class_mask], reduction="mean"
                    )

        if not torch.isfinite(loss):
            raise RuntimeError("NaN/Inf in LM loss")

        # ---------------- optional debug logging -------------------
        if self.logger is not None:
            # every 4× interval: log a decoded prediction / target pair
            if self.step % (self.log_interval * 4) == 0:
                # Use original tensors for debug output (shifted tensors cause misalignment)
                pred_ids = shift_logits.argmax(dim=-1)[0].to(device="cpu")
                tgt_ids = shift_labels[0].to(device="cpu")
                
                pred_str, tgt_str = decode_pred_gt(pred_ids, tgt_ids, self.tokenizer)
                
                # NEW: Extract and log actual GT objects for monitoring
                gt_objects = self._extract_gt_objects(tgt_ids)
                
                self.logger.info({"sample_pred": pred_str, "sample_gt": tgt_str})
                self.logger.info({"gt_objects": gt_objects})  # Log parsed GT objects
                self.logger.info({
                    "loss": loss.item(),
                    "aux_numeric": (self.lambda_digit * aux_loss_digit).item(),
                    "aux_class": (self.lambda_class * aux_loss_class).item(),
                })

        self.step += 1
        # Combine losses (main CE + weighted auxiliaries)
        total_loss = (
            loss
            + self.lambda_digit * aux_loss_digit
            + self.lambda_class * aux_loss_class
        )

        return total_loss

    def _extract_gt_objects(self, gt_ids):
        """
        Extract ground truth objects from label token IDs for logging.
        
        Args:
            gt_ids: Ground truth token IDs (potentially with -100 ignore tokens)
            
        Returns:
            List of parsed GT objects with class names and coordinates
        """
        import re
        
        # Clean up ignore tokens
        clean_ids = [t for t in gt_ids if t != self.ignore_index]
        
        if not clean_ids:
            return []
        
        # Decode to text to parse objects
        try:
            gt_text = self.tokenizer.decode(clean_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
            
            # Extract detection fragments using regex
            fragment_pattern = r'<\|bbob\|>([^<]*?)</\|bbob\|>'
            matches = re.findall(fragment_pattern, gt_text)
            
            objects = []
            for match in matches:
                try:
                    # Parse class and coordinates
                    if ':' in match:
                        class_name, coords_part = match.split(':', 1)
                        class_name = class_name.strip()
                        
                        # Extract coordinates using regex
                        coord_pattern = r'[-+]?\d*\.?\d+'
                        coords = re.findall(coord_pattern, coords_part)
                        coords = [float(c) for c in coords[:4]]  # Take first 4 coordinates
                        
                        if len(coords) == 4:
                            # Format coordinates to 3 decimal places
                            coords_str = f"[{coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}, {coords[3]:.3f}]"
                            objects.append(f"{class_name}: {coords_str}")
                        else:
                            objects.append(f"{class_name}: [incomplete coords]")
                    else:
                        objects.append(f"[malformed]: {match}")
                        
                except Exception as e:
                    objects.append(f"[parse error]: {match}")
            
            return objects
            
        except Exception as e:
            return [f"[decode error]: {str(e)}"]


# ----------------------------------------------------------------------
# Public factory – keeps the old name so training scripts remain unchanged
# ----------------------------------------------------------------------

def create_compute_loss_func(tokenizer, **kw):
    """Return a ready-to-call ``BBOBLoss`` instance.

    This wrapper keeps external training scripts unchanged: they can still
    call ``create_compute_loss_func`` and receive the updated loss object
    with numeric, punctuation, and class auxiliary supervision.
    """
    return BBOBLoss(tokenizer, **kw)