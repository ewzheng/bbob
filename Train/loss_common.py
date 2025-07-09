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
                 lambda_punct: float = 0.05,  # weight for punctuation/syntax aux CE
                 lambda_class: float = 0.25,  # weight for object-class token aux CE
                 lambda_box: float = 0.1,    # weight for box-level CIoU auxiliary loss
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
        self.lambda_punct = lambda_punct
        self.lambda_class = lambda_class
        self.lambda_box = lambda_box
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
        # Only cache punctuation token IDs if punct loss is enabled
        # ------------------------------------------------------------------
        if self.lambda_punct > 0:
            punct_token_ids = []
            
            # Get punctuation tokens, handling cases where they might not exist
            for token in [TAG_OPEN, TAG_CLOSE, ",", ":", "[", "]"]:
                tid = tokenizer.convert_tokens_to_ids(token)
                if tid is not None and tid not in (tokenizer.unk_token_id, -1):
                    punct_token_ids.append(tid)
            
            # Add EOS token if it exists
            if tokenizer.eos_token_id is not None:
                punct_token_ids.append(tokenizer.eos_token_id)
            
            if punct_token_ids:
                self._raw_punct_ids = punct_token_ids
            else:
                if logger is not None:
                    logger.warning("No punctuation tokens found in tokenizer vocabulary; punct loss will be disabled")
                self.lambda_punct = 0.0  # Disable punct loss if no tokens found

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
            
        if self._raw_punct_ids and self.lambda_punct > 0:
            self.punct_ids = torch.tensor(self._raw_punct_ids, dtype=torch.long, device=device)
            
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
        aux_loss_punct = torch.tensor(0.0, device=logits.device)
        aux_loss_class = torch.tensor(0.0, device=logits.device)
        aux_loss_box = torch.tensor(0.0, device=logits.device)

        # Only compute auxiliary losses if their lambda values are non-zero
        # Use pre-computed flat tensors to avoid redundant computations
        if self.lambda_digit > 0 or self.lambda_punct > 0 or self.lambda_class > 0 or self.lambda_box > 0:
            # Auxiliary loss focused on numeric tokens
            if self.lambda_digit > 0 and self.numeric_ids is not None:
                # OPTIMIZED: No repeated .to(device) calls - tensors already on correct device
                numeric_mask = (flat_labels >= 0) & torch.isin(flat_labels, self.numeric_ids)
                # CRITICAL: Use .sum() > 0 instead of .any() to avoid GPU-CPU sync
                if numeric_mask.sum() > 0:
                    aux_loss_digit = F.cross_entropy(
                        flat_logits[numeric_mask], flat_labels[numeric_mask], reduction="mean"
                    )

            # Auxiliary loss focused on punctuation tokens
            if self.lambda_punct > 0 and self.punct_ids is not None:
                # OPTIMIZED: No repeated .to(device) calls - tensors already on correct device
                punct_mask = (flat_labels >= 0) & torch.isin(flat_labels, self.punct_ids)
                # CRITICAL: Use .sum() > 0 instead of .any() to avoid GPU-CPU sync
                if punct_mask.sum() > 0:
                    aux_loss_punct = F.cross_entropy(
                        flat_logits[punct_mask], flat_labels[punct_mask], reduction="mean"
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

            # ------------------------------------------------------------------
            # Box-level CIoU auxiliary – uses model *predictions* (arg-max)
            # ------------------------------------------------------------------
            if self.lambda_box > 0:
                # 1. Decode predicted & GT strings (cheap CPU op)
                pred_ids = shift_logits.argmax(dim=-1)  # (B,S)
                gt_ids = shift_labels  # (B,S)

                pred_filtered = [
                    [int(t) for t in row.tolist() if t != self.ignore_index]
                    for row in pred_ids
                ]
                gt_filtered = [
                    [int(t) for t in row.tolist() if t != self.ignore_index]
                    for row in gt_ids
                ]

                pred_strs = self.tokenizer.batch_decode(pred_filtered, skip_special_tokens=False,
                                                         clean_up_tokenization_spaces=True)
                gt_strs = self.tokenizer.batch_decode(gt_filtered, skip_special_tokens=False,
                                                       clean_up_tokenization_spaces=True)

                # ---------------- vectorised using ids_to_boxes_labels ----------------
                pred_ids = shift_logits.argmax(dim=-1)
                # move to CPU for cheap Python-side decode; keeps GPU free
                pred_coords_all, _, pred_ptr = ids_to_boxes_labels(
                    pred_ids.cpu(), self.tokenizer, ignore_index=self.ignore_index
                )

                gt_coords_all, _, gt_ptr = ids_to_boxes_labels(
                    shift_labels.cpu(), self.tokenizer, ignore_index=self.ignore_index
                )

                pb_all: list[torch.Tensor] = []
                gb_all: list[torch.Tensor] = []

                B = shift_logits.size(0)
                for b in range(B):
                    p_slice = slice(pred_ptr[b], pred_ptr[b + 1])
                    g_slice = slice(gt_ptr[b], gt_ptr[b + 1])

                    if p_slice.start == p_slice.stop or g_slice.start == g_slice.stop:
                        continue  # no boxes for this sample

                    p_boxes = pred_coords_all[p_slice]
                    g_boxes = gt_coords_all[g_slice]

                    pairs = hungarian_match(p_boxes, g_boxes)
                    if not pairs:
                        continue

                    pb_all.append(p_boxes[[i for i, _ in pairs]])
                    gb_all.append(g_boxes[[j for _, j in pairs]])

                if pb_all:
                    pb_cat = torch.cat(pb_all, dim=0).to(device=logits.device)
                    gb_cat = torch.cat(gb_all, dim=0).to(device=logits.device)

                    pb_xyxy = xywh_to_xyxy(pb_cat)
                    gb_xyxy = xywh_to_xyxy(gb_cat)

                    if _ciou_loss_fn is not None:
                        aux_loss_box = _ciou_loss_fn(pb_xyxy, gb_xyxy, reduction="mean")
                    else:
                        giou = _g_box_iou(pb_xyxy, gb_xyxy).diag()
                        aux_loss_box = (1.0 - giou).mean()

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
                self.logger.info({"sample_pred": pred_str, "sample_gt": tgt_str})
                self.logger.info({
                    "loss": loss.item(),
                    "aux_numeric": (self.lambda_digit * aux_loss_digit).item(),
                    "aux_punct": (self.lambda_punct * aux_loss_punct).item(),
                    "aux_class": (self.lambda_class * aux_loss_class).item(),
                    "aux_box": (self.lambda_box * aux_loss_box).item(),
                })

        self.step += 1
        # Combine losses (main CE + weighted auxiliaries)
        total_loss = (
            loss
            + self.lambda_digit * aux_loss_digit
            + self.lambda_punct * aux_loss_punct
            + self.lambda_class * aux_loss_class
            + self.lambda_box * aux_loss_box
        )

        return total_loss


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