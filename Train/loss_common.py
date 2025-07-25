import torch
import torch.nn.functional as F
from .train_common import format_coordinate
from .loss_helpers import (
    decode_pred_gt,
    TAG_OPEN,
    TAG_CLOSE,
)
import re

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
    def __call__(self, outputs, labels, input_ids=None, **kwargs):
        """Return autoregressive CE loss.

        Parameters
        ----------
        outputs : transformers.modeling_outputs.BaseModelOutputWithPast
            Must expose ``logits`` of shape (B, S, V).
        labels : LongTensor (B, S)
            Target token IDs with masking (ignored positions == ``ignore_index``).
        input_ids : LongTensor (B, S), optional
            Input token IDs (including noise) for logging purposes.
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

                # ------------------------------------------------------------
                # Pick the *first* row that actually contains at least one
                # supervised GT token so logs are meaningful even when the
                # first batch element is an empty MS-crop.
                # ------------------------------------------------------------
                row_idx = 0
                for b in range(labels.size(0)):
                    if (labels[b] != self.ignore_index).any():
                        row_idx = b
                        break

                # Compute arg-max predictions for that batch row
                raw_pred_ids = shift_logits.argmax(dim=-1)[row_idx].to(device="cpu")

                # Use *original* labels (no shift) for GT – they contain -100 at noise positions
                tgt_ids = labels[row_idx].to(device="cpu")

                # ------------------------------------------------------------------
                # NEW: Filter out positions where the GT label is *ignore_index* so we
                # only log predictions that correspond to *real* ground-truth tokens
                # (i.e. the object fragments).  This avoids showing the model’s free
                # predictions on noise positions, which are untrained and therefore
                # can look like random “noise classes”.
                # ------------------------------------------------------------------
                # FIXED: Use shifted labels to match the shifted predictions
                shifted_tgt_ids = tgt_ids[1:]  # Shift to match raw_pred_ids
                keep_mask = shifted_tgt_ids != self.ignore_index
                pred_ids = raw_pred_ids
                
                # Also get input sequence to see noise interleaving (same row)
                if input_ids is not None:
                    actual_input_ids = input_ids[row_idx].to(device="cpu")  # Actual input with noise
                    # Filter out -100 tokens before decoding to avoid overflow error
                    clean_input_ids = actual_input_ids[actual_input_ids != self.ignore_index]
                    input_str = self.tokenizer.decode(clean_input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
                else:
                    input_str = "[input_ids not available]"
                
                pred_str, tgt_str = decode_pred_gt(pred_ids, tgt_ids, self.tokenizer)

                # NEW: Extract and log actual GT objects for monitoring (same row)
                gt_objects = self._extract_gt_objects(actual_input_ids, labels[row_idx].to(device="cpu"))
                
                self.logger.info({"sample_pred": pred_str, "sample_gt": tgt_str, "sample_input": input_str})
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

    def _extract_gt_objects(self, input_ids_row, labels_row):
        """Return list of GT *text* fragments ignoring noise.

        Parameters
        ----------
        input_ids_row : LongTensor (S,)
            The full input sequence (image-placeholder + instruction + GT + noise).
        labels_row : LongTensor (S,)
            The label sequence where GT tokens are real IDs and noise tokens are
            set to ``ignore_index`` (-100).
        """

        # --------------------------------------------------------------
        # SAFETY: Collator prepends a *single* image placeholder token to
        # `input_ids` but pads *VIS_TOKENS* (64) positions with -100 in
        # `labels` to ignore the vision embeddings.  This means
        # `labels_row` is typically 63 tokens longer than `input_ids_row`.
        # Instead of aborting, we realign by trimming the *leading* portion
        # of the longer tensor so both have equal length and tokens remain
        # in sync from the *end* (where all detection fragments live).
        # --------------------------------------------------------------
        if input_ids_row.numel() != labels_row.numel():
            diff = labels_row.numel() - input_ids_row.numel()
            if diff > 0:
                # labels is longer → drop the extra *leading* tokens
                labels_row = labels_row[diff:]
            elif diff < 0:
                # input_ids is longer (should not happen but handle) → trim
                input_ids_row = input_ids_row[-diff:]
            # If still mismatched, bail out to avoid mis-alignment
            if input_ids_row.numel() != labels_row.numel():
                return []

        # Keep only positions where label != ignore_index → these belong to GT
        kept = [int(t) for t, lab in zip(input_ids_row.tolist(), labels_row.tolist()) if lab != self.ignore_index]

        if not kept:
            return []

        try:
            gt_text = self.tokenizer.decode(kept, skip_special_tokens=False, clean_up_tokenization_spaces=True)
            frag_pat = r'<\|bbob\|>([^<]*?)</\|bbob\|>'
            matches = re.findall(frag_pat, gt_text)
            objects = []
            for m in matches:
                if ':' not in m:
                    objects.append(f"[malformed]: {m}")
                    continue
                cls, coord_part = m.split(':', 1)
                cls = cls.strip()
                nums = re.findall(r'[-+]?\d*\.?\d+', coord_part)[:4]
                if len(nums) != 4:
                    objects.append(f"{cls}: [incomplete coords]")
                    continue
                coords_f = [float(x) for x in nums]
                coords_str = f"[{format_coordinate(coords_f[0])}, {format_coordinate(coords_f[1])}, {format_coordinate(coords_f[2])}, {format_coordinate(coords_f[3])}]"
                objects.append(f"{cls}: {coords_str}")
            return objects
        except Exception as e:
            return [f"[decode error]: {e}"]


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