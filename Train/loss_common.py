import torch
import torch.nn.functional as F
from .loss_helpers import decode_pred_gt, TAG_OPEN, TAG_CLOSE

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
                 lambda_class: float = 0.15,  # weight for object-class token aux CE
                 **kwargs):
        self.tok = tokenizer
        self.ignore_index = ignore_index
        self.logger = logger
        self.log_interval = max(1, log_interval)
        self.step = 0
        self.lambda_digit = float(lambda_digit)
        self.lambda_punct = float(lambda_punct)
        self.lambda_class = float(lambda_class)

        # Initialize token ID caches as None - only populate if needed
        self.numeric_ids = None
        self.digit_ids = None  # Backward compatibility alias
        self.punct_ids = None
        self._tok_open = None
        self._tok_colon = None

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

            self.numeric_ids = torch.tensor(num_ids, dtype=torch.long)
            # For backward-compatibility with Trainer guidance
            self.digit_ids = self.numeric_ids

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
                self.punct_ids = torch.tensor(punct_token_ids, dtype=torch.long)
            else:
                if logger is not None:
                    logger.warning("No punctuation tokens found in tokenizer vocabulary; punct loss will be disabled")
                self.lambda_punct = 0.0  # Disable punct loss if no tokens found

        # ------------------------------------------------------------------
        # Only cache class parsing tokens if class loss is enabled
        # ------------------------------------------------------------------
        if self.lambda_class > 0:
            self._tok_open = tokenizer.convert_tokens_to_ids(TAG_OPEN)
            self._tok_colon = tokenizer.convert_tokens_to_ids(":")
            
            # Check if required tokens exist
            if (self._tok_open is None or self._tok_open == tokenizer.unk_token_id or 
                self._tok_colon is None or self._tok_colon == tokenizer.unk_token_id):
                if logger is not None:
                    logger.warning("Required class parsing tokens not found in tokenizer vocabulary; class loss will be disabled")
                self.lambda_class = 0.0  # Disable class loss if tokens not found

        # ------------------------------------------------------------------
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
        
        # OPTIMIZED: Pre-compute shifted tensors once
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        flat_logits = shift_logits.view(-1, vocab)
        flat_labels = shift_labels.view(-1)
        
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

        # Only compute auxiliary losses if their lambda values are non-zero
        # Use pre-computed flat tensors to avoid redundant computations
        if self.lambda_digit > 0 or self.lambda_punct > 0 or self.lambda_class > 0:
            # Auxiliary loss focused on numeric tokens
            if self.lambda_digit > 0 and self.numeric_ids is not None:
                numeric_mask = (flat_labels >= 0) & torch.isin(
                    flat_labels,
                    self.numeric_ids.to(flat_labels.device),
                )
                if numeric_mask.any():
                    aux_loss_digit = F.cross_entropy(
                        flat_logits[numeric_mask], flat_labels[numeric_mask], reduction="mean"
                    )

            # Auxiliary loss focused on punctuation tokens
            if self.lambda_punct > 0 and self.punct_ids is not None:
                punct_mask = (flat_labels >= 0) & torch.isin(
                    flat_labels,
                    self.punct_ids.to(flat_labels.device),
                )
                if punct_mask.any():
                    aux_loss_punct = F.cross_entropy(
                        flat_logits[punct_mask], flat_labels[punct_mask], reduction="mean"
                    )

            # Class-token auxiliary CE – combats wrong / hallucinated labels
            if self.lambda_class > 0 and self._tok_open is not None and self._tok_colon is not None:
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
                if class_mask.any():
                    aux_loss_class = F.cross_entropy(
                        flat_logits[class_mask], flat_labels[class_mask], reduction="mean"
                    )

        if not torch.isfinite(loss):
            raise RuntimeError("NaN/Inf in LM loss")

        # ---------------- optional debug logging -------------------
        if self.logger is not None:
            # every 4× interval: log a decoded prediction / target pair
            if self.step % (self.log_interval * 4) == 0:
                pred_ids = logits.argmax(dim=-1)[0].detach().cpu()
                tgt_ids  = labels[0].detach().cpu()
                pred_str, tgt_str = decode_pred_gt(pred_ids, tgt_ids, self.tok)
                self.logger.info({"sample_pred": pred_str, "sample_gt": tgt_str})
                self.logger.info({
                    "loss": loss.item(),
                    "aux_numeric": (self.lambda_digit * aux_loss_digit).item(),
                    "aux_punct": (self.lambda_punct * aux_loss_punct).item(),
                    "aux_class": (self.lambda_class * aux_loss_class).item(),
                })

        self.step += 1
        # Combine losses (main CE + weighted auxiliaries)
        total_loss = (
            loss
            + self.lambda_digit * aux_loss_digit
            + self.lambda_punct * aux_loss_punct
            + self.lambda_class * aux_loss_class
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