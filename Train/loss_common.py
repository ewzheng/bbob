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
        # ------------------------------------------------------------------
        # Numeric-bin token IDs  ("0.000" … "1.000", inclusive)
        # ------------------------------------------------------------------
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

        # For backward-compatibility with Trainer guidance that still refers
        # to `.digit_ids`, we alias the attribute.  The name remains the same
        # but now points to the list of *numeric* token IDs.
        self.digit_ids = self.numeric_ids

        self.punct_ids = torch.tensor([tokenizer.convert_tokens_to_ids(TAG_OPEN), tokenizer.convert_tokens_to_ids(TAG_CLOSE), 
                                       tokenizer.convert_tokens_to_ids(","), 
                                       tokenizer.convert_tokens_to_ids(":"),
                                       tokenizer.convert_tokens_to_ids("["),
                                       tokenizer.convert_tokens_to_ids("]")], dtype=torch.long)
        
        if tokenizer.eos_token_id is not None:
            self.punct_ids = torch.cat([self.punct_ids, torch.tensor([tokenizer.eos_token_id], dtype=torch.long)])

        # Token IDs that mark the start/end of the class label segment
        self._tok_open = tokenizer.convert_tokens_to_ids(TAG_OPEN)
        self._tok_colon = tokenizer.convert_tokens_to_ids(":")

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
        # Shift so that predictions at time t are compared against label t+1
        loss = F.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, vocab),
            labels[..., 1:].contiguous().view(-1),
            ignore_index=self.ignore_index,
            reduction="mean",
        )

        # Auxiliary loss focused on numeric tokens (0-9 and decimal point)
        flat_logits = logits[..., :-1, :].contiguous().view(-1, vocab)
        flat_labels = labels[..., 1:].contiguous().view(-1)
        numeric_mask = (flat_labels >= 0) & torch.isin(
            flat_labels,
            self.numeric_ids.to(flat_labels.device),
        )

        if numeric_mask.any():
            aux_loss_digit = F.cross_entropy(
                flat_logits[numeric_mask], flat_labels[numeric_mask], reduction="mean"
            )
        else:
            aux_loss_digit = torch.tensor(0.0, device=logits.device)
        
        punct_mask = (flat_labels >= 0) & torch.isin(
            flat_labels,
            self.punct_ids.to(flat_labels.device),
        )

        if punct_mask.any():
            aux_loss_punct = F.cross_entropy(
                flat_logits[punct_mask], flat_labels[punct_mask], reduction="mean"
            )
        else:
            aux_loss_punct = torch.tensor(0.0, device=logits.device)

        # ---------------------------------------------------------
        # NEW: class-token auxiliary CE – combats wrong / hallucinated labels
        # ---------------------------------------------------------
        B = labels.size(0)
        seq_len = labels.size(1) - 1  # because of shift (labels[...,1:])
        class_mask_list = []
        for b in range(B):
            row = labels[b, 1:]
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
        else:
            aux_loss_class = torch.tensor(0.0, device=logits.device)

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