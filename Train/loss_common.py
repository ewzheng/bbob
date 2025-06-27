import torch
import torch.nn.functional as F
from .loss_helpers import decode_pred_gt

class BBOBLoss:
    """Minimal language-model cross-entropy loss for BBOB.

    This treats the entire detection string as ordinary text; supervision is
    plain autoregressive CE with an ignore-index mask supplied by the collator.
    The class still exposes a ``digit_ids`` attribute so that BBOBTrainer's
    guided-sampling helper continues to work unchanged.
    """

    def __init__(self, tokenizer, *, ignore_index: int = -100, logger=None, log_interval: int = 100, **_):
        self.tok = tokenizer
        self.ignore_index = ignore_index
        self.logger = logger
        self.log_interval = max(1, log_interval)
        self.step = 0
        # Cache tensor of digit token IDs – needed by BBOBTrainer for the
        # optional guidance loss in the first ``total_tf_steps``.
        ids = [tokenizer.convert_tokens_to_ids(str(i)) for i in range(10)]
        if any(i in (None, tokenizer.unk_token_id, -1) for i in ids):
            raise ValueError("Tokenizer must contain explicit '0'..'9' tokens")
        self.digit_ids = torch.tensor(ids, dtype=torch.long)

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

        self.step += 1
        return loss


# ----------------------------------------------------------------------
# Public factory – keeps the old name so training scripts remain unchanged
# ----------------------------------------------------------------------

def create_compute_loss_func(tokenizer, **kw):
    """Return a ready-to-call ``LMLoss`` instance."""
    return BBOBLoss(tokenizer, **kw) 