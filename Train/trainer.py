"""bbob_trainer.py

Custom HuggingFace Trainer that switches the data-collator between training
and evaluation so that teacher-forcing is enabled only for the training
DataLoader.  The same collator instances are reused across epochs, therefore
`persistent_workers=True` causes no process leaks and the teacher-forcing
schedule is not reset.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from torch.utils.data import DataLoader
from transformers import Trainer
import math
import random
import torch
import torch.nn.functional as F
import numpy as np


class BBOBTrainer(Trainer):
    """Trainer that uses separate collators for train / eval.

    Parameters
    ----------
    train_collator : callable
        Collator to use for training batches.  Must implement `.train()` and
        `.eval()` switching methods (as `BBOBCollator` does).
    eval_collator : callable | None
        Collator to use for evaluation.  If *None*, ``train_collator`` is used
        with `.eval()` mode.
    tf_start_p : float
        Start probability for teacher-forcing schedule.
    tf_end_p : float
        End probability for teacher-forcing schedule.
    total_tf_steps : int
        Total number of steps for teacher-forcing schedule.
    tf_schedule : str
        Schedule type for teacher-forcing.
    compute_loss_func : callable | None
        Loss function to use for training. If *None*, the default loss function
        is used.
    guidance_strength : float
        Strength of guided sampling.
    tf_ramp_ratio : float
        Ramp ratio for teacher-forcing schedule.
    All other positional / keyword arguments are forwarded to
    ``transformers.Trainer``.
    """

    def __init__(
        self,
        *args: Any,
        train_collator: Optional[Any] = None,
        eval_collator: Optional[Any] = None,
        # Teacher-forcing schedule params
        tf_start_p: float = 1.0,
        tf_end_p: float = 0.0,
        total_tf_steps: int = 0,
        total_gd_steps: int = 0,
        tf_schedule: str = "cosine",
        tf_ramp_ratio: float = 0.8,
        compute_loss_func: Optional[Any] = None,
        guidance_strength: float = 1.5,
        **kwargs: Any,
    ) -> None:
        # If caller did not pass an explicit data_collator we insert the train-one
        if "data_collator" not in kwargs and train_collator is not None:
            kwargs["data_collator"] = train_collator

        # Forward the loss callable to HF Trainer (if supported) _and_ store local ref
        if compute_loss_func is not None:
            kwargs["compute_loss_func"] = compute_loss_func

        super().__init__(*args, **kwargs)

        self._loss_func = compute_loss_func

        # Store collators (fallback: use the same for both)
        self._train_collator = train_collator or self.data_collator
        self._eval_collator = eval_collator or self._train_collator

        # Store schedule parameters
        self._tf_start_p = float(tf_start_p)
        self._tf_end_p   = float(tf_end_p)
        self._tf_total   = int(total_tf_steps) if total_tf_steps > 0 else 1
        self._tf_sched   = tf_schedule

        # store ramp ratio (fraction of steps used for linear/other decay)
        self._tf_ramp = max(1e-6, float(tf_ramp_ratio))

        # guided sampling uses the same warm-up length as teacher-forcing
        self._guidance_steps = total_gd_steps
        self._guidance_strength = guidance_strength

    # ------------------------------------------------------------------
    # Overridden DataLoader builders
    # ------------------------------------------------------------------

    def get_train_dataloader(self) -> DataLoader:  # noqa: D401
        # Ensure correct mode before workers are spawned
        if hasattr(self._train_collator, "train"):
            self._train_collator.train()
            self._train_collator.reseed()
        dl = super().get_train_dataloader()
        dl.collate_fn = self._train_collator  # make sure
        return dl

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:  # noqa: D401
        # Switch to eval mode before new workers are forked
        if hasattr(self._eval_collator, "eval"):
            self._eval_collator.eval()
        dl = super().get_eval_dataloader(eval_dataset)
        dl.collate_fn = self._eval_collator
        return dl

    # Same treatment for prediction
    def get_test_dataloader(self, test_dataset) -> DataLoader:  # noqa: D401
        if hasattr(self._eval_collator, "eval"):
            self._eval_collator.eval()
        dl = super().get_test_dataloader(test_dataset)
        dl.collate_fn = self._eval_collator
        return dl

    # ---------------- teacher-forcing schedule helper ------------------

    def _tf_prob(self, step: int) -> float:
        """Return teacher-forcing probability at *global* optimiser step."""
        # Normalised progress (0‥1) with 80 % ramp window — parentheses are
        # critical: without them division precedes multiplication leading to
        # wildly out-of-range values.
        denom = self._tf_ramp * self._tf_total
        t = min(step, denom) / denom
        if self._tf_sched == "linear":
            return self._tf_start_p + t * (self._tf_end_p - self._tf_start_p)
        if self._tf_sched == "cosine":
            return self._tf_end_p + 0.5 * (self._tf_start_p - self._tf_end_p) * (1 + math.cos(math.pi * t))
        if self._tf_sched == "exp":
            k = 5.0
            return self._tf_end_p + (self._tf_start_p - self._tf_end_p) * math.exp(-k * t)
        # fallback linear
        return self._tf_start_p + t * (self._tf_end_p - self._tf_start_p)

    # ---------------- batch preprocessing override --------------------

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
        """Override to apply teacher-forcing logic after base device transfer.
        
        Label alignment is now handled by the collator, so this method only
        needs to handle teacher-forcing replacement.
        """
        # First, run the base implementation which moves tensors to the right device
        inputs = super().prepare_inputs(inputs)

        # Apply teacher-forcing replacement
        inputs = self._apply_teacher_forcing(inputs)

        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
        labels = inputs.get("labels")
        
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})

        guidance_loss = None
        if self.model.training and self.state.global_step < self._guidance_steps and labels is not None and self._loss_func is not None:
            logits = outputs.logits  # (B, S, V)
            flat_logits = logits.view(-1, logits.size(-1))
            flat_labels = labels.view(-1)
            
            # CRITICAL: Ensure loss function tensors are initialized on correct device
            self._loss_func._ensure_device_tensors(flat_labels.device)
            
            # CRITICAL: Only proceed if we have valid digit/punct IDs
            if self._loss_func.digit_ids is not None and self._loss_func.punct_ids is not None:
                # CRITICAL: Cache concatenated tensor instead of recreating every forward pass
                # This was a major performance bottleneck!
                if not hasattr(self._loss_func, '_digit_punct_cache') or self._loss_func._digit_punct_cache is None:
                    self._loss_func._digit_punct_cache = torch.cat([self._loss_func.digit_ids, self._loss_func.punct_ids])
                
                # Move cached tensor to correct device (much cheaper than cat+to every time)
                digit_or_punct = self._loss_func._digit_punct_cache.to(flat_labels.device, non_blocking=True)
                
                # OPTIMIZED: Use vectorized masking without .any() which causes CPU sync
                valid_mask = (flat_labels >= 0)
                digit_punct_mask = torch.isin(flat_labels, digit_or_punct)
                mask = valid_mask & digit_punct_mask
                
                # CRITICAL: Use mask.sum() > 0 instead of mask.any() to avoid GPU-CPU sync during training
                # Only compute guidance loss if we have valid tokens (without blocking)
                mask_sum = mask.sum()
                if mask_sum > 0:
                    # OPTIMIZED: Direct indexing without nonzero() which blocks GPU-CPU sync
                    masked_logits = flat_logits[mask]
                    masked_labels = flat_labels[mask]
                    
                    if masked_logits.numel() > 0:  # Safety check
                        base_loss = F.cross_entropy(masked_logits, masked_labels, reduction="mean")
                        factor = self._guidance_strength * (1.0 - self.state.global_step / self._guidance_steps)
                        guidance_loss = factor * base_loss

        if self._loss_func is not None:
            loss = self._loss_func(outputs, labels)
            if guidance_loss is not None:
                loss = loss + guidance_loss
        else:
            # Use standard cross-entropy loss when no custom loss function is provided
            if labels is not None and hasattr(outputs, "logits"):
                logits = outputs.logits
                # Shift logits and labels for causal LM loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="mean"
                )
            else:
                loss = outputs.loss if hasattr(outputs, "loss") else None

        if return_outputs:
            return loss, outputs
        return loss 

    # ------------------------------------------------------------------
    # Helper – re-used TF logic extracted from the old prepare_inputs body
    # ------------------------------------------------------------------
    def _apply_teacher_forcing(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply teacher-forcing token replacement on *inputs* in-place and
        return the dict.  Separated so we can call after the new length-fix
        code as well as in the training loop.
        """

        # ------------------------------------------------------------
        # Scheduled sampling (teacher forcing) – per-token Bernoulli
        # ------------------------------------------------------------
        if self.model.training and self.state.global_step < self._tf_total:
            if "input_ids" in inputs and "labels" in inputs:
                p = self._tf_prob(self.state.global_step)
                if p > 0.0:
                    ids = inputs["input_ids"].clone()
                    lbl = inputs["labels"]
                    # create Bernoulli mask only where a ground-truth label exists
                    bern = torch.rand_like(lbl, dtype=torch.float, device=lbl.device) < p
                    mask = (lbl != -100) & bern
                    ids[mask] = lbl[mask]
                    inputs["input_ids"] = ids

        return inputs 