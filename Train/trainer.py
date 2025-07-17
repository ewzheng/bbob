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
        force: bool = False,
        compute_loss_func: Optional[Any] = None,
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
        self.force = force

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
        input_ids = inputs.get("input_ids")
        
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})

        # CRITICAL: Labels are already aligned by the collator, so use them as-is
        # The collator handles the visual token alignment, so we don't need to do it again
        aligned_labels = labels

        if self._loss_func is not None:
            # CRITICAL: Use aligned labels for main loss function to ensure tensor shape consistency
            loss_labels = aligned_labels if aligned_labels is not None else labels
            # Pass input_ids to loss function for proper logging
            loss = self._loss_func(outputs, loss_labels, input_ids=input_ids)
        else:
            # Use standard cross-entropy loss when no custom loss function is provided
            if labels is not None and hasattr(outputs, "logits"):
                logits = outputs.logits
                # Shift logits and labels for causal LM loss
                shift_logits = logits[..., :-1, :]
                shift_labels = labels[..., 1:]
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
        if "input_ids" in inputs and "labels" in inputs:
            if self.force and self.model.training:
                ids = inputs["input_ids"].clone()
                lbl = inputs["labels"]
                # Copy all ground truth tokens (where label != -100)
                mask = lbl != -100  # Fixed: was -10, should be -100
                ids[mask] = lbl[mask]
                inputs["input_ids"] = ids
            else:
                # Fully student-forced: hide every supervised token so the
                # model must generate them on its own.  Replace with the
                # tokenizer pad token (fall back to 0) to keep length.
                ids = inputs["input_ids"].clone()
                lbl = inputs["labels"]

                # Determine replacement token - use EOS token when teacher forcing is disabled
                try:
                    tokenizer = self.model.get_tokenizer()
                    # Use EOS token ID instead of pad token ID
                    replacement_id = tokenizer.eos_token_id
                    if replacement_id is None:
                        # Fall back to pad token if EOS is not set
                        replacement_id = tokenizer.pad_token_id
                    if replacement_id is None:
                        # Last resort: use 0
                        replacement_id = 0
                except Exception:
                    # If any exception occurs, use 0 as fallback
                    replacement_id = 0

                mask = lbl != -100
                if mask.any():
                    ids[mask] = replacement_id
                    inputs["input_ids"] = ids

        return inputs 