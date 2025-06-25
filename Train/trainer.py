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
        tf_schedule: str = "cosine",
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

        # guided sampling uses the same warm-up length as teacher-forcing
        self._guidance_steps = self._tf_total
        self._guidance_strength = guidance_strength

    # ------------------------------------------------------------------
    # Overridden DataLoader builders
    # ------------------------------------------------------------------

    def get_train_dataloader(self) -> DataLoader:  # noqa: D401
        # Ensure correct mode before workers are spawned
        if hasattr(self._train_collator, "train"):
            self._train_collator.train()
        dl = super().get_train_dataloader()
        dl.collate_fn = self._train_collator  # make sure
        # Ensure loss function is in *train* mode
        return dl

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:  # noqa: D401
        # Switch to eval mode before new workers are forked
        if hasattr(self._eval_collator, "eval"):
            self._eval_collator.eval()
        dl = super().get_eval_dataloader(eval_dataset)
        dl.collate_fn = self._eval_collator
        # Switch loss function to *eval* mode for correct logging behaviour
        if self._loss_func is not None and hasattr(self._loss_func, "is_eval"):
            self._loss_func.is_eval = True
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
        t = min(step, self._tf_total) / self._tf_total
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
        inputs = super().prepare_inputs(inputs)
        if self.model.training and self.state.global_step < self._tf_total:
            p = self._tf_prob(self.state.global_step)
            if torch.rand(1, device=inputs["input_ids"].device).item() < p:
                if "input_ids" in inputs and "labels" in inputs:
                    ids = inputs["input_ids"].clone()
                    lbl = inputs["labels"]
                    mask = lbl != -100
                    ids[mask] = lbl[mask]
                    inputs["input_ids"] = ids
        return inputs 

    # ---------------- inject guided sampling before loss -----------------

    def training_step(self, model, inputs, *args, **kwargs):
        """Thin wrapper around HF Trainer.training_step that stays compatible
        with both the old (<4.40) two-argument signature and the newer one
        that passes an additional *num_items_in_batch* parameter.  Any extra
        positional/keyword arguments are forwarded verbatim to the parent
        implementation so we do not break when the upstream API changes
        again in future releases.
        """

        # Make sure the custom loss function logs in *train* mode
        if self._loss_func is not None and hasattr(self._loss_func, "is_eval"):
            self._loss_func.is_eval = False

        # Call the parent method with the detected signature
        try:
            return super().training_step(model, inputs, *args, **kwargs)
        except TypeError:
            # Fallback for older transformers that expect only 2 positional args
            return super().training_step(model, inputs)

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval", **kwargs):
        if self._loss_func is not None and hasattr(self._loss_func, "is_eval"):
            self._loss_func.is_eval = True
        return super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})

        guidance_loss = None
        if self.model.training and self.state.global_step < self._guidance_steps and labels is not None and self._loss_func is not None:
            logits = outputs.logits  # (B, S, V)
            flat_logits = logits.view(-1, logits.size(-1))
            flat_labels = labels.view(-1)
            digit_ids = self._loss_func.digit_ids.to(flat_labels.device)
            mask = (flat_labels >= 0) & torch.isin(flat_labels, digit_ids)
            if mask.any():
                idx = mask.nonzero(as_tuple=False).squeeze(1)
                correct = flat_labels[idx]
                base_loss = F.cross_entropy(flat_logits[idx], correct, reduction="mean")
                factor = self._guidance_strength * (1.0 - self.state.global_step / self._guidance_steps)
                guidance_loss = factor * base_loss

        if self._loss_func is not None:
            loss = self._loss_func(outputs, labels)
            if guidance_loss is not None:
                loss = loss + guidance_loss
        else:
            loss = outputs.loss if hasattr(outputs, "loss") else None

        return (loss, outputs) if return_outputs else loss 