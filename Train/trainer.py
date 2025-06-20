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
        **kwargs: Any,
    ) -> None:
        # If caller did not pass an explicit data_collator we insert the train-one
        if "data_collator" not in kwargs and train_collator is not None:
            kwargs["data_collator"] = train_collator
        super().__init__(*args, **kwargs)

        # Store collators (fallback: use the same for both)
        self._train_collator = train_collator or self.data_collator
        self._eval_collator = eval_collator or self._train_collator

        # Store schedule parameters
        self._tf_start_p = float(tf_start_p)
        self._tf_end_p   = float(tf_end_p)
        self._tf_total   = int(total_tf_steps) if total_tf_steps > 0 else 1
        self._tf_sched   = tf_schedule

    # ------------------------------------------------------------------
    # Overridden DataLoader builders
    # ------------------------------------------------------------------

    def get_train_dataloader(self) -> DataLoader:  # noqa: D401
        # Ensure correct mode before workers are spawned
        if hasattr(self._train_collator, "train"):
            self._train_collator.train()
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
        if self.model.training:
            p = self._tf_prob(self.state.global_step)
            if random.random() < p:
                if "input_ids" in inputs and "labels" in inputs:
                    ids = inputs["input_ids"].clone()
                    lbl = inputs["labels"]
                    mask = lbl != -100
                    ids[mask] = lbl[mask]
                    inputs["input_ids"] = ids
        return inputs 