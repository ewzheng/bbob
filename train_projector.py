'''
File: train_projector.py
Author: Elias Zheng and Claude
Description: Train only the **projector** of the refactored BBOB model. The
vision tower is instantiated internally, so no external path is required.
Usage: python train_projector.py -m <base_llm_path> -d <dataset_name> -e <epochs> -i <instruction_text>
'''

import torch
import argparse
from datetime import datetime
import math

# utils
import sys
import os
import multiprocess as mp
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
# training
from trl import SFTTrainer, SFTConfig
from Utils import get_logger, LoggingCallback
from Model import build_BBOB 
from Train import load_and_prepare_dataset

def make_collate_fn(pad_token_id: int):
    """Return a collate function capturing the pad id from the tokenizer."""

    def _collate(batch):
        from torch.nn.utils.rnn import pad_sequence
        import torch

        # keep PIL images untouched; TRL Trainer will move them to device later
        images = [item["images"] for item in batch]

        merged_input_ids = []
        merged_labels = []

        for item in batch:
            instr_ids = item["input_ids"]
            tgt_ids   = item["target_text"]

            # drop padding tokens that were added during preprocessing
            instr_ids = instr_ids[instr_ids != pad_token_id]
            tgt_ids   = tgt_ids[tgt_ids   != pad_token_id]

            # concatenate instruction + target ⇒ model input
            ids = torch.cat([instr_ids, tgt_ids], dim=0)

            # labels: ignore instruction tokens; learn on target tokens only
            lbl = ids.clone()
            lbl[: instr_ids.size(0)] = -100

            merged_input_ids.append(ids)
            merged_labels.append(lbl)

        # pad to max length in batch
        input_ids_padded = pad_sequence(merged_input_ids, batch_first=True, padding_value=pad_token_id)
        labels_padded    = pad_sequence(merged_labels,    batch_first=True, padding_value=-100)

        attention_mask = (input_ids_padded != pad_token_id).long()

        return {
            "images": images,
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
        }

    return _collate


def train(
    model,
    train_dataset,
    val_dataset,
    output_dir,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    grad_acc_steps: int = 1,
    logger=None,
    warmup_steps: int = 0,
    num_training_steps: int = 0,
):
    """
    Train projector.

    Parameters:
        - model: BBOB instance (vision tower + base LLM already loaded).
        - train_dataset: processed dataset created by ``load_and_prepare_dataset``.
        - val_dataset:   validation dataset (same structure).
        - output_dir: str – directory for checkpoints.
        - epochs: int – number of epochs.
        - batch_size: int – per-device batch size.
        - lr: float – learning rate for AdamW.
        - grad_acc_steps: int – gradient-accumulation steps (default 1).

    Returns:
        - None – model is saved to *output_dir* on completion.
    """

    model.freeze_model()                              
    model.unfreeze_projector()                  

    # config
    cuda = torch.cuda.is_available()
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    steps_per_epoch = math.ceil(len(train_dataset) / (batch_size * grad_acc_steps))
    steps_per_epoch = max(steps_per_epoch, 1)  # safety guard

    cfg = SFTConfig(
        output_dir                  = output_dir,
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        gradient_accumulation_steps = grad_acc_steps,
        learning_rate               = lr,
        bf16                        = bf16_supported,
        fp16                        = cuda and not bf16_supported,
        eval_strategy               = "steps",
        eval_steps                  = max(512 // grad_acc_steps, 1),
        save_strategy               = "steps",
        save_steps                  = max(steps_per_epoch // 3, 1),
        logging_steps               = max(128 // grad_acc_steps, 1),
        report_to                   = "none",
    )

    # custom collator that injects labels based on *target_text*
    collate_fn = make_collate_fn(model.get_tokenizer().pad_token_id)

    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=lr)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=steps_per_epoch*epochs-warmup_steps, num_cycles=epochs)

    trainer = SFTTrainer(
        model          = model,
        train_dataset  = train_dataset,
        eval_dataset   = val_dataset,
        data_collator  = collate_fn,
        args           = cfg,
        callbacks      = [LoggingCallback(logger)] if logger is not None else None,
        optimizers     = (optimizer, scheduler),
        processing_class = model.get_tokenizer(),
    )

    logger.info("Starting training of projector...")

    trainer.train()

    return

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Train BBOB projector")
    parser.add_argument("-m", "--model", required=True, help="Model location/path")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset location/path")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Maximum number of training epochs (default: 1)")
    parser.add_argument("-i", "--instruction", required=True, help="Instruction text to add to dataset examples")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for projector training (default: 2e-5)")
    parser.add_argument("--bnb_config", type=str, default=None, help="Bits and bytes configuration (default: None)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--warmup_steps", type=int, default=16, help="Number of warmup steps for scheduler (default: 16)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before optimizer step (default: 1)")
    args = parser.parse_args()
    
    # create output directory 
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"Output/{current_datetime}"
    os.makedirs(output_dir, exist_ok=True)

    # logging
    logger = get_logger(output_dir, "projector_training.log") 

    logger.info(f"Loading base language model from: {args.model}")
    logger.info(f"Loading dataset from: {args.dataset}")
    logger.info(f"Training for max {args.epochs} epochs")

    model = build_BBOB(args.model, args.bnb_config)

    train_dataset, val_dataset = load_and_prepare_dataset(
        args.dataset,
        tokenizer=model.get_tokenizer(),
        instruction=args.instruction,
        dtype=model.dtype
    )

    train(
        model,
        train_dataset,
        val_dataset,
        output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_acc_steps=args.gradient_accumulation_steps,
        logger=logger,
    )

    logger.info("Projector training is complete, model successfully saved.")
    
    return

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main()