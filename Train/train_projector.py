'''
File: train_projector.py
Author: Elias Zheng and Claude
Description: Train only the **projector** of the refactored BBOB model. The
vision tower is instantiated internally, so no external path is required.
Usage: python train_projector.py -m <base_llm_path> -d <dataset_name> -e <epochs> -i <instruction_text>
'''

import torch
import argparse
import time
import logging
from datetime import datetime
from torch.utils.data import DataLoader


# utils
import sys
import os
import multiprocess as mp

from train_common import load_and_prepare_dataset, load_model

# Hugging Face / TRL trainer imports
from trl import SFTTrainer, SFTConfig

# configure logging (if not already configured)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Dynamic collator that builds "input_ids" = [instruction | target] and
# corresponding "labels" that mask the instruction tokens (−100) so the loss is
# computed **only** on the *target_text* segment.  No subclassing of Trainer is
# required – the stock loss from AutoModelForCausalLM handles everything.
# -----------------------------------------------------------------------------

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
):
    """
    Fine-tune *only* the **projector** through TRL's ``SFTTrainer``.

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
    model.vision_tower.eval()
    model.vision_tower.freeze()           
    model.unfreeze_projector()                  

    # prepare training config and trainer ------------------------------------------------
    cfg = SFTConfig(
        output_dir                  = output_dir,
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        gradient_accumulation_steps = grad_acc_steps,
        learning_rate               = lr,
        fp16                        = torch.cuda.is_available(),
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",
        logging_steps               = 50,
        weight_decay                = 0.0,
        report_to                   = "none",
    )

    # custom collator that injects labels based on *target_text*
    collate_fn = make_collate_fn(model.get_tokenizer().pad_token_id)

    trainer = SFTTrainer(
        model          = model,
        train_dataset  = train_dataset,
        eval_dataset   = val_dataset,
        data_collator  = collate_fn,
        args           = cfg,
    )

    trainer.train()
    model.save_pretrained(output_dir)

    return

def main():
    """
    Main entry point for projector training script
    
    Parameters:
        - None (uses command line arguments)
        
    Returns:
        - None (saves trained model to output directory)
    """
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Train BBOB projector")
    parser.add_argument("-m", "--model", required=True, help="Model location/path")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset location/path")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Maximum number of training epochs (default: 1)")
    parser.add_argument("-i", "--instruction", required=True, help="Instruction text to add to dataset examples")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for projector training (default: 2e-5)")
    parser.add_argument("--bnb_config", type=str, default=None, help="Bits and bytes configuration (default: None)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before optimizer step (default: 1)")
    args = parser.parse_args()
    
    # create output directory 
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"Output/{current_datetime}"
    os.makedirs(output_dir, exist_ok=True)

    # add file handler for logging to file
    logfile = os.path.join(output_dir, "training.log")
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(logfile) for h in logger.handlers):
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    logger.info(f"Loading base language model from: {args.model}")
    logger.info(f"Loading dataset from: {args.dataset}")
    logger.info(f"Training for max {args.epochs} epochs")

    model = load_model(args.model, args.bnb_config)

    model.freeze_model()                           
    for p in model.vision_tower.parameters():      
        p.requires_grad = False
    model.vision_tower.eval()
    model.unfreeze_projector()                     

    train_dataset, val_dataset = load_and_prepare_dataset(
        args.dataset,
        tokenizer=model.get_tokenizer(),
        instruction=args.instruction,
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
    )

    logger.info("Projector training is complete, model successfully saved.")
    return

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main()