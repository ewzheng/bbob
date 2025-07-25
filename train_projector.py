'''
File: train_projector.py
Author: Elias Zheng and Claude
Description: Train only the projector. Everything else is frozen.
Usage: python train_projector.py -m <base_llm_path> -d <dataset_name> -e <epochs> -i <instruction_text>
'''

import os
# Set CUDA memory allocator to use expandable segments to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import argparse
from datetime import datetime

# utils
import sys
import multiprocess as mp
import math
import numpy as np

# training
from transformers import Trainer, TrainingArguments
from Utils import get_logger, LoggingCallback, model_size_breakdown, create_metrics_functions   
from Model import build_BBOB 
from Train import load_and_prepare_dataset, clean_tokenizer_config, make_collate_fn

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
    warmup_ratio: float = 0.0,
    num_workers: int = 4,
):
    '''
    fine-tune projector weights only.

    parameters:
        - model (BBOB): model with frozen lm & vision tower.
        - train_dataset (Dataset): processed train split.
        - val_dataset (Dataset): validation split.
        - output_dir (str): path to save checkpoints.
        - epochs (int): training epochs.
        - batch_size (int): per-device batch size.
        - lr (float): learning rate.
        - grad_acc_steps (int): gradient accumulation.
        - logger (logging.Logger|None): optional logger.
        - warmup_ratio (float): lr warmup ratio.
    '''

    model.freeze_model()                              
    model.unfreeze_projector()    
    model.freeze_vision_tower()              

    # config
    cuda = torch.cuda.is_available()
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    steps_per_epoch = math.ceil(len(train_dataset) / (batch_size * grad_acc_steps))
    steps_per_epoch = max(steps_per_epoch, 1)  # safety guard

    cfg = TrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        gradient_accumulation_steps = grad_acc_steps,
        learning_rate               = lr,
        optim                       = "adamw_torch",
        weight_decay                = 0.1,
        max_grad_norm               = 1.0,
        adam_beta1                  = 0.9,
        adam_beta2                  = 0.95,
        bf16                        = bf16_supported,
        fp16                        = cuda and not bf16_supported,  
        eval_strategy               = "steps",
        eval_steps                  = max(steps_per_epoch // 4, 1),
        eval_accumulation_steps     = 4,  # Memory optimization for evaluation
        save_strategy               = "steps",
        save_steps                  = max(steps_per_epoch // 3, 1),
        logging_steps               = max(batch_size // grad_acc_steps, 1),
        report_to                   = "none",
        remove_unused_columns       = False,
        dataloader_num_workers      = num_workers,
        dataloader_persistent_workers=True,
        dataloader_pin_memory       = True, 
        save_total_limit            = 2,
        save_safetensors            = True,
        lr_scheduler_type           = "linear",   
        warmup_ratio                = warmup_ratio,
        torch_empty_cache_steps     = max(512 // grad_acc_steps, 1) + max(batch_size // grad_acc_steps, 1), # flush cache after eval
        include_num_input_tokens_seen = True,  # Enable token counting for metrics
    )

    # guarantee pad token exists
    tokenizer = model.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # sanitise dtype entries so tokenizer can be saved by Trainer
    try:
        clean_tokenizer_config(tokenizer)
    except Exception as _e:
        # do not fail training if helper import fails – just log once
        if logger is not None:
            logger.warning("Tokenizer config sanitisation skipped: %s", _e)

    logger.info(model_size_breakdown(model))
    # ---------------- visual token length ---------------------------
    vis_tokens = getattr(model, "vis_length", 64)

    # Keep Train.train_common constant in sync (used by preprocessing helpers)
    import Train.train_common as tc
    tc.VIS_TOKENS = vis_tokens  # type: ignore[attr-defined]

    # custom collator that injects labels based on *target_text*
    collate_fn = make_collate_fn(
        tokenizer.pad_token_id,
        tokenizer=model.get_tokenizer(),
        image_processor=model.get_image_processor(),
        on_the_fly=True,
        vis_tokens=vis_tokens
    )

    # Create metrics functions with shared state (no global variables)
    # This creates two functions that share closure variables for accumulating metrics
    compute_metrics, preprocess_logits_for_metrics = create_metrics_functions(tokenizer=model.get_tokenizer())

    trainer = Trainer(
        model          = model,
        train_dataset  = train_dataset,
        eval_dataset   = val_dataset,
        data_collator  = collate_fn,
        args           = cfg,
        callbacks      = [LoggingCallback(logger)] if logger is not None else None,
        processing_class = tokenizer,
        compute_metrics = compute_metrics,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics
    )   
    
    logger.info("Starting training of projector...")

    trainer.train()

    return

def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(description='train bbob projector')
    parser.add_argument("-m", "--model", required=True, help="Model location/path")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset location/path")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Maximum number of training epochs (default: 1)")
    parser.add_argument("-i", "--instruction", required=True, help="Instruction text to add to dataset examples")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for projector training (default: 2e-5)")
    parser.add_argument("--bnb_config", type=str, default=None, help="Bits and bytes configuration (default: None)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--warmup_ratio", type=float, default=0.2, help="Warmup ratio for scheduler (default: 0.2)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before optimizer step (default: 1)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of CPU workers for DataLoader preprocessing (default: 4)")
    parser.add_argument("--output_dir", type=str, default=None, help="Name of output directory")
    args = parser.parse_args()
    
    # create output directory 
    if args.output_dir is None:
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"Output/{current_datetime}"
    else:
        output_dir = args.output_dir
        
    os.makedirs(output_dir, exist_ok=True)

    # logging
    logger = get_logger(output_dir, "projector_training.log") 

    logger.info(f"Loading base language model from: {args.model}")
    logger.info(f"Loading dataset from: {args.dataset}")
    logger.info(f"Training for max {args.epochs} epochs")

    model = build_BBOB(args.model, args.bnb_config)
    
    # Clean GPU memory after model construction to prevent fragmentation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("Cleared GPU cache after model initialization")

    train_dataset, val_dataset = load_and_prepare_dataset(
        args.dataset,
        tokenizer=model.get_tokenizer(),
        instruction=args.instruction,
        image_processor=model.get_image_processor(),
        dtype=model.dtype,
        on_the_fly=True
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
        num_workers=args.num_workers,
        warmup_ratio=args.warmup_ratio
    )

    logger.info("Projector training is complete, model successfully saved.")
    
    return

if __name__ == "__main__":
    # Prevent "tokenizers parallelism" fork warnings inside multiprocess map
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")    

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main()  