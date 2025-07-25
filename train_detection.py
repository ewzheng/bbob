''' 
File: train_detection.py
Author: Elias Zheng and Claude
Description: End-to-end training (vision tower + projector + language model) using the composite detection + language modelling loss.  
Usage: python train_detection.py -m <checkpoint_with_projector> -d <dataset_name> -i <instruction_text> [other args]
'''

import os
# Set CUDA memory allocator to use expandable segments to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import argparse
from datetime import datetime
import math, multiprocess as mp
import random
import numpy as np

from transformers import TrainingArguments

from Utils import get_logger, LoggingCallback, create_metrics_functions, model_size_breakdown
from Model import build_BBOB
from Train import load_and_prepare_dataset, clean_tokenizer_config, make_collate_fn, create_compute_loss_func, BBOBTrainer


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
    """End-to-end training of the whole BBOB model with composite loss."""


    model.unfreeze_model()
    model.unfreeze_projector()
    model.freeze_vision_tower()

    cuda = torch.cuda.is_available()
    bf16_supported = cuda and torch.cuda.is_bf16_supported()

    batches_per_epoch = max(math.ceil(len(train_dataset) / batch_size), 1)
    optim_steps_per_epoch = max(math.ceil(batches_per_epoch / grad_acc_steps), 1)
    total_optim_steps = epochs * optim_steps_per_epoch

    # Keep the old variable name for backward-compatibility
    steps_per_epoch = optim_steps_per_epoch

    cfg = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        learning_rate=lr,
        optim="paged_adamw_8bit",
        weight_decay=0.1,
        max_grad_norm=1.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        bf16=bf16_supported,
        fp16=cuda and not bf16_supported,
        eval_strategy="steps",
        eval_steps=max(steps_per_epoch, 0.05),
        save_strategy="steps",
        save_steps=max(steps_per_epoch, 0.05),
        logging_steps=batch_size,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=num_workers,
        dataloader_prefetch_factor=8,
        dataloader_persistent_workers=False,
        dataloader_pin_memory=True,
        save_total_limit=8,
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=False,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        include_num_input_tokens_seen=True,
        torch_empty_cache_steps     = max(steps_per_epoch // 2, 50),
    )

    tokenizer = model.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    clean_tokenizer_config(tokenizer)

    logger.info(model_size_breakdown(model))

    logger.info("Preparing dataset …")

    # Visual-token count from model
    vis_tokens = getattr(model, "vis_length", 64)

    # Update train_common global for downstream preprocessing heuristics
    import Train.train_common as tc
    tc.VIS_TOKENS = vis_tokens  # type: ignore[attr-defined]

    # Collator uses the dynamic vis_tokens value
    collate_fn = make_collate_fn(
        tokenizer.pad_token_id,
        tokenizer=tokenizer,
        image_processor=model.get_image_processor(),
        logger=logger,
        on_the_fly=False,
        vis_tokens=vis_tokens,
    )
    eval_collate_fn = make_collate_fn(
        tokenizer.pad_token_id,
        tokenizer=tokenizer,
        image_processor=model.get_image_processor(),
        logger=logger,
        on_the_fly=False,
        vis_tokens=vis_tokens,
    )

    eval_collate_fn.eval()
    # Composite loss callable
    compute_loss_fn = create_compute_loss_func(tokenizer, logger=logger, log_interval = max(steps_per_epoch//4, 0.05))  

    # Create metrics functions with shared state (no global variables)
    # This creates two functions that share closure variables for accumulating metrics
    compute_metrics, preprocess_logits_for_metrics = create_metrics_functions(tokenizer, do_detection_metrics=True, logger=logger)

    trainer = BBOBTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        train_collator=collate_fn,
        eval_collator=eval_collate_fn,  # same collator works for eval
        args=cfg,
        callbacks=[LoggingCallback(logger)] if logger is not None else None,
        compute_loss_func=compute_loss_fn,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        force=True
    )

    logger.info("Starting end-to-end training …")
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="End-to-end fine-tune BBOB with composite loss")
    parser.add_argument("-m", "--model", required=True, help="Path to projector-trained checkpoint")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset path or HF name")
    parser.add_argument("-i", "--instruction", required=True, help="Instruction to prepend in dataset")
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--bnb_config", type=str, default=None) 
    
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"Output/detection_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(args.output_dir, exist_ok=True)

    if args.num_workers == -1:
        num_workers = min(mp.cpu_count() - 2, 12)  # Use more workers for CPU-intensive collation
    else:
        num_workers = args.num_workers

    logger = get_logger(args.output_dir, "vision_training.log")

    logger.info(f"Loading model from {args.model}")
    model = build_BBOB(args.model, args.bnb_config, load=True)
    
    # Clean GPU memory after model construction to prevent fragmentation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("Cleared GPU cache after model initialization")

    # Get tokenizer and ensure it has a defined pad token (defaults to EOS).
    tokenizer = model.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------------
    # Always add the opening/closing <|bbob|> tags. Optionally add
    # 1 001 numeric-bin tokens ("0.000" … "1.000") when the user
    # passes --bin=True.  These are declared as *additional* special
    # tokens so the tokenizer will treat each string as an atomic
    # token, never split by the underlying BPE.
    # --------------------------------------------------------------

    extra_tokens = ["<|bbob|>", "</|bbob|>"] + [f"{i/1000:.3f}" for i in range(1001)]  # 0.000…1.000

    num_added = tokenizer.add_special_tokens({"additional_special_tokens": extra_tokens})

    if num_added > 0:
        logger.info(f"Added {num_added} special tokens to tokenizer – resizing model embeddings")
        try:
            model.base_model.resize_token_embeddings(len(tokenizer))
        except AttributeError:
            model.resize_token_embeddings(len(tokenizer))

    # Sanitize config after modification so Trainer can save it
    clean_tokenizer_config(tokenizer)

    logger.info("Preparing dataset …")
    train_ds, val_ds = load_and_prepare_dataset(
        args.dataset,
        tokenizer=tokenizer,
        instruction=args.instruction,
        image_processor=model.get_image_processor(),
        augment=True,
        dtype=model.dtype,
        on_the_fly=False,
    )

    train(
        model,
        train_ds,
        val_ds,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_acc_steps=args.gradient_accumulation_steps,
        logger=logger,
        num_workers=num_workers,
        warmup_ratio=args.warmup_ratio,
    )

    logger.info("Vision training complete. Model saved.")


if __name__ == "__main__":
    # Prevent "tokenizers parallelism" fork warnings inside multiprocess map
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")    

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main()  