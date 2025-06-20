''' 
File: train_vision.py
Author: Auto-generated
Description: End-to-end training (vision tower + projector + language model) using the composite detection + language modelling loss.  
Usage: python train_vision.py -m <checkpoint_with_projector> -d <dataset_name> -i <instruction_text> [other args]
'''

import torch
import argparse
from datetime import datetime
import os, math, multiprocess as mp

from transformers import TrainingArguments

from Utils import get_logger, LoggingCallback, create_metrics_functions
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
    lm_target: float = 2.0,
):
    """End-to-end training of the whole BBOB model with composite loss."""


    model.unfreeze_model()
    model.unfreeze_projector()

    cuda = torch.cuda.is_available()
    bf16_supported = cuda and torch.cuda.is_bf16_supported()
    steps_per_epoch = max(math.ceil(len(train_dataset) / (batch_size * grad_acc_steps)), 1)

    cfg = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        learning_rate=lr,
        optim="adamw_torch",
        weight_decay=0.1,
        max_grad_norm=1.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        bf16=bf16_supported,
        fp16=cuda and not bf16_supported,
        eval_strategy="steps",
        eval_steps=max(steps_per_epoch // 4, 1),
        save_strategy="epoch",  
        logging_steps=max(batch_size // grad_acc_steps, 1),
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=num_workers,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        save_total_limit=2,
        save_safetensors=True,
        lr_scheduler_type="cosine_with_restarts",
        warmup_ratio=warmup_ratio,
        lr_scheduler_kwargs={"num_cycles": epochs},
        include_num_input_tokens_seen=True,
    )

    tokenizer = model.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    clean_tokenizer_config(tokenizer)

    # ------------------------------------------------------------------
    # Add <bbob> tokens so the language model can emit them
    # ------------------------------------------------------------------

    special = {
        "additional_special_tokens": ["<bbob>", "</bbob>"]
    }
    num_added = tokenizer.add_special_tokens(special)
    if num_added > 0:
        logger.info(f"Added {num_added} special tokens to tokenizer – resizing model embeddings")
        try:
            model.base_model.resize_token_embeddings(len(tokenizer))
        except AttributeError:
            # Fallback for different attribute names
            model.resize_token_embeddings(len(tokenizer))

    logger.info("Preparing dataset …")

    # Collator always hides targets; TF decision moved to BBOBTrainer
    collate_fn = make_collate_fn(tokenizer.pad_token_id, tokenizer, tf_start_p=0.0, tf_end_p=0.0, schedule="linear", logger=logger, log_interval=max(batch_size // grad_acc_steps, 1))
    # Composite loss callable
    compute_loss_fn = create_compute_loss_func(tokenizer, logger=logger, log_interval = max(batch_size // grad_acc_steps, 1), lm_target=lm_target)  

    # Create metrics functions with shared state (no global variables)
    # This creates two functions that share closure variables for accumulating metrics
    compute_metrics, preprocess_logits_for_metrics = create_metrics_functions(tokenizer)

    total_tf_steps = int(warmup_ratio * epochs * steps_per_epoch)
    trainer = BBOBTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        train_collator=collate_fn,
        eval_collator=collate_fn,  # same collator works for eval
        tf_start_p=1.0,
        tf_end_p=0.0,
        total_tf_steps=total_tf_steps,
        tf_schedule="cosine",
        args=cfg,
        callbacks=[LoggingCallback(logger)] if logger is not None else None,
        compute_loss_func=compute_loss_fn,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
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
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--bnb_config", type=str, default=None)
    parser.add_argument("--lm_target", type=float, default=2.75)
    
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"Output/vision_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(args.output_dir, "vision_training.log")

    logger.info(f"Loading model from {args.model}")
    model = build_BBOB(args.model, args.bnb_config, load=True)

    logger.info("Preparing dataset …")
    train_ds, val_ds = load_and_prepare_dataset(
        args.dataset,
        tokenizer=model.get_tokenizer(),
        instruction=args.instruction,
        image_processor=model.get_image_processor(),
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
        num_workers=args.num_workers,
        warmup_ratio=args.warmup_ratio,
        lm_target=args.lm_target,
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