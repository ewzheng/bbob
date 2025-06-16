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

# utils
import sys
import os
import multiprocess as mp
import math
import numpy as np

# training
from transformers import Trainer, TrainingArguments
from Utils import get_logger, LoggingCallback, model_size_breakdown, create_metrics_functions   
from Model import build_BBOB 
from Train import load_and_prepare_dataset, clean_tokenizer_config

# img / tensor utilities
from torchvision.transforms.functional import pil_to_tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# Prevent "tokenizers parallelism" fork warnings inside multiprocess map
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")    

def make_collate_fn(pad_token_id: int, tokenizer):
    '''
    factory that builds a custom collate_fn for sfttrainer.

    parameters:
        - pad_token_id (int): tokenizer pad id.

    returns: callable(list[dict]) -> batch dict.
    '''

    def _collate(batch):
        # find image key dynamically
        img_key = None
        for cand in ("images", "image", "pixel_values"):
            if cand in batch[0]:
                img_key = cand
                break
        if img_key is None:
            raise KeyError("Batch items lack an 'images'/'image'/'pixel_values' field")

        target_size = (256, 256)
        # Stay on CPU inside worker; main process/Accelerate will move to GPU
        device = "cpu"

        processed = []
        for img in [item[img_key] for item in batch]:
            # Case-1: pre-normalised numpy array (3,H,W) or tensor
            if isinstance(img, torch.Tensor):
                t = img.to(dtype=torch.float32)
            elif isinstance(img, np.ndarray):
                t = torch.as_tensor(img, dtype=torch.float32)
            else:
                # Fallback: PIL → tensor path (rare after preprocessing refactor)
                t = pil_to_tensor(img).float().div_(255.0).to(device)

            # Ensure channel-first shape (3, H, W)
            if t.dim() == 2:  # grayscale H×W
                t = t.unsqueeze(0).expand(3, -1, -1)  # repeat channels
            elif t.dim() == 3:
                if t.shape[0] == 3:  # C,H,W RGB
                    pass
                elif t.shape[0] == 1:  # C=1, H, W  -> replicate channel
                    t = t.expand(3, -1, -1)
                elif t.shape[2] == 3:  # H,W,C RGB
                    t = t.permute(2, 0, 1)
                elif t.shape[2] == 1:  # H,W,1 grayscale
                    t = t.permute(2, 0, 1).expand(3, -1, -1)
                else:
                    raise RuntimeError(f"Unexpected image shape {t.shape}; cannot determine channel dimension")
            else:
                raise RuntimeError(f"Unsupported tensor dim {t.dim()} for image input")

            _, H, W = t.shape
            if (H, W) != target_size:
                scale = min(target_size[1] / H, target_size[0] / W)
                # Clamp to at least 1 px to avoid zero-dimension resize errors
                nh = max(1, int(H * scale))
                nw = max(1, int(W * scale))
                t = F.interpolate(t.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False)[0]

                canvas = 0.5 * torch.ones(3, *target_size)
                dh = (target_size[1] - nh) // 2
                dw = (target_size[0] - nw) // 2
                canvas[:, dh:dh+nh, dw:dw+nw] = t
                t = canvas

            processed.append(t)

        pixel_values = torch.stack(processed, 0)

        merged_input_ids = []
        merged_labels = []

        for item in batch:
            # --- instruction tokens ---
            if "input_ids" in item:
                instr_ids = torch.as_tensor(item["input_ids"], dtype=torch.long).flatten()
            else:
                text = item.get("text", "")
                # Leave room for the 64 visual tokens the model will prepend
                VIS_TOKENS = 64
                max_txt_len = tokenizer.model_max_length - VIS_TOKENS
                tokens = tokenizer(text, return_tensors="pt", max_length=max_txt_len, truncation=True)
                instr_ids = tokens["input_ids"].squeeze(0)

            if "target_text" in item:
                tgt_ids = torch.as_tensor(item["target_text"], dtype=torch.long)
            else:
                tgt_ids = torch.tensor([], dtype=torch.long)

            # ensure both are 1-D
            instr_ids = instr_ids.flatten()
            tgt_ids   = tgt_ids.flatten()

            # drop padding tokens that were added during preprocessing
            instr_ids = instr_ids[instr_ids != pad_token_id]
            tgt_ids   = tgt_ids[tgt_ids   != pad_token_id]

            # -------------------------------------------------------------
            # Ensure combined text length stays within positional budget
            # -------------------------------------------------------------
            max_txt_len = tokenizer.model_max_length - VIS_TOKENS

            if instr_ids.size(0) > max_txt_len:
                # Extremely long instruction – keep the *last* segment so that
                # targets remain aligned to the end.
                instr_ids = instr_ids[-max_txt_len:]
                tgt_ids = torch.tensor([], dtype=torch.long)
            else:
                remaining = max_txt_len - instr_ids.size(0)
                if tgt_ids.size(0) > remaining:
                    tgt_ids = tgt_ids[:remaining]

            # concatenate instruction + (possibly truncated) target ⇒ model input (text only)
            ids = torch.cat([instr_ids, tgt_ids], dim=0)

            # build labels: prepend placeholders for visual tokens (64) and mask instruction
            visual_ignore = torch.full((VIS_TOKENS,), -100, dtype=torch.long)
            lbl = torch.cat([visual_ignore, ids.clone()])
            lbl[: VIS_TOKENS + instr_ids.size(0)] = -100  # ignore vision + instruction

            merged_input_ids.append(ids)
            merged_labels.append(lbl)

        # pad to max length first
        input_ids_padded = pad_sequence(merged_input_ids, batch_first=True, padding_value=pad_token_id)
        labels_padded    = pad_sequence(merged_labels,    batch_first=True, padding_value=-100)

        # build attention mask AFTER padding – **visual tokens should be *visible***
        # The language model receives the visual embeddings prepended internally
        # by `BBOB._merge_multimodal_inputs`, which itself creates a 1-filled
        # attention mask for those tokens.  Hence we only need to pass the text
        # mask here.  Setting the visual part to 0 would make the model ignore
        # the image information entirely.

        attention_mask = (input_ids_padded != pad_token_id).long()

        return {
            "images": pixel_values,
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
        dataloader_pin_memory       = True, 
        save_total_limit            = 2,
        lr_scheduler_type           = "cosine",   
        warmup_ratio                = warmup_ratio,
        lr_scheduler_kwargs         = {"num_cycles": 0.2},
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
    # custom collator that injects labels based on *target_text*
    collate_fn = make_collate_fn(tokenizer.pad_token_id, tokenizer)

    # Create metrics functions with shared state (no global variables)
    # This creates two functions that share closure variables for accumulating metrics
    compute_metrics, preprocess_logits_for_metrics = create_metrics_functions()

    trainer = Trainer(
        model          = model,
        train_dataset  = train_dataset,
        eval_dataset   = val_dataset,
        data_collator  = collate_fn,
        args           = cfg,
        callbacks      = [LoggingCallback(logger    )] if logger is not None else None,
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
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main()  