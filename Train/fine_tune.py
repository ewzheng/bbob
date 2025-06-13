'''
File: fine_tune.py
Author: Elias Zheng and Claude
Description: This script fine-tunes the BBOB model using LoRA adapters and a custom composite loss for structured vision-language tasks. Supports gradient accumulation, learning rate warmup, and mixed precision training. Designed for multimodal datasets with bounding box and class label supervision.
Usage: python fine_tune.py -m <model_path> -d <dataset_name> -v <vision_tower_path> -i <instruction_text> [other options]
'''

import os
import sys
import time
import argparse
import logging
from datetime import datetime
import math

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
import multiprocess as mp

from train_common import collate, load_and_prepare_dataset, compute_gradient_norm, compute_parameter_norm
from loss_common import CompositeLoss
from Model.model import BBOB

# configure logging (if not already configured)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def lora_loss(model, batch, composite_loss_fn, tokenizer, return_components=False, class_map=None):
    """Compute joint language modeling and detection/classification loss for fine-tuning."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    vision_features = batch.get("vision_features", None)
    target_labels = batch.get("target_labels", None)
    target_boxes = batch.get("target_boxes", None)
    target_text = batch.get("target_text", None)
    # Forward pass with detection head
    outputs = model(
        vision_features=vision_features,
        input_ids=input_ids,
        attention_mask=attention_mask,
        detection=True,
        target_labels=target_labels
    )
    # Language modeling logits and labels
    lm_logits = outputs["outputs"].logits  # [B, seq_len_total, vocab_size]
    # Construct lm_labels to match visual+text sequence
    num_visual_tokens = outputs["class_logits"].shape[1]
    batch_size, num_text_tokens = input_ids.shape
    device = input_ids.device
    visual_labels = torch.full((batch_size, num_visual_tokens), -100, dtype=input_ids.dtype, device=device)
    text_labels = input_ids.clone()
    text_labels[attention_mask == 0] = -100
    lm_labels = torch.cat([visual_labels, text_labels], dim=1)
    # Detection/classification outputs
    class_logits = outputs["class_logits"]  # [B, num_visual_tokens, num_classes]
    box_preds = outputs["box_preds"]        # [B, num_visual_tokens, 4]
    # Compute composite loss
    result = composite_loss_fn(
        lm_logits, lm_labels, class_logits, box_preds, target_labels, target_boxes, target_text, class_map=class_map, return_components=return_components
    )
    return result

def find_lora_target_modules(model):
    """
    Dynamically determine LoRA target modules based on the model architecture.
    """
    # Try to get the model type from config if available
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    class_name = model.__class__.__name__.lower()

    # Llama and similar
    if model_type is not None and "llama" in model_type:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if "llama" in class_name:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # GPT-2 and similar
    if model_type == "gpt2" or "gpt2" in class_name:
        return ["c_attn", "c_proj", "c_fc"]

    # Falcon
    if model_type == "falcon" or "falcon" in class_name:
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

    # MPT
    if model_type == "mpt" or "mpt" in class_name:
        return ["Wqkv", "out_proj", "fc1", "fc2"]

    # OPT
    if model_type == "opt" or "opt" in class_name:
        return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

    # Bloom
    if model_type == "bloom" or "bloom" in class_name:
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

    # Default fallback: try to find all linear modules
    import torch
    linear_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_modules.add(name.split('.')[-1])
    if linear_modules:
        return list(linear_modules)

    # If all else fails, return an empty list
    return []

def fine_tune(model, train_dataset, test_dataset, lora_rank, lora_alpha, lora_dropout, learning_rate, bias, batch_size, gradient_accumulation_steps, warmup_steps, max_steps, output_dir, epochs=1, class_map=None):
    """Fine-tune BBOB with LoRA adapters and composite loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # For GPT-2, use the standard LoRA target modules
    target_modules = find_lora_target_modules(model.base_model)
    logger.info(f"Using target modules for LoRA: {target_modules}")
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )
    
    
    if hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable()
        if hasattr(model.base_model, "config") and hasattr(model.base_model.config, "use_cache"):
            model.base_model.config.use_cache = False
        
    tokenizer = model.base_tokenizer
    model.base_model = prepare_model_for_kbit_training(model.base_model)
    model.base_model = get_peft_model(model.base_model, lora_config)
    model.base_model.print_trainable_parameters()

    scaler = GradScaler("cuda") if torch.cuda.is_available() and not next(model.base_model.parameters()).dtype == torch.bfloat16 else None
    logger.info(f"Mixed precision training: {'Enabled' if scaler is not None else 'Disabled'}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=2, prefetch_factor=1, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=2, prefetch_factor=1, persistent_workers=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate, weight_decay=0
    )

    steps_per_epoch = (len(train_loader) + gradient_accumulation_steps - 1) // gradient_accumulation_steps
    val_freq = 512//gradient_accumulation_steps
    logger.info(f"Starting LoRA fine-tuning for {epochs} epochs, max {max_steps} steps")
    logger.info(f"Device: {device}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(test_loader)}")
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=(steps_per_epoch*epochs)-warmup_steps,
        num_cycles=epochs
    )

    loss_fn = CompositeLoss()

    best_val_loss = float('inf')
    start_time = time.time()
    checkpoint_dir = f"{output_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    initial_memory = torch.cuda.memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0
    logger.info(f"Initial GPU memory usage: {initial_memory:.2f}GB")
    logger.info("Beginning fine tuning.")
    optimizer.zero_grad()
    global_step = 0  # Only increments when optimizer.step() is called

    for epoch in range(epochs):
        if global_step >= max_steps:
            break
        epoch_start_time = time.time()

        model.train()

        total_train_loss = 0
        total_cls_correct = 0
        total_cls_total = 0
        total_iou_sum = 0.0
        total_iou_count = 0
        total_l1 = 0
        accumulation_counter = 0
        for batch_idx, batch in enumerate(train_loader):
            if global_step >= max_steps:
                break
            batch_start_time = time.time()
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            if scaler is not None:
                with autocast("cuda"):
                    loss_tuple = lora_loss(model, batch, loss_fn, tokenizer, return_components=True, class_map=class_map)
                loss = loss_tuple[0] / gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                loss_tuple = lora_loss(model, batch, loss_fn, tokenizer, return_components=True, class_map=class_map)
                loss = loss_tuple[0] / gradient_accumulation_steps
                loss.backward()

            # Accumulate metrics from loss_tuple
            total_train_loss += loss_tuple[0].item() * gradient_accumulation_steps
            total_cls_correct += loss_tuple[1]
            total_cls_total += loss_tuple[2]
            total_iou_sum += loss_tuple[3]
            total_iou_count += loss_tuple[4]
            total_l1 += loss_tuple[5]
            # Period metrics
            if batch_idx % 128 == 0:
                period_cls_correct = loss_tuple[1]
                period_cls_total = loss_tuple[2]
                period_iou_sum = loss_tuple[3]
                period_iou_count = loss_tuple[4]
            else:
                if 'period_cls_correct' not in locals():
                    period_cls_correct = 0
                    period_cls_total = 0
                    period_iou_sum = 0.0
                    period_iou_count = 0
                period_cls_correct += loss_tuple[1]
                period_cls_total += loss_tuple[2]
                period_iou_sum += loss_tuple[3]
                period_iou_count += loss_tuple[4]
            accumulation_counter += 1
            logger.info(f"Accumulation counter: {accumulation_counter}")
            if accumulation_counter == gradient_accumulation_steps or (batch_idx + 1 == len(train_loader)):
                logger.info(f"==> OPTIMIZER STEP about to start at batch {batch_idx}, global_step {global_step}")
                if scaler is not None:
                    logger.info("Calling backward() (mixed precision)")
                    # backward already called above
                    scaler.unscale_(optimizer)
                    grad_norm = compute_gradient_norm(model)
                    logger.info("Finished backward() (mixed precision)")
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logger.info("Calling backward() (fp32)")
                    # backward already called above
                    grad_norm = compute_gradient_norm(model)
                    logger.info("Finished backward() (fp32)")
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                logger.info(f"==> OPTIMIZER STEP DONE at batch {batch_idx}, global_step {global_step}")
                accumulation_counter = 0
                global_step += 1

                batch_time = time.time() - batch_start_time
                samples_per_sec = len(batch["input_ids"]) / max(batch_time, 1e-6)
                period_cls_acc = period_cls_correct / max(period_cls_total, 1)
                period_mean_iou = period_iou_sum / max(period_iou_count, 1)
                period_mean_iou = min(max(period_mean_iou, 0.0), 1.0)
                percent_matched = (period_iou_count / period_cls_total) if period_cls_total > 0 else 0.0
                logger.info(f"[TRAIN] Epoch {epoch+1}/{epochs} | Step {global_step} | Batch {batch_idx}/{len(train_loader)} | "
                            f"Loss: {loss_tuple[0].item() * gradient_accumulation_steps:.4f} | ClsAcc: {period_cls_acc:.4f} | MeanIoU: {period_mean_iou:.4f} | "
                            f"IoU Matches: {period_iou_count} | GT Objects: {period_cls_total} | %Matched: {percent_matched:.2%} | "
                            f"GradNorm: {grad_norm:.2e} | Speed: {samples_per_sec:.1f} samples/s | LR: {scheduler.get_last_lr()[0]:.2e}")
                # Reset period counters
                period_cls_correct = 0
                period_cls_total = 0
                period_iou_sum = 0.0
                period_iou_count = 0
                if global_step > 0 and global_step % val_freq == 0:
                    model.eval()
                    total_val_loss = 0
                    total_val_cls_correct = 0
                    total_val_cls_total = 0
                    total_val_iou_sum = 0.0
                    total_val_iou_count = 0
                    total_val_l1 = 0
                    val_batches = 0
                    # Initialize period IoU matches
                    period_iou_matches = 0
                    with torch.no_grad():
                        for val_batch_idx, val_batch in enumerate(test_loader):
                            val_batch_time = time.time()
                            val_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in val_batch.items()}
                            if scaler is not None:
                                with autocast("cuda"):
                                    val_loss_tuple = lora_loss(model, val_batch, CompositeLoss(), tokenizer, return_components=True, class_map=class_map)
                            else:
                                val_loss_tuple = lora_loss(model, val_batch, CompositeLoss(), tokenizer, return_components=True, class_map=class_map)
                            total_val_loss += val_loss_tuple[0]
                            total_val_cls_correct += val_loss_tuple[1]
                            total_val_cls_total += val_loss_tuple[2]
                            total_val_iou_sum += val_loss_tuple[3]
                            total_val_iou_count += val_loss_tuple[4]
                            total_val_l1 += val_loss_tuple[5]
                            val_batches += 1
                            period_iou_matches = val_loss_tuple[4]  # Non-cumulative, per-batch
                            if val_batch_idx % 32 == 0:
                                val_bt = time.time() - val_batch_time
                                val_sps = len(val_batch["input_ids"]) / max(val_bt, 1e-6)
                                val_cls_acc = total_val_cls_correct / max(total_val_cls_total, 1)
                                val_mean_iou = total_val_iou_sum / max(total_val_iou_count, 1)
                                val_mean_iou = min(max(val_mean_iou, 0.0), 1.0)
                                val_percent_matched = (period_iou_matches / period_cls_total) if period_cls_total > 0 else 0.0
                                logger.info(f"[VAL]   Epoch {epoch+1}/{epochs} | Step {global_step} | ValBatch {val_batch_idx}/{len(test_loader)} | "
                                            f"Loss: {val_loss_tuple[0]:.4f} | ClsAcc: {val_cls_acc:.4f} | MeanIoU: {val_mean_iou:.4f} | IoU Matches: {period_iou_matches} | GT Objects: {val_loss_tuple[2]} | L1: {val_loss_tuple[5]:.4f} | %Matched: {val_percent_matched:.2%} | Speed: {val_sps:.1f} samples/s")
                                # Log a sample of the model's output
                                try:
                                    sample_idx = 0
                                    input_ids = val_batch["input_ids"][sample_idx]
                                    input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                                    gt_boxes = val_batch.get("target_boxes", [None])[sample_idx]
                                    gt_labels = val_batch.get("target_labels", [None])[sample_idx]
                                    # Get the model's predictions for this batch
                                    outputs = model(
                                        vision_features=val_batch.get("vision_features", None),
                                        input_ids=val_batch["input_ids"],
                                        attention_mask=val_batch["attention_mask"],
                                        detection=True,
                                        target_labels=val_batch.get("target_labels", None)
                                    )
                                    class_logits = outputs["class_logits"][sample_idx]
                                    box_preds = outputs["box_preds"][sample_idx]
                                    pred_classes = class_logits.argmax(dim=-1).cpu().tolist()
                                    pred_boxes = box_preds.cpu().tolist()
                                    logger.info(f"[VAL SAMPLE] Batch {val_batch_idx} | Input: {input_text}")
                                    logger.info(f"[VAL SAMPLE] GT Labels: {gt_labels}")
                                    logger.info(f"[VAL SAMPLE] GT Boxes: {gt_boxes}")
                                    logger.info(f"[VAL SAMPLE] Pred Classes: {pred_classes[:5]} ...")
                                    logger.info(f"[VAL SAMPLE] Pred Boxes: {pred_boxes[:5]} ...")
                                except Exception as e:
                                    logger.warning(f"[VAL SAMPLE] Could not log sample output: {e}")
                    avg_val_loss = total_val_loss / max(val_batches, 1)
                    avg_val_cls_acc = total_val_cls_correct / max(total_val_cls_total, 1)
                    avg_val_mean_iou = total_val_iou_sum / max(total_val_iou_count, 1)
                    avg_val_mean_iou = min(max(avg_val_mean_iou, 0.0), 1.0)
                    avg_val_l1 = total_val_l1 / max(val_batches, 1)
                    avg_val_percent_matched = (total_val_iou_count / total_val_cls_total) if total_val_cls_total > 0 else 0.0
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model_dir = f"{checkpoint_dir}/best_model-step_{global_step}"
                        os.makedirs(best_model_dir, exist_ok=True)
                        model.save_pretrained(best_model_dir)
                        if hasattr(model.base_model, 'save_pretrained'):
                            model.base_model.save_pretrained(os.path.join(best_model_dir, 'lora_adapter'))
                        logger.info(f"[CHECKPOINT] NEW BEST MODEL at step {global_step} | Val Loss: {avg_val_loss:.4f} | ClsAcc: {avg_val_cls_acc:.4f} | MeanIoU: {avg_val_mean_iou:.4f} | L1: {avg_val_l1:.4f} | %Matched: {avg_val_percent_matched:.2%}")
                    current_dir = f"{checkpoint_dir}/latest-step_{global_step}"
                    os.makedirs(current_dir, exist_ok=True)
                    model.save_pretrained(current_dir)
                    if hasattr(model.base_model, 'save_pretrained'):
                        model.base_model.save_pretrained(os.path.join(current_dir, 'lora_adapter'))
                    logger.info(f"[CHECKPOINT] Step {global_step} checkpoint saved | Val Loss: {avg_val_loss:.4f} | Best Val Loss: {best_val_loss:.4f}")
        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        avg_train_cls_acc = total_cls_correct / max(total_cls_total, 1)
        avg_train_mean_iou = total_iou_sum / max(total_iou_count, 1)
        avg_train_mean_iou = min(max(avg_train_mean_iou, 0.0), 1.0)
        avg_train_l1 = total_l1 / max(len(train_loader), 1)
        param_norm = compute_parameter_norm(model)
        epoch_time = time.time() - epoch_start_time
        avg_percent_matched = (total_iou_count / total_cls_total) if total_cls_total > 0 else 0.0
        logger.info(f"=== EPOCH {epoch+1}/{epochs} SUMMARY ===")
        logger.info(f"Losses: Train={avg_train_loss:.4f}, ClsAcc={avg_train_cls_acc:.4f}, MeanIoU={avg_train_mean_iou:.4f}, L1={avg_train_l1:.4f}, %Matched={avg_percent_matched:.2%}")
        logger.info(f"Model: ParamNorm={param_norm:.3f}, LR={scheduler.get_last_lr()[0]:.2e}")
        logger.info(f"Performance: EpochTime={epoch_time:.1f}s")
        # End of validation phase
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    model.eval()
    total_val_loss = 0
    total_val_cls_correct = 0
    total_val_cls_total = 0
    total_val_iou_sum = 0.0
    total_val_iou_count = 0
    total_val_l1 = 0
    val_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch_time = time.time()
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            # Use lora_loss logic for evaluation
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            vision_features = batch.get("vision_features", None)
            target_labels = batch.get("target_labels", None)
            target_boxes = batch.get("target_boxes", None)
            target_text = batch.get("target_text", None)
            outputs = model(
                vision_features=vision_features,
                input_ids=input_ids,
                attention_mask=attention_mask,
                detection=True,
                target_labels=target_labels
            )
            lm_logits = outputs["outputs"].logits
            num_visual_tokens = outputs["class_logits"].shape[1]
            batch_size_, num_text_tokens = input_ids.shape
            device_ = input_ids.device
            visual_labels = torch.full((batch_size_, num_visual_tokens), -100, dtype=input_ids.dtype, device=device_)
            text_labels = input_ids.clone()
            text_labels[attention_mask == 0] = -100
            lm_labels = torch.cat([visual_labels, text_labels], dim=1)
            class_logits = outputs["class_logits"]
            box_preds = outputs["box_preds"]
            # Compute composite loss and metrics
            val_loss_tuple = composite_loss_fn(
                lm_logits, lm_labels, class_logits, box_preds, target_labels, target_boxes, target_text, class_map=class_map, return_components=True
            )
            total_val_loss += val_loss_tuple[0]
            total_val_cls_correct += val_loss_tuple[1]
            total_val_cls_total += val_loss_tuple[2]
            total_val_iou_sum += val_loss_tuple[3]
            total_val_iou_count += val_loss_tuple[4]
            total_val_l1 += val_loss_tuple[5]
            val_batches += 1
            period_iou_matches = val_loss_tuple[4]  # Non-cumulative, per-batch
            if batch_idx % 32 == 0:
                val_bt = time.time() - batch_time
                val_sps = len(batch["input_ids"]) / max(val_bt, 1e-6)
                val_cls_acc = total_val_cls_correct / max(total_val_cls_total, 1)
                val_mean_iou = total_val_iou_sum / max(total_val_iou_count, 1)
                val_mean_iou = min(max(val_mean_iou, 0.0), 1.0)
                val_percent_matched = (period_iou_matches / val_loss_tuple[2]) if val_loss_tuple[2] > 0 else 0.0
                logger.info(f"[EVAL] Batch {batch_idx+1}/{len(test_loader)} | Loss: {val_loss_tuple[0]:.4f} | ClsAcc: {val_cls_acc:.4f} | MeanIoU: {val_mean_iou:.4f} | IoU Matches: {period_iou_matches} | GT Objects: {val_loss_tuple[2]} | L1: {val_loss_tuple[5]:.4f} | %Matched: {val_percent_matched:.2%} | Speed: {val_sps:.1f} samples/s")
    eval_time = time.time() - batch_time
    avg_val_loss = total_val_loss / max(val_batches, 1)
    avg_val_cls_acc = total_val_cls_correct / max(total_val_cls_total, 1)
    avg_val_mean_iou = total_val_iou_sum / max(total_val_iou_count, 1)
    avg_val_mean_iou = min(max(avg_val_mean_iou, 0.0), 1.0)
    avg_val_l1 = total_val_l1 / max(val_batches, 1)
    avg_val_percent_matched = (total_val_iou_count / total_val_cls_total) if total_val_cls_total > 0 else 0.0
    logger.info(f"=== FINAL EVALUATION SUMMARY ===")
    logger.info(f"Loss: {avg_val_loss:.4f} | ClsAcc: {avg_val_cls_acc:.4f} | MeanIoU: {avg_val_mean_iou:.4f} | L1: {avg_val_l1:.4f} | %Matched: {avg_val_percent_matched:.2%}")
    logger.info(f"Evaluation time: {eval_time:.1f}s")

def denormalize_bboxes(bboxes, img_w, img_h):
    """Convert normalized bboxes [N, 4] to COCO pixel coordinates [x, y, w, h]."""
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    return [[x*img_w, y*img_h, w*img_w, h*img_h] for x, y, w, h in bboxes]

def evaluate_on_test_set(model, test_dataset, batch_size=32, class_map=None):
    """Evaluate on test set using CompositeLoss, matching training/validation evaluation."""
    from torch.utils.data import DataLoader
    from loss_common import CompositeLoss
    import torch
    import time
    model.eval()
    device = next(model.parameters()).device
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=2, prefetch_factor=1, persistent_workers=True)
    logger.info("Beginning evaluation on final test set...")
    eval_start_time = time.time()
    composite_loss_fn = CompositeLoss()
    tokenizer = model.get_tokenizer() if hasattr(model, 'get_tokenizer') else None
    total_val_loss = 0
    total_val_cls_correct = 0
    total_val_cls_total = 0
    total_val_iou_sum = 0.0
    total_val_iou_count = 0
    total_val_l1 = 0
    val_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch_time = time.time()
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            # Use lora_loss logic for evaluation
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            vision_features = batch.get("vision_features", None)
            target_labels = batch.get("target_labels", None)
            target_boxes = batch.get("target_boxes", None)
            target_text = batch.get("target_text", None)
            outputs = model(
                vision_features=vision_features,
                input_ids=input_ids,
                attention_mask=attention_mask,
                detection=True,
                target_labels=target_labels
            )
            lm_logits = outputs["outputs"].logits
            num_visual_tokens = outputs["class_logits"].shape[1]
            batch_size_, num_text_tokens = input_ids.shape
            device_ = input_ids.device
            visual_labels = torch.full((batch_size_, num_visual_tokens), -100, dtype=input_ids.dtype, device=device_)
            text_labels = input_ids.clone()
            text_labels[attention_mask == 0] = -100
            lm_labels = torch.cat([visual_labels, text_labels], dim=1)
            class_logits = outputs["class_logits"]
            box_preds = outputs["box_preds"]
            # Compute composite loss and metrics
            val_loss_tuple = composite_loss_fn(
                lm_logits, lm_labels, class_logits, box_preds, target_labels, target_boxes, target_text, class_map=class_map, return_components=True
            )
            total_val_loss += val_loss_tuple[0]
            total_val_cls_correct += val_loss_tuple[1]
            total_val_cls_total += val_loss_tuple[2]
            total_val_iou_sum += val_loss_tuple[3]
            total_val_iou_count += val_loss_tuple[4]
            total_val_l1 += val_loss_tuple[5]
            val_batches += 1
            period_iou_matches = val_loss_tuple[4]  # Non-cumulative, per-batch
            if batch_idx % 32 == 0:
                val_bt = time.time() - batch_time
                val_sps = len(batch["input_ids"]) / max(val_bt, 1e-6)
                val_cls_acc = total_val_cls_correct / max(total_val_cls_total, 1)
                val_mean_iou = total_val_iou_sum / max(total_val_iou_count, 1)
                val_mean_iou = min(max(val_mean_iou, 0.0), 1.0)
                val_percent_matched = (period_iou_matches / val_loss_tuple[2]) if val_loss_tuple[2] > 0 else 0.0
                logger.info(f"[EVAL] Batch {batch_idx+1}/{len(test_loader)} | Loss: {val_loss_tuple[0]:.4f} | ClsAcc: {val_cls_acc:.4f} | MeanIoU: {val_mean_iou:.4f} | IoU Matches: {period_iou_matches} | GT Objects: {val_loss_tuple[2]} | L1: {val_loss_tuple[5]:.4f} | %Matched: {val_percent_matched:.2%} | Speed: {val_sps:.1f} samples/s")
    eval_time = time.time() - eval_start_time
    avg_val_loss = total_val_loss / max(val_batches, 1)
    avg_val_cls_acc = total_val_cls_correct / max(total_val_cls_total, 1)
    avg_val_mean_iou = total_val_iou_sum / max(total_val_iou_count, 1)
    avg_val_mean_iou = min(max(avg_val_mean_iou, 0.0), 1.0)
    avg_val_l1 = total_val_l1 / max(val_batches, 1)
    avg_val_percent_matched = (total_val_iou_count / total_val_cls_total) if total_val_cls_total > 0 else 0.0
    logger.info(f"=== FINAL EVALUATION SUMMARY ===")
    logger.info(f"Loss: {avg_val_loss:.4f} | ClsAcc: {avg_val_cls_acc:.4f} | MeanIoU: {avg_val_mean_iou:.4f} | L1: {avg_val_l1:.4f} | %Matched: {avg_val_percent_matched:.2%}")
    logger.info(f"Evaluation time: {eval_time:.1f}s")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune BBOB with LoRA")
    parser.add_argument("-m", "--model", required=True, help="Model location/path")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset name from HuggingFace Hub")  
    parser.add_argument("-v", "--vision_tower", required=False, help="Vision tower model location/path")
    parser.add_argument("-i", "--instruction", required=True, help="Instruction text to add to dataset examples")
    parser.add_argument("--bnb_config", type=str, default="8bit", help="Bits and bytes configuration (default: 8bit)")
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank (default: 64)")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha (default: 128)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout (default: 0.1)")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Learning rate (default: 6e-4)")
    parser.add_argument("--bias", type=str, default="lora_only", help="LoRA bias type: 'none', 'all', or 'lora_only' (default: 'lora_only')")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--warmup_steps", type=int, default=64, help="Warmup steps (default: 64)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of training epochs (default: 1)")
    parser.add_argument("--max_steps", type=int, default=2048, help="Maximum number of training steps (default: 2048)")
    parser.add_argument("--label_file", type=str, default=None, help="Optional path to YAML label file")
    parser.add_argument("--lora_adapter", type=str, default=None, help="Optional path to a LoRA adapter directory to load into the model at startup")

    args = parser.parse_args()

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"Tuning/{current_datetime}"
    os.makedirs(output_dir, exist_ok=True)

    # Add file handler for logging to file in output_dir
    logfile = os.path.join(output_dir, "training.log")
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(logfile) for h in logger.handlers):
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    logger.info(f"Loading model from: {args.model}")
    logger.info(f"Loading dataset from: {args.dataset}")
    logger.info(f"Vision tower: {args.vision_tower}")
    logger.info(f"Instruction: {args.instruction}")
    logger.info(f"LoRA config: rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    logger.info(f"Training config: lr={args.learning_rate}, batch_size={args.batch_size}, grad_accum={args.gradient_accumulation_steps}")
    logger.info(f"Bias: {args.bias}")
    logger.info(f"\n Beginning fine tuning on {args.dataset}")

    label_dict = None
    if args.label_file is not None:
        from train_common import load_labels_from_yaml
        label_dict = load_labels_from_yaml(args.label_file)
        logger.info(f"Loaded labels from {args.label_file}: {label_dict}")

    model = BBOB.from_pretrained(args.model, num_classes=80, bnb_config=args.bnb_config)
    if args.lora_adapter is not None:
        logger.info(f"Loading LoRA adapter from {args.lora_adapter}")
        model.load_lora_adapter(args.lora_adapter)
    train_dataset, test_dataset = load_and_prepare_dataset(
        args.dataset,
        model.get_tokenizer(),
        model.image_processor,
        model.vision_encoder,
        args.instruction
    )

    split = test_dataset.train_test_split(test_size=0.1, seed=42)
    validation_dataset = split['train']
    final_test_dataset = split['test']

    logger.info(f"Train set: {len(train_dataset)} samples, Validation set: {len(validation_dataset)} samples, Test set: {len(final_test_dataset)} samples")
    logger.info(f"{model.get_model_size()}")
    
    model.unfreeze_model()
    model.freeze_vision_tower() 
    model.unfreeze_projector()
    model.unfreeze_heads()

    fine_tune(
        model,
        train_dataset,
        validation_dataset,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
        args.learning_rate,
        args.bias,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.warmup_steps,
        args.max_steps,
        output_dir,
        epochs=args.epochs,
        class_map=label_dict
    )

    logger.info(f"Saving final model to: {output_dir}")
    model.save_pretrained(output_dir)
    if hasattr(model.base_model, 'save_pretrained'):
        model.base_model.save_pretrained(os.path.join(output_dir, 'lora_adapter'))
    logger.info("Fine tuning is complete, model successfully saved.")

    evaluate_on_test_set(model, final_test_dataset, batch_size=args.batch_size, class_map=label_dict)

    return

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main()