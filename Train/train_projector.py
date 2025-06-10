'''
File: train_projector.py
Author: Elias Zheng and Claude
Description: This script trains the projector component of the BBOB model.
Usage: python train_projector.py -m <model_path> -d <dataset_path> -e <epochs> -v <vision_tower_path> -i <instruction_text>
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import argparse
import time
import logging
from datetime import datetime
from torch.utils.data import DataLoader


# utils
import sys
import os
import multiprocessing as mp

from train_common import load_and_prepare_dataset, load_model, collate, compute_embedding_similarity, compute_gradient_norm, compute_parameter_norm
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

# configure logging (if not already configured)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def projector_loss(model, vision_features, input_ids, attention_mask):
    """
    Compute language modeling loss using model's forward pass
    This is the correct LLaVA Stage 1 approach
    
    Parameters:
        - model: BBOB model with projector and language model
        - vision_features: raw vision features from encoder [batch, seq_len, hidden_dim]  
        - input_ids: tokenized text input [batch, seq_len]
        - attention_mask: attention mask for text [batch, seq_len]
        
    Returns:
        - language modeling loss from model forward pass
    """
    # Project vision features to get the number of visual tokens
    projected_vision = model.projector(vision_features)
    num_visual_tokens = projected_vision.shape[1]  # Number of visual tokens per batch
    
    # Create labels accounting for visual tokens
    # Visual tokens should be ignored in loss (set to -100)
    batch_size = input_ids.shape[0]
    device = input_ids.device
    
    # Create visual token labels (all -100 to ignore in loss)
    visual_labels = torch.full(
        (batch_size, num_visual_tokens), 
        -100, 
        dtype=input_ids.dtype, 
        device=device
    )
    
    # Create text labels (same as input_ids, with -100 for padding)
    text_labels = input_ids.clone()
    text_labels[attention_mask == 0] = -100
    
    # Concatenate visual and text labels to match the combined sequence
    labels = torch.cat([visual_labels, text_labels], dim=1)
    
    # Forward pass through the model with vision features and text
    outputs = model(
        vision_features=vision_features,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    return outputs.loss



def train(model, train, test, epochs, output_dir, learning_rate, batch_size, gradient_accumulation_steps=1):
    """
    Train the projector component with comprehensive logging and monitoring
    
    Parameters:
        - model: BBOB model with frozen base model and vision encoder
        - train: training dataset with preprocessed features
        - test: validation dataset with preprocessed features  
        - epochs: number of training epochs
        - output_dir: directory for saving checkpoints and logs
        - learning_rate: learning rate for optimizer
        - batch_size: batch size for training
        - gradient_accumulation_steps: number of steps to accumulate gradients before optimizer step
    Returns:
        - None (saves best model checkpoint during training)
    """
    # setup device and move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # verify all model components are on the same device
    logger.info(f"Model device: {device}")
    logger.info(f"Projector device: {next(model.projector.parameters()).device}")
    logger.info(f"Base model device: {next(model.base_model.parameters()).device}")
    logger.info(f"Vision encoder device: {next(model.vision_encoder.parameters()).device}")
    
    # set frozen components to eval mode once (they stay frozen throughout training)
    model.base_model.eval()
    model.vision_encoder.eval()
    
    optimizer = optim.AdamW(
        model.projector.parameters(),
        lr=learning_rate,
        weight_decay=0.0
    )

    # Scheduler setup (cosine with hard restarts and warmup)
    steps_per_epoch = len(train) // 32 if hasattr(train, '__len__') else 1000  # fallback if train is not a Dataset
    num_training_steps = epochs * steps_per_epoch
    warmup_steps = max(100, int(0.05 * num_training_steps))
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=1
    )

    # initialize mixed precision scaler
    scaler = GradScaler("cuda") if torch.cuda.is_available() else None
    logger.info(f"Mixed precision training: {'Enabled' if scaler is not None else 'Disabled'}")
    
    # create data loaders with custom collate function to handle variable-sized tensors
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate, pin_memory=True, 
                             num_workers=2, persistent_workers=True, prefetch_factor=1) 
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate, pin_memory=True,
                            num_workers=2, persistent_workers=True, prefetch_factor=1)
    
    # training tracking variables
    best_train_loss = float('inf')
    start_time = time.time()
    
    # create checkpoint directory
    checkpoint_dir = f"{output_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # log initial memory usage (baseline) 
    initial_memory = torch.cuda.memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0
    logger.info(f"Initial GPU memory usage: {initial_memory:.2f}GB")
    
    logger.info(f"Starting projector training for {epochs} epochs")
    logger.info(f"Device: {device}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(test_loader)}")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # set projector to training mode 
        model.projector.train()
        
        # training phase
        total_train_loss = 0
        total_similarity = 0
        similarity_count = 0
        accumulation_counter = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()
            
            # move data to device
            vision_features = batch["vision_features"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # compute loss using language modeling approach with mixed precision
            if scaler is not None:
                # mixed precision forward pass
                with autocast("cuda"):
                    loss = projector_loss(model, vision_features, input_ids, attention_mask)
                loss = loss / gradient_accumulation_steps
                
                with torch.no_grad():
                    text_embeddings = model.base_model.get_input_embeddings()(input_ids)
                    projected_vision = model.projector(vision_features)
                    similarity = compute_embedding_similarity(projected_vision, text_embeddings)
                similarity_count += 1
            
                # mixed precision backward pass
                scaler.scale(loss).backward()
            else:
                # standard precision training (fallback)
                loss = projector_loss(model, vision_features, input_ids, attention_mask)
                loss = loss / gradient_accumulation_steps

                with torch.no_grad():
                    text_embeddings = model.base_model.get_input_embeddings()(input_ids)
                    projected_vision = model.projector(vision_features)
                    similarity = compute_embedding_similarity(projected_vision, text_embeddings)
                similarity_count += 1

                loss.backward()
            
            # accumulate statistics (use unscaled loss for reporting)
            total_train_loss += loss.item() * gradient_accumulation_steps
            total_similarity += similarity
            accumulation_counter += 1
            logger.info(f"Accumulation counter: {accumulation_counter}")
            
            do_step = (accumulation_counter == gradient_accumulation_steps) or (batch_idx + 1 == len(train_loader))
            if do_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    grad_norm = compute_gradient_norm(model)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = compute_gradient_norm(model)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accumulation_counter = 0
                batch_time = time.time() - batch_start_time
                samples_per_sec = len(vision_features) / max(batch_time, 1e-6)
                current_lr = scheduler.get_last_lr()[0]
                logger.info(f"Epoch {epoch+1}/{epochs} Batch {batch_idx}/{len(train_loader)}: "
                           f"Loss={loss.item() * gradient_accumulation_steps:.4f}, Sim={similarity:.3f}, "
                           f"GradNorm={grad_norm:.2e}, Speed={samples_per_sec:.1f} samples/s, LR={current_lr:.2e}")
            
            # checkpoint every 512 batches using training loss
            if batch_idx > 0 and batch_idx % 512 == 0 and do_step:
                # use current training loss for checkpointing (maintains train/val separation)
                current_train_loss = loss.item() * gradient_accumulation_steps
                # save checkpoint if this is the best training loss so far
                if current_train_loss < best_train_loss:
                    best_train_loss = current_train_loss
                    best_model_dir = f"{checkpoint_dir}/best_model-batch_{batch_idx}"
                    os.makedirs(best_model_dir, exist_ok=True)
                    model.save_pretrained(best_model_dir)
                    logger.info(f"NEW BEST MODEL at batch {batch_idx}: Train Loss={current_train_loss:.4f}")
        
        # validation phase
        model.projector.eval()
        total_val_loss = 0
        total_val_similarity = 0
        with torch.no_grad():
            for batch in test_loader:
                vision_features = batch["vision_features"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                # compute validation loss using language modeling with mixed precision
                if scaler is not None:
                    with autocast("cuda"):
                        loss = projector_loss(model, vision_features, input_ids, attention_mask)
                        # compute similarity for monitoring
                        text_embeddings = model.base_model.get_input_embeddings()(input_ids)
                        projected_vision = model.projector(vision_features)
                        similarity = compute_embedding_similarity(projected_vision, text_embeddings)
                else:
                    # standard precision validation (fallback)
                    loss = projector_loss(model, vision_features, input_ids, attention_mask)
                    # compute similarity for monitoring
                    text_embeddings = model.base_model.get_input_embeddings()(input_ids)
                    projected_vision = model.projector(vision_features)
                    similarity = compute_embedding_similarity(projected_vision, text_embeddings)
                total_val_loss += loss.item()
                total_val_similarity += similarity
            # clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # calculate epoch statistics
        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        avg_val_loss = total_val_loss / max(len(test_loader), 1)
        avg_train_similarity = total_similarity / max(similarity_count, 1)  # only divide by batches where similarity was computed
        avg_val_similarity = total_val_similarity / max(len(test_loader), 1)
        # compute model statistics
        param_norm = compute_parameter_norm(model)
        epoch_time = time.time() - epoch_start_time
        # compute convergence indicators
        train_val_gap = avg_train_loss - avg_val_loss
        improvement = best_train_loss - avg_train_loss if best_train_loss != float('inf') else 0
        # log comprehensive epoch statistics
        logger.info(f"=== EPOCH {epoch+1}/{epochs} SUMMARY ===")
        logger.info(f"Losses: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}, Gap={train_val_gap:.4f}")
        logger.info(f"Similarities: Train={avg_train_similarity:.3f}, Val={avg_val_similarity:.3f}")
        logger.info(f"Model: ParamNorm={param_norm:.3f}, LR={optimizer.param_groups[0]['lr']:.2e}")
        logger.info(f"Performance: EpochTime={epoch_time:.1f}s, Improvement={improvement:.4f}")
        # save epoch checkpoint (always save latest, validation is for monitoring only)
        current_dir = f"{checkpoint_dir}/latest-{epoch+1}"
        os.makedirs(current_dir, exist_ok=True)
        model.save_pretrained(current_dir)
        logger.info(f"Epoch {epoch+1} checkpoint saved. Current Val Loss: {avg_val_loss:.4f}, Best Train Loss: {best_train_loss:.4f}")
    # final training summary
    total_time = time.time() - start_time
    logger.info(f"=== TRAINING COMPLETE ===")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"Best training loss: {best_train_loss:.4f}")
    logger.info(f"Final train/val gap: {train_val_gap:.4f}")

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
    parser.add_argument("-v", "--vision_tower", required=True, help="Vision tower model location/path")
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

    logger.info(f"Loading model from: {args.model}")
    logger.info(f"Loading dataset from: {args.dataset}")
    logger.info(f"Training for max {args.epochs} epochs")
    logger.info(f"Vision tower: {args.vision_tower}")
    logger.info(f"Instruction: {args.instruction}")

    # load model
    model = load_model(args.model, args.vision_tower, args.bnb_config)
    logger.info(f"Model size: {model.get_model_size()}")
    train_dataset, test_dataset = load_and_prepare_dataset(args.dataset, model.get_tokenizer(), model.image_processor, model.vision_encoder, args.instruction, None)
    
    # freeze model, vision tower, train only the projector 
    model.freeze_model()
    model.freeze_vision_tower() 
    model.unfreeze_projector()

    train(model, train_dataset, test_dataset, args.epochs, output_dir, args.lr, args.batch_size, args.gradient_accumulation_steps)

    # save final model in main output directory
    logger.info(f"Saving final model to: {output_dir}")
    model.save_pretrained(output_dir)

    logger.info("Projector training is complete, model successfully saved.")
    return

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    main()