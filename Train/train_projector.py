'''
File: train_projector.py
Author: Elias Zheng and Claude
Description: This script trains the projector component of the BBOB model.
Usage: python train_projector.py -m <model_path> -d <dataset_path> -e <epochs> -v <vision_tower_path> -i <instruction_text>
'''

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time
from datetime import datetime
from torch.utils.data import DataLoader
import torch._logging as logging

# utils
from model.model import BBOB
from train_common import load_and_prepare_dataset, load_model

# configure torch logging
logging.set_logs(all=logging.INFO)
logger = torch._logging.getArtifactLogger(__name__, "training_stats")

def projector_loss(vision_features, text_embeddings, temperature=0.07):
    """
    Compute contrastive loss between vision and text embeddings
    
    Parameters:
        - vision_features: projected vision features tensor
        - text_embeddings: text embedding vectors
        - temperature: scaling parameter for contrastive learning
        
    Returns:
        - cross entropy loss for vision-text alignment
    """
    vision_norm = nn.functional.normalize(vision_features, dim=-1)
    text_norm = nn.functional.normalize(text_embeddings, dim=-1)
    
    logits = torch.matmul(vision_norm, text_norm.T) / temperature
    labels = torch.arange(len(logits)).to(logits.device)
    
    return nn.functional.cross_entropy(logits, labels)

def compute_embedding_similarity(vision_features, text_embeddings):
    """
    Compute average cosine similarity between vision and text embeddings
    
    Parameters:
        - vision_features: projected vision features tensor
        - text_embeddings: text embedding vectors
        
    Returns:
        - average cosine similarity score as float
    """
    vision_norm = nn.functional.normalize(vision_features, dim=-1)
    text_norm = nn.functional.normalize(text_embeddings, dim=-1)
    
    # compute pairwise similarities for diagonal (matching pairs)
    similarities = torch.sum(vision_norm * text_norm, dim=-1)
    return similarities.mean().item()

def compute_gradient_norm(model):
    """
    Compute gradient norm for monitoring gradient health
    
    Parameters:
        - model: BBOB model with projector parameters
        
    Returns:
        - L2 norm of gradients across all projector parameters
    """
    total_norm = 0
    param_count = 0
    for p in model.projector.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    return (total_norm ** 0.5) if param_count > 0 else 0.0

def compute_parameter_norm(model):
    """
    Compute parameter norm to track weight magnitudes
    
    Parameters:
        - model: BBOB model with projector parameters
        
    Returns:
        - L2 norm of all projector parameters
    """
    total_norm = 0
    for p in model.projector.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train(model, train, test, epochs, output_dir):
    """
    Train the projector component with comprehensive logging and monitoring
    
    Parameters:
        - model: BBOB model with frozen base model and vision encoder
        - train: training dataset with preprocessed features
        - test: validation dataset with preprocessed features  
        - epochs: number of training epochs
        - output_dir: directory for saving checkpoints and logs
        
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
    
    optimizer = torch.optim.AdamW(
        model.projector.parameters(),
        lr=1e-4,       
        weight_decay=0.01  
    )
    
    # create data loaders with shuffling
    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    test_loader = DataLoader(test, batch_size=32, shuffle=False)
    
    # training tracking variables
    best_val_loss = float('inf')
    start_time = time.time()
    
    # create checkpoint directory
    checkpoint_dir = f"{output_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"Starting projector training for {epochs} epochs")
    logger.info(f"Device: {device}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(test_loader)}")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # training phase
        model.projector.train()
        total_train_loss = 0
        total_similarity = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()
            
            # move data to device
            vision_features = batch["vision_features"].to(device)
            input_ids = batch["input_ids"].to(device)
            
            # get text embeddings and project vision features
            with torch.cuda.device(device) if torch.cuda.is_available() else torch.no_grad():
                text_embeddings = model.base_model.get_input_embeddings()(input_ids)
            projected_vision = model.projector(vision_features)
            
            assert projected_vision.device == text_embeddings.device == device, f"Device mismatch: projected_vision={projected_vision.device}, text_embeddings={text_embeddings.device}, expected={device}"
            
            # compute loss and similarity
            optimizer.zero_grad()
            loss = projector_loss(projected_vision, text_embeddings)
            similarity = compute_embedding_similarity(projected_vision, text_embeddings)
            
            loss.backward()
            
            # compute gradient norm before clipping
            grad_norm = compute_gradient_norm(model)
            
            optimizer.step()
            
            # accumulate statistics
            total_train_loss += loss.item()
            total_similarity += similarity
            
            # log batch-level statistics every 10 batches
            if batch_idx % 10 == 0:
                batch_time = time.time() - batch_start_time
                samples_per_sec = len(vision_features) / max(batch_time, 1e-6)
                memory_used = torch.cuda.memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0
                
                logger.info(f"Epoch {epoch+1}/{epochs} Batch {batch_idx}/{len(train_loader)}: "
                           f"Loss={loss.item():.4f}, Sim={similarity:.3f}, "
                           f"GradNorm={grad_norm:.3f}, Speed={samples_per_sec:.1f} samples/s, "
                           f"Memory={memory_used:.2f}GB")
        
        # validation phase
        model.projector.eval()
        total_val_loss = 0
        total_val_similarity = 0
        
        with torch.no_grad():
            for batch in test_loader:
                vision_features = batch["vision_features"].to(device)
                input_ids = batch["input_ids"].to(device)
                
                text_embeddings = model.base_model.get_input_embeddings()(input_ids)
                projected_vision = model.projector(vision_features)
                
                loss = projector_loss(projected_vision, text_embeddings)
                similarity = compute_embedding_similarity(projected_vision, text_embeddings)
                
                total_val_loss += loss.item()
                total_val_similarity += similarity
        
        # calculate epoch statistics
        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        avg_val_loss = total_val_loss / max(len(test_loader), 1)
        avg_train_similarity = total_similarity / max(len(train_loader), 1)
        avg_val_similarity = total_val_similarity / max(len(test_loader), 1)
        
        # compute model statistics
        param_norm = compute_parameter_norm(model)
        epoch_time = time.time() - epoch_start_time
        
        # compute convergence indicators
        train_val_gap = avg_train_loss - avg_val_loss
        improvement = best_val_loss - avg_val_loss if best_val_loss != float('inf') else 0
        
        # log comprehensive epoch statistics
        logger.info(f"=== EPOCH {epoch+1}/{epochs} SUMMARY ===")
        logger.info(f"Losses: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}, Gap={train_val_gap:.4f}")
        logger.info(f"Similarities: Train={avg_train_similarity:.3f}, Val={avg_val_similarity:.3f}")
        logger.info(f"Model: ParamNorm={param_norm:.3f}, LR={optimizer.param_groups[0]['lr']:.2e}")
        logger.info(f"Performance: EpochTime={epoch_time:.1f}s, Improvement={improvement:.4f}")
        
        # save best model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_dir = f"{checkpoint_dir}/best_model"
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
            logger.info(f"NEW BEST MODEL: Val Loss improved to {avg_val_loss:.4f}")
        else:
            logger.info(f"No improvement. Best Val Loss: {best_val_loss:.4f}")
    
    # final training summary
    total_time = time.time() - start_time
    logger.info(f"=== TRAINING COMPLETE ===")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
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
    parser.add_argument("-e", "--epochs", type=int, default=5, help="Maximum number of training epochs (default: 5)")
    parser.add_argument("-v", "--vision_tower", required=True, help="Vision tower model location/path")
    parser.add_argument("-i", "--instruction", required=True, help="Instruction text to add to dataset examples")
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model}")
    print(f"Loading dataset from: {args.dataset}")
    print(f"Training for max {args.epochs} epochs")
    print(f"Vision tower: {args.vision_tower}")
    print(f"Instruction: {args.instruction}")

    # create output directory
    current_date = datetime.now().strftime("%Y-%m-%d")
    output_dir = f"Output/{current_date}"
    os.makedirs(output_dir, exist_ok=True)

    # load model
    model = load_model(args.model, args.vision_tower, None)
    train, test = load_and_prepare_dataset(args.dataset, model.tokenizer, model.image_processor, model.vision_encoder, args.instruction, None)
    
    # freeze model, vision tower, train only the projector
    model.freeze_model()
    model.freeze_vision_tower()
    model.unfreeze_projector()

    train(model, train, test, args.epochs, output_dir)

    # save final model in main output directory
    print(f"Saving final model to: {output_dir}")
    model.save_pretrained(output_dir)

    print("Projector training is complete, model successfully saved.")
    return

if __name__ == "__main__":
    main()