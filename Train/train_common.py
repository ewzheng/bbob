'''
File: train_common.py
Author: Elias Zheng and Claude
Description: This script contains common training functions
'''

import torch
import torch.nn as nn
import datasets
import aiohttp
import transformers
from torch.nn.utils.rnn import pad_sequence

# utils
from functools import partial
import os
import time
import multiprocessing
import math

# Image processing imports for dynamic resizer
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms
import warnings

# import model components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import Model.model as model

import logging
import yaml

def jitter_bboxes(bboxes, img_width, img_height, jitter_ratio=0.05):
    """
    Randomly jitter bounding boxes by a fraction of their size (COCO format).
    Args:
        bboxes: Tensor or list of [N, 4] boxes (x, y, w, h) (COCO format)
        img_width: Width of the image
        img_height: Height of the image
        jitter_ratio: Max fraction of box size to jitter (default 5%)
    Returns:
        Jittered bboxes (same shape/type as input, COCO format)
    """
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.clone().detach().cpu().numpy()
    bboxes_jittered = []
    
    for box in bboxes:
        x, y, w, h = box  # COCO format
        # Convert to corners
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        # Jitter center and size
        jitter_cx = cx + np.random.uniform(-jitter_ratio, jitter_ratio) * w
        jitter_cy = cy + np.random.uniform(-jitter_ratio, jitter_ratio) * h
        jitter_w = w * (1 + np.random.uniform(-jitter_ratio, jitter_ratio))
        jitter_h = h * (1 + np.random.uniform(-jitter_ratio, jitter_ratio))
        # Clamp to image bounds
        new_x1 = np.clip(jitter_cx - jitter_w / 2, 0, img_width - 1)
        new_y1 = np.clip(jitter_cy - jitter_h / 2, 0, img_height - 1)
        new_x2 = np.clip(jitter_cx + jitter_w / 2, 0, img_width - 1)
        new_y2 = np.clip(jitter_cy + jitter_h / 2, 0, img_height - 1)
        # Convert back to COCO format
        new_x = new_x1
        new_y = new_y1
        new_w = new_x2 - new_x1
        new_h = new_y2 - new_y1
        bboxes_jittered.append([new_x, new_y, new_w, new_h])

    return torch.tensor(bboxes_jittered, dtype=torch.float32)

def normalize_coco_bboxes(bboxes, img_width, img_height):
    """
    Normalize COCO bounding boxes ([x, y, w, h]) to [x/img_w, y/img_h, w/img_w, h/img_h].
    Args:
        bboxes: Tensor or list of [N, 4] boxes (x, y, w, h)
        img_width: Width of the image
        img_height: Height of the image
    Returns:
        Normalized bboxes (same shape/type as input)
    """
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.clone().detach().cpu().numpy()
    bboxes_norm = []
    for box in bboxes:
        x, y, w, h = box
        bboxes_norm.append([
            x / img_width,
            y / img_height,
            w / img_width,
            h / img_height
        ])
    return torch.tensor(bboxes_norm, dtype=torch.float32)

def letterbox_image(image, target_size=(256, 256)):
    """Resize image with unchanged aspect ratio using padding."""
    iw, ih = image.size
    w, h = target_size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', target_size, (128, 128, 128))
    pad_w = (w - nw) // 2
    pad_h = (h - nh) // 2
    new_image.paste(image, (pad_w, pad_h))
    return new_image, scale, pad_w, pad_h

def adjust_boxes_for_letterbox(boxes, scale, pad_w, pad_h, orig_w, orig_h, target_w, target_h):
    """Adjust [x, y, w, h] boxes for letterbox resize and normalize to padded image size."""
    adjusted = []
    for x, y, w, h in boxes:
        x = x * scale + pad_w
        y = y * scale + pad_h
        w = w * scale
        h = h * scale
        # normalize to new image size
        adjusted.append([
            x / target_w,
            y / target_h,
            w / target_w,
            h / target_h
        ])
    return torch.tensor(adjusted, dtype=torch.float32)

def preprocess_batch(batch, tokenizer, image_processor, vision_encoder, category_mapping=None, gpu_batch_size=64, bbox_jitter_ratio=0.05, training=False):
    """
    Process a batch of multimodal data through vision encoder and tokenizer
    
    Parameters:
        - batch: dictionary containing image, text and other fields
        - tokenizer: text tokenizer for processing text inputs
        - image_processor: vision processor for handling images
        - vision_encoder: vision model for extracting image features
        - category_mapping: optional mapping from category IDs to names
        - gpu_batch_size: batch size for GPU processing
        - bbox_jitter_ratio: ratio for jittering bounding boxes
        - training: boolean indicating whether this is a training batch
        
    Returns:
        - processed batch with vision_features, input_ids, attention_mask and preserved fields
    """

    # detect image field name dynamically
    image_field = None
    for field in ["image", "img", "images", "picture", "photo", "jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"]:
        if field in batch:
            image_field = field
            break
    
    if image_field is None:
        raise ValueError(f"No image field found in batch. Available fields: {list(batch.keys())}")
    
    images = batch[image_field]
    text = batch["text"]
    
    # only handle format conversion - let image processor handle resizing
    try:
        processed_images = []
        image_sizes = []
        letterbox_params = []
        for img in images:
            try:
                if hasattr(img, 'mode') and img.mode != 'RGB':
                    if img.mode == 'RGBA':
                        img = Image.new('RGB', img.size, (255, 255, 255))
                        img.paste(img, mask=img.split()[-1])
                    else:
                        img = img.convert('RGB')
                orig_size = img.size
                img, scale, pad_w, pad_h = letterbox_image(img, target_size=(256, 256))
                processed_images.append(img)
                image_sizes.append(orig_size)  # store original size for box adjustment
                letterbox_params.append((scale, pad_w, pad_h, orig_size[0], orig_size[1], 256, 256))
            except Exception as e:
                warnings.warn(f"Format conversion failed: {e}. Using fallback.")
                try:
                    if hasattr(img, 'convert'):
                        img_rgb = img.convert('RGB')
                        orig_size = img_rgb.size
                        img_rgb, scale, pad_w, pad_h = letterbox_image(img_rgb, target_size=(256, 256))
                        processed_images.append(img_rgb)
                        image_sizes.append(orig_size)
                        letterbox_params.append((scale, pad_w, pad_h, orig_size[0], orig_size[1], 256, 256))
                    else:
                        processed_images.append(img)
                        image_sizes.append((256, 256))
                        letterbox_params.append((1.0, 0, 0, 256, 256, 256, 256))
                except:
                    processed_images.append(Image.new('RGB', (256, 256), (0, 0, 0)))
                    image_sizes.append((256, 256))
                    letterbox_params.append((1.0, 0, 0, 256, 256, 256, 256))
    except Exception as e:
        warnings.warn(f"Image processing failed: {e}. Using original images.")
        processed_images = images
        image_sizes = [(256, 256)] * len(images)
        letterbox_params = [(1.0, 0, 0, 256, 256, 256, 256)] * len(images)
    
    # process images through vision encoder in GPU sub-batches
    device = next(vision_encoder.parameters()).device
    all_vision_features = []
    total_images = len(processed_images)
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        # process in smaller GPU sub-batches to manage VRAM
        for start_idx in range(0, total_images, gpu_batch_size):
            end_idx = min(start_idx + gpu_batch_size, total_images)
            sub_batch_images = processed_images[start_idx:end_idx]
            
            try:
                processor_output = image_processor(sub_batch_images, return_tensors="pt")
                pixel_values = processor_output["pixel_values"].to(device)
                
                vision_outputs = vision_encoder(pixel_values)
                vision_features = vision_outputs.last_hidden_state.cpu()
                
                all_vision_features.append(vision_features)
                
                # clear GPU memory after each sub-batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                warnings.warn(f"Vision encoder processing failed for sub-batch {start_idx}-{end_idx}: {e}")
                # create dummy features for failed sub-batch
                sub_batch_size = len(sub_batch_images)
                dummy_features = torch.zeros(sub_batch_size, 256, 768)
                all_vision_features.append(dummy_features)
    
    # concatenate all sub-batch results
    if all_vision_features:
        vision_features = torch.cat(all_vision_features, dim=0)
    else:
        # fallback if everything failed
        vision_features = torch.zeros(total_images, 256, 768)  
        
    # ensure tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenized_text = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True)
    
    result = {
        "vision_features": vision_features,
        "input_ids": tokenized_text["input_ids"].cpu(),
        "attention_mask": tokenized_text["attention_mask"].cpu(),
    }

    # apply augmentations to objects
    if "objects" in batch:
        result["target_boxes"] = []
        result["target_labels"] = []
        for i, sample in enumerate(batch["objects"]):
            bboxes = sample["bbox"]
            scale, pad_w, pad_h, orig_w, orig_h, target_w, target_h = letterbox_params[i]
            if training:
                bboxes = jitter_bboxes(bboxes, orig_w, orig_h, jitter_ratio=bbox_jitter_ratio)
            bboxes = adjust_boxes_for_letterbox(bboxes, scale, pad_w, pad_h, orig_w, orig_h, target_w, target_h)
            sample_boxes = []
            sample_labels = []
            for bbox, category in zip(bboxes, sample["category"]):
                sample_boxes.append(bbox)
                sample_labels.append(category)
            result["target_boxes"].append(sample_boxes)
            result["target_labels"].append(sample_labels)

    # Add target_text from COCO sentences if present
    if "sentences" in batch:
        # batch["sentences"] is a list of lists of dicts (one per sample)
        target_texts = []
        for sent_list in batch["sentences"]:
            if isinstance(sent_list, list):
                all_texts = [s["raw"] for s in sent_list if "raw" in s]
                target_text = " ".join(all_texts)
            else:
                # fallback: single dict
                target_text = sent_list.get("raw", "")
            # Tokenize
            tokenized = tokenizer(target_text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
            target_texts.append(tokenized["input_ids"].squeeze(0))
        result["target_text"] = torch.stack(target_texts)

    # Store image sizes for denormalization during evaluation
    result["image_sizes"] = image_sizes  # original sizes
    result["padded_image_sizes"] = [(256, 256)] * len(processed_images)  # always 256x256 for letterbox

    return result

def preprocess_dataset(dataset, tokenizer, image_processor, vision_encoder, instruction, category_mapping=None, is_training=False):
    """
    Process entire dataset through image resizing and feature extraction
    
    Parameters:
        - dataset: HuggingFace dataset containing images and text
        - tokenizer: text tokenizer for processing text inputs
        - image_processor: vision processor for handling images  
        - vision_encoder: vision model for extracting image features
        - instruction: instruction text to add to each example
        - category_mapping: optional mapping from category IDs to names
        - is_training: boolean indicating whether this is a training set
        
    Returns:
        - processed dataset with vision_features and tokenized text
    """
    
    # add instruction to each sample to create "text" field
    dataset = dataset.map(lambda x: x.update({"text": instruction}) or x)
    
    max_workers = min(multiprocessing.cpu_count() - 1, 4)
    print(f"Using CPU multiprocessing with {max_workers} workers...")
    
    # separate CPU and GPU batch sizes for memory management
    if torch.cuda.is_available():
        gpu_batch_size = calculate_optimal_batch_size(vision_encoder, tokenizer, safety_margin=0.15)
        cpu_batch_size = 64  # conservative CPU batch for RAM safety
        print(f"Using GPU batch size: {gpu_batch_size}, CPU batch size: {cpu_batch_size}")
    else:
        gpu_batch_size = 64
        cpu_batch_size = 64
        print(f"CPU mode - using batch size: {cpu_batch_size}")
    
    _preprocessing_function = partial(
        preprocess_batch,
        tokenizer=tokenizer,
        image_processor=image_processor,
        vision_encoder=vision_encoder,
        category_mapping=category_mapping,
        gpu_batch_size=gpu_batch_size,
        training=is_training
    )

    dataset = dataset.map(
        _preprocessing_function, 
        batched=True, 
        batch_size=cpu_batch_size,  # smaller CPU batch for RAM safety
        remove_columns=dataset.column_names,
        num_proc=max_workers,
        desc=f"Processing images and text ({max_workers} workers, CPU batch={cpu_batch_size}, GPU batch={gpu_batch_size})",
        load_from_cache_file=True
    )

    return dataset

def load_and_prepare_dataset(dataset_name, tokenizer, image_processor, vision_encoder, instruction, category_mapping=None):
    """
    Load dataset from HuggingFace hub and create train/test splits
    
    Parameters:
        - dataset_name: HuggingFace dataset identifier
        - tokenizer: text tokenizer for processing text inputs
        - image_processor: vision processor for handling images
        - vision_encoder: vision model for extracting image features
        - instruction: instruction text to add to each example
        - category_mapping: optional mapping from category IDs to names
        
    Returns:
        - train, test datasets with extracted features
    """
    print(f"Loading dataset {dataset_name}")
    
    dataset = datasets.load_dataset(
        dataset_name, 
        trust_remote_code=True,
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=10000)}}
    )



    # handle different dataset structures automatically
    if isinstance(dataset, datasets.DatasetDict):
        # check if dataset already has both train and validation/val splits
        if 'train' in dataset and 'validation' in dataset:
            # use existing pre-split datasets (like COCO8, COCO8-pose)
            train = dataset['train']
            test = dataset['validation']
            print(f"Using existing splits: {len(train)} train, {len(test)} validation")
        elif 'train' in dataset and 'val' in dataset:
            # some datasets use 'val' instead of 'validation'
            train = dataset['train'] 
            test = dataset['val']
            print(f"Using existing splits: {len(train)} train, {len(test)} val")
        elif 'train' in dataset:
            # only has train split, need to split it manually
            base_dataset = dataset['train']
            try:
                dataset_length = len(base_dataset)
                print(f"Splitting train set of {dataset_length} entries...")
            except:
                print("Splitting train set (length unknown)...")
            split = base_dataset.train_test_split(test_size=0.2, seed=42)
            train = split["train"]
            test = split["test"]
            print(f"Created splits: {len(train)} train, {len(test)} test")
        else:
            raise ValueError(f"Dataset {dataset_name} has unknown split structure: {list(dataset.keys())}")
    else:
        # single dataset without predefined splits, need to split manually
        try:
            dataset_length = len(dataset)
            print(f"Splitting dataset of {dataset_length} entries...")
        except:
            print("Splitting dataset (length unknown)...")
        split = dataset.train_test_split(test_size=0.2, seed=42)
        train = split["train"]
        test = split["test"]
        print(f"Created splits: {len(train)} train, {len(test)} test")


    print("Preprocessing train dataset...")
    train = preprocess_dataset(train, tokenizer, image_processor, vision_encoder, instruction, category_mapping, is_training=True)
    print("Preprocessing test dataset...")
    test = preprocess_dataset(test, tokenizer, image_processor, vision_encoder, instruction, category_mapping, is_training=False)

    return train, test

def load_model(src, vision_tower, bnb_config):
    """
    Load BBOB model with specified base model and vision encoder
    
    Parameters:
        - src: path or identifier for base language model
        - vision_tower: path or identifier for vision encoder
        - bnb_config: quantization configuration for model loading
        
    Returns:
        - initialized BBOB model instance
    """

    # clear GPU memory before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model.BBOB(src, vision_tower, bnb_config)

def calculate_optimal_batch_size(vision_encoder, tokenizer, safety_margin=0.15, min_batch_size=32, max_batch_size=8192):
    """
    Automatically calculate optimal batch size based on available VRAM
    
    Parameters:
        - vision_encoder: vision model for memory estimation
        - tokenizer: text tokenizer for memory estimation  
        - safety_margin: fraction of VRAM to leave as headroom (default: 15%)
        - min_batch_size: minimum allowed batch size
        - max_batch_size: maximum allowed batch size
        
    Returns:
        - optimal_batch_size: calculated batch size with safety margin
    """
    if not torch.cuda.is_available():
        print("No CUDA available, using default batch size: 64")
        return 64
    
    # get VRAM information
    device_props = torch.cuda.get_device_properties(0)
    total_vram = device_props.total_memory
    allocated_vram = torch.cuda.memory_allocated(0)
    available_vram = total_vram - allocated_vram
    
    print(f"VRAM Analysis:")
    print(f"Total VRAM: {total_vram/1024**3:.1f}GB")
    print(f"Currently allocated: {allocated_vram/1024**3:.1f}GB")
    print(f"Available: {available_vram/1024**3:.1f}GB")
    
    # estimate memory per sample
    try:
        # get vision encoder output dimensions
        device = next(vision_encoder.parameters()).device
        with torch.no_grad():
            dummy_image = torch.randn(1, 3, 256, 256).to(device)
            vision_output = vision_encoder(dummy_image)
            vision_features = vision_output.last_hidden_state
            
            if vision_features.dim() == 4:  # [batch, height, width, channels]
                # flatten spatial dimensions like the projector does
                vision_features = vision_features.flatten(2).transpose(1, 2)
                
            vision_memory_per_sample = vision_features.numel() * 4  # float32 = 4 bytes
            
        # estimate text memory (rough approximation)
        max_text_length = 1024  # from tokenizer max_length
        text_memory_per_sample = max_text_length * 4  # approximate
        
        # total memory per sample (including overhead for processing, intermediate tensors, fragmentation)
        memory_per_sample = (vision_memory_per_sample + text_memory_per_sample) * 5  # 5x for realistic GPU overhead
        
        print(f"Estimated memory per sample: {memory_per_sample/1024**2:.1f}MB")
        
    except Exception as e:
        print(f"Could not estimate memory usage: {e}")
        # fallback estimate: ~2MB per sample
        memory_per_sample = 2 * 1024 * 1024
        print(f"Using fallback estimate: {memory_per_sample/1024**2:.1f}MB per sample")
    
    # calculate usable VRAM (leaving safety margin)
    usable_vram = available_vram * (1 - safety_margin)
    
    optimal_batch_size = int(usable_vram / memory_per_sample) 
    
    # apply limits
    optimal_batch_size = max(min_batch_size, min(optimal_batch_size, max_batch_size))
    
    # round down to nearest power of 2 for optimal GPU efficiency
    if optimal_batch_size >= 2:
        power = int(math.log2(optimal_batch_size))
        optimal_batch_size = 2 ** power
    
    estimated_usage = (optimal_batch_size * memory_per_sample) / 1024**3
    vram_percentage = (estimated_usage / (total_vram / 1024**3)) * 100
    
    print(f"Calculated optimal batch size: {optimal_batch_size}")
    print(f"Estimated VRAM usage: {estimated_usage:.1f}GB ({vram_percentage:.1f}%)")
    print(f"Safety margin: {safety_margin*100:.0f}% ({(available_vram * safety_margin)/1024**3:.1f}GB)")
    
    return optimal_batch_size

def collate(batch):
    """
    Custom collate function to handle variable-sized vision features and text embeddings
    
    Parameters:
        - batch: list of dictionaries with keys ['vision_features', 'input_ids', 'attention_mask']
        
    Returns:
        - collated batch with padded tensors
    """
    # separate the different fields
    vision_features = [item['vision_features'] for item in batch]
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    
    # handle additional fields like target_boxes and target_labels
    additional_fields = {}
    for key in batch[0].keys():
        if key not in ['vision_features', 'input_ids', 'attention_mask']:
            additional_fields[key] = [item[key] for item in batch]
    
    # Pad and stack target_labels if present
    if "target_labels" in additional_fields:
        label_tensors = []
        for lbl in additional_fields["target_labels"]:
            if isinstance(lbl, list):
                lbl = torch.tensor(lbl)
            if not isinstance(lbl, torch.Tensor):
                lbl = torch.tensor(lbl)
            label_tensors.append(lbl)
        # Pad to the same length with ignore index -100
        target_labels_padded = pad_sequence(label_tensors, batch_first=True, padding_value=-100)
        # Clamp to valid range [0, 79] or set to -100 if out of range
        num_classes = 80  # COCO
        mask = (target_labels_padded != -100)
        target_labels_padded[mask & (target_labels_padded < 0)] = -100
        target_labels_padded[mask & (target_labels_padded >= num_classes)] = -100
        additional_fields["target_labels"] = target_labels_padded
    # Pad and stack target_boxes if present
    if "target_boxes" in additional_fields:
        box_tensors = []
        for bx in additional_fields["target_boxes"]:
            if isinstance(bx, list):
                bx = torch.tensor(bx)
            if not isinstance(bx, torch.Tensor):
                bx = torch.tensor(bx)
            box_tensors.append(bx)
        # Pad to the same length with 0.0
        target_boxes_padded = pad_sequence(box_tensors, batch_first=True, padding_value=0.0)
        additional_fields["target_boxes"] = target_boxes_padded
    
    # convert to tensors and handle dimensions properly
    vision_tensors = []
    for vf in vision_features:
        if isinstance(vf, list):
            vf = torch.tensor(vf)
        if not isinstance(vf, torch.Tensor):
            vf = torch.tensor(vf)
        # ensure 2D: [seq_len, hidden_dim]
        if vf.dim() == 3:
            vf = vf.squeeze(0)
        elif vf.dim() == 1:
            vf = vf.unsqueeze(0)
        vision_tensors.append(vf)
    
    input_tensors = []
    for ids in input_ids:
        if isinstance(ids, list):
            ids = torch.tensor(ids)
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids)
        # ensure 1D: [seq_len]
        if ids.dim() == 2:
            ids = ids.squeeze(0)
        input_tensors.append(ids)
    
    mask_tensors = []
    for mask in attention_masks:
        if isinstance(mask, list):
            mask = torch.tensor(mask)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)
        # ensure 1D: [seq_len]
        if mask.dim() == 2:
            mask = mask.squeeze(0)
        mask_tensors.append(mask)
    
    # pad sequences to same length
    vision_features_padded = pad_sequence(vision_tensors, batch_first=True, padding_value=0.0)
    input_ids_padded = pad_sequence(input_tensors, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(mask_tensors, batch_first=True, padding_value=0)
    
    result = {
        'vision_features': vision_features_padded,
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded
    }
    
    # add additional fields
    result.update(additional_fields)
    
    return result

def compute_embedding_similarity(vision_features, text_embeddings):
    """
    Compute average cosine similarity between vision and text embeddings
    
    Parameters:
        - vision_features: projected vision features tensor [batch, seq_len, hidden_dim]
        - text_embeddings: text embedding vectors [batch, seq_len, hidden_dim]
        
    Returns:
        - average cosine similarity score as float
    """
    # pool features to get single representation per sample
    vision_pooled = vision_features.mean(dim=1)  # [batch, hidden_dim]
    text_pooled = text_embeddings.mean(dim=1)    # [batch, hidden_dim]
    
    # normalize for cosine similarity
    vision_norm = nn.functional.normalize(vision_pooled, dim=-1)
    text_norm = nn.functional.normalize(text_pooled, dim=-1)
    
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
    # More robust gradient norm computation as per PyTorch forums
    parameters = [p for p in model.projector.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        return 0.0
    
    # Use the faster concatenation method for better performance
    grads = [param.grad.detach().flatten() for param in parameters]
    if len(grads) == 0:
        return 0.0
    
    total_norm = torch.cat(grads).norm().item()
    return total_norm

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

def load_labels_from_yaml(yaml_path):
    """Load label dictionary from a YAML file and return class_name -> index mapping."""
    if not os.path.exists(yaml_path):
        logging.error(f"Label YAML file not found: {yaml_path}")
        raise FileNotFoundError(f"Label YAML file not found: {yaml_path}")
    try:
        with open(yaml_path, 'r') as f:
            labels = yaml.safe_load(f)
        if not isinstance(labels, dict) or 'names' not in labels:
            raise ValueError("YAML file must contain a 'names' dictionary of labels.")
        names = labels['names']
        # Invert mapping: class_name -> index
        class_map = {v: int(k) for k, v in names.items()}
        return class_map
    except Exception as e:
        logging.error(f"Error loading labels from YAML: {e}")
        raise
