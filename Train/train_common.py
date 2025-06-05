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

def dynamic_image_resize(image, target_size=None, max_size=512, min_size=224):
    """
    Dynamically resize images with robust format handling to prevent channel dimension errors
    
    Parameters:
        - image: PIL Image, numpy array, or torch tensor to resize
        - target_size: tuple (width, height) for specific size, or None for auto-sizing
        - max_size: maximum dimension size for auto-sizing
        - min_size: minimum dimension size for auto-sizing
        
    Returns:
        - PIL Image in RGB format, ready for processing
    """
    # convert to PIL Image if needed
    if hasattr(image, 'numpy'):  # torch tensor
        image = Image.fromarray((image.numpy() * 255).astype(np.uint8))
    elif isinstance(image, np.ndarray):  # numpy array
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # ensure RGB format (fixes channel dimension issues)
    if image.mode != 'RGB':
        if image.mode == 'RGBA':
            # handle transparency by adding white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])  # use alpha channel as mask
            image = background
        elif image.mode in ['L', 'P', '1']:  # grayscale or palette modes
            image = image.convert('RGB')
        else:
            image = image.convert('RGB')
    
    if target_size is None:
        # auto-size based on aspect ratio
        width, height = image.size
        aspect_ratio = width / height
        
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(max_size / aspect_ratio)
            else:
                new_height = max_size
                new_width = int(max_size * aspect_ratio)
        elif min(width, height) < min_size:
            if width < height:
                new_width = min_size
                new_height = int(min_size / aspect_ratio)
            else:
                new_height = min_size
                new_width = int(min_size * aspect_ratio)
        else:
            # image is within acceptable range
            return image
            
        target_size = (new_width, new_height)
    
    # resize with high-quality resampling
    try:
        resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
    except AttributeError:
        # fallback for older PIL versions
        resized_image = image.resize(target_size, Image.ANTIALIAS)
    
    return resized_image

def preprocess_batch(batch, tokenizer, image_processor, vision_encoder, category_mapping=None, gpu_batch_size=64):
    """
    Process a batch of multimodal data through vision encoder and tokenizer
    
    Parameters:
        - batch: dictionary containing image, text and other fields
        - tokenizer: text tokenizer for processing text inputs
        - image_processor: vision processor for handling images
        - vision_encoder: vision model for extracting image features
        - category_mapping: optional mapping from category IDs to names
        
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
        for img in images:
            try:
                # convert to RGB if needed (prevent format errors)
                if hasattr(img, 'mode') and img.mode != 'RGB':
                    if img.mode == 'RGBA':
                        # remove alpha channel
                        img = Image.new('RGB', img.size, (255, 255, 255))
                        img.paste(img, mask=img.split()[-1])
                    else:
                        img = img.convert('RGB')
                
                processed_images.append(img)
            except Exception as e:
                warnings.warn(f"Format conversion failed: {e}. Using fallback.")
                # fallback conversion
                try:
                    if hasattr(img, 'convert'):
                        processed_images.append(img.convert('RGB'))
                    else:
                        processed_images.append(img)
                except:
                    # create blank image if all else fails
                    processed_images.append(Image.new('RGB', (256, 256), (0, 0, 0)))
    except Exception as e:
        warnings.warn(f"Image processing failed: {e}. Using original images.")
        processed_images = images
    
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
        "attention_mask": tokenized_text["attention_mask"].cpu()
    }
    
    # process any additional fields in batch (generic approach)
    for key, value in batch.items():
        if key not in [image_field, "text"]:  # skip already processed fields
            if key == "objects" and value is not None:
                # extract and vectorize bounding boxes and labels
                batch_bboxes = []
                batch_labels = []
                
                for sample_objects in value:
                    sample_bboxes = []
                    sample_labels = []
                    
                    if sample_objects:  # check if objects exist for this sample
                        for obj in sample_objects:
                            if obj and isinstance(obj, dict):
                                # extract bbox coordinates
                                if 'bbox' in obj:
                                    bbox = obj['bbox']
                                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                                        sample_bboxes.append(bbox)
                                        
                                        # extract category label
                                        if 'category' in obj:
                                            category = obj['category']
                                            if isinstance(category, list):
                                                sample_labels.append(category[0])  # take first category if multiple
                                            else:
                                                sample_labels.append(category)
                                        else:
                                            sample_labels.append(0)  # default category
                    
                    # convert to tensors, handle empty cases
                    if sample_bboxes:
                        batch_bboxes.append(torch.tensor(sample_bboxes, dtype=vision_features.dtype))
                        batch_labels.append(torch.tensor(sample_labels, dtype=tokenized_text["input_ids"].dtype))
                    else:
                        # empty sample - add dummy bbox and label
                        batch_bboxes.append(torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=vision_features.dtype))
                        batch_labels.append(torch.tensor([0], dtype=tokenized_text["input_ids"].dtype))
                
                # store vectorized bboxes and labels
                result["target_boxes"] = batch_bboxes  # list of tensors [num_objects_per_sample, 4]
                result["target_labels"] = batch_labels  # list of tensors [num_objects_per_sample]
                
                # also keep original objects for debugging/reference
                result["objects_original"] = value
            else:
                # pass through other fields unchanged
                result[key] = value
    
    return result

def preprocess_dataset(dataset, tokenizer, image_processor, vision_encoder, instruction, category_mapping=None):
    """
    Process entire dataset through image resizing and feature extraction
    
    Parameters:
        - dataset: HuggingFace dataset containing images and text
        - tokenizer: text tokenizer for processing text inputs
        - image_processor: vision processor for handling images  
        - vision_encoder: vision model for extracting image features
        - instruction: instruction text to add to each example
        - category_mapping: optional mapping from category IDs to names
        
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
    
    _preprocessing_function = partial(preprocess_batch, tokenizer=tokenizer, image_processor=image_processor, vision_encoder=vision_encoder, category_mapping=category_mapping, gpu_batch_size=gpu_batch_size)
    
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
    train = preprocess_dataset(train, tokenizer, image_processor, vision_encoder, instruction, category_mapping)
    print("Preprocessing test dataset...")
    test = preprocess_dataset(test, tokenizer, image_processor, vision_encoder, instruction, category_mapping)

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
