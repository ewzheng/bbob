'''
File: train_common.py
Author: Elias Zheng and Claude
Description: This script contains common training functions
'''

import torch
import torch.nn as nn
import datasets
import transformers

# utils
from functools import partial
import os
import time

# import model components
import model.model as model

def preprocess_batch(batch, tokenizer, image_processor, vision_encoder, category_mapping=None):
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
    images = batch["image"]
    text = batch["text"]
    
    # process images through vision encoder
    with torch.no_grad():
        processed_images = image_processor(images, return_tensors="pt")
        pixel_values = processed_images["pixel_values"]
        
        # ensure vision encoder and pixel values are on same device
        device = next(vision_encoder.parameters()).device
        pixel_values = pixel_values.to(device)
        
        vision_outputs = vision_encoder(pixel_values)
        vision_features = vision_outputs.last_hidden_state
        
    # process text through tokenizer
    tokenized_text = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True)
    
    result = {
        "vision_features": vision_features.cpu(),
        "input_ids": tokenized_text["input_ids"].cpu(),
        "attention_mask": tokenized_text["attention_mask"].cpu()
    }
    
    # process any additional fields in batch (generic approach)
    for key, value in batch.items():
        if key not in ["image", "text"]:  # skip already processed fields
            if key == "objects" and value is not None:
                # special handling for objects field (common in object detection datasets)
                processed_objects = []
                for obj in value:
                    if obj and isinstance(obj, dict):
                        processed_obj = dict(obj)  # copy the object dict
                        
                        # add category names if mapping provided and categories exist
                        if category_mapping and 'category' in obj:
                            categories = obj['category']
                            if isinstance(categories, list):
                                category_names = [category_mapping.get(cat, f"category_{cat}") for cat in categories]
                            else:
                                category_names = category_mapping.get(categories, f"category_{categories}")
                            processed_obj["category_names"] = category_names
                            
                    else:
                        processed_obj = obj  # keep as-is if not a dict or is None
                        
                    processed_objects.append(processed_obj)
                result[key] = processed_objects
            else:
                # pass through other fields unchanged
                result[key] = value
    
    return result

def preprocess_dataset(dataset, tokenizer, image_processor, vision_encoder, instruction, category_mapping=None):
    """
    Preprocess an entire dataset split with instruction mapping and feature extraction
    
    Parameters:
        - dataset: HuggingFace dataset split to preprocess
        - tokenizer: text tokenizer for processing text inputs
        - image_processor: vision processor for handling images
        - vision_encoder: vision model for extracting image features
        - instruction: instruction text to add to each example
        - category_mapping: optional mapping from category IDs to names
        
    Returns:
        - preprocessed dataset with extracted features and shuffled order
    """
    print("Preprocessing " + str(len(dataset)) + " entries in split...")

    # map instruction to dataset using lambda with update
    dataset = dataset.map(lambda x: x.update({"text": instruction}) or x)
    # preprocess split
    _preprocessing_function = partial(preprocess_batch, tokenizer=tokenizer, image_processor=image_processor, vision_encoder=vision_encoder, category_mapping=category_mapping)
    dataset = dataset.map(_preprocessing_function, batched=True, remove_columns=dataset.column_names)

    # shuffle dataset
    dataset = dataset.shuffle(seed=time.time_ns())

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
    dataset = datasets.load_dataset(dataset_name)

    # handle different dataset structures
    if hasattr(dataset, 'train'):
        # dataset has predefined splits, use train split
        base_dataset = dataset['train']
    elif hasattr(dataset, '__getitem__'):
        # dataset is directly indexable
        base_dataset = dataset
    else:
        raise ValueError(f"Unable to access dataset {dataset_name}")

    try:
        dataset_length = len(base_dataset)
        print(f"Loading {dataset_length} entries in dataset...")
    except:
        print("Loading dataset (length unknown)...")

    split = base_dataset.train_test_split(test_size=0.2)
    train = split["train"]
    test = split["test"]

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
    # load bbob
    return model.BBOB(src, vision_tower, bnb_config)
