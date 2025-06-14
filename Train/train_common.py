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
import multiprocess as mp
import math

# Image processing imports for dynamic resizer
from PIL import Image
import numpy as np
import os

import logging
import yaml
import psutil

def jitter_bboxes(bboxes, img_width, img_height, dtype, jitter_ratio=0.05):
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

    return torch.tensor(bboxes_jittered, dtype=dtype)

def normalize_coco_bboxes(bboxes, img_width, img_height, dtype):
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
    return torch.tensor(bboxes_norm, dtype=dtype)

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

def adjust_boxes_for_letterbox(boxes, scale, pad_w, pad_h, orig_w, orig_h, target_w, target_h, dtype):
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
    return torch.tensor(adjusted, dtype=dtype)

def preprocess_batch(batch, tokenizer, gpu_batch_size=64, bbox_jitter_ratio=0.05, training=False, target_size=(256, 256), dtype=torch.float32):
    """
    Convert a raw *batch* of samples into model-ready tensors.

    Steps performed per sample
        1.  Convert image → RGB and **letter-box** it to *target_size* while
            preserving aspect-ratio (padding = 128 grey).
        2.  Tokenise the instructional *text* field with padding/truncation.
        3.  (Optional) Jitter and re-scale COCO-style bounding-boxes, then
            normalise them to the padded image size.

    Parameters:
        - batch: dict with keys like ``image`` / ``text`` / ``objects`` …
        - vision_tower: VisionTower object for processing images
        - tokenizer: HuggingFace tokenizer (will auto-add pad token if missing)
        - gpu_batch_size: unused here – kept for backwards compatibility.
        - bbox_jitter_ratio: float, jitter amplitude for bboxes when *training*.
        - training: bool, enables bbox jitter.
        - target_size: tuple(int, int), final (W, H) after letter-boxing.

    Returns:
        dict containing:
            • images              – list[ PIL.Image ] (letter-boxed)
            • input_ids           – LongTensor[B, T]
            • attention_mask      – LongTensor[B, T]
            • *optional* target_boxes / target_labels / target_text …
    """

    processed_images = []      # PIL.Image (letter-boxed)
    image_sizes = []           # original (w, h) per image
    padded_image_sizes = []    # after letter-box (should all be target_size)
    lb_params = []             # (scale, pad_w, pad_h) per image

    images_field = "images" if "images" in batch else "image" if "image" in batch else None
    if images_field is None:
        raise KeyError("Batch dict must contain an 'images' or 'image' key with PIL images")

    for img in batch[images_field]:
        if not isinstance(img, Image.Image):
            raise ValueError("Images must be PIL.Image objects")

        rgb = img.convert("RGB")

        # apply letter-box resize
        lb_img, scale, pad_w, pad_h = letterbox_image(rgb, target_size=target_size)

        processed_images.append(lb_img)
        image_sizes.append(rgb.size)          # (orig_w, orig_h)
        padded_image_sizes.append(target_size)
        lb_params.append((scale, pad_w, pad_h))

    text = batch["text"]
    
    # ensure tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenized_text = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True)
    
    # Convert tensors to python lists (1-D per sample) – we slice row-wise
    input_id_rows = [row.tolist() for row in tokenized_text["input_ids"]]
    attention_mask_rows = [row.tolist() for row in tokenized_text["attention_mask"]]

    result = {
        "images": processed_images,
        "input_ids": input_id_rows,
        "attention_mask": attention_mask_rows,
    }

    # apply augmentations to objects
    if "objects" in batch:
        result["target_boxes"] = []
        result["target_labels"] = []
        for i, sample in enumerate(batch["objects"]):
            bboxes = sample["bbox"]

            # 1) optional jitter in *original* image space
            if training:
                bboxes = jitter_bboxes(
                    bboxes,
                    img_width=image_sizes[i][0],
                    img_height=image_sizes[i][1],
                    dtype=dtype,
                    jitter_ratio=bbox_jitter_ratio,
                )

            # 2) adjust to letter-boxed, padded coordinates then normalise
            scale, pad_w, pad_h = lb_params[i]
            tgt_w, tgt_h = target_size
            bboxes = adjust_boxes_for_letterbox(
                bboxes,
                scale=scale,
                pad_w=pad_w,
                pad_h=pad_h,
                orig_w=image_sizes[i][0],
                orig_h=image_sizes[i][1],
                target_w=tgt_w,
                target_h=tgt_h,
                dtype=dtype,
            )
            sample_boxes = []
            sample_labels = []
            for bbox, category in zip(bboxes, sample["category"]):
                sample_boxes.append(bbox)
                sample_labels.append(category)
            result["target_boxes"].append(sample_boxes)
            result["target_labels"].append(sample_labels)

    if "sentences" in batch:
        target_texts = []
        for sent_list in batch["sentences"]:
            if isinstance(sent_list, list):
                all_texts = [s["raw"] for s in sent_list if "raw" in s]
                target_text = " ".join(all_texts)
            else:
                target_text = sent_list.get("raw", "")

            tokenized = tokenizer(
                target_text,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding="max_length",
            )

            # store as python list for HF datasets compatibility
            target_texts.append(tokenized["input_ids"].squeeze(0).tolist())

        result["target_text"] = target_texts

    # Store image sizes for denormalization during evaluation
    result["image_sizes"] = image_sizes
    result["padded_image_sizes"] = padded_image_sizes

    return result


def preprocess_dataset(dataset, tokenizer, instruction, is_training=False, dtype=torch.float32):
    """
    Process entire dataset through image resizing and feature extraction
    
    Parameters:
        - dataset: HuggingFace dataset containing images and text
        - tokenizer: text tokenizer for processing text inputs
        - instruction: instruction text to add to each example
        - is_training: boolean indicating whether this is a training set
        
    Returns:
        - processed dataset with images and tokenized text
    """

    # add instruction to each sample to create "text" field
    dataset = dataset.map(lambda x: x.update({"text": instruction}) or x)
    
    max_workers = min(mp.cpu_count() - 1, 16)
    
    # determine optimal batch sizes ----------------------------------------------------
    gpu_batch_size, cpu_batch_size = calculate_optimal_batch_size(workers=max_workers, safety_margin=0.15)

    if torch.cuda.is_available():
        print(f"Using GPU batch size: {gpu_batch_size}, CPU batch size: {cpu_batch_size}")
    else:
        print(f"CPU mode - using batch size: {cpu_batch_size}")
    
    _preprocessing_function = partial(
        preprocess_batch,
        tokenizer=tokenizer,
        gpu_batch_size=gpu_batch_size,
        training=is_training,
        dtype=dtype,
    )

    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        batch_size=cpu_batch_size,  # smaller CPU batch for RAM safety
        remove_columns=dataset.column_names,
        num_proc=max_workers,
        desc=f"Processing images and text ({max_workers} workers, CPU batch={cpu_batch_size}, GPU batch={gpu_batch_size})",
        load_from_cache_file=False,  # force reprocessing after code changes
    )

    return dataset

def load_and_prepare_dataset(dataset_name, tokenizer, instruction, dtype=torch.float32):
    """
    Load dataset from HuggingFace hub and create train/test splits
    
    Parameters:
        - dataset_name: HuggingFace dataset identifier
        - tokenizer: text tokenizer for processing text inputs
        - instruction: instruction text to add to each example
        
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
    train = preprocess_dataset(train, tokenizer, instruction, is_training=True, dtype=dtype)
    print("Preprocessing test dataset...")
    test = preprocess_dataset(test, tokenizer, instruction, is_training=False, dtype=dtype)

    return train, test

def calculate_optimal_batch_size(
    workers = 1,
    safety_margin= 0.15,
    min_batch_size = 8,
    max_batch_size = 32768,
):
    """Compute **both** GPU and CPU batch sizes.

    GPU: uses free VRAM; CPU: uses available system RAM & logical cores.

    Returns
    -------
    (gpu_bs, cpu_bs) : tuple[int | None, int]
        • *gpu_bs* – optimal per-device GPU batch size or *None* when CUDA is
          not available.
        • *cpu_bs* – optimal CPU-side batch size for multiprocessing image
          preprocessing.
    """

    # gpu batch
    gpu_batch_size: int | None = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory
        allocated = torch.cuda.memory_allocated(0)
        available_vram = total_vram - allocated

        print("VRAM Analysis:")
        print(f"  Total VRAM:       {total_vram/1024**3:.1f} GB")
        print(f"  Currently alloc.: {allocated/1024**3:.1f} GB")
        print(f"  Available:        {available_vram/1024**3:.1f} GB")

        memory_per_sample_gpu = 4 * 1024 * 1024  # 4 MB heuristic per sample
        gpu_batch_size = int(available_vram / memory_per_sample_gpu)
        gpu_batch_size = max(min_batch_size, min(gpu_batch_size, max_batch_size*(workers//2)))

        if gpu_batch_size >= 2:
            gpu_batch_size = 2 ** int(math.log2(gpu_batch_size))

        est_usage = (gpu_batch_size * memory_per_sample_gpu) / 1024**3
        pct = (est_usage / (total_vram / 1024**3)) * 100
        print(f"  → GPU batch size:  {gpu_batch_size}  (≈{est_usage:.1f} GB, {pct:.1f}% of VRAM)")

    # cpu batch
    if psutil is not None:
        vm = psutil.virtual_memory()
        total_ram = vm.total
        available_ram = vm.available
    else:
        # fallback: assume 8 GB total with 50 % free
        total_ram = 8 * 1024**3
        available_ram = total_ram * (1 - safety_margin)

    print("RAM Analysis:")
    print(f"  Total RAM:        {total_ram/1024**3:.1f} GB")
    print(f"  Available RAM:    {available_ram/1024**3:.1f} GB")

    memory_per_sample_cpu = 2 * 1024 * 1024  
    # ram usage per worker
    cpu_bs_mem = int(available_ram * (1 - safety_margin) / memory_per_sample_cpu) * workers

    cpu_batch_size = min(cpu_bs_mem, max_batch_size)
    if cpu_batch_size >= 2:
        cpu_batch_size = 2 ** int(math.log2(cpu_batch_size))

    est_cpu_usage = (cpu_batch_size * memory_per_sample_cpu) / 1024**3
    print(f"  → CPU batch size:  {cpu_batch_size}  (≈{est_cpu_usage:.1f} GB RAM)")

    return gpu_batch_size, cpu_batch_size

def collate(batch):
    """
    Minimal collate-fn that:
        • keeps the list of **PIL images** untouched (key ``images``)
        • pads ``input_ids`` and ``attention_mask`` to the max length in the batch
        • pads / stacks optional detection targets (``target_boxes`` / ``target_labels``)
    """

    images = [item["images"] for item in batch]          # list[ PIL.Image ]
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]

    # -------- text padding --------
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    result = {
        "images": images,
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
    }

    # -------- optional detection targets --------
    additional_keys = [k for k in batch[0].keys() if k not in result]
    for key in additional_keys:
        elems = [item[key] for item in batch]

        if key == "target_boxes":
            # pad to same number of boxes per image (0.0 -> blank box)
            tensors = [torch.tensor(e) if not isinstance(e, torch.Tensor) else e for e in elems]
            result[key] = pad_sequence(tensors, batch_first=True, padding_value=0.0)
        elif key == "target_labels":
            tensors = [torch.tensor(e) if not isinstance(e, torch.Tensor) else e for e in elems]
            result[key] = pad_sequence(tensors, batch_first=True, padding_value=-100)
        elif key == "target_text":
            # already token ids – pad exactly like input_ids (value = pad_token_id==0)
            tensors = [torch.tensor(e) if not isinstance(e, torch.Tensor) else e for e in elems]
            result[key] = pad_sequence(tensors, batch_first=True, padding_value=0)
        else:
            result[key] = elems  # leave as list

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
