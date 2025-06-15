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

import torchvision.transforms.functional as TF
import torch.nn.functional as F

import logging
import yaml
import psutil

from transformers import MobileViTImageProcessor

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
    nw = max(1, int(iw * scale))
    nh = max(1, int(ih * scale))
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

def adjust_boxes_resize_crop(bboxes, orig_w, orig_h, target=256, dtype=torch.float32):
    """Resize bbox coordinates using MobileViT shortest-edge resize followed by
    center crop to `target`×`target`, then normalise to 0-1.

    Parameters
    ----------
    bboxes : Tensor | list
        Boxes in COCO format [x, y, w, h].
    orig_w, orig_h : int
        Original image dimensions.
    target : int
        Final square side after crop – 256 for MobileViT default.
    dtype : torch.dtype
        Output tensor dtype.
    """

    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.clone().detach().cpu().numpy()

    ratio = target / min(orig_w, orig_h)
    resized_w, resized_h = orig_w * ratio, orig_h * ratio

    # integer pixel dims
    resized_w_i, resized_h_i = int(round(resized_w)), int(round(resized_h))

    crop_left = max((resized_w_i - target) // 2, 0)
    crop_top  = max((resized_h_i - target) // 2, 0)

    adjusted = []
    for x, y, w, h in bboxes:
        x = x * ratio - crop_left
        y = y * ratio - crop_top
        w = w * ratio
        h = h * ratio

        # clamp to crop area
        x = max(0, min(x, target - 1))
        y = max(0, min(y, target - 1))
        w = max(0, min(w, target - x))
        h = max(0, min(h, target - y))

        adjusted.append([
            x / target,
            y / target,
            w / target,
            h / target,
        ])

    return torch.tensor(adjusted, dtype=dtype)

def preprocess_batch(batch, tokenizer, image_processor: MobileViTImageProcessor, gpu_batch_size=64, bbox_jitter_ratio=0.05, training=False, target_size=(256, 256), dtype=torch.float32):
    '''
    build vision-language features for one raw dataset batch.

    pre: batch must include keys `image`/`images` and `text`.

    parameters:
        - batch (dict): incoming dataset slice.
        - tokenizer (PreTrainedTokenizer): hf tokenizer.
        - image_processor: MobileViTImageProcessor instance for image processing
        - gpu_batch_size (int): images processed per cuda chunk.
        - bbox_jitter_ratio (float): amplitude of bbox noise.
        - training (bool): enables bbox jitter.
        - target_size (tuple[int,int]): final (w, h) resolution.
        - dtype (torch.dtype): dtype for tensors.

    returns: dict with image tensors, token ids, masks and optional targets.
    '''

    processed_images = []      # PIL.Image (letter-boxed)
    image_sizes = []           # original (w, h) per image
    padded_image_sizes = []    # after letter-box (should all be target_size)
    lb_params = []             # (scale, pad_w, pad_h) per image

    images_field = "images" if "images" in batch else "image" if "image" in batch else None
    if images_field is None:
        raise KeyError("Batch dict must contain an 'images' or 'image' key with PIL images")

    # store raw rgb images; resizing will be done in collate_fn on gpu
    for img in batch[images_field]:
        if not isinstance(img, Image.Image):
            raise ValueError("images must be PIL.Image objects")

        rgb = img.convert("RGB")

        # --- MobileViT image processor ---
        px = image_processor(
            rgb,
            return_tensors="np",
        )["pixel_values"][0]  # (3, 256, 256) float32 already normalised

        processed_images.append(px)

        image_sizes.append(rgb.size)
        padded_image_sizes.append(target_size)

        # store params for bbox transform
        orig_w, orig_h = rgb.size
        ratio = target_size[0] / min(orig_w, orig_h)
        resized_w, resized_h = int(round(orig_w * ratio)), int(round(orig_h * ratio))
        crop_left = max((resized_w - target_size[0]) // 2, 0)
        crop_top = max((resized_h - target_size[1]) // 2, 0)
        lb_params.append((ratio, crop_left, crop_top))

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
            ratio, crop_left, crop_top = lb_params[i]
            bboxes = adjust_boxes_resize_crop(
                bboxes,
                orig_w = image_sizes[i][0],
                orig_h = image_sizes[i][1],
                target = target_size[0],
                dtype = dtype,
            )
            sample_boxes = []
            sample_labels = []
            for bbox, category in zip(bboxes, sample["category"]):
                # convert tensors to plain python lists to prevent Arrow encoding errors
                if isinstance(bbox, torch.Tensor):
                    bbox = bbox.tolist()
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


def preprocess_dataset(dataset, tokenizer, image_processor: MobileViTImageProcessor, instruction, is_training=False, dtype=torch.float32, max_workers: int | None = None):
    """
    Process entire dataset through image resizing and feature extraction
    
    Parameters:
        - dataset: HuggingFace dataset containing images and text
        - tokenizer: text tokenizer for processing text inputs
        - image_processor: MobileViTImageProcessor instance for image processing
        - instruction: instruction text to add to each example
        - is_training: boolean indicating whether this is a training set
        
    Returns:
        - processed dataset with images and tokenized text
    """

    # add instruction to each sample to create "text" field
    dataset = dataset.map(lambda x: x.update({"text": instruction}) or x)
    
    if max_workers is None:
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
        image_processor=image_processor,
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
        load_from_cache_file=True
    )

    return dataset

def load_and_prepare_dataset(
    dataset_name,
    tokenizer,
    instruction,
    *,
    image_processor: MobileViTImageProcessor | None = None,
    dtype: torch.dtype = torch.float32,
    on_the_fly: bool = False,
):
    """
    Load dataset from HuggingFace hub and create train/test splits
    
    Parameters:
        - dataset_name: HuggingFace dataset identifier
        - tokenizer: text tokenizer for processing text inputs
        - instruction: instruction text to add to each example
        - image_processor: MobileViTImageProcessor instance for image processing
        - dtype: dtype for tensors
        - on_the_fly: boolean indicating whether to skip heavy preprocessing
        
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

    if image_processor is None:
        image_processor = MobileViTImageProcessor.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")

    if on_the_fly:
        # Light transform: just add instruction text; collate will tokenize & process images
        def _insert_instruction(example):
            """Add the user instruction and (optionally) build `target_text`.

            • `text`        → raw system/instruction prompt (always present).
            • `target_text` → *token-id list* for the caption that the model
              should predict.  If a COCO sample has several captions we join
              them with spaces before tokenising.  When no caption is found
              we store an empty list so the collate-fn can fall back to
              producing an all-ignored label sequence (loss will be zero).
            """

            example["text"] = instruction

            raw_caption = ""

            # HF COCO format → list[{raw: str, ...}] under "sentences"
            if "sentences" in example and example["sentences"]:
                if isinstance(example["sentences"], list):
                    raw_caption = " ".join(
                        s.get("raw", "") for s in example["sentences"] if isinstance(s, dict)
                    ).strip()
                elif isinstance(example["sentences"], dict):
                    raw_caption = example["sentences"].get("raw", "").strip()

            # Some variants expose a single string field, e.g. ``caption``
            elif "caption" in example:
                raw_caption = str(example["caption"]).strip()

            # Tokenise *without* padding ─ we'll pad in the collator
            if raw_caption:
                ids = tokenizer(raw_caption, return_tensors="pt", truncation=True, max_length=128)[
                    "input_ids"
                ].squeeze(0).tolist()
                example["target_text"] = ids
            else:
                example["target_text"] = []  # will yield all-ignored labels

            return example

        train = train.map(_insert_instruction)
        test  = test.map(_insert_instruction)
    else:
        print("Preprocessing train dataset...")
        train = preprocess_dataset(train, tokenizer, image_processor, instruction, is_training=True, dtype=dtype)
        print("Preprocessing test dataset...")
        test = preprocess_dataset(test, tokenizer, image_processor, instruction, is_training=False, dtype=dtype)

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

    # After introducing MobileViT pre-processing each sample is ~6 MB in RAM.
    memory_per_sample_cpu = 6 * 1024 * 1024  

    cpu_bs_mem = int(available_ram * (1 - safety_margin) / memory_per_sample_cpu) // workers

    cpu_batch_size = min(cpu_bs_mem, max_batch_size)
    if cpu_batch_size >= 2:
        cpu_batch_size = 2 ** int(math.log2(cpu_batch_size))

    est_cpu_usage = (cpu_batch_size * memory_per_sample_cpu) / 1024**3
    print(f"  → CPU batch size:  {cpu_batch_size}  (≈{est_cpu_usage:.1f} GB RAM)")

    return gpu_batch_size, cpu_batch_size

def collate(batch):
    '''
    join pre-processed samples into a single trainer batch.

    pre: each sample has `images`, `input_ids`, `attention_mask` keys.

    parameters:
        - batch (list[dict]): mini-batch from torch DataLoader.

    returns: dict ready for `model(**batch)`.
    '''

    from torch.nn.utils.rnn import pad_sequence
    import torch

    # stack image tensors -> (B, 3, 256, 256)
    images = torch.stack([item["images"] if isinstance(item["images"], torch.Tensor)
                          else torch.as_tensor(item["images"], dtype=torch.float32)
                          for item in batch], 0)

    # -------- text padding --------
    input_id_seqs = [torch.as_tensor(itm["input_ids"], dtype=torch.long)
                     for itm in batch]
    attn_seqs     = [torch.as_tensor(itm["attention_mask"], dtype=torch.long)
                     for itm in batch]

    input_ids_padded      = pad_sequence(input_id_seqs, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attn_seqs,     batch_first=True, padding_value=0)

    result = {
        "images": images,
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
    }

    # -------- optional detection / caption targets --------
    additional_keys = [k for k in batch[0].keys() if k not in result]
    for key in additional_keys:
        elems = [item[key] for item in batch]

        if key == "target_boxes":
            tensors = [torch.as_tensor(e, dtype=torch.float32) for e in elems]
            result[key] = pad_sequence(tensors, batch_first=True, padding_value=0.0)
        elif key == "target_labels":
            tensors = [torch.as_tensor(e, dtype=torch.long) for e in elems]
            result[key] = pad_sequence(tensors, batch_first=True, padding_value=-100)
        elif key == "target_text":
            tensors = [torch.as_tensor(e, dtype=torch.long) for e in elems]
            result[key] = pad_sequence(tensors, batch_first=True, padding_value=0)
        else:
            result[key] = elems  # leave as list – Trainer will ignore unknowns

    return result


def compute_embedding_similarity(vision_features, text_embeddings):
    """Average cosine similarity between vision-tower features and text embeddings."""
    import torch.nn.functional as F

    vision_pooled = vision_features.mean(dim=1)
    text_pooled   = text_embeddings.mean(dim=1)

    sims = F.cosine_similarity(vision_pooled, text_pooled, dim=-1)
    return sims.mean().item()


def compute_gradient_norm(model):
    """Return global L2 norm of gradients on the projector parameters."""
    import torch

    grads = [p.grad.detach().flatten() for p in model.projector.parameters()
             if p.grad is not None and p.requires_grad]
    if not grads:
        return 0.0
    return torch.cat(grads).norm().item()


def compute_parameter_norm(model):
    """Return global L2 norm of the projector parameters themselves."""
    import math

    total = 0.0
    for p in model.projector.parameters():
        total += p.data.norm(2).item() ** 2
    return math.sqrt(total)


def load_labels_from_yaml(yaml_path):
    """Load class-name → index mapping from a YOLO-style YAML file."""
    import os, yaml, logging

    if not os.path.exists(yaml_path):
        logging.error(f"Label YAML file not found: {yaml_path}")
        raise FileNotFoundError(yaml_path)

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "names" not in data:
        raise ValueError("YAML file must contain a 'names' section")

    return {v: int(k) for k, v in data["names"].items()}