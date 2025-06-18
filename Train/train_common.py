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
import yaml 

# Image processing imports for dynamic resizer
from PIL import Image
import numpy as np

# img / tensor utilities
from torchvision.transforms.functional import pil_to_tensor
import torch.nn.functional as F

import psutil

# Image augmentations
from .train_augments import apply_weather_augmentations, apply_camera_augmentations, apply_batch_augmentations

# Constants
VIS_TOKENS = 64  # Visual tokens that will be prepended by the model
DEFAULT_TARGET_SIZE = (256, 256)
MAX_TARGET_TEXT_LENGTH = 128
DEFAULT_BBOX_JITTER_RATIO = 0.05
MEMORY_SAFETY_MARGIN = 0.15
MIN_BATCH_SIZE = 8
MAX_BATCH_SIZE = 32768
GPU_MEMORY_PER_SAMPLE = 4 * 1024 * 1024  # 4 MB heuristic
CPU_MEMORY_PER_SAMPLE = 6 * 1024 * 1024  # 6 MB heuristic

def jitter_bboxes(bboxes, img_width, img_height, dtype, jitter_ratio=DEFAULT_BBOX_JITTER_RATIO):
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
        bboxes = bboxes.clone().detach().float().cpu().numpy()
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
        bboxes = bboxes.clone().detach().float().cpu().numpy()
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

def letterbox_image(image, target_size=DEFAULT_TARGET_SIZE):
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
        bboxes = bboxes.clone().detach().float().cpu().numpy()

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

def preprocess_batch(batch, tokenizer, image_processor, gpu_batch_size=64, bbox_jitter_ratio=DEFAULT_BBOX_JITTER_RATIO, training=False, target_size=DEFAULT_TARGET_SIZE, dtype=torch.float32, label_lookup=None):
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
        - label_lookup (dict): mapping from class indices to names

    returns: dict with image tensors, token ids, masks and optional targets.
    '''

    processed_images = []      # pixel arrays after processor (np.float32)
    image_sizes = []           # original (w, h) per augmented sample
    padded_image_sizes = []    # after letter-box / resize
    lb_params = []             # (ratio, crop_left, crop_top) per augmented sample
    sample_replication = []    # map augmented index -> original sample idx

    images_field = "images" if "images" in batch else "image" if "image" in batch else None
    if images_field is None:
        raise KeyError("Batch dict must contain an 'images' or 'image' key with PIL images")

    for idx, img in enumerate(batch[images_field]):
        if not isinstance(img, Image.Image):
            raise ValueError("images must be PIL.Image objects")

        base_rgb = img.convert("RGB")

        # ------------------------------------------------------------------
        # Generate list of images: original + each augmentation (training only)
        # ------------------------------------------------------------------
        img_versions = [base_rgb]

        if training:
            try:
                aug_versions = apply_batch_augmentations(
                    [base_rgb],
                    weather_intensity="medium",
                    camera_intensity="medium",
                    weather_enabled=True,
                    camera_enabled=True,
                    max_augmentations_per_type=2,
                )
                if aug_versions:
                    img_versions.extend(aug_versions)
            except Exception as e:
                print(f"Augmentation error (continuing with original image): {e}")

        for rgb in img_versions:
            # --- MobileViT image processor ---
            try:
                px = image_processor(rgb, return_tensors="np")["pixel_values"][0]
                processed_images.append(px)
            except Exception as e:
                print(f"Image processing error: {e}")
                # Fallback to basic processing
                rgb_resized = rgb.resize(target_size, Image.BICUBIC)
                px = np.array(rgb_resized, dtype=np.float32) / 255.0
                px = px.transpose(2, 0, 1)  # HWC to CHW
                processed_images.append(px)

            image_sizes.append(rgb.size)
            padded_image_sizes.append(target_size)

            # store params (based on original image dims)
            orig_w, orig_h = base_rgb.size
            ratio = target_size[0] / min(orig_w, orig_h)
            resized_w, resized_h = int(round(orig_w * ratio)), int(round(orig_h * ratio))
            crop_left = max((resized_w - target_size[0]) // 2, 0)
            crop_top = max((resized_h - target_size[1]) // 2, 0)
            lb_params.append((ratio, crop_left, crop_top))

            sample_replication.append(idx)

    text = batch["text"]
    
    # ensure tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Reserve room for the visual tokens that will be prepended later in the
    # model forward pass.  MobileViT-v2 @256 px yields 8×8 = 64 tokens.
    max_txt_len = tokenizer.model_max_length - VIS_TOKENS
    if max_txt_len <= 0:
        raise ValueError("Tokenizer max_length is too small to fit visual tokens")

    tokenized_text = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_txt_len,
        truncation=True,
        padding=True,
    )
    
    # Replicate token rows for each augmented variant
    base_input_id_rows = [row.tolist() for row in tokenized_text["input_ids"]]
    base_attention_rows = [row.tolist() for row in tokenized_text["attention_mask"]]

    input_id_rows = [base_input_id_rows[i] for i in sample_replication]
    attention_mask_rows = [base_attention_rows[i] for i in sample_replication]

    result = {
        "images": processed_images,
        "input_ids": input_id_rows,
        "attention_mask": attention_mask_rows,
    }

    # apply augmentations to objects
    if "objects" in batch:
        result["target_boxes"] = []
        result["target_labels"] = []
        for aug_idx, orig_idx in enumerate(sample_replication):
            sample = batch["objects"][orig_idx]
            bboxes = sample["bbox"]

            # 1) optional jitter in *original* image space
            if training:
                bboxes = jitter_bboxes(
                    bboxes,
                    img_width=image_sizes[aug_idx][0],
                    img_height=image_sizes[aug_idx][1],
                    dtype=dtype,
                    jitter_ratio=bbox_jitter_ratio,
                )

            # 2) adjust to letter-boxed, padded coordinates then normalise
            ratio, crop_left, crop_top = lb_params[aug_idx]
            bboxes = adjust_boxes_resize_crop(
                bboxes,
                orig_w = image_sizes[aug_idx][0],
                orig_h = image_sizes[aug_idx][1],
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

            # -------------------------------------------------------------
            # Build detection target string: <bbob>class:bbox</bbob>
            # -------------------------------------------------------------
            if label_lookup is None:
                label_lookup = {}

            detection_fragments = []
            for bbox, cat in zip(sample_boxes, sample_labels):
                # Using lookup when possible; fallback = str(cat)
                label = label_lookup.get(cat, str(cat)) if isinstance(cat, int) else str(cat)

                # bbox components already 0-1 normalised; format with 3 decimals
                bbox_txt = " ".join(f"{v:.3f}" for v in bbox)
                detection_fragments.append(f"<bbob>{label}:{bbox_txt}</bbob>")

            detection_text = " ".join(detection_fragments)

            # Tokenise—NOTE: no padding here; collate will pad.
            if detection_text:
                ids = tokenizer(
                    detection_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_TARGET_TEXT_LENGTH,
                )["input_ids"].squeeze(0).tolist()
            else:
                ids = []

            # store per-sample
            if "target_text" not in result:
                result["target_text"] = []
            result["target_text"].append(ids)

    if "sentences" in batch and "target_text" not in result:
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
                max_length=MAX_TARGET_TEXT_LENGTH,
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


def preprocess_dataset(dataset, tokenizer, image_processor, instruction, is_training=False, dtype=torch.float32, max_workers=None, label_lookup=None):
    """
    Process entire dataset through image resizing and feature extraction
    
    Parameters:
        - dataset: HuggingFace dataset containing images and text
        - tokenizer: text tokenizer for processing text inputs
        - image_processor: MobileViTImageProcessor instance for image processing
        - instruction: instruction text to add to each example
        - is_training: boolean indicating whether this is a training set
        - dtype: torch dtype for tensors
        - max_workers: number of workers for multiprocessing
        - label_lookup: mapping from class indices to names
        
    Returns:
        - processed dataset with images and tokenized text
    """

    # add instruction to each sample to create "text" field
    dataset = dataset.map(lambda x: x.update({"text": instruction}) or x)
    
    if max_workers is None:
        max_workers = min(mp.cpu_count() - 1, 8)
    
    # determine optimal batch sizes ----------------------------------------------------
    gpu_batch_size, cpu_batch_size = calculate_optimal_batch_size(workers=max_workers, safety_margin=MEMORY_SAFETY_MARGIN)

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
        label_lookup=label_lookup,
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

def _insert_instruction(example, tokenizer, instruction):
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
        ids = tokenizer(
            raw_caption, 
            return_tensors="pt", 
            truncation=True, 
            max_length=MAX_TARGET_TEXT_LENGTH
        )["input_ids"].squeeze(0).tolist()
        example["target_text"] = ids
    else:
        example["target_text"] = []  # will yield all-ignored labels

    return example

def load_and_prepare_dataset(
    dataset_name,
    tokenizer,
    instruction,
    image_processor,
    dtype=torch.float32,
    on_the_fly=False,
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

    if on_the_fly:
        # Fixed: Use fn_kwargs for map function
        train = train.map(
            _insert_instruction, 
            fn_kwargs={"tokenizer": tokenizer, "instruction": instruction}
        )
        test = test.map(
            _insert_instruction, 
            fn_kwargs={"tokenizer": tokenizer, "instruction": instruction}
        )
    else:
        label_lookup = None
        # ------------------------------------------------------------------
        # 1) Try to extract label names directly from dataset features
        # ------------------------------------------------------------------
        try:
            sample_split = train if isinstance(train, datasets.Dataset) else next(iter(dataset.values()))
            names = _find_classlabel_names(sample_split.features)
            if names:
                label_lookup = {i: n for i, n in enumerate(names)}
        except Exception:
            pass

        dataset_id = dataset_name.split("/")[-1]
        yaml_path = os.path.join(os.path.dirname(__file__), "Labels", f"{dataset_id}.yaml")

        # ------------------------------------------------------------------
        # 2) Fallback: read YAML if feature-derived names not available
        # ------------------------------------------------------------------
        if label_lookup is None:
            try:
                name_to_idx = load_labels_from_yaml(yaml_path)
                label_lookup = {idx: name for name, idx in name_to_idx.items()}
            except Exception:
                print("No labels found – bounding box text will use numeric IDs.")

        print("Preprocessing train dataset…")
        train = preprocess_dataset(
            train,
            tokenizer,
            image_processor,
            instruction,
            is_training=True,
            dtype=dtype,
            label_lookup=label_lookup,
        )

        print("Preprocessing test dataset…")
        test = preprocess_dataset(
            test,
            tokenizer,
            image_processor,
            instruction,
            is_training=False,
            dtype=dtype,
            label_lookup=label_lookup,
        )

    return train, test

def calculate_optimal_batch_size(
    workers=1,
    safety_margin=MEMORY_SAFETY_MARGIN,
    min_batch_size=MIN_BATCH_SIZE,
    max_batch_size=MAX_BATCH_SIZE,
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
    gpu_batch_size = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory
        allocated = torch.cuda.memory_allocated(0)
        available_vram = total_vram - allocated

        print("VRAM Analysis:")
        print(f"  Total VRAM:       {total_vram/1024**3:.1f} GB")
        print(f"  Currently alloc.: {allocated/1024**3:.1f} GB")
        print(f"  Available:        {available_vram/1024**3:.1f} GB")

        gpu_batch_size = int(available_vram / GPU_MEMORY_PER_SAMPLE)
        gpu_batch_size = max(min_batch_size, min(gpu_batch_size, max_batch_size*(workers//2)))

        if gpu_batch_size >= 2:
            gpu_batch_size = 2 ** int(math.log2(gpu_batch_size))

        est_usage = (gpu_batch_size * GPU_MEMORY_PER_SAMPLE) / 1024**3
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

    cpu_bs_mem = int(available_ram * (1 - safety_margin) / CPU_MEMORY_PER_SAMPLE) // workers // 4

    cpu_batch_size = min(cpu_bs_mem, max_batch_size)
    if cpu_batch_size >= 2:
        cpu_batch_size = 2 ** int(math.log2(cpu_batch_size))

    est_cpu_usage = (cpu_batch_size * CPU_MEMORY_PER_SAMPLE) / 1024**3
    print(f"  → CPU batch size:  {cpu_batch_size}  (≈{est_cpu_usage:.1f} GB RAM)")

    return gpu_batch_size, cpu_batch_size


def load_labels_from_yaml(yaml_path):
    """Load class-name → index mapping from a YOLO-style YAML file."""

    if not os.path.exists(yaml_path):
        print(f"Label YAML file not found: {yaml_path}")
        raise FileNotFoundError(yaml_path)

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "names" not in data:
        raise ValueError("YAML file must contain a 'names' section")

    return {v: int(k) for k, v in data["names"].items()}


def _convert_dtype_to_str(value):
    """Return value with torch.dtype objects replaced by their string name."""

    if isinstance(value, torch.dtype):
        return str(value).split(".")[-1]
    if isinstance(value, dict):
        return {k: _convert_dtype_to_str(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_convert_dtype_to_str(v) for v in value]
    return value


def clean_tokenizer_config(tokenizer):
    """In-place convert any torch.dtype values inside `tokenizer.init_kwargs`.

    Call this once after the tokenizer is created/before `Trainer` starts so
    that checkpoints can be saved without hitting JSON serialization errors.
    """

    if not hasattr(tokenizer, "init_kwargs") or not isinstance(tokenizer.init_kwargs, dict):
        return

    tokenizer.init_kwargs = _convert_dtype_to_str(tokenizer.init_kwargs)

def make_collate_fn(pad_token_id: int, tokenizer):
    '''
    factory that builds a custom collate_fn for trainer.

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
                # Fallback: PIL → tensor path
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

def _find_classlabel_names(features):
    """Recursively search a datasets.Features tree for a ClassLabel and return its names list."""
    from datasets import ClassLabel, Features, Sequence

    if isinstance(features, ClassLabel):
        return features.names

    if isinstance(features, dict):
        for v in features.values():
            names = _find_classlabel_names(v)
            if names:
                return names
    elif isinstance(features, (list, tuple)):
        for v in features:
            names = _find_classlabel_names(v)
            if names:
                return names
    elif isinstance(features, Sequence):
        return _find_classlabel_names(features.feature)

    return None