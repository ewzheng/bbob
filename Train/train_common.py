'''
File: train_common.py
Author: Elias Zheng and Claude
Description: This script contains common training functions
'''

import torch
import torch.nn as nn
import datasets
from datasets import ClassLabel, Sequence
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
import random

# Image processing imports for dynamic resizer
from PIL import Image
import numpy as np

# img / tensor utilities
from torchvision.transforms.functional import pil_to_tensor
import torch.nn.functional as F

import psutil

# Image augmentations
from .train_augments import apply_batch_augmentations, apply_ms_crop
from Utils.class_id_map import init_from_labels, get_id

# Constants
VIS_TOKENS = 64  # Visual tokens that will be prepended by the model
DEFAULT_TARGET_SIZE = (256, 256)
MAX_TARGET_TEXT_LENGTH = 256
MEMORY_SAFETY_MARGIN = 0.15
MIN_BATCH_SIZE = 8
MAX_BATCH_SIZE = 4096
GPU_MEMORY_PER_SAMPLE = 16 * 1024 * 1024  # 16 MB heuristic (was 4 MB)
MAX_PREPROC_GPU_BATCH = 256
CPU_MEMORY_PER_SAMPLE = 6 * 1024 * 1024  # 6 MB heuristic
AUG_PROB = 0.05
# NEW: number of Pix2Seq-style multi-scale crops to generate per original image
MS_CROPS_PER_SAMPLE = 4  # increase to e.g. 3 for more geometric diversity

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

        # Skip boxes that end up with zero area (fully outside the crop)
        if w <= 0 or h <= 0:
            continue

        adjusted.append([
            x / target,
            y / target,
            w / target,
            h / target,
        ])

    return torch.tensor(adjusted, dtype=dtype)

def preprocess_batch(batch, tokenizer, image_processor, training=False, augment=False, target_size=DEFAULT_TARGET_SIZE, dtype=torch.float32, label_lookup=None):
    '''
    build vision-language features for one raw dataset batch.

    pre: batch must include keys `image`/`images` and `text`.

    parameters:
        - batch (dict): incoming dataset slice.
        - tokenizer (PreTrainedTokenizer): hf tokenizer.
        - image_processor: MobileViTImageProcessor instance for image processing
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

    # Store per-augmented-image ground-truth boxes / labels and a flag
    # indicating whether geometric adjustment is still required.  For
    # Pix2Seq-style MS-crop the boxes are already normalised to the 256×256
    # output, so no further adjust is needed.
    aug_boxes_all: list[list] = []
    aug_labels_all: list[list] = []
    aug_need_adjust: list[bool] = []
    # Track whether a particular augmented view is already normalised by MS-crop
    is_ms_crop_flags: list[bool] = []

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
        boxes_versions: list[list] = []
        labels_versions: list[list] = []

        # Load base GT if available
        if "objects" in batch:
            sample_obj = batch["objects"][idx]
            base_boxes = sample_obj.get("bbox", [])
            base_labels = sample_obj.get("category", [])
        else:
            base_boxes, base_labels = [], []

        boxes_versions.append(base_boxes)
        labels_versions.append(base_labels)
        is_ms_crop_flags.append(False)  # original view needs geometric adjust

        
        if training:
            if augment and random.random() < AUG_PROB:
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
                        for _ in aug_versions:
                            boxes_versions.append(base_boxes)  # same geometry
                            labels_versions.append(base_labels)
                            is_ms_crop_flags.append(False)  # photometric aug – still needs adjust
                except Exception as e:
                    print(f"Augmentation error (continuing with original image): {e}")

            # ---------------- multi-scale crop augmentation -------------------
            for _ in range(MS_CROPS_PER_SAMPLE):
                try:
                    img_c, boxes_c, labels_c = apply_ms_crop(base_rgb, base_boxes, base_labels)
                    img_versions.append(img_c)
                    boxes_versions.append(boxes_c)
                    labels_versions.append(labels_c)
                    is_ms_crop_flags.append(True)  # MS-crop view is already normalised
                except Exception as e:
                    print(f"MS-crop augmentation failed (continuing): {e}")

        # --- iterate paired variants -----------------------------------------
        for aug_idx, (rgb, boxes_this, labels_this) in enumerate(zip(img_versions, boxes_versions, labels_versions)):
            # -------------------------------------------------------------
            # Step 1: prepare a 256×256 input for MobileViT *without*
            # centre-cropping. Non-MS-crop views are letter-boxed to
            # preserve every object; MS-crop views already match 256×256.
            # -------------------------------------------------------------
            if is_ms_crop_flags[aug_idx]:
                rgb_proc = rgb          # already 256×256
                scale, pad_w, pad_h = 1.0, 0, 0
            else:
                rgb_proc, scale, pad_w, pad_h = letterbox_image(rgb, target_size)

            # --- MobileViT image processor ---
            try:
                px = image_processor(
                    rgb_proc,
                    return_tensors="pt",
                    do_center_crop=False,  # centre-crop disabled globally
                    do_resize=False,       # we performed resize/pad ourselves
                )["pixel_values"][0]
                processed_images.append((px * 255).astype(np.uint8))
            except Exception as e:
                print(f"Image processing error: {e}")
                # Fallback to basic processing – resize shortest edge then letter-box
                fallback, _, _, _ = letterbox_image(rgb, target_size)
                px = np.array(fallback, dtype=np.uint8).transpose(2, 0, 1)
                processed_images.append(px)

            # Store *original* image dims (before letter-box) for bbox adjust
            image_sizes.append(base_rgb.size)
            padded_image_sizes.append(target_size)

            # Save scale & padding so we can transform boxes later
            lb_params.append((scale, pad_w, pad_h))

            sample_replication.append(idx)

            # store GT boxes/labels for this view
            aug_boxes_all.append(boxes_this)
            aug_labels_all.append(labels_this)
            # No geometric adjustment required for MS-crop views (already 0-1)
            need_adj = not is_ms_crop_flags[aug_idx]
            aug_need_adjust.append(need_adj)

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
    for aug_idx, orig_idx in enumerate(sample_replication):
        # Retrieve stored boxes / labels for this augmented view
        bboxes_raw = aug_boxes_all[aug_idx]
        labels_raw = aug_labels_all[aug_idx]

        if aug_need_adjust[aug_idx] and bboxes_raw:
            # Letter-box adjustment (non-MS-crop views)
            scale, pad_w, pad_h = lb_params[aug_idx]
            bboxes = adjust_boxes_for_letterbox(
                bboxes_raw,
                scale,
                pad_w,
                pad_h,
                orig_w=image_sizes[aug_idx][0],
                orig_h=image_sizes[aug_idx][1],
                target_w=target_size[0],
                target_h=target_size[1],
                dtype=dtype,
            )
        else:
            bboxes = torch.as_tensor(bboxes_raw, dtype=dtype)

        sample_boxes = []
        sample_label_ids = []   # canonical integer IDs (multi-token aware)
        sample_label_strs = []  # human-readable strings (for text)

        for bbox, category in zip(bboxes, labels_raw):
            # Convert tensors to plain python lists
            if isinstance(bbox, torch.Tensor):
                bbox = bbox.tolist()

            # --- map category to string --------------------------------
            if isinstance(category, int):
                label_str = label_lookup.get(category, str(category)) if label_lookup else str(category)
            else:
                label_str = str(category)

            # --- tokenise *whole* phrase and map to deterministic ID --
            sub_tok_ids = tokenizer(label_str, add_special_tokens=False)["input_ids"]
            cls_id = get_id(sub_tok_ids)

            sample_boxes.append(bbox)
            sample_label_ids.append(cls_id)
            sample_label_strs.append(label_str)

        # -------------------------------------------------------------
        # Build detection target string: <bbob>class:bbox</bbob>
        # -------------------------------------------------------------
        detection_fragments = []
        for bbox, label in zip(sample_boxes, sample_label_strs):
            # bbox components are already 0-1; keep them as 3-decimal floats
            bbox_txt = ", ".join(f"{v:.3f}" for v in bbox)
            detection_fragments.append(f"<|bbob|>{label}: [{bbox_txt}]</|bbob|>")

        # ---------------------------------------------------------
        # Build the textual target *and* decide how many detections
        # can be kept so that text, boxes and labels stay in sync.
        # ---------------------------------------------------------

        kept_n: int = 0  # how many objects remain after truncation
        ids: list[int]

        if detection_fragments:
            frags = detection_fragments  # alias for brevity

            # Try dropping fragments from the *end* until the encoded
            # length fits.  This guarantees that we always keep a
            # prefix of the object list, so slicing boxes / labels with
            # kept_n is safe.
            while frags:
                candidate = " ".join(frags)
                enc = tokenizer(candidate, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
                if enc.size(0) <= MAX_TARGET_TEXT_LENGTH:
                    ids = enc.tolist()
                    kept_n = len(frags)
                    break
                frags.pop()  # drop last <bbob>… fragment and retry
            else:
                # Even the first fragment does not fit – encode the
                # first one with hard truncation, keep_n = 1
                first = detection_fragments[0]
                ids = tokenizer(first, return_tensors="pt", truncation=True,
                                 max_length=MAX_TARGET_TEXT_LENGTH)["input_ids"].squeeze(0).tolist()
                kept_n = 1
        else:
            ids = []
            kept_n = 0

        # ---------------- boxes / labels -----------------------
        # Slice to *kept_n* so geometric targets match the text.
        result.setdefault("target_boxes", []).append(sample_boxes[:kept_n])
        result.setdefault("target_labels", []).append(sample_label_ids[:kept_n])
        result.setdefault("target_label_strs", []).append(sample_label_strs[:kept_n])
        # ---------------- store token ids ----------------------
        result.setdefault("target_text", []).append(ids)

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


def preprocess_dataset(dataset, tokenizer, image_processor, instruction, is_training=False, augment=False, dtype=torch.float32, max_workers=None, label_lookup=None):
    """
    Process entire dataset through image resizing and feature extraction
    
    Parameters:
        - dataset: HuggingFace dataset containing images and text
        - tokenizer: text tokenizer for processing text inputs
        - image_processor: MobileViTImageProcessor instance for image processing
        - instruction: instruction text to add to each example
        - is_training: boolean indicating whether this is a training set
        - augment: boolean indicating whether to apply augmentations
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
    cpu_batch_size = calculate_optimal_batch_size(workers=max_workers, safety_margin=MEMORY_SAFETY_MARGIN)

    print(f"CPU batch size (image preprocessing): {cpu_batch_size}")
    
    _preprocessing_function = partial(
        preprocess_batch,
        tokenizer=tokenizer,
        image_processor=image_processor,
        training=is_training,
        dtype=dtype,
        label_lookup=label_lookup,
        augment=augment,
    )

    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        batch_size=cpu_batch_size,  # smaller CPU batch for RAM safety
        remove_columns=dataset.column_names,
        num_proc=max_workers,
        desc=f"Processing images and text ({max_workers} workers, CPU batch={cpu_batch_size})",
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
    augment=False,
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
    # ------------------------------------------------------------------
    # Ensure the tokenizer inserts one BOS and one EOS automatically so we
    # do not have to add them manually when building detection strings.
    # ------------------------------------------------------------------

    if hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = True
    if hasattr(tokenizer, "add_eos_token"):
        tokenizer.add_eos_token = True

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

        # ----------------------------------------------------------
        # Build deterministic *token-sequence → integer* map once so
        # both dataset preprocessing *and* runtime loss can convert
        # multi-token class phrases into a single ID.
        # ----------------------------------------------------------
        if label_lookup is not None:
            try:
                init_from_labels(label_lookup, tokenizer)
            except Exception as e:
                print(f"[WARN] Could not initialise class-id map: {e}")

        print("Preprocessing train dataset…")
        train = preprocess_dataset(
            train,
            tokenizer,
            image_processor,
            instruction,
            is_training=True,
            augment=augment,
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
            augment=augment,
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
    """Return an optimal CPU batch size for image preprocessing.

    The heuristic estimates how many samples fit comfortably into RAM
    given the number of preprocessing workers.  GPU memory is no longer
    considered because all heavy preprocessing happens on the CPU.

    Returns
    -------
    int
        Optimal CPU-side batch size.
    """

    # ------------------------------------------------------------------
    # CPU batch size heuristic
    # ------------------------------------------------------------------
    if psutil is not None:
        vm = psutil.virtual_memory()
        total_ram = vm.total
        available_ram = vm.available
    else:
        # Fallback: assume 8 GB total with 50 % free
        total_ram = 8 * 1024**3
        available_ram = total_ram * (1 - safety_margin)

    print("RAM Analysis:")
    print(f"  Total RAM:        {total_ram/1024**3:.1f} GB")
    print(f"  Available RAM:    {available_ram/1024**3:.1f} GB")

    # Conservative estimate: divide by number of workers and apply margin
    cpu_bs_mem = int(available_ram * (1 - safety_margin) / CPU_MEMORY_PER_SAMPLE) // max(workers, 1)

    # Clamp to valid range and round to nearest power-of-two for efficiency
    cpu_batch_size = min(max(cpu_bs_mem, min_batch_size), max_batch_size)
    if cpu_batch_size >= 2:
        cpu_batch_size = 2 ** int(math.log2(cpu_batch_size))

    est_cpu_usage = (cpu_batch_size * CPU_MEMORY_PER_SAMPLE) / 1024**3
    print(f"  → CPU batch size:  {cpu_batch_size}  (≈{est_cpu_usage:.1f} GB RAM)")

    return cpu_batch_size


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

def _find_classlabel_names(features):
    """Recursively search a datasets.Features tree for a ClassLabel and return its names list."""

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