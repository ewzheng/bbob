"""train_collate.py

Data-collator utilities for BBOB training.

This file contains `make_collate_fn`, a factory that returns a collate
callable tailored for multimodal batches (image + text) and compatible
with HuggingFace `Trainer`.  Its behaviour matches the latest logic that
• normalises/letterboxes images,
• prepends 64 visual placeholders to labels,
• *hides* target-text tokens from the model inputs so teacher-forcing is
  avoided by default.

The function can be extended with scheduled-sampling options but starts
with the deterministic "always hide targets" policy adopted on 2024-06-18.
"""

from __future__ import annotations

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import io
from .loss_helpers import TAG_OPEN, TAG_CLOSE
from .train_common import format_coordinate
import random
import torch.utils.data as tud
import re

# Constants (kept in sync with Train.train_common)
VIS_TOKENS: int = 64           # number of visual tokens the model prepends
TARGET_SIZE = (256, 256)       # spatial resolution used by MobileViT-v2

# ---------------------------------------------------------------------------
# Coordinate safety margin – keep jittered / synthetic boxes away from exactly
# 0.000 so that the model does not over-fit to the left/top border token.
# Matches the 0.001 resolution of the numeric vocabulary.
# ---------------------------------------------------------------------------
MIN_COORD = 1.0 / 1000.0  # 0.001

# ====================================================================
# Scheduled-sampling data collator
# --------------------------------------------------------------------
# Implements a probability-based teacher-forcing regime:
#   • At step 0 the probability of feeding ground-truth targets can be
#     `tf_start_p` (default 1.0 = always teacher force).
#   • The probability decays towards `tf_end_p` over `total_steps`.
#   • Supports three schedule shapes: linear, cosine, exponential.
#
# Usage in training script:
#   collator = BBOBCollator(pad_token_id, tokenizer,
#                           tf_start_p=1.0, tf_end_p=0.0,
#                           total_steps=20_000, schedule="linear")
#   trainer = Trainer(..., data_collator=collator)
# ====================================================================

import math


class BBOBCollator:  # noqa: N801
    """Minimal collator for BBOB – no teacher-forcing logic."""

    def __init__(
        self,
        pad_token_id: int,
        tokenizer,
        image_processor,
        *,
        logger=None,
        on_the_fly=False,
        noise_prob=1.0,  # ALWAYS add noise objects for proper Pix2Seq training
        max_noise_boxes=32,  # Max number of noise boxes to add
        noise_ratio_range=(1, 1.25),  # Range for noise count as fraction of GT count
        **kwargs,  # accept legacy tf_* kwargs but ignore them
    ):
        self.pad_id = pad_token_id
        self.tokenizer = tokenizer
        self.processor = image_processor
        self.logger = logger

        self.rngjesus = np.random.default_rng()

        # teacher-forcing parameters removed – keep placeholders for API compat
        self.is_eval = False
        self.on_the_fly = on_the_fly
        self.noise_prob = noise_prob
        self.max_noise_boxes = max_noise_boxes
        self.noise_ratio_range = noise_ratio_range

        # Pre-compute placeholder id differing from pad id.
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id != pad_token_id:
            self.placeholder_id = tokenizer.eos_token_id
        elif tokenizer.unk_token_id is not None and tokenizer.unk_token_id != pad_token_id:
            self.placeholder_id = tokenizer.unk_token_id
        else:
            self.placeholder_id = pad_token_id

        self.open_id = tokenizer.convert_tokens_to_ids(TAG_OPEN)
        self.close_id = tokenizer.convert_tokens_to_ids(TAG_CLOSE)
        # No need to cache a dedicated space token – we will prefix a literal
        # space to fragment strings before tokenisation so the tokenizer can
        # handle it according to its own rules (works for BPE and sentencepiece).
        
        # OPTIMIZATION: Cache for coordinate formatting to avoid repeated string operations
        # Using LRU cache with reasonable size limit to balance memory vs speed
        from functools import lru_cache
        self._coord_cache_size = 1000  # Cache up to 1000 unique coordinate values
        self._fmt_coord_cached = lru_cache(maxsize=self._coord_cache_size)(self._fmt_coord_impl)

    # OPTIMIZATION: Cached coordinate formatting implementation
    def _fmt_coord_impl(self, v):
        """Internal implementation of coordinate formatting with caching."""
        return format_coordinate(v)
    
    def fmt_coord(self, v):
        """Cached coordinate formatting method."""
        return self._fmt_coord_cached(float(v))  # Ensure hashable input for cache

    # ---------------- main callable ------------------------------------

    def __call__(self, batch):
        return self._make_batch(
            batch,
            pad_token_id=self.pad_id,
            placeholder_id=self.placeholder_id,
            on_the_fly=self.on_the_fly,
        )

    # BBOBTrainer toggles these; keep as no-ops
    def eval(self):
        self.logger.info("Forked a eval worker")
        self.is_eval = True

    def train(self):
        self.logger.info("Forked a training worker")
        self.is_eval = False

    def reseed(self):
        random.seed()

    def _generate_noise_boxes_batch(self, batch_gt_boxes, batch_label_strs, batch_noise_counts):
        """
        OPTIMIZED: Generate noise boxes for entire batch using vectorized operations.
        FIXED: Proper label isolation - each sample's noise uses only its own GT labels.
        
        Args:
            batch_gt_boxes: List of GT box tensors for each sample in batch
            batch_label_strs: List of label string lists for each sample
            batch_noise_counts: List of noise counts for each sample
            
        Returns:
            List of (noise_boxes, noise_labels) tuples for each sample
        """
        # Pre-initialize results list with correct size
        batch_size = len(batch_gt_boxes)
        batch_results = [None] * batch_size
        
        # Collect all GT boxes and track which sample they belong to
        all_gt_boxes = []
        all_gt_labels = []
        box_to_sample_map = []  # Maps each box index to its sample index
        sample_label_pools = {}  # Per-sample label pools for proper isolation
        
        for sample_idx, (gt_boxes, label_strs, num_noise) in enumerate(zip(batch_gt_boxes, batch_label_strs, batch_noise_counts)):
            if num_noise <= 0:
                # Use proper device for empty tensors
                device = gt_boxes.device if gt_boxes.numel() > 0 else torch.device('cpu')
                batch_results[sample_idx] = (torch.empty((0, 4), device=device), [])
                continue
            
            # Store this sample's label pool for later use
            sample_label_pools[sample_idx] = label_strs
            
            # Add GT boxes for this sample to the batch
            for i, (box, label) in enumerate(zip(gt_boxes, label_strs)):
                all_gt_boxes.append(box)
                all_gt_labels.append(label)
                box_to_sample_map.append(sample_idx)  # Track which sample this box came from
        
        if not all_gt_boxes:
            # No GT boxes in entire batch - handle remaining samples
            for sample_idx, num_noise in enumerate(batch_noise_counts):
                if batch_results[sample_idx] is None:
                    device = batch_gt_boxes[sample_idx].device if batch_gt_boxes[sample_idx].numel() > 0 else torch.device('cpu')
                    noise_boxes, noise_labels = self._generate_noise_boxes(num_noise, ["object"], torch.empty((0, 4), device=device))
                    batch_results[sample_idx] = (noise_boxes, noise_labels)
            return batch_results
        
        # Convert to tensors for vectorized operations
        all_gt_boxes_tensor = torch.stack(all_gt_boxes)
        
        # VECTORIZED NOISE GENERATION
        total_noise_needed = sum(num for num in batch_noise_counts if num > 0)
        if total_noise_needed == 0:
            return batch_results
        
        # Generate noise categories vectorized
        n_jittered = max(1, int(total_noise_needed * 0.4))
        n_shifted = max(1, int(total_noise_needed * 0.3))  
        n_random = total_noise_needed - n_jittered - n_shifted
        
        all_noise_boxes = []
        all_noise_labels = []
        noise_sample_assignments = []  # Track which sample each noise box should go to
        
        # 1. VECTORIZED JITTERED BOXES
        if len(all_gt_boxes) > 0 and n_jittered > 0:
            # Randomly select GT boxes to jitter (with replacement)
            jitter_indices = torch.randint(0, len(all_gt_boxes), (n_jittered,))
            boxes_to_jitter = all_gt_boxes_tensor[jitter_indices]
            
            # Vectorized jitter application
            x, y, w, h = boxes_to_jitter[:, 0], boxes_to_jitter[:, 1], boxes_to_jitter[:, 2], boxes_to_jitter[:, 3]
            
            # Generate all jitter values at once with consistent device
            device = boxes_to_jitter.device
            jitter_x = (torch.rand(n_jittered, device=device) * 2.0 - 1.0) * 0.4 * w
            jitter_y = (torch.rand(n_jittered, device=device) * 2.0 - 1.0) * 0.4 * h
            jitter_w = (torch.rand(n_jittered, device=device) * 2.0 - 1.0) * 0.3 * w
            jitter_h = (torch.rand(n_jittered, device=device) * 2.0 - 1.0) * 0.3 * h
            
            # Apply jitter and clamp vectorized
            jittered_x = x + jitter_x
            jittered_y = y + jitter_y
            jittered_w = w + jitter_w
            jittered_h = h + jitter_h
            
            # Clamp coordinates and sizes properly
            new_x = torch.clamp_min(jittered_x, 0.0)
            new_y = torch.clamp_min(jittered_y, 0.0)
            new_w = torch.clamp_min(jittered_w, 0.01)
            new_h = torch.clamp_min(jittered_h, 0.01)
            
            # Ensure boxes stay within image bounds
            new_x = torch.minimum(new_x, 1.0 - new_w)
            new_y = torch.minimum(new_y, 1.0 - new_h)
            new_w = torch.minimum(new_w, 1.0 - new_x)
            new_h = torch.minimum(new_h, 1.0 - new_y)
            
            jittered_boxes = torch.stack([new_x, new_y, new_w, new_h], dim=1)
            all_noise_boxes.append(jittered_boxes)
            
            # FIXED: Sample labels from correct sample's label pool
            jittered_labels = []
            for idx in jitter_indices.tolist():
                source_sample = box_to_sample_map[idx]
                # Pix2Seq: noise fragments carry the dedicated "noise" class token
                jittered_labels.append("noise")
                noise_sample_assignments.append(source_sample)
            all_noise_labels.extend(jittered_labels)
        
        # 2. VECTORIZED SHIFTED BOXES
        if len(all_gt_boxes) > 0 and n_shifted > 0:
            shift_indices = torch.randint(0, len(all_gt_boxes), (n_shifted,))
            boxes_to_shift = all_gt_boxes_tensor[shift_indices]
            
            # Extract widths and heights
            w, h = boxes_to_shift[:, 2], boxes_to_shift[:, 3]
            
            # Generate random centers vectorized with consistent device
            device = boxes_to_shift.device
            cx = torch.rand(n_shifted, device=device) * (1.0 - w) + w/2
            cy = torch.rand(n_shifted, device=device) * (1.0 - h) + h/2
            
            # Convert back to (x, y, w, h)
            new_x = max(MIN_COORD, cx - w/2)
            new_y = max(MIN_COORD, cy - h/2)
            
            shifted_boxes = torch.stack([new_x, new_y, w, h], dim=1)
            all_noise_boxes.append(shifted_boxes)
            
            # FIXED: Sample labels from correct sample's label pool
            shifted_labels = []
            for idx in shift_indices.tolist():
                source_sample = box_to_sample_map[idx]
                # Use original label from the source box
                shifted_labels.append("noise")
                noise_sample_assignments.append(source_sample)
            all_noise_labels.extend(shifted_labels)
        
        # 3. VECTORIZED RANDOM BOXES
        if n_random > 0:
            # Generate completely random boxes vectorized with consistent device
            device = all_gt_boxes_tensor.device
            x = torch.rand(n_random, device=device) * 0.8
            y = torch.rand(n_random, device=device) * 0.8
            # Calculate max width/height to stay within bounds
            max_w = torch.minimum(torch.full_like(x, 0.5), 1.0 - x)
            max_h = torch.minimum(torch.full_like(y, 0.5), 1.0 - y)
            w = torch.rand(n_random, device=device) * max_w
            h = torch.rand(n_random, device=device) * max_h
            
            random_boxes = torch.stack([x, y, w, h], dim=1)
            all_noise_boxes.append(random_boxes)
            
            # FIXED: Assign random boxes to samples and sample labels from correct pools
            samples_needing_noise = [idx for idx, count in enumerate(batch_noise_counts) if count > 0]
            for _ in range(n_random):
                # Pick a random sample that needs noise
                target_sample = random.choice(samples_needing_noise)
                # Sample label from that specific sample's label pool
                target_labels = sample_label_pools[target_sample]
                all_noise_labels.append("noise")
                noise_sample_assignments.append(target_sample)
        
        # Combine all noise boxes
        if all_noise_boxes:
            combined_noise_boxes = torch.cat(all_noise_boxes, dim=0)
        else:
            device = all_gt_boxes_tensor.device
            combined_noise_boxes = torch.empty((0, 4), device=device)
        
        # DISTRIBUTE NOISE BACK TO SAMPLES based on assignments
        sample_noise_lists = {idx: ([], []) for idx in range(batch_size) if batch_noise_counts[idx] > 0}
        
        for noise_idx, (box, label, target_sample) in enumerate(zip(combined_noise_boxes, all_noise_labels, noise_sample_assignments)):
            if target_sample in sample_noise_lists:
                sample_noise_lists[target_sample][0].append(box)
                sample_noise_lists[target_sample][1].append(label)
        
        # Convert lists back to tensors and assign to results
        for sample_idx, (boxes_list, labels_list) in sample_noise_lists.items():
            if boxes_list:
                sample_boxes = torch.stack(boxes_list)
                # Truncate to requested count if we generated too many
                requested_count = batch_noise_counts[sample_idx]
                if len(boxes_list) > requested_count:
                    sample_boxes = sample_boxes[:requested_count]
                    labels_list = labels_list[:requested_count]
                batch_results[sample_idx] = (sample_boxes, labels_list)
            else:
                device = batch_gt_boxes[sample_idx].device if batch_gt_boxes[sample_idx].numel() > 0 else torch.device('cpu')
                batch_results[sample_idx] = (torch.empty((0, 4), device=device), [])
        
        return batch_results

    def _generate_noise_boxes(self, num_boxes, label_strs, gt_boxes):
        """Generate noise boxes using Pix2Seq approach: jittered, shifted, and random boxes."""
        if num_boxes <= 0:
            return torch.empty((0, 4)), []
        
        noise_boxes = []
        noise_labels = []
        
        # Split noise into categories like Pix2Seq
        # 40% jittered GT boxes (near duplicates), 30% shifted boxes, 30% random boxes
        n_jittered = max(1, int(num_boxes * 0.4))
        n_shifted = max(1, int(num_boxes * 0.3))  
        n_random = num_boxes - n_jittered - n_shifted
        
        # 1. Jittered boxes (near duplicates of GT boxes)
        if len(gt_boxes) > 0:
            for _ in range(n_jittered):
                # Pick random GT box to jitter
                gt_idx = random.randint(0, len(gt_boxes) - 1)
                x, y, w, h = gt_boxes[gt_idx].tolist()
                
                # Add large jitter to create clear negative examples 
                jitter_x = random.uniform(-0.4, 0.4) * w  # 40% jitter 
                jitter_y = random.uniform(-0.4, 0.4) * h  # 40% jitter
                jitter_w = random.uniform(-0.3, 0.3) * w  # 30% size jitter
                jitter_h = random.uniform(-0.3, 0.3) * h  # 30% size jitter
                
                # Apply jitter and clamp to [0, 1]
                new_x = max(0.0, min(1.0 - w, x + jitter_x))
                new_y = max(0.0, min(1.0 - h, y + jitter_y))
                new_w = max(0.01, min(1.0 - new_x, w + jitter_w))
                new_h = max(0.01, min(1.0 - new_y, h + jitter_h))
                
                noise_boxes.append([new_x, new_y, new_w, new_h])
                # Pix2Seq: dedicated noise label
                noise_labels.append("noise")
        else:
            # No GT boxes available, make them random instead
            n_random += n_jittered
            n_jittered = 0
        
        # 2. Shifted boxes (same size as GT, different position)
        if len(gt_boxes) > 0:
            for _ in range(n_shifted):
                # Pick random GT box to shift
                gt_idx = random.randint(0, len(gt_boxes) - 1)
                _, _, w, h = gt_boxes[gt_idx].tolist()
                
                # Generate new random center position
                cx = random.uniform(w/2, 1.0 - w/2)
                cy = random.uniform(h/2, 1.0 - h/2)
                
                # Convert back to (x, y, w, h)
                new_x = cx - w/2
                new_y = cy - h/2
                
                noise_boxes.append([new_x, new_y, w, h])
                # Use real class token from GT
                noise_labels.append("noise")
        else:
            # No GT boxes available, make them random instead
            n_random += n_shifted
            n_shifted = 0
        
        # 3. Completely random boxes
        for _ in range(n_random):
            # Generate truly random box coordinates
            x = random.uniform(MIN_COORD, 0.8)  # Leave some margin and avoid 0
            y = random.uniform(MIN_COORD, 0.8)
            w = random.uniform(0.1, min(0.5, 1.0 - x))  # Reasonable size
            h = random.uniform(0.1, min(0.5, 1.0 - y))
            
            noise_boxes.append([x, y, w, h])
            # Use real class token
            noise_labels.append("noise")
        
        # Convert to tensor
        noise_boxes_tensor = torch.tensor(noise_boxes, dtype=torch.float32)
        return noise_boxes_tensor, noise_labels

    def _build_and_process_sequence(self, boxes, labels, noise_mask, max_length, _unused=None):
        """Fast fragment→token conversion with perfect Pix2Seq alignment.

        • Tokenise *all* fragments in one batch call (big HF speed-up).
        • Prepend a space token to every fragment except the first (matches
          " ".join()).
        • Build input/target lists in pure Python, then convert to tensors.
        """
        if not labels:
            empty = torch.empty(0, dtype=torch.long, device=boxes.device)
            return empty, empty

        # ---- 1. Build fragment strings ----------------------------------
        fragments = []
        is_noise_list = []
        for bb, lab, is_noise in zip(boxes.tolist(), labels, (noise_mask if noise_mask is not None else [False]*len(labels))):
            coord_txt = ", ".join(self.fmt_coord(v) for v in bb)
            fragments.append(f"<|bbob|>{lab}: [{coord_txt}]</|bbob|>")
            is_noise_list.append(bool(is_noise))

        # ---- 2. Tokenise all fragments in one batch ----------------------
        fragments_pref = [fragments[0]] + [" " + f for f in fragments[1:]]
        tok_lists = self.tokenizer(fragments_pref, add_special_tokens=False).input_ids  # list[list[int]]

        # ---- 3. Build flat input / target lists --------------------------
        inp_flat: list[int] = []
        tgt_flat: list[int] = []
        for toks, is_noise in zip(tok_lists, is_noise_list):
            inp_flat.extend(toks)

            # --------------------------------------------------------------
            # Pix2Seq noise handling:
            #   – Supervise the *class* token(s) of a noise fragment so the
            #     decoder learns to identify a background / "noise" object.
            #   – Ignore (mask) the coordinate tokens so the loss does not
            #     force any particular numbers for the fake box.
            # --------------------------------------------------------------
            if not is_noise:
                # Regular GT fragment → keep full supervision
                tgt_flat.extend(toks)
                continue

            # --- noise fragment ------------------------------------------
            # Detect where the coordinate list begins.  It always starts at
            # the first token that contains a '[' character.
            tok_texts = self.tokenizer.convert_ids_to_tokens(toks)
            coord_start = next((idx for idx, txt in enumerate(tok_texts) if "[" in txt), len(toks))

            # Copy label-and-punctuation tokens up to the '[' (excluded)
            tgt_flat.extend(toks[:coord_start])
            # Mask the coordinate tokens so they do not contribute to loss
            tgt_flat.extend([-100] * (len(toks) - coord_start))

        # ---- 4. Length truncation safety ---------------------------------
        if len(inp_flat) > max_length:
            inp_flat = inp_flat[:max_length]
            tgt_flat = tgt_flat[:max_length]

        # ---- 5. Convert to tensors ---------------------------------------
        input_ids = torch.tensor(inp_flat, dtype=torch.long, device=boxes.device)
        tgt_ids   = torch.tensor(tgt_flat, dtype=torch.long, device=boxes.device)

        return input_ids, tgt_ids

    def _truncate_at_fragment_boundary(self, token_ids, max_length):
        """
        Truncate token sequence at complete fragment boundaries to avoid cutting boxes mid-sequence.
        
        Args:
            token_ids: Token sequence to truncate
            max_length: Maximum allowed length
            
        Returns:
            Truncated token sequence that ends at a complete fragment boundary
        """
        if token_ids.size(0) <= max_length:
            return token_ids
            
        if self.close_id == -1 or self.close_id is None or self.open_id is None:
            # No fragment markers in vocab - fall back to simple truncation
            return token_ids[:max_length]
            
        # Find the last complete fragment that fits within max_length
        tokens = token_ids.tolist()
        last_complete_end = 0
        
        i = 0
        while i < min(len(tokens), max_length):
            if tokens[i] == self.open_id:
                # Found start of fragment - find its end
                depth = 1
                j = i + 1
                while j < len(tokens) and depth > 0:
                    if tokens[j] == self.open_id:
                        depth += 1
                    elif tokens[j] == self.close_id:
                        depth -= 1
                    j += 1
                
                # Check if complete fragment fits within limit
                if j <= max_length:
                    last_complete_end = j
                    i = j
                else:
                    # Fragment would exceed limit - stop here
                    break
            else:
                i += 1
        
        # Truncate to last complete fragment
        if last_complete_end > 0:
            return token_ids[:last_complete_end]
        else:
            # No complete fragments fit - return empty
            return torch.empty(0, dtype=token_ids.dtype, device=token_ids.device)


    def jitter_bboxes_norm(self, bboxes, dtype, jitter_ratio=0.05):
        """Jitter *normalised* (x,y,w,h) boxes in 0‥1 space.

        Each box is perturbed independently:
        – centre moved by up to ``jitter_ratio × w/h``
        – width/height scaled by ±``jitter_ratio``.
        The result is clamped so that the box stays inside 0…1 and keeps
        ``w,h ≥ 0``.
        """
        # ------------------------------------------------------------------
        # Fast vectorised implementation (torch-only, no Python loop).
        # ------------------------------------------------------------------

        # Accept list / np.ndarray inputs *without* moving to GPU – all collate
        # operations are meant to stay on the CPU to avoid needless GPU load.
        if not isinstance(bboxes, torch.Tensor):
            bboxes = torch.as_tensor(bboxes, dtype=dtype)

        if bboxes.numel() == 0:
            return bboxes.to(dtype=dtype)

        # Keep subsequent ops on the same device (should be CPU in dataloader)
        device = bboxes.device  # typically "cpu"
        bx = bboxes.to(dtype=dtype, device=device).clone()

        # centre (cx,cy) and size (w,h)
        cxcy = bx[:, :2] + 0.5 * bx[:, 2:]
        wh   = bx[:, 2:]

        # random jitter on centre: ±jitter_ratio * (w,h)
        trans_rng = (torch.rand_like(cxcy, device=device) * 2.0 - 1.0) * jitter_ratio
        cxcy = cxcy + trans_rng * wh

        # random scaling on size: ±jitter_ratio
        scale_rng = (torch.rand_like(wh, device=device) * 2.0 - 1.0) * jitter_ratio
        wh = wh * (1.0 + scale_rng)

        # back to xywh (top-left origin)
        xy = cxcy - 0.5 * wh
        # Clamp with MIN_COORD lower bound so x/y never hit exactly 0.
        xy = torch.clamp(xy, MIN_COORD, 1.0)

        # clamp sizes so that the box remains inside [0,1] after the xy clamp
        # torch.clamp cannot mix a scalar *min* with a tensor *max* – we therefore
        # do the operation in two steps: first enforce the lower bound (>=0),
        # then the upper bound (<= 1 – coordinate).

        w_nonneg = torch.clamp_min(wh[:, 0], 0.0)
        h_nonneg = torch.clamp_min(wh[:, 1], 0.0)

        w_clamped = torch.minimum(w_nonneg, 1.0 - xy[:, 0])
        h_clamped = torch.minimum(h_nonneg, 1.0 - xy[:, 1])
        wh = torch.stack([w_clamped, h_clamped], dim=1)

        out = torch.cat([xy, wh], dim=1)
        return out

    def _make_batch(self, batch, *, pad_token_id: int, placeholder_id: int, on_the_fly=False):
        """Core functional routine"""

        # 1) locate image field
        img_key = None
        for cand in ("images", "image", "pixel_values"):
            if cand in batch[0]:
                img_key = cand
                break
        if img_key is None:
            raise KeyError("Batch items lack an 'images'/'image'/'pixel_values' field")

        device = "cpu"
        
        if on_the_fly:
            processed = []
            for img in (item[img_key] for item in batch):
                # Convert to PIL Image first
                if isinstance(img, torch.Tensor):
                    # Convert tensor to PIL for consistent processing
                    if img.dim() == 3 and img.shape[0] in (1, 3):
                        img = img.permute(1, 2, 0)  # CHW -> HWC
                    if img.dtype != torch.uint8:
                        img = (img * 255).clamp(0, 255).to(torch.uint8)
                    img = Image.fromarray(img.cpu().numpy())
                elif isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                elif isinstance(img, list):
                    img = Image.fromarray(np.array(img, dtype=np.uint8))
                # else: img is already PIL Image

                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Let the image processor handle all resizing and preprocessing
                try:
                    pv = self.processor(
                        img,
                        return_tensors="pt",
                        do_center_crop=False,
                        do_resize=True,
                        size=TARGET_SIZE,  # Explicitly specify target size
                    )["pixel_values"][0]
                    
                    # Debug: Check the actual output size
                    if self.logger and pv.shape[-2:] != TARGET_SIZE[::-1]:
                        self.logger.warning(f"Image processor output size {pv.shape[-2:]} != expected {TARGET_SIZE[::-1]}")
                    
                    processed.append(pv.to(device))
                except Exception as e:
                    # Log the error to understand why image processor is failing
                    if self.logger:
                        self.logger.warning(f"Image processor failed: {e}. Image type: {type(img)}, size: {img.size}")
                    # Fallback to manual processing
                    t = pil_to_tensor(img).float().div_(255.0).to(device)
                    processed.append(t)

            # Debug: Check tensor shapes before stacking
            if self.logger and len(processed) > 1:
                shapes = [p.shape for p in processed]
                if len(set(shapes)) > 1:
                    self.logger.warning(f"Different tensor shapes before stacking: {shapes}")
            
            pixel_values = torch.stack(processed, 0)
        else:
            # -------------------------------------------------------------
            # Fast-path: images were pre-processed upstream (e.g. by
            # Train.train_common.preprocess_batch) and are stored under
            # `images` / `pixel_values` in CHW format.  We simply stack and
            # normalise.
            # -------------------------------------------------------------
            img_tensors = []
            for itm in batch:
                img_data = itm[img_key]

                if isinstance(img_data, (bytes, bytearray)):
                    # Decode JPEG/PNG bytes back to CHW tensor
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    arr = torch.as_tensor(np.array(img), device=device)  # HWC uint8
                    img_tensors.append(arr.permute(2, 0, 1))  # CHW
                else:
                    img_tensors.append(torch.as_tensor(img_data, device=device))

            # Ensure float32 + 0‥1 range once for the whole list (cheap).
            ref = img_tensors[0]
            needs_cast = ref.dtype != torch.float32 or ref.max() > 1.1
            if needs_cast:
                img_tensors = [t.to(dtype=torch.float32, device=device).div_(255.0) for t in img_tensors]

            pixel_values = torch.stack(img_tensors, 0)

        # OPTIMIZED: Batch-vectorized bbox processing
        # Extract all bboxes from batch at once
        all_bboxes = []
        
        for i, item in enumerate(batch):
            if "target_boxes" in item and not self.is_eval:
                bx_raw = torch.as_tensor(item["target_boxes"], dtype=torch.float32, device=device)
                if bx_raw.numel() > 0:  # Only process non-empty bbox lists
                    all_bboxes.append(bx_raw)
        
        # OPTIMIZED: Vectorized bbox jittering for all bboxes at once
        jittered_bboxes = []
        if all_bboxes:
            # Concatenate all bboxes into one tensor
            combined_bboxes = torch.cat(all_bboxes, dim=0)
            # Jitter all bboxes in one vectorized operation
            jittered_combined = self.jitter_bboxes_norm(combined_bboxes, dtype=torch.float32)
            
            # Split back into per-sample groups
            start_idx = 0
            for i, bx_raw in enumerate(all_bboxes):
                end_idx = start_idx + bx_raw.size(0)
                jittered_bboxes.append(jittered_combined[start_idx:end_idx])
                start_idx = end_idx

        # OPTIMIZED: Prepare data for vectorized noise generation
        batch_gt_boxes = []
        batch_label_strs = []
        batch_noise_counts = []

        for i, item in enumerate(batch):
            if "target_boxes" in item and not self.is_eval:
                bx_raw = torch.as_tensor(item["target_boxes"], dtype=torch.float32, device=device)
                label_strs = item.get("target_label_strs", ["obj"] * bx_raw.size(0))[: bx_raw.size(0)]
                
                # Determine noise count
                if random.random() <= self.noise_prob and not self.is_eval:
                    num_gt_boxes = len(label_strs)
                    if num_gt_boxes > 0:
                        noise_ratio = random.uniform(*self.noise_ratio_range)
                        num_noise = max(1, min(self.max_noise_boxes, int(num_gt_boxes * noise_ratio)))
                    else:
                        num_noise = random.randint(1, min(2, self.max_noise_boxes))
                else:
                    num_noise = 0
                
                batch_gt_boxes.append(bx_raw)
                batch_label_strs.append(label_strs)
                batch_noise_counts.append(num_noise)
            else:
                batch_gt_boxes.append(torch.empty((0, 4)))
                batch_label_strs.append([])
                batch_noise_counts.append(0)

        # OPTIMIZED: Generate noise boxes for the entire batch at once
        batch_noise_results = self._generate_noise_boxes_batch(batch_gt_boxes, batch_label_strs, batch_noise_counts)

        # OPTIMIZED: Pre-allocate output lists with known batch size
        batch_size = len(batch)
        merged_input_ids = [None] * batch_size
        merged_labels = [None] * batch_size

        # CRITICAL: Pre-compute common tokens with correct device specification
        image_placeholder = torch.tensor([placeholder_id], dtype=torch.long, device=device)
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        max_txt_len = self.tokenizer.model_max_length - VIS_TOKENS
        
        # OVERRIDE: Force reasonable sequence limits to prevent OOM
        # Many tokenizers have very high model_max_length (like 1M+) which is impractical
        reasonable_max_length = 2048  # Reasonable limit for detection training
        if max_txt_len > reasonable_max_length - VIS_TOKENS:
            max_txt_len = reasonable_max_length - VIS_TOKENS
            if self.logger and not hasattr(self, '_logged_override'):
                self.logger.warning(f"Overriding tokenizer max_length to reasonable limit: {reasonable_max_length}")
                self._logged_override = True

        # DEBUG: Log sequence length constraints
        if self.logger and hasattr(self, '_logged_constraints'):
            if not self._logged_constraints:
                self.logger.info(f"Sequence constraints: model_max_length={self.tokenizer.model_max_length}, VIS_TOKENS={VIS_TOKENS}, max_txt_len={max_txt_len}")
                self._logged_constraints = True
        elif self.logger and not hasattr(self, '_logged_constraints'):
            self.logger.info(f"Sequence constraints: model_max_length={self.tokenizer.model_max_length}, VIS_TOKENS={VIS_TOKENS}, max_txt_len={max_txt_len}")
            self._logged_constraints = True
        
        # OPTIMIZED: Pre-compute common tensors outside the loop to avoid repeated creation
        bos_tensor = torch.tensor([bos_id], dtype=torch.long, device=device) if bos_id is not None else None
        eos_tensor = torch.tensor([eos_id], dtype=torch.long, device=device) if eos_id is not None else None
        empty_tensor = torch.empty(0, dtype=torch.long, device=device)
        
        # OPTIMIZED: Pre-allocate ignore tensor for visual tokens and reuse
        visual_ignore = torch.full((VIS_TOKENS,), -100, dtype=torch.long, device=device)

        # OPTIMIZED: Batch tokenize all instruction texts at once
        batch_texts_to_tokenize = []
        batch_has_input_ids = []
        for item in batch:
            if "input_ids" in item:
                batch_has_input_ids.append(True)
                batch_texts_to_tokenize.append("")  # Placeholder
            else:
                batch_has_input_ids.append(False)
                batch_texts_to_tokenize.append(item.get("text", ""))
        
        # Pre-tokenize all text instructions individually (batch tokenization needs padding for different lengths)
        texts_needing_tokenization = [text for text, has_ids in zip(batch_texts_to_tokenize, batch_has_input_ids) if not has_ids]
        if texts_needing_tokenization:
            # Tokenize each text individually to avoid length mismatch issues
            individual_tokenized = []
            for text in texts_needing_tokenization:
                tokens = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    max_length=max_txt_len, 
                    truncation=True, 
                    add_special_tokens=False
                )["input_ids"].squeeze(0)
                individual_tokenized.append(tokens)
            tokenized_iter = iter(individual_tokenized)
        else:
            tokenized_iter = iter([])

        bbox_idx = 0
        for i, item in enumerate(batch):
            if batch_has_input_ids[i]:
                instr_ids = torch.as_tensor(item["input_ids"], dtype=torch.long, device=device).flatten()
            else:
                instr_ids = next(tokenized_iter).to(device)  # Move to correct device

            # Decide ground-truth detection text ----------------------------------
            if "target_boxes" in item and not self.is_eval:
                # TRAIN MODE → use pre-jittered bboxes with vectorized noise
                bx_raw = torch.as_tensor(item["target_boxes"], dtype=torch.float32, device=device)
                if bx_raw.numel() > 0 and bbox_idx < len(jittered_bboxes):
                    bx = jittered_bboxes[bbox_idx]
                    bbox_idx += 1
                    
                    # rebuild textual fragments so tokens match jittered coords
                    label_strs = item.get("target_label_strs", ["obj"] * bx.size(0))[: bx.size(0)]

                    # Compute remaining space once and apply same truncation to both sequences
                    remaining_space = max_txt_len - instr_ids.size(0)
                    
                    # OPTIMIZED: Use pre-computed noise from vectorized batch generation
                    noise_boxes, noise_labels = batch_noise_results[i]
                    if noise_boxes.numel() > 0:
                        # Combine GT and noise boxes first
                        combined_boxes  = torch.cat([bx, noise_boxes], dim=0)
                        combined_labels = label_strs + noise_labels
                        noise_boundary  = len(label_strs)
                    else:
                        combined_boxes  = bx
                        combined_labels = label_strs
                        noise_boundary  = len(combined_labels)  # no noise – boundary at end

                    # ----------------------------------------------------
                    # Pix2Seq *tail-noise* ordering:
                    #   1. Shuffle **only** the GT fragments (0‥noise_boundary-1).
                    #   2. Append all noise fragments afterwards so they are
                    #      generated last.  We may optionally shuffle *within*
                    #      the noise tail, but the block itself stays at the
                    #      end of the sequence.
                    # ----------------------------------------------------
                    if combined_boxes.numel() > 0:
                        # --- 1. Permute GT part --------------------------------
                        gt_perm = torch.randperm(noise_boundary, device=combined_boxes.device)

                        # --- 2. Keep noise indices contiguous after GT ---------
                        if len(combined_labels) > noise_boundary:
                            noise_indices = torch.arange(noise_boundary, len(combined_labels), device=combined_boxes.device)
                            # Optionally shuffle inside tail to avoid fixed order
                            # noise_indices = noise_indices[torch.randperm(noise_indices.size(0), device=combined_boxes.device)]
                            new_order = torch.cat([gt_perm, noise_indices], dim=0)
                        else:
                            new_order = gt_perm  # no noise present

                        combined_boxes  = combined_boxes[new_order]
                        combined_labels = [combined_labels[idx] for idx in new_order.tolist()]

                        # Boolean mask: True for positions belonging to noise tail
                        noise_mask = new_order >= noise_boundary
                    else:
                        noise_mask = None

                    # Build fragments and tokenize efficiently
                    input_ids_det, tgt_ids = self._build_and_process_sequence(
                        combined_boxes, combined_labels, noise_mask, remaining_space, None
                    )
                else:
                    # Empty bboxes case
                    input_ids_det = empty_tensor.clone()
                    tgt_ids = empty_tensor.clone()
            else:
                # EVAL MODE or boxes absent – use stored token list as is
                input_ids_det = torch.as_tensor(item.get("target_text", []), dtype=torch.long, device=device).flatten()
                tgt_ids = torch.as_tensor(item.get("target_text", []), dtype=torch.long, device=device).flatten()

            #------ now same instr truncation logic uses tgt_ids variable ----
            input_ids_det = input_ids_det[input_ids_det != pad_token_id]
            tgt_ids = tgt_ids[tgt_ids != pad_token_id]

            if instr_ids.size(0) > max_txt_len:
                instr_ids = instr_ids[-max_txt_len:]
                input_ids_det = empty_tensor.clone()
                tgt_ids = empty_tensor.clone()  # Use pre-computed empty tensor
            else:
                # Length alignment is now handled in _build_and_process_sequence
                # Just ensure both sequences are the same length (safety check)
                min_len = min(input_ids_det.size(0), tgt_ids.size(0))
                input_ids_det = input_ids_det[:min_len]
                tgt_ids = tgt_ids[:min_len]

            # CRITICAL: Handle BOS/EOS tokens consistently to maintain alignment
            
            # --- ensure a single BOS token starts the instruction sequence ---
            if bos_tensor is not None:
                if instr_ids.numel() == 0:
                    # create sequence containing only <bos>
                    instr_ids = bos_tensor.clone()
                elif instr_ids[0] != bos_id:
                    # prepend or replace first token with <bos> depending on space
                    total_space_needed = instr_ids.size(0) + 1 + tgt_ids.size(0)  # +1 for potential BOS
                    if total_space_needed > self.tokenizer.model_max_length:
                        # no room left – overwrite first token
                        instr_ids[0] = bos_id
                    else:
                        instr_ids = torch.cat([bos_tensor, instr_ids])

            # --- ensure a single EOS token terminates the target sequence ---
            if eos_tensor is not None:
                if tgt_ids.numel() == 0:
                    # create sequence containing only <eos>
                    tgt_ids = eos_tensor.clone()
                elif tgt_ids[-1] != eos_id:
                    # append or replace last token with <eos> depending on space
                    total_space_needed = instr_ids.size(0) + tgt_ids.size(0) + 1  # +1 for potential EOS
                    if total_space_needed > self.tokenizer.model_max_length:
                        # no room left – overwrite last token
                        tgt_ids[-1] = eos_id
                    else:
                        tgt_ids = torch.cat([tgt_ids, eos_tensor])

            # SAFETY: Ensure detection sequences are exactly the same length
            # This is critical since input_ids uses input_ids_det and labels uses tgt_ids
            if input_ids_det.size(0) != tgt_ids.size(0):
                min_det_len = min(input_ids_det.size(0), tgt_ids.size(0))
                input_ids_det = input_ids_det[:min_det_len]
                tgt_ids = tgt_ids[:min_det_len]

            # OPTIMIZED: Efficient sequence construction with pre-allocated tensors
            # NOTE: input_ids still uses placeholder (1 token) which the model will replace with VIS_TOKENS visual tokens
            
            # SAFETY: Final check for total sequence length before concatenation
            total_length = 1 + instr_ids.size(0) + input_ids_det.size(0)  # placeholder + instruction + detection
            if total_length > 2048:  # Hard limit to prevent OOM
                if self.logger:
                    self.logger.warning(f"Final sequence too long ({total_length} tokens), truncating detection part")
                # Aggressively truncate detection sequence to fit
                max_det_len = 2048 - 1 - instr_ids.size(0)
                if max_det_len > 0:
                    input_ids_det = input_ids_det[:max_det_len]
                    tgt_ids = tgt_ids[:max_det_len]
                else:
                    # If instruction is too long, clear detection entirely
                    input_ids_det = empty_tensor.clone()
                    tgt_ids = empty_tensor.clone()
            
            # OPTIMIZED: Single tensor allocation for input_ids
            input_ids = torch.cat([image_placeholder, instr_ids, input_ids_det], dim=0)
            
            # OPTIMIZED: Efficient labels construction using pre-allocated tensors
            # Account for visual token replacement: 1 placeholder becomes VIS_TOKENS visual tokens
            instr_size = instr_ids.size(0)
            
            # Create ignore labels for instruction (reuse pre-allocated visual_ignore)
            if instr_size > 0:
                instr_ignore = torch.full((instr_size,), -100, dtype=torch.long, device=device)
                labels = torch.cat([visual_ignore, instr_ignore, tgt_ids], dim=0)
            else:
                labels = torch.cat([visual_ignore, tgt_ids], dim=0)

            # OPTIMIZED: Direct assignment instead of append
            merged_input_ids[i] = input_ids
            merged_labels[i] = labels

        input_ids_padded = pad_sequence(merged_input_ids, batch_first=True, padding_value=pad_token_id)

        # -------------------------------------------------------------
        # Spam-penalty: positions that are added only by batch padding
        # now *expect* the first EOS token.  Any non-EOS prediction
        # therefore incurs cross-entropy, so the model is encouraged to
        # stop once it has reproduced all ground-truth boxes.
        # (Visual tokens and instruction prefix remain -100 and are still
        # ignored by the loss.)
        # -------------------------------------------------------------
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_value = eos_id if eos_id is not None else -100
        labels_padded = pad_sequence(merged_labels, batch_first=True, padding_value=pad_value)

        attention_mask = (input_ids_padded != pad_token_id).long()

        batch_out = {
            "images": pixel_values,
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
        }

        return batch_out


def make_collate_fn(pad_token_id: int, tokenizer, image_processor, **kwargs):
    """Create a collator.  Pass-through extra kwargs such as
    `noise_prob`, `max_noise_boxes`, `noise_ratio_range` for optional features."""

    return BBOBCollator(
        pad_token_id,
        tokenizer,
        image_processor,
        on_the_fly=kwargs.get("on_the_fly", False),
        noise_prob=kwargs.get("noise_prob", 1),
        max_noise_boxes=kwargs.get("max_noise_boxes", 10),
        noise_ratio_range=kwargs.get("noise_ratio_range", (1, 1.25)),
        logger=kwargs.get("logger"),
    ) 
