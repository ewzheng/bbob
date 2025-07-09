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
from .loss_helpers import TAG_OPEN, TAG_CLOSE
import random
import torch.utils.data as tud

# Constants (kept in sync with Train.train_common)
VIS_TOKENS: int = 64           # number of visual tokens the model prepends
TARGET_SIZE = (256, 256)       # spatial resolution used by MobileViT-v2

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
import numpy as np


class BBOBCollator:  # noqa: N801
    """Minimal collator for BBOB – no teacher-forcing logic."""

    def __init__(
        self,
        pad_token_id: int,
        tokenizer,
        image_processor,
        *,
        logger=None,
        log_interval=128,
        on_the_fly=False,
        noise_prob=0.3,  # Probability of adding noise boxes
        max_noise_boxes=10,  # Max number of noise boxes to add
        use_noise_class=False,  # Use dedicated "noise" class instead of GT labels
        noise_ratio_range=(0.25, 0.75),  # Range for noise count as fraction of GT count
        **kwargs,  # accept legacy tf_* kwargs but ignore them
    ):
        self.pad_id = pad_token_id
        self.tokenizer = tokenizer
        self.processor = image_processor
        self.logger = logger

        self.rngjesus = np.random.default_rng()

        # teacher-forcing parameters removed – keep placeholders for API compat
        self.log_interval = log_interval
        self.is_eval = False
        self.on_the_fly = on_the_fly
        self.noise_prob = noise_prob
        self.max_noise_boxes = max_noise_boxes
        self.use_noise_class = use_noise_class
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
        self.space_id = tokenizer.convert_tokens_to_ids(" ")

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

    def _generate_noise_boxes(self, num_boxes, label_strs, device):
        """Generate random noisy bounding boxes for noise injection."""
        if num_boxes <= 0:
            return [], []
        
        noise_boxes = []
        noise_labels = []
        
        for _ in range(num_boxes):
            # Generate random box coordinates (x, y, w, h) in [0, 1]
            x = random.uniform(0.0, 0.8)
            y = random.uniform(0.0, 0.8) 
            w = random.uniform(0.1, min(0.3, 1.0 - x))
            h = random.uniform(0.1, min(0.3, 1.0 - y))
            
            noise_boxes.append([x, y, w, h])
            
            # Pick a random label from existing ground truth labels, or use "noise"
            if self.use_noise_class or not label_strs or random.random() > 0.7:
                # Use dedicated noise class
                noise_labels.append("object")  # Generic noise label
            else:
                # Sample from GT labels (makes task harder)
                noise_labels.append(random.choice(label_strs))
        
        return noise_boxes, noise_labels

    def _mask_noise_tokens(self, tgt_ids, target_text, combined_labels, noise_mask):
        """
        Selectively mask tokens corresponding to noise boxes in the target sequence.
        
        This implements the Pix2Seq approach: input has GT+noise mixed, target has
        GT content but noise positions are masked with -100.
        
        Args:
            tgt_ids: Target token sequence to modify in-place
            target_text: Full target text string  
            combined_labels: List of all labels (GT + noise)
            noise_mask: Boolean tensor indicating which positions are noise
        """
        if not any(noise_mask):
            return  # No noise to mask
            
        # PRECISE APPROACH: Tokenize the actual target text fragments to get exact positions
        
        # Split target text into individual detection fragments using regex to preserve structure
        import re
        
        # Find all detection fragments: <|bbob|>label: [coords]</|bbob|>
        # Improved regex to handle spaces and coordinate content properly
        fragment_pattern = r'<\|bbob\|>[^<]*?</\|bbob\|>'
        matches = list(re.finditer(fragment_pattern, target_text))
        
        # Tokenize each fragment to find exact token boundaries
        cumulative_tokens = 0
        
        for frag_idx, match in enumerate(matches):
            if frag_idx >= len(noise_mask):
                break
                
            fragment_text = match.group(0)  # Full fragment including tags
            
            # Tokenize this exact fragment
            frag_tokens = self.tokenizer(fragment_text, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
            frag_token_count = len(frag_tokens)
            
            if noise_mask[frag_idx]:
                # This fragment is noise - mask its tokens
                start_idx = cumulative_tokens
                end_idx = min(start_idx + frag_token_count, len(tgt_ids))
                tgt_ids[start_idx:end_idx] = -100
            
            cumulative_tokens += frag_token_count
            
            # Safety check to avoid going beyond sequence length
            if cumulative_tokens >= len(tgt_ids):
                break
                
        # Handle any space tokens between fragments
        # The above handles the main detection tokens, spaces are typically not significant



    def jitter_bboxes_norm(self, bboxes, dtype, jitter_ratio=0.15):
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
        xy = torch.clamp(xy, 0.0, 1.0)

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

    def _shuffle_fragments(self, ids: torch.Tensor) -> torch.Tensor:
        """Return a copy with <bbob> … </bbob> snippets shuffled."""
        if self.open_id == -1 or self.close_id == -1:
            return ids  # tags not in vocab → nothing to shuffle

        seq = ids.tolist()
        # Find first <bbob>
        ptr = 0
        prefix = []
        while ptr < len(seq) and seq[ptr] != self.open_id:
            prefix.append(seq[ptr])
            ptr += 1

        if ptr == len(seq):
            return ids  # no fragments

        fragments: list[list[int]] = []
        while ptr < len(seq):
            if seq[ptr] != self.open_id:
                break
            start = ptr
            depth = 1
            ptr += 1
            while ptr < len(seq) and depth > 0:
                if seq[ptr] == self.open_id:
                    depth += 1
                elif seq[ptr] == self.close_id:
                    depth -= 1
                ptr += 1
            frag = seq[start:ptr]
            fragments.append(frag)
            # skip whitespace between fragments (kept as single space later)
            while ptr < len(seq) and seq[ptr] == self.space_id:
                ptr += 1

        suffix = seq[ptr:]

        if len(fragments) <= 1:
            return ids  # nothing to shuffle

        self.rngjesus.shuffle(fragments)

        new_seq = prefix[:]
        first = True
        for frag in fragments:
            if not first and self.space_id != -1:
                new_seq.append(self.space_id)
            new_seq.extend(frag)
            first = False
        new_seq.extend(suffix)
        return torch.tensor(new_seq, dtype=torch.long)


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
                try:
                    pv = self.processor(img, return_tensors="pt", do_center_crop=False)["pixel_values"][0]  # (3,256,256) float32 0-1
                    processed.append(pv.to(device))  # Ensure on correct device
                    continue  # success – skip manual fallback
                except Exception:
                    # Fallback to legacy manual branch for rare failures
                    pass

                # ---------------- manual fallback -------------------
                if isinstance(img, torch.Tensor):
                    t = img.to(dtype=torch.float32, device=device).div_(255.0)
                elif isinstance(img, np.ndarray):
                    t = torch.as_tensor(img, dtype=torch.float32, device=device).div_(255.0)
                elif isinstance(img, list):
                    t = torch.as_tensor(np.array(img, dtype=np.uint8), dtype=torch.float32, device=device).div_(255.0)
                else:
                    t = pil_to_tensor(img).float().div_(255.0).to(device)

                # channel / shape normalisation (same as old code)
                if t.dim() == 2:
                    t = t.unsqueeze(0).expand(3, -1, -1)
                elif t.dim() == 3 and t.shape[0] not in (1, 3):
                    t = t.permute(2, 0, 1)
                    if t.shape[0] == 1:
                        t = t.expand(3, -1, -1)
                elif t.dim() != 3:
                    raise RuntimeError(f"Unsupported image tensor dim {t.dim()}")

                _, H, W = t.shape
                if (H, W) != TARGET_SIZE:
                    scale = min(TARGET_SIZE[1] / H, TARGET_SIZE[0] / W)
                    nh = max(1, int(H * scale))
                    nw = max(1, int(W * scale))
                    t = F.interpolate(t.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False)[0]

                    canvas = 0.5 * torch.ones(3, *TARGET_SIZE, device=device)  # FIXED: specify device
                    dh = (TARGET_SIZE[1] - nh) // 2
                    dw = (TARGET_SIZE[0] - nw) // 2
                    canvas[:, dh : dh + nh, dw : dw + nw] = t
                    t = canvas

                processed.append(t)

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
                    import io
                    from PIL import Image
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
        # Extract all bboxes and labels from batch at once
        all_bboxes = []
        all_labels = []
        batch_indices = []  # Track which sample each bbox belongs to
        
        for i, item in enumerate(batch):
            if "target_boxes" in item and not self.is_eval:
                bx_raw = torch.as_tensor(item["target_boxes"], dtype=torch.float32, device=device)
                if bx_raw.numel() > 0:  # Only process non-empty bbox lists
                    all_bboxes.append(bx_raw)
                    label_strs = item.get("target_label_strs", ["obj"] * bx_raw.size(0))[: bx_raw.size(0)]
                    all_labels.extend(label_strs)
                    batch_indices.extend([i] * bx_raw.size(0))
        
        # Vectorized bbox jittering for all bboxes at once
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

        merged_input_ids, merged_labels = [], []

        # CRITICAL: Pre-compute common tokens with correct device specification
        image_placeholder = torch.tensor([placeholder_id], dtype=torch.long, device=device)
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        max_txt_len = self.tokenizer.model_max_length - VIS_TOKENS
        
        # OPTIMIZED: Pre-compute common tensors outside the loop to avoid repeated creation
        bos_tensor = torch.tensor([bos_id], dtype=torch.long, device=device) if bos_id is not None else None
        eos_tensor = torch.tensor([eos_id], dtype=torch.long, device=device) if eos_id is not None else None
        empty_tensor = torch.empty(0, dtype=torch.long, device=device)

        # OPTIMIZED: Pre-compute coordinate formatter to avoid lambda creation in loop
        def fmt_coord(v):
            return f"{max(0.0, min(1.0, v)):.3f}"

        bbox_idx = 0
        for i, item in enumerate(batch):
            if "input_ids" in item:
                instr_ids = torch.as_tensor(item["input_ids"], dtype=torch.long, device=device).flatten()
            else:
                text = item.get("text", "")
                tokens = self.tokenizer(text, return_tensors="pt", max_length=max_txt_len, truncation=True)
                instr_ids = tokens["input_ids"].squeeze(0).to(device)

            # Decide ground-truth detection text ----------------------------------
            if "target_boxes" in item and not self.is_eval:
                # TRAIN MODE → use pre-jittered bboxes
                bx_raw = torch.as_tensor(item["target_boxes"], dtype=torch.float32, device=device)
                if bx_raw.numel() > 0 and bbox_idx < len(jittered_bboxes):
                    bx = jittered_bboxes[bbox_idx]
                    bbox_idx += 1
                    
                    # rebuild textual fragments so tokens match jittered coords
                    label_strs = item.get("target_label_strs", ["obj"] * bx.size(0))[: bx.size(0)]

                    frags = [
                        f"<|bbob|>{lab}: [{', '.join(fmt_coord(v) for v in bb)}]</|bbob|>"
                        for bb, lab in zip(bx.tolist(), label_strs)
                    ]
                    gt_det_text = " ".join(frags)
                    
                    # Generate noisy input vs clean target with proper alignment
                    if random.random() < self.noise_prob and not self.is_eval:
                        # IMPROVED: Pix2Seq-style noisy sequencing with selective masking
                        # Create aligned input/target sequences where only noise positions are masked
                        
                        # IMPROVED: Generate noise boxes proportional to GT count
                        # Examples with noise_ratio_range=(0.25, 0.75):
                        # - 1 GT box → 1 noise box (25-75% of 1, min 1)
                        # - 4 GT boxes → 1-3 noise boxes (25-75% of 4 = 1-3)  
                        # - 8 GT boxes → 2-6 noise boxes (25-75% of 8 = 2-6, capped by max_noise_boxes)
                        # - 0 GT boxes → 1-2 noise boxes (fallback case)
                        num_gt_boxes = len(label_strs)
                        if num_gt_boxes > 0:
                            # Add noise proportional to GT count (configurable range)
                            noise_ratio = random.uniform(*self.noise_ratio_range)
                            num_noise = max(1, min(self.max_noise_boxes, int(num_gt_boxes * noise_ratio)))
                        else:
                            # If no GT boxes, add 1-2 noise boxes
                            num_noise = random.randint(1, min(2, self.max_noise_boxes))
                        
                        # Optional logging for debugging noise generation
                        if self.logger:
                            self._noise_log_counter = getattr(self, '_noise_log_counter', 0) + 1
                            if self._noise_log_counter % 100 == 0:  # Log every 100 samples
                                self.logger.info(f"Noise stats: {num_gt_boxes} GT boxes → {num_noise} noise boxes (ratio: {noise_ratio:.2f})")
                        noise_boxes, noise_labels = self._generate_noise_boxes(num_noise, label_strs, device)
                        
                        # Create combined sequence with GT and noise mixed together
                        combined_boxes = torch.cat([bx, noise_boxes], dim=0)
                        combined_labels = label_strs + noise_labels
                        
                        # Create shuffled order for mixing GT and noise
                        num_total = len(combined_labels)
                        shuffle_indices = torch.randperm(num_total)
                        
                        # Apply shuffle to both boxes and labels
                        combined_boxes = combined_boxes[shuffle_indices]
                        combined_labels = [combined_labels[i] for i in shuffle_indices.tolist()]
                        
                        # Track which positions are noise (after shuffling)
                        noise_mask = shuffle_indices >= len(label_strs)  # True for noise positions
                        
                        # Build combined input sequence
                        input_frags = [
                            f"<|bbob|>{lab}: [{', '.join(fmt_coord(v) for v in bb)}]</|bbob|>"
                            for bb, lab in zip(combined_boxes.tolist(), combined_labels)
                        ]
                        input_det_text = " ".join(input_frags)
                        
                        # Build target sequence with IDENTICAL structure to input
                        # KEY INSIGHT: Input and target have same text, but we selectively mask noise tokens
                        # This is the core of Pix2Seq's approach: learn to generate GT from noisy input
                        target_det_text = input_det_text
                        
                        # Tokenize both sequences (they're identical)
                        input_ids_det = self.tokenizer(input_det_text, return_tensors="pt", truncation=False)["input_ids"].squeeze(0).to(device)
                        target_ids_det = self.tokenizer(target_det_text, return_tensors="pt", truncation=False)["input_ids"].squeeze(0).to(device)
                        
                        # Create selective mask: mark ONLY noise positions as -100
                        tgt_ids = target_ids_det.clone()
                        
                        # Find token spans corresponding to noise boxes and mask them
                        self._mask_noise_tokens(tgt_ids, target_det_text, combined_labels, noise_mask)
                        
                        # Don't shuffle fragments when using aligned sequences
                        # input_ids_det already contains the mixed sequence
                        
                    else:
                        # No noise injection - normal training
                        input_det_text = gt_det_text
                        target_det_text = gt_det_text
                        
                        # Tokenize INPUT sequence (clean)
                        input_ids_det = self.tokenizer(input_det_text, return_tensors="pt", truncation=False)["input_ids"].squeeze(0).to(device)
                        input_ids_det = self._shuffle_fragments(input_ids_det)
                        
                        # Tokenize TARGET sequence (same as input when no noise)  
                        tgt_ids = self.tokenizer(target_det_text, return_tensors="pt", truncation=False)["input_ids"].squeeze(0).to(device)
                        tgt_ids = self._shuffle_fragments(tgt_ids)
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
                # Handle input sequence (may be longer due to noise)
                remaining_input = max_txt_len - instr_ids.size(0)
                if input_ids_det.size(0) > remaining_input:
                    input_ids_det = input_ids_det[:remaining_input]
                
                # Handle target sequence  
                remaining_target = max_txt_len - instr_ids.size(0)
                if tgt_ids.size(0) > remaining_target:
                    tgt_ids = tgt_ids[:remaining_target]

                # CRITICAL: Fix device mismatches in BOS/EOS token creation
                # --- ensure a single BOS token starts the instruction sequence ---
                if bos_tensor is not None:
                    if instr_ids.numel() == 0:
                        # create sequence containing only <bos>
                        instr_ids = bos_tensor.clone()
                    elif instr_ids[0] != bos_id:
                        # prepend or replace first token with <bos> depending on space
                        if instr_ids.size(0) >= self.tokenizer.model_max_length - tgt_ids.size(0):
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
                        if tgt_ids.size(0) >= self.tokenizer.model_max_length - instr_ids.size(0):
                            # no room left – overwrite last token
                            tgt_ids[-1] = eos_id
                        else:
                            tgt_ids = torch.cat([tgt_ids, eos_tensor])

            # OPTIMIZED: Concatenate all at once instead of multiple torch.cat calls
            # NOTE: input_ids still uses placeholder (1 token) which the model will replace with VIS_TOKENS visual tokens
            input_ids = torch.cat([image_placeholder, instr_ids, input_ids_det], dim=0)
            
            # OPTIMIZED: Create labels more efficiently using sizes and correct device
            # Account for visual token replacement: 1 placeholder becomes VIS_TOKENS visual tokens
            instr_size = instr_ids.size(0)
            
            # Create ignore labels for visual tokens and instruction
            # Visual tokens don't need supervision - they're just processed features
            total_prefix_size = VIS_TOKENS + instr_size  # visual tokens + instruction
            label_ignore = torch.full((total_prefix_size,), -100, dtype=torch.long, device=device)
            labels = torch.cat([label_ignore, tgt_ids], dim=0)

            merged_input_ids.append(input_ids)
            merged_labels.append(labels)

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
        noise_prob=kwargs.get("noise_prob", 0.3),
        max_noise_boxes=kwargs.get("max_noise_boxes", 10),
        use_noise_class=kwargs.get("use_noise_class", False),
        noise_ratio_range=kwargs.get("noise_ratio_range", (0.25, 0.75)),
        logger=kwargs.get("logger"),
        log_interval=kwargs.get("log_interval", 0),
    ) 
