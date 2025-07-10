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
import re

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
        on_the_fly=False,
        noise_prob=0.0,  # TEMPORARILY DISABLED: Probability of adding noise boxes
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
        # Note: space_id no longer needed - tokenizer handles spacing naturally

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

    def _generate_noise_boxes(self, num_boxes, label_strs):
        """Generate random noisy bounding boxes for noise injection."""
        if num_boxes <= 0:
            return torch.empty((0, 4)), []
        
        noise_boxes = []
        noise_labels = []
        
        for _ in range(num_boxes):
            # Generate truly random box coordinates (x, y, w, h) in full [0, 1] range
            # This creates maximally challenging noise for the model to learn to ignore
            x = random.uniform(0.0, 1.0)
            y = random.uniform(0.0, 1.0) 
            w = random.uniform(0.0, 1.0 - x)  # Ensure box stays within image bounds
            h = random.uniform(0.0, 1.0 - y)  # Ensure box stays within image bounds
            
            noise_boxes.append([x, y, w, h])
            
            # Pick a random label from existing ground truth labels, or use "noise"
            if self.use_noise_class or not label_strs or random.random() > 0.7:
                # Use dedicated noise class
                noise_labels.append("object")  # Generic noise label
            else:
                # Sample from GT labels (makes task harder)
                noise_labels.append(random.choice(label_strs))
        
        # Convert to tensor for concatenation with GT boxes
        noise_boxes_tensor = torch.tensor(noise_boxes, dtype=torch.float32)
        return noise_boxes_tensor, noise_labels

    def _build_and_process_sequence(self, boxes, labels, noise_mask, max_length, fmt_coord):
        """
        Efficiently build, tokenize, truncate and mask detection sequence in one pass.
        
        Args:
            boxes: Tensor of box coordinates 
            labels: List of label strings
            noise_mask: Boolean tensor indicating noise positions (None if no noise)
            max_length: Maximum sequence length allowed
            fmt_coord: Coordinate formatting function
            
        Returns:
            input_ids_det, tgt_ids: Processed token sequences
        """
        if len(labels) == 0:
            empty = torch.empty(0, dtype=torch.long, device=boxes.device)
            return empty.clone(), empty.clone()
        
        # Build fragments efficiently 
        fragments = [
            f"<|bbob|>{lab}: [{', '.join(fmt_coord(v) for v in bb)}]</|bbob|>"
            for bb, lab in zip(boxes.tolist(), labels)
        ]
        
        # Apply shuffling if no noise (fragment-level shuffling)
        if noise_mask is None:
            # Shuffle fragments for no-noise case
            indices = torch.randperm(len(fragments))
            fragments = [fragments[i] for i in indices.tolist()]
        
        # Build full text with natural spaces (let tokenizer handle spacing)
        full_text = " ".join(fragments)
        
        # Tokenize the full text once
        full_tokens = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
        
        # SAFETY: Emergency sequence length check to prevent OOM
        if len(full_tokens) > 8192:  # Emergency limit well beyond typical sequences
            if self.logger:
                self.logger.warning(f"Extremely long sequence detected ({len(full_tokens)} tokens), truncating aggressively")
                self.logger.warning(f"Fragment count: {len(fragments)}, text preview: {full_text[:200]}...")
            # Force truncation to safe length
            truncated_tokens = full_tokens[:min(max_length, 4096)]
        elif len(full_tokens) > max_length:
            # Use fragment-boundary truncation if possible
            truncated_tokens = self._truncate_at_fragment_boundary(full_tokens, max_length)
            if self.logger and len(full_tokens) > max_length * 2:  # Log if significantly over limit
                self.logger.warning(f"Long sequence truncated: {len(full_tokens)} -> {len(truncated_tokens)} tokens")
        else:
            truncated_tokens = full_tokens
        
        input_ids_det = truncated_tokens.to(boxes.device)
        tgt_ids = input_ids_det.clone()
        
        # Apply noise masking if needed - NEW APPROACH: Direct token-level masking
        if noise_mask is not None and len(input_ids_det) > 0:
            # Instead of decoding/re-encoding (which corrupts coordinates),
            # mask tokens directly by finding fragment boundaries in the token sequence
            self._mask_noise_tokens_direct(tgt_ids, fragments, noise_mask)
        
        return input_ids_det, tgt_ids

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



    def _mask_noise_tokens_direct(self, tgt_ids, fragments, noise_mask):
        """
        Selectively mask tokens corresponding to noise boxes in the target sequence.
        
        This implements the Pix2Seq approach: input has GT+noise mixed, target has
        GT content but noise positions are masked with -100.
        
        Args:
            tgt_ids: Target token sequence to modify in-place
            fragments: List of strings, each representing a detection fragment
            noise_mask: Boolean tensor indicating which positions are noise
        """
        if not any(noise_mask):
            return  # No noise to mask

        # MUCH SIMPLER APPROACH: Tokenize each fragment individually and track positions
        current_pos = 0
        
        for frag_idx, frag_str in enumerate(fragments):
            if frag_idx >= len(noise_mask):
                break
                
            # Tokenize this fragment to get its length
            frag_tokens = self.tokenizer(frag_str, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
            frag_len = len(frag_tokens)
            
            # Check if we need to account for space tokens
            # The space comes BEFORE this fragment (except for the first one)
            if frag_idx > 0:
                # Add space token(s) - tokenize a single space to see how many tokens it creates
                space_tokens = self.tokenizer(" ", return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
                space_len = len(space_tokens)
                current_pos += space_len
            
            # Now handle the fragment itself
            if noise_mask[frag_idx] and current_pos < len(tgt_ids):
                # This fragment is noise - mask its tokens
                end_pos = min(current_pos + frag_len, len(tgt_ids))
                tgt_ids[current_pos:end_pos] = -100
            
            current_pos += frag_len
            
            # Safety check to avoid going beyond sequence
            if current_pos >= len(tgt_ids):
                break


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
        # Extract all bboxes from batch at once
        all_bboxes = []
        
        for i, item in enumerate(batch):
            if "target_boxes" in item and not self.is_eval:
                bx_raw = torch.as_tensor(item["target_boxes"], dtype=torch.float32, device=device)
                if bx_raw.numel() > 0:  # Only process non-empty bbox lists
                    all_bboxes.append(bx_raw)
        
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

                    # Compute remaining space once and apply same truncation to both sequences
                    remaining_space = max_txt_len - instr_ids.size(0)
                    
                    # STREAMLINED PIPELINE: Build → Shuffle → Truncate+Mask in one pass
                    if random.random() < self.noise_prob and not self.is_eval:
                        # Generate noise and create combined box+label lists
                        num_gt_boxes = len(label_strs)
                        if num_gt_boxes > 0:
                            noise_ratio = random.uniform(*self.noise_ratio_range)
                            num_noise = max(1, min(self.max_noise_boxes, int(num_gt_boxes * noise_ratio)))
                        else:
                            num_noise = random.randint(1, min(2, self.max_noise_boxes))
                        
                        noise_boxes, noise_labels = self._generate_noise_boxes(num_noise, label_strs)
                        combined_boxes = torch.cat([bx, noise_boxes], dim=0)
                        combined_labels = label_strs + noise_labels
                        
                        # Shuffle at box level
                        shuffle_indices = torch.randperm(len(combined_labels))
                        combined_boxes = combined_boxes[shuffle_indices]
                        combined_labels = [combined_labels[i] for i in shuffle_indices.tolist()]
                        noise_mask = shuffle_indices >= len(label_strs)
                        
                        # Build fragments and tokenize efficiently
                        input_ids_det, tgt_ids = self._build_and_process_sequence(
                            combined_boxes, combined_labels, noise_mask, remaining_space, fmt_coord
                        )
                        
                    else:
                        # No noise - simpler path
                        input_ids_det, tgt_ids = self._build_and_process_sequence(
                            bx, label_strs, None, remaining_space, fmt_coord
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
        noise_prob=kwargs.get("noise_prob", 1),
        max_noise_boxes=kwargs.get("max_noise_boxes", 10),
        use_noise_class=kwargs.get("use_noise_class", False),
        noise_ratio_range=kwargs.get("noise_ratio_range", (0.25, 0.75)),
        logger=kwargs.get("logger"),
    ) 
