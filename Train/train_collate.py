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
from .train_common import DEFAULT_TARGET_SIZE, letterbox_image
from functools import lru_cache

# ---------------------------------------------------------------------------
# Helper utilities 
# ---------------------------------------------------------------------------
from .det_helpers import (
    _generate_noise_boxes_batch as _det_generate_noise_boxes_batch,
    _generate_noise_boxes as _det_generate_noise_boxes,
    _build_and_process_sequence as _det_build_and_process_sequence,
    _truncate_at_fragment_boundary as _det_truncate_at_fragment_boundary,
    jitter_bboxes_norm as det_jitter_bboxes_norm,
)

# Constants (default value only – actual value comes from model.vis_length)
DEFAULT_VIS_TOKENS: int = 64   # fallback if not specified

# ---------------------------------------------------------------------------
# Coordinate safety margin – keep jittered / synthetic boxes away from exactly
# 0.000 so that the model does not over-fit to the left/top border token.
# Matches the 0.001 resolution of the numeric vocabulary.
# ---------------------------------------------------------------------------
MIN_COORD = 1.0 / 1000.0  # 0.001

import math


class BBOBCollator:  # noqa: N801
    """Minimal collator for BBOB, performs pix2seq GT shuffling for detection.
    
        On the fly mode processes images at runtime.
    """

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
        vis_tokens: int | None = None,
        **kwargs,  # accept legacy tf_* kwargs but ignore them
    ):
        self.pad_id = pad_token_id
        self.tokenizer = tokenizer
        self.processor = image_processor
        self.logger = logger

        self.rngjesus = np.random.default_rng()

      
        self.is_eval = False
        self.on_the_fly = on_the_fly
        self.noise_prob = noise_prob
        self.max_noise_boxes = max_noise_boxes
        self.noise_ratio_range = noise_ratio_range

        # Visual-token count – fall back to default when not provided
        self.vis_tokens = int(vis_tokens) if vis_tokens is not None else DEFAULT_VIS_TOKENS

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
        self._coord_cache_size = 1000  # Cache up to 1000 unique coordinate values
        self._fmt_coord_cached = lru_cache(maxsize=self._coord_cache_size)(self._fmt_coord_impl)

    # OPTIMIZATION: Cached coordinate formatting implementation
    def _fmt_coord_impl(self, v):
        """Internal implementation of coordinate formatting with caching."""
        return format_coordinate(v)
    
    def fmt_coord(self, v):
        """Cached coordinate formatting method."""
        return self._fmt_coord_cached(float(v))  # Ensure hashable input for cache

    # ------------------------------------------------------------------
    # Helper: shuffle GT and noise fragments independently, then concat.
    # ------------------------------------------------------------------
    def _merge_shuffle_gt_noise(
        self,
        gt_boxes: torch.Tensor,
        gt_labels: list[str],
        noise_boxes: torch.Tensor,
        noise_labels: list[str],
    ) -> tuple[torch.Tensor, list[str], torch.Tensor | None]:
        """Return shuffled GT + noise tensors and a Boolean noise mask.

        GT and noise fragments are shuffled *within* their groups.  The
        concatenated output always places GT first, noise afterwards so the
        decoder still emits noise fragments last (Pix2Seq convention).
        """

        # PATCH 1: determine device robustly even when both tensors are empty
        if gt_boxes.numel() > 0:
            device = gt_boxes.device
        elif noise_boxes.numel() > 0:
            device = noise_boxes.device
        else:
            device = torch.device("cpu")  # safe fallback when both inputs empty

        # shuffle GT
        if gt_boxes.numel() > 0:
            perm_gt = torch.randperm(gt_boxes.size(0), device=device)
            gt_boxes = gt_boxes[perm_gt]
            gt_labels = [gt_labels[idx] for idx in perm_gt.tolist()]

        # shuffle noise
        if noise_boxes.numel() > 0:
            perm_noise = torch.randperm(noise_boxes.size(0), device=device)
            noise_boxes = noise_boxes[perm_noise]
            noise_labels = [noise_labels[idx] for idx in perm_noise.tolist()]

        if gt_boxes.numel() == 0 and noise_boxes.numel() == 0:
            return torch.empty((0, 4), device=device), [], None

        boxes_out = torch.cat([gt_boxes, noise_boxes], dim=0)
        labels_out = gt_labels + noise_labels

        # noise mask: False for GT positions, True for noise
        noise_mask = torch.tensor(
            [False] * gt_boxes.size(0) + [True] * noise_boxes.size(0),
            dtype=torch.bool,
            device=device,
        )

        return boxes_out, labels_out, noise_mask

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

    def reseed(self, seed: int | None = None):
        """Reseed the internal NumPy RNG.

        Parameters
        ----------
        seed : int | None, default None
            If *None* we draw a fresh seed from ``np.random.SeedSequence`` so
            every worker gets a unique stream.  Otherwise we use the provided
            integer for deterministic behaviour (useful in tests).
        """

        if seed is None:
            # Derive a unique seed based on the global RNG state so that each
            # dataloader worker (or epoch) gets an independent stream.
            seed = np.random.SeedSequence().entropy

        # Re-initialise the generator with the new seed
        self.rngjesus = np.random.default_rng(int(seed))

        # ------------------------------------------------------------------
    # Compact wrappers delegating heavy-lifting to Train.det_helpers ------
        # ------------------------------------------------------------------

    def _generate_noise_boxes_batch(self, batch_gt_boxes, batch_label_strs, batch_noise_counts):
        return _det_generate_noise_boxes_batch(self, batch_gt_boxes, batch_label_strs, batch_noise_counts)

    def _generate_noise_boxes(self, num_boxes, label_strs, gt_boxes):
        return _det_generate_noise_boxes(self, num_boxes, label_strs, gt_boxes)

    def _build_and_process_sequence(self, boxes, labels, noise_mask, max_length, _unused=None):
        return _det_build_and_process_sequence(self, boxes, labels, noise_mask, max_length, _unused)

    def _truncate_at_fragment_boundary(self, token_ids, max_length):
        return _det_truncate_at_fragment_boundary(self, token_ids, max_length)

    def jitter_bboxes_norm(self, bboxes, dtype, jitter_ratio=0.05):
        return det_jitter_bboxes_norm(self, bboxes, dtype, jitter_ratio)

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

                img, scale, pad_w, pad_h = letterbox_image(img, DEFAULT_TARGET_SIZE)

                # Let the image processor handle all resizing and preprocessing
                try:
                    pv = self.processor(
                        img,
                        return_tensors="pt",
                        do_center_crop=False,
                        do_resize=False,
                    )["pixel_values"][0]
                    
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
                    # log at DEBUG level only
                    self.logger.debug("Different tensor shapes before stacking: %s", shapes)

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
            # Determine if sample *has the field* (may still be empty)
            has_boxes_field = (
                "target_boxes" in item
                and isinstance(item["target_boxes"], (list, np.ndarray, torch.Tensor))
                and not self.is_eval
            )

            if has_boxes_field:
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
            # Determine if sample *has the field* (may still be empty)
            has_boxes_field = (
                "target_boxes" in item
                and isinstance(item["target_boxes"], (list, np.ndarray, torch.Tensor))
                and not self.is_eval
            )

            if has_boxes_field:
                bx_raw = torch.as_tensor(item["target_boxes"], dtype=torch.float32, device=device)
                label_strs = item.get("target_label_strs", ["obj"] * bx_raw.size(0))[: bx_raw.size(0)]
                
                # Determine noise count
                if random.random() <= self.noise_prob and not self.is_eval:
                    num_gt_boxes = len(label_strs)
                    if num_gt_boxes > 0:
                        noise_ratio = random.uniform(*self.noise_ratio_range)
                        num_noise = max(1, min(self.max_noise_boxes, int(num_gt_boxes * noise_ratio)))
                    else:
                        # PATCH 2: guarantee *some* noise boxes even when there are
                        # no GT boxes – use 25% of max_noise_boxes as minimum.
                        num_noise = max(1, int(self.max_noise_boxes * 0.25))
                else:
                    num_noise = 0
                
                batch_gt_boxes.append(bx_raw)
                batch_label_strs.append(label_strs)
                batch_noise_counts.append(num_noise)
            else:
                batch_gt_boxes.append(torch.empty((0, 4)))
                batch_label_strs.append([])
                batch_noise_counts.append(0)

        # PATCH 3: Skip heavy noise generation during evaluation
        if not self.is_eval:
            batch_noise_results = self._generate_noise_boxes_batch(
                batch_gt_boxes, batch_label_strs, batch_noise_counts
            )
        else:
            batch_noise_results = [
                (torch.empty((0, 4), device=device), []) for _ in batch
            ]

        # OPTIMIZED: Pre-allocate output lists with known batch size
        batch_size = len(batch)
        merged_input_ids = [None] * batch_size
        merged_labels = [None] * batch_size

        # CRITICAL: Pre-compute common tokens with correct device specification
        image_placeholder = torch.tensor([placeholder_id], dtype=torch.long, device=device)
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        max_txt_len = self.tokenizer.model_max_length - self.vis_tokens
        
        # OVERRIDE: Force reasonable sequence limits to prevent OOM
        # Many tokenizers have very high model_max_length (like 1M+) which is impractical
        reasonable_max_length = 4096  # Reasonable limit for detection training
        if max_txt_len > reasonable_max_length - self.vis_tokens:
            max_txt_len = reasonable_max_length - self.vis_tokens
            if self.logger and not hasattr(self, '_logged_override'):
                self.logger.warning(f"Overriding tokenizer max_length to reasonable limit: {reasonable_max_length}")
                self._logged_override = True
        
        # OPTIMIZED: Pre-compute common tensors outside the loop to avoid repeated creation
        bos_tensor = torch.tensor([bos_id], dtype=torch.long, device=device) if bos_id is not None else None
        eos_tensor = torch.tensor([eos_id], dtype=torch.long, device=device) if eos_id is not None else None
        empty_tensor = torch.empty(0, dtype=torch.long, device=device)
        
        # OPTIMIZED: Pre-allocate ignore tensor for visual tokens and reuse
        visual_ignore = torch.full((self.vis_tokens,), -100, dtype=torch.long, device=device)

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

                    # Fallback: if vectorised generator unexpectedly returned no boxes
                    # for this sample although a positive noise count was requested,
                    # regenerate noise locally to guarantee training supervision.
                    if noise_boxes.numel() == 0 and batch_noise_counts[i] > 0:
                        # PATCH 4: emit warning so we can trace rare failures
                        if self.logger:
                            self.logger.warning(
                                f"Noise generation failed for sample {i}, count={batch_noise_counts[i]} – regenerating locally"
                            )

                        noise_boxes, noise_labels = self._generate_noise_boxes(
                            batch_noise_counts[i], label_strs, bx
                        )
                    if noise_boxes.numel() > 0:
                        # Combine, shuffle, build mask in one step
                        combined_boxes, combined_labels, noise_mask = self._merge_shuffle_gt_noise(
                            bx, label_strs, noise_boxes, noise_labels
                        )

                        # Build fragments and tokenize efficiently
                        input_ids_det, tgt_ids = self._build_and_process_sequence(
                            combined_boxes, combined_labels, noise_mask, remaining_space, None
                        )
                    else:
                        # If noise generation failed, use empty tensors
                        input_ids_det = empty_tensor.clone()
                        tgt_ids = empty_tensor.clone()
                else:
                    # Empty bboxes case
                    input_ids_det = empty_tensor.clone()
                    tgt_ids = empty_tensor.clone()
            else:
                # Caption-only path (no detection boxes).  Use stored
                # target_text tokens directly as inputs so the model sees the
                # caption prompt during both training and evaluation.
                input_ids_det = torch.as_tensor(item.get("target_text", []), dtype=torch.long, device=device).flatten()
                tgt_ids       = input_ids_det.clone()

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

            # CRITICAL: Handle/EOS tokens consistently to maintain alignment
            
            # --- ensure a single BOS token starts the instruction sequence ---
            if bos_tensor is not None and bos_id != eos_id:
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
            # NOTE: input_ids still uses placeholder (1 token) which the model will replace with self.vis_tokens visual tokens
            
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
            # Account for visual token replacement: 1 placeholder becomes self.vis_tokens visual tokens
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
        # Supervise ONLY the *first* EOS token.
        # Padding as well as any extra tokens after the first EOS are
        # masked out with IGNORE_INDEX (-100) so they do not contribute
        # to the loss.  This prevents the model from over-optimising on
        # EOS/PAD and prematurely terminating generation.
        # -------------------------------------------------------------

        IGNORE_INDEX = -100
        labels_padded = pad_sequence(merged_labels, batch_first=True, padding_value=IGNORE_INDEX)

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            # keep only the first EOS per sample
            for row in labels_padded:
                eos_positions = (row == eos_id).nonzero(as_tuple=False).flatten()
                if eos_positions.numel() > 0:
                    # ignore all EOS after the first
                    row[eos_positions[1:]] = IGNORE_INDEX

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
        vis_tokens=kwargs.get("vis_tokens"),
        logger=kwargs.get("logger"),
    ) 
