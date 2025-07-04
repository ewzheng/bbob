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
        self.rngjesus = np.random.default_rng()

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

        # Accept list / np.ndarray inputs
        if not isinstance(bboxes, torch.Tensor):
            bboxes = torch.as_tensor(bboxes, dtype=dtype)

        if bboxes.numel() == 0:
            return bboxes.to(dtype=dtype)

        bx = bboxes.to(dtype=dtype).clone()

        # centre (cx,cy) and size (w,h)
        cxcy = bx[:, :2] + 0.5 * bx[:, 2:]
        wh   = bx[:, 2:]

        # random jitter on centre: ±jitter_ratio * (w,h)
        trans_rng = (torch.rand_like(cxcy) * 2.0 - 1.0) * jitter_ratio
        cxcy = cxcy + trans_rng * wh

        # random scaling on size: ±jitter_ratio
        scale_rng = (torch.rand_like(wh) * 2.0 - 1.0) * jitter_ratio
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
                    pv = self.processor(img, return_tensors="pt")["pixel_values"][0]  # (3,256,256) float32 0-1
                    processed.append(pv)
                    continue  # success – skip manual fallback
                except Exception:
                    # Fallback to legacy manual branch for rare failures
                    pass

                # ---------------- manual fallback -------------------
                if isinstance(img, torch.Tensor):
                    t = img.to(dtype=torch.float32).div_(255.0)
                elif isinstance(img, np.ndarray):
                    t = torch.as_tensor(img, dtype=torch.float32).div_(255.0)
                elif isinstance(img, list):
                    t = torch.as_tensor(np.array(img, dtype=np.uint8), dtype=torch.float32).div_(255.0)
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

                    canvas = 0.5 * torch.ones(3, *TARGET_SIZE)
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
            img_tensors = [torch.as_tensor(item[img_key]) for item in batch]

            # Ensure float32 + 0‥1 range once for the whole list (cheap).
            ref = img_tensors[0]
            needs_cast = ref.dtype != torch.float32 or ref.max() > 1.1
            if needs_cast:
                img_tensors = [t.to(dtype=torch.float32).div_(255.0) for t in img_tensors]

            pixel_values = torch.stack(img_tensors, 0)

        merged_input_ids, merged_labels = [], []

        for item in batch:
            if "input_ids" in item:
                instr_ids = torch.as_tensor(item["input_ids"], dtype=torch.long).flatten()
            else:
                text = item.get("text", "")
                max_txt_len = self.tokenizer.model_max_length - VIS_TOKENS
                tokens = self.tokenizer(text, return_tensors="pt", max_length=max_txt_len, truncation=True)
                instr_ids = tokens["input_ids"].squeeze(0)

            # Decide ground-truth detection text ----------------------------------
            if "target_boxes" in item and not self.is_eval:
                # TRAIN MODE → jitter
                bx_raw = torch.as_tensor(item["target_boxes"], dtype=torch.float32)
                bx = self.jitter_bboxes_norm(bx_raw, dtype=torch.float32)

                # rebuild textual fragments so tokens match jittered coords
                label_strs = item.get("target_label_strs", ["obj"] * bx.size(0))[: bx.size(0)]


                fmt_coord = lambda v: f"{max(0.0, min(1.0, v)):.3f}"

                frags = [
                    f"<|bbob|>{lab}: [{', '.join(fmt_coord(v) for v in bb)}]</|bbob|>"
                    for bb, lab in zip(bx.tolist(), label_strs)
                ]
                det_text = " ".join(frags)
                tgt_ids = self.tokenizer(det_text, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
                tgt_ids = self._shuffle_fragments(tgt_ids)
            else:
                # EVAL MODE or boxes absent – use stored token list as is
                tgt_ids = torch.as_tensor(item.get("target_text", []), dtype=torch.long).flatten()


            #------ now same instr truncation logic uses tgt_ids variable ----
            tgt_ids = tgt_ids[tgt_ids != pad_token_id]

            max_txt_len = self.tokenizer.model_max_length - VIS_TOKENS
            if instr_ids.size(0) > max_txt_len:
                instr_ids = instr_ids[-max_txt_len:]
                tgt_ids = torch.tensor([], dtype=torch.long)
            else:
                remaining = max_txt_len - instr_ids.size(0)
                if tgt_ids.size(0) > remaining:
                    tgt_ids = tgt_ids[:remaining]

                # --- ensure a single BOS token starts the instruction sequence ---
                bos_id = getattr(self.tokenizer, "bos_token_id", None)
                if bos_id is not None:
                    if instr_ids.numel() == 0:
                        # create sequence containing only <bos>
                        instr_ids = torch.tensor([bos_id], dtype=torch.long)
                    elif instr_ids[0] != bos_id:
                        # prepend or replace first token with <bos> depending on space
                        if instr_ids.size(0) >= self.tokenizer.model_max_length - tgt_ids.size(0):
                            # no room left – overwrite first token
                            instr_ids[0] = bos_id
                        else:
                            instr_ids = torch.cat([torch.tensor([bos_id], dtype=torch.long), instr_ids])

                # --- ensure a single EOS token terminates the target sequence ---
                eos_id = getattr(self.tokenizer, "eos_token_id", None)
                if eos_id is not None:
                    if tgt_ids.numel() == 0:
                        # create sequence containing only <eos>
                        tgt_ids = torch.tensor([eos_id], dtype=torch.long)
                    elif tgt_ids[-1] != eos_id:
                        # append or replace last token with <eos> depending on space
                        if tgt_ids.size(0) >= self.tokenizer.model_max_length - instr_ids.size(0):
                            # no room left – overwrite last token
                            tgt_ids[-1] = eos_id
                        else:
                            tgt_ids = torch.cat([tgt_ids, torch.tensor([eos_id], dtype=torch.long)])

            placeholder_ids = torch.full((tgt_ids.size(0),), placeholder_id, dtype=torch.long)
            input_ids = torch.cat([instr_ids, placeholder_ids], dim=0)

            visual_ignore = torch.full((VIS_TOKENS,), -100, dtype=torch.long)
            lbl_text = torch.cat([instr_ids.clone(), tgt_ids.clone()], dim=0)
            labels = torch.cat([visual_ignore, lbl_text])
            labels[: VIS_TOKENS + instr_ids.size(0)] = -100

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
    `bin_numeric_tokens` for optional features."""

    return BBOBCollator(
        pad_token_id,
        tokenizer,
        image_processor,
        on_the_fly=kwargs.get("on_the_fly", False),
        logger=kwargs.get("logger"),
        log_interval=kwargs.get("log_interval", 0),
    ) 
