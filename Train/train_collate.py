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
import random


class BBOBCollator:  # noqa: N801
    """Minimal collator for BBOB – no teacher-forcing logic."""

    def __init__(
        self,
        pad_token_id: int,
        tokenizer,
        *,
        logger=None,
        log_interval=128,
        **kwargs,  # accept legacy tf_* kwargs but ignore them
    ):
        self.pad_id = pad_token_id
        self.tokenizer = tokenizer
        self.logger = logger

        # teacher-forcing parameters removed – keep placeholders for API compat
        self.log_interval = log_interval
        self.is_eval = False

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
            tokenizer=self.tokenizer,
            placeholder_id=self.placeholder_id,
        )

    # BBOBTrainer toggles these; keep as no-ops
    def eval(self):
        self.is_eval = True

    def train(self):
        self.is_eval = False

    def jitter_bboxes_norm(self, bboxes, dtype, jitter_ratio=0.05):
        """Jitter *normalised* (x,y,w,h) boxes in 0‥1 space.

        Each box is perturbed independently:
        – centre moved by up to ``jitter_ratio × w/h``
        – width/height scaled by ±``jitter_ratio``.
        The result is clamped so that the box stays inside 0…1 and keeps
        ``w,h ≥ 0``.
        """
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.clone().detach().float().cpu().numpy()

        out: list[list[float]] = []
        for x, y, w, h in bboxes:
            # centre coords
            cx, cy = x + w * 0.5, y + h * 0.5

            # random jitter on centre and size
            cx += np.random.uniform(-jitter_ratio, jitter_ratio) * w
            cy += np.random.uniform(-jitter_ratio, jitter_ratio) * h
            w  *= 1 + np.random.uniform(-jitter_ratio, jitter_ratio)
            h  *= 1 + np.random.uniform(-jitter_ratio, jitter_ratio)

            # convert back to top-left origin and clamp
            x_new = max(0.0, cx - w * 0.5)
            y_new = max(0.0, cy - h * 0.5)
            w  = max(0.0, min(w, 1.0 - x_new))
            h  = max(0.0, min(h, 1.0 - y_new))

            out.append([x_new, y_new, w, h])

        return torch.tensor(out, dtype=dtype)

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

        random.shuffle(fragments)

        new_seq = prefix[:]
        first = True
        for frag in fragments:
            if not first and self.space_id != -1:
                new_seq.append(self.space_id)
            new_seq.extend(frag)
            first = False
        new_seq.extend(suffix)
        return torch.tensor(new_seq, dtype=torch.long)


    def _make_batch(self, batch, *, pad_token_id: int, tokenizer, placeholder_id: int):
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

        processed = []
        for img in (item[img_key] for item in batch):
            if isinstance(img, torch.Tensor):
                t = img.to(dtype=torch.float32).div_(255.0)
            elif isinstance(img, np.ndarray):
                t = torch.as_tensor(img, dtype=torch.float32).div_(255.0)
            elif isinstance(img, list):
                t = torch.as_tensor(np.array(img, dtype=np.uint8), dtype=torch.float32).div_(255.0)
            else:
                t = pil_to_tensor(img).float().div_(255.0).to(device)

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

        merged_input_ids, merged_labels = [], []

        for item in batch:
            if "input_ids" in item:
                instr_ids = torch.as_tensor(item["input_ids"], dtype=torch.long).flatten()
            else:
                text = item.get("text", "")
                max_txt_len = tokenizer.model_max_length - VIS_TOKENS
                tokens = tokenizer(text, return_tensors="pt", max_length=max_txt_len, truncation=True)
                instr_ids = tokens["input_ids"].squeeze(0)

            # Decide ground-truth detection text ----------------------------------
            if "target_boxes" in item and not self.is_eval:
                # TRAIN MODE → jitter
                bx_raw = torch.as_tensor(item["target_boxes"], dtype=torch.float32)
                bx = self.jitter_bboxes_norm(bx_raw, dtype=torch.float32)

                # rebuild textual fragments so tokens match jittered coords
                label_strs = item.get("target_label_strs", ["obj"] * bx.size(0))[: bx.size(0)]

                frags = [f"<|bbob|>{lab}: [{', '.join(f'{v:.3f}' for v in bb)}]</|bbob|>" for bb, lab in zip(bx.tolist(), label_strs)]
                det_text = " ".join(frags)
                tgt_ids = self.tokenizer(det_text, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
            else:
                # EVAL MODE or boxes absent – use stored token list as is
                tgt_ids = torch.as_tensor(item.get("target_text", []), dtype=torch.long).flatten()

            
            tgt_ids = self._shuffle_fragments(tgt_ids)

            #------ now same instr truncation logic uses tgt_ids variable ----
            tgt_ids = tgt_ids[tgt_ids != pad_token_id]

            max_txt_len = tokenizer.model_max_length - VIS_TOKENS
            if instr_ids.size(0) > max_txt_len:
                instr_ids = instr_ids[-max_txt_len:]
                tgt_ids = torch.tensor([], dtype=torch.long)
            else:
                remaining = max_txt_len - instr_ids.size(0)
                if tgt_ids.size(0) > remaining:
                    tgt_ids = tgt_ids[:remaining]

                # --- ensure a single EOS token terminates the target sequence ---
                eos_id = getattr(tokenizer, "eos_token_id", None)
                if eos_id is not None:
                    if tgt_ids.numel() == 0:
                        # create sequence containing only <eos>
                        tgt_ids = torch.tensor([eos_id], dtype=torch.long)
                    elif tgt_ids[-1] != eos_id:
                        # append or replace last token with <eos> depending on space
                        if tgt_ids.size(0) >= tokenizer.model_max_length - instr_ids.size(0):
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
        labels_padded = pad_sequence(merged_labels, batch_first=True, padding_value=-100)
        attention_mask = (input_ids_padded != pad_token_id).long()

        batch_out = {
            "images": pixel_values,
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
        }

        return batch_out


def make_collate_fn(pad_token_id: int, tokenizer, **kwargs):
    """Create a collator – all teacher-forcing args are ignored for back-compat."""

    return BBOBCollator(
        pad_token_id,
        tokenizer,
        logger=kwargs.get("logger"),
        log_interval=kwargs.get("log_interval", 0),
    ) 
