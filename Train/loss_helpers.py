"""Utility helpers shared by the optimized detection-loss pipeline.

This module is intentionally self-contained so that future refactors can
import from it without dragging in the full loss class.
"""

from __future__ import annotations

import itertools
import math
import re
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment  # type: ignore
from torchvision.ops import box_iou as _box_iou  # type: ignore

# -----------------------------------------------------------------------------
# Constants & compiled regex -----------------------------------------------------------------------------
EPSILON: float = 1e-6
TAG_OPEN, TAG_CLOSE = "<|bbob|>", "</|bbob|>"
NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+")

NUM_DIGITS: int = 3              # expected digit tokens per coordinate
COORD_SCALE: float = float(10 ** NUM_DIGITS)

# -----------------------------------------------------------------------------
# Parsing helpers -----------------------------------------------------------------------------

def _parse_boxes(logits: torch.Tensor, tokenizer) -> List[List[str]]:
    """Extract `<bbob>` snippets without decoding entire sequences.

    Parameters
    ----------
    logits : FloatTensor (B,S,V)
        Raw model logits; we only need arg-max ids.
    tokenizer : PreTrainedTokenizer
        Tokeniser providing `convert_tokens_to_ids` and `batch_decode`.

    Returns
    -------
    list[list[str]]
        For every batch sample a list of decoded snippet texts (whitespace-stripped).
    """
    if logits.dim() != 3:
        raise ValueError(f"Expected logits with shape (B,S,V) but got {logits.shape}")

    token_ids = logits.argmax(dim=-1)  # (B,S)
    id_open, id_close = tokenizer.convert_tokens_to_ids([TAG_OPEN, TAG_CLOSE])

    snippets_per_batch: List[List[str]] = [[] for _ in range(token_ids.size(0))]

    ids_cpu = token_ids.cpu().numpy()
    for b, row in enumerate(ids_cpu):
        opens = (row == id_open).nonzero()[0]
        closes = (row == id_close).nonzero()[0]
        for st in opens:
            after = closes[closes > st]
            if after.size == 0:
                snippets_per_batch[b].append("")
                continue
            ed = after[0]
            snippet_ids = row[st + 1 : ed].tolist() if ed - st > 1 else []
            decoded = tokenizer.batch_decode([snippet_ids], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            snippets_per_batch[b].append(decoded.strip())

    return snippets_per_batch

def parse_detection_string(det: str) -> Tuple[List[float], float]:
    """Parse a single `<bbob>` snippet.

    Returns
    -------
    coords : list[float]  length-4, clamped to 0‥1
    fmt_err : float       formatting error in [0, +inf)
    """
    parts = det.split(":", 1)
    if len(parts) != 2:
        return [0, 0, 0, 0], 1.0

    blank_pen = 0.25 if parts[0].strip() == "" else 0.0

    nums = [max(0.0, min(1.0, float(s))) for s in NUMERIC_PATTERN.findall(parts[1])[:4]]
    missing = 4 - len(nums)
    nums.extend([0.5] * missing)

    clip_err = sum(abs(v - max(0.0, min(1.0, v))) for v in nums) / 4.0
    fmt_err = missing / 4.0 + clip_err + blank_pen

    # penalise zero-area boxes
    if nums[2] == 0.0 or nums[3] == 0.0:
        fmt_err += 0.25

    return nums, fmt_err


def decode_batch(parsed_strs: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorise detection-string parsing for a whole batch.

    Parameters
    ----------
    parsed_strs : list[list[str]]
        Output of `_parse_boxes`; length B, with K₍ᵦ₎ strings each.

    Returns
    -------
    coords_all : FloatTensor  (K,4)  concatenated coordinates
    fmt_tensor : FloatTensor  (K,)   formatting errors
    batch_ptr  : LongTensor   (B+1,) prefix sum so that slice
                  `coords_all[ptr[b]:ptr[b+1]]` gives sample *b*.
    """
    coords, fmt = [], []
    counts = [len(lst) for lst in parsed_strs]
    for det in itertools.chain.from_iterable(parsed_strs):
        c, f = parse_detection_string(det)
        coords.append(c)
        fmt.append(f)

    if coords:
        coords_all = torch.tensor(coords, dtype=torch.float32)
        fmt_tensor = torch.tensor(fmt, dtype=torch.float32)
    else:
        coords_all = torch.zeros((0, 4), dtype=torch.float32)
        fmt_tensor = torch.zeros((0,), dtype=torch.float32)

    batch_ptr = torch.tensor(np.concatenate(([0], np.cumsum(counts))), dtype=torch.long)
    return coords_all, fmt_tensor, batch_ptr

# -----------------------------------------------------------------------------
# Geometry helpers -----------------------------------------------------------------------------

def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    return torch.stack((x, y, x + w, y + h), dim=-1)


def iou_matrix_xywh(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))

    if _box_iou is not None:
        return _box_iou(xywh_to_xyxy(boxes1), xywh_to_xyxy(boxes2))

    # simple vectorised fallback (both tensors on same device)
    x1, y1, w1, h1 = boxes1.unbind(-1)
    x2, y2, w2, h2 = boxes2.unbind(-1)

    x1b, y1b = x1 + w1, y1 + h1
    x2b, y2b = x2 + w2, y2 + h2

    inter_x1 = torch.max(x1.unsqueeze(1), x2.unsqueeze(0))
    inter_y1 = torch.max(y1.unsqueeze(1), y2.unsqueeze(0))
    inter_x2 = torch.min(x1b.unsqueeze(1), x2b.unsqueeze(0))
    inter_y2 = torch.min(y1b.unsqueeze(1), y2b.unsqueeze(0))

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (w1 * h1).unsqueeze(1)
    area2 = (w2 * h2).unsqueeze(0)
    union = area1 + area2 - inter_area + EPSILON
    return inter_area / union

# -----------------------------------------------------------------------------
# Matching ----------------------------------------------------------------------------

def hungarian_match(pred: torch.Tensor, gt: torch.Tensor) -> List[Tuple[int, int]]:
    """Return Hungarian assignment pairs using IoU-based cost."""
    if pred.numel() == 0 or gt.numel() == 0:
        return []

    pred = pred[(pred[:, 2] > EPSILON) & (pred[:, 3] > EPSILON)]
    gt = gt[(gt[:, 2] > EPSILON) & (gt[:, 3] > EPSILON)]
    if pred.numel() == 0 or gt.numel() == 0:
        return []

    cost = (1.0 - iou_matrix_xywh(pred, gt)).cpu().float().numpy()
    if cost.size == 0:
        return []
    row_ind, col_ind = linear_sum_assignment(cost)
    return list(zip(row_ind.tolist(), col_ind.tolist()))

# -----------------------------------------------------------------------------
# High-level helpers for evaluation / logging
# -----------------------------------------------------------------------------

def decode_pred_gt(
    pred_ids: "torch.Tensor | list[int]",
    gt_ids: "torch.Tensor | list[int]",
    tokenizer,
    *,
    ignore_index: int = -100,
) -> tuple[str, str]:
    """Decode model prediction and ground-truth IDs to text.

    Parameters
    ----------
    pred_ids : 1-D LongTensor or list[int]
        Token IDs predicted by the model (e.g. after arg-max or beam search).
    gt_ids   : 1-D LongTensor or list[int]
        Ground-truth token IDs, possibly containing *ignore_index* paddings.
    tokenizer : PreTrainedTokenizerBase
        Tokenizer used by the model.
    ignore_index : int, default -100
        IDs with this value are removed from *gt_ids* before decoding.

    Returns
    -------
    pred_str, gt_str : tuple[str,str]
        Decoded prediction and ground-truth strings with *special tokens kept* so
        that "<|bbob|>…</|bbob|>" snippets remain visible for inspection.
    """
    import torch  # local import to avoid hard dependency when torch absent

    # Convert tensors to Python lists
    if isinstance(pred_ids, torch.Tensor):
        pred_ids = pred_ids.tolist()
    if isinstance(gt_ids, torch.Tensor):
        gt_ids = gt_ids.tolist()

    # Remove padding / ignore_index from ground truth
    gt_ids_clean = [t for t in gt_ids if t != ignore_index]

    pred_str = tokenizer.decode(pred_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
    gt_str = tokenizer.decode(gt_ids_clean, skip_special_tokens=False, clean_up_tokenization_spaces=True)
    return pred_str, gt_str 