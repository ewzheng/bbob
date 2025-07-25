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
# Prefer GPU batch solver; fall back to SciPy if not installed / on CPU
try:
    from torch_linear_assignment import linear_sum_assignment as _tla_lsa  # type: ignore
    _HAS_TLA = True
except ImportError:  # pragma: no cover
    _HAS_TLA = False
    from scipy.optimize import linear_sum_assignment as _scipy_lsa  # type: ignore
from torchvision.ops import box_iou as _box_iou  # type: ignore

# -----------------------------------------------------------------------------
# Constants & compiled regex -----------------------------------------------------------------------------
EPSILON: float = 1e-6
TAG_OPEN, TAG_CLOSE = "<|bbob|>", "</|bbob|>"
NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+")

NUM_DIGITS: int = 3              # expected digit tokens per coordinate
COORD_SCALE: float = float(10 ** NUM_DIGITS)

# -----------------------------------------------------------------------------
# Snippet extraction & conversion – shared by training & evaluation
# -----------------------------------------------------------------------------


def _extract_snippets(text: str) -> List[str]:
    """Return every substring between TAG_OPEN and TAG_CLOSE (whitespace-stripped)."""
    snippets: List[str] = []
    start = 0
    while True:
        st = text.find(TAG_OPEN, start)
        if st == -1:
            break
        st_end = st + len(TAG_OPEN)
        ed = text.find(TAG_CLOSE, st_end)
        if ed == -1:
            break  # unmatched open – ignore tail
        snippets.append(text[st_end:ed].strip())
        start = ed + len(TAG_CLOSE)
    return snippets


def _split_snippet(det: str) -> Tuple[str, List[float]]:
    """Return *(label, xywh)* for one <bbob> snippet (label lower-cased)."""
    parts = det.split(":", 1)
    label = parts[1].strip().lower() if len(parts) == 2 else ""
    xywh, _ = parse_detection_string(det)
    return label, xywh


def _snippets_to_boxes_labels(snippets: List[str]) -> Tuple[torch.Tensor, List[str]]:
    """Convert list of snippets to tensor boxes and parallel label list."""
    coords: List[List[float]] = []
    labels: List[str] = []
    for det in snippets:
        lab, xywh = _split_snippet(det)
        coords.append(xywh)
        labels.append(lab)

    if coords:
        return torch.tensor(coords, dtype=torch.float32), labels
    return torch.zeros((0, 4), dtype=torch.float32), labels

# -----------------------------------------------------------------------------
# Batch utilities – convert (B,S) token IDs to concatenated box tensors
# -----------------------------------------------------------------------------


def ids_to_boxes_labels(
    token_ids: "torch.Tensor",  # (B,S)
    tokenizer,
    *,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """Vectorised helper that turns a batch of token IDs into one coordinate tensor.

    Parameters
    ----------
    token_ids : LongTensor (B,S)
        Model predictions *or* ground-truth IDs.
    tokenizer : PreTrainedTokenizerBase
        Tokeniser used to decode IDs.
    ignore_index : int, default -100
        IDs with this value are removed before decoding.

    Returns
    -------
    coords_all : FloatTensor  (N,4)  concatenated xywh in 0‥1
    labels_all : list[str]    parallel class labels (lower-cased)
    batch_ptr  : LongTensor   (B+1,) prefix sum indices for slicing per-image.
    """
    if token_ids.ndim != 2:
        raise ValueError("token_ids must be (B,S)")

    # 1) Remove ignore_index paddings
    filtered: List[List[int]] = [
        [int(t) for t in row.tolist() if t != ignore_index] for row in token_ids
    ]

    # 2) Batch-decode once (efficient in HF)
    decoded: List[str] = tokenizer.batch_decode(
        filtered, skip_special_tokens=False, clean_up_tokenization_spaces=True
    )

    # 3) Extract snippets & convert to tensors (vectorised decode_batch)
    snippets_per_batch: List[List[str]] = [_extract_snippets(txt) for txt in decoded]

    coords_all, _fmt_dummy, batch_ptr = decode_batch(snippets_per_batch)

    # Decode labels alongside coords (cheap Python loop over snippets text)
    labels_all: List[str] = []
    for snippets in snippets_per_batch:
        for det in snippets:
            lab, _ = _split_snippet(det)
            labels_all.append(lab)

    return coords_all, labels_all, batch_ptr

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

    Expected format: ``[x1, y1, x2, y2]: class`` (bracketed coords first).

    Returns
    -------
    coords : list[float]  length-4, clamped to 0‥1
    fmt_err : float       formatting error in [0, +inf)
    """
    parts = det.split(":", 1)
    if len(parts) != 2:
        return [0, 0, 0, 0], 1.0

    # coords are in the *first* part, potentially inside [...] brackets
    coord_part = parts[0]
    label_part = parts[1]

    blank_pen = 0.25 if label_part.strip() == "" else 0.0

    # Robust numeric parse – clamp to 0..1 and skip bad tokens
    nums_raw = []
    for tok in NUMERIC_PATTERN.findall(coord_part)[:4]:
        try:
            val = float(tok)
        except ValueError:
            continue
        nums_raw.append(val)

    nums = [max(0.0, min(1.0, v)) for v in nums_raw]
    missing = 4 - len(nums)
    nums.extend([0.5] * missing)

    clip_err = sum(abs(v - max(0.0, min(1.0, v))) for v in nums) / 4.0
    fmt_err = missing / 4.0 + clip_err + blank_pen

    # penalise zero-area
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
# New helper – IoU on (x1,y1,x2,y2) boxes (already absolute corners)
# -----------------------------------------------------------------------------

def iou_matrix_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Return pairwise IoU for two *xyxy* box tensors.

    Falls back to a vectorised implementation if the optional C++/CUDA
    `torchvision.ops._box_iou` is unavailable (as on some CPU-only setups).
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))

    if _box_iou is not None:
        return _box_iou(boxes1, boxes2)

    # Manual vectorised IoU
    x1, y1, x2, y2 = boxes1.unbind(-1)
    x1b, y1b, x2b, y2b = boxes2.unbind(-1)

    # Expand for pairwise max/min operations
    inter_x1 = torch.max(x1.unsqueeze(1), x1b.unsqueeze(0))
    inter_y1 = torch.max(y1.unsqueeze(1), y1b.unsqueeze(0))
    inter_x2 = torch.min(x2.unsqueeze(1), x2b.unsqueeze(0))
    inter_y2 = torch.min(y2.unsqueeze(1), y2b.unsqueeze(0))

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = ((x2 - x1) * (y2 - y1)).unsqueeze(1)
    area2 = ((x2b - x1b) * (y2b - y1b)).unsqueeze(0)
    union = area1 + area2 - inter_area + EPSILON
    return inter_area / union

# -----------------------------------------------------------------------------
# Matching ----------------------------------------------------------------------------

def hungarian_match(pred: torch.Tensor, gt: torch.Tensor) -> List[Tuple[int, int]]:
    """Return Hungarian assignment pairs using IoU-based cost.

    If *torch_linear_assignment* is available we use its CUDA-aware single-instance
    solver; otherwise we fall back to SciPy (CPU).
    """
    if pred.numel() == 0 or gt.numel() == 0:
        return []

    pred = pred[(pred[:, 2] > EPSILON) & (pred[:, 3] > EPSILON)]
    gt   = gt[(gt[:, 2] > EPSILON) & (gt[:, 3] > EPSILON)]
    if pred.numel() == 0 or gt.numel() == 0:
        return []

    cost = 1.0 - iou_matrix_xywh(pred, gt)

    if _HAS_TLA:
        row_ind, col_ind = _tla_lsa(cost)  # works on tensors directly
        if isinstance(row_ind, torch.Tensor):
            row_ind = row_ind.cpu()
            col_ind = col_ind.cpu()
        return list(zip(row_ind.tolist(), col_ind.tolist()))
    else:
        cost_np = cost.cpu().float().numpy()
        row_ind, col_ind = _scipy_lsa(cost_np)
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
    # Convert tensors to Python lists
    if isinstance(pred_ids, torch.Tensor):
        pred_ids = pred_ids.tolist()
    if isinstance(gt_ids, torch.Tensor):
        gt_ids = gt_ids.tolist()

    eos_id = tokenizer.eos_token_id

    # Helper: keep tokens up to and incl. first EOS
    def _trim(ids):
        if eos_id is not None and eos_id in ids:
            return ids[: ids.index(eos_id) + 1]
        return ids

    pred_ids_trim = _trim(pred_ids)

    # Remove padding / ignore_index from ground-truth then trim
    gt_ids_clean = [t for t in gt_ids if t != ignore_index]
    gt_ids_trim = _trim(gt_ids_clean)

    pred_str = tokenizer.decode(pred_ids_trim, skip_special_tokens=False, clean_up_tokenization_spaces=True)
    gt_str = tokenizer.decode(gt_ids_trim, skip_special_tokens=False, clean_up_tokenization_spaces=True)
    
    return pred_str, gt_str 