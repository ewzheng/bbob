from __future__ import annotations

"""Object-detection evaluation helpers used during validation.

The functions here work purely on *token IDs* so they can be called from the
memory-efficient evaluation loop that keeps logits on the GPU only long enough
for an ``argmax``.  They decode the prediction / ground-truth sequences,
extract the ``<|bbob|>…</|bbob|>`` fragments, convert them to numeric xywh
boxes in the 0‥1 range, and finally run Hungarian matching to compute mean IoU
and recall.
"""

from typing import Dict, List, Tuple

import torch

from Train.loss_helpers import (
    TAG_OPEN,
    TAG_CLOSE,
    parse_detection_string,
    iou_matrix_xywh,
    hungarian_match,
)

__all__ = ["detection_metrics_batch"]

# -----------------------------------------------------------------------------
# Helper – snippet extraction & conversion
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


def _snippets_to_boxes(snippets: List[str]) -> torch.Tensor:
    """Convert list of ``<bbob>`` detection snippets to a (N,4) tensor."""
    coords: List[List[float]] = []
    for det in snippets:
        xywh, _ = parse_detection_string(det)
        coords.append(xywh)
    if coords:
        return torch.tensor(coords, dtype=torch.float32)
    return torch.zeros((0, 4), dtype=torch.float32)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def detection_metrics_batch(
    pred_ids: torch.Tensor,  # (B,S)
    gt_ids: torch.Tensor,    # (B,S)
    tokenizer,
    *,
    iou_thresh: float = 0.5,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """Compute mean IoU and recall for a batch of predictions.

    Parameters
    ----------
    pred_ids : LongTensor (B,S)
        Token IDs predicted by the model after arg-max.
    gt_ids : LongTensor (B,S)
        Ground-truth token IDs; positions equal to ``ignore_index`` are ignored
        when decoding.
    tokenizer : transformers.PreTrainedTokenizerBase
        Needed to turn IDs back into text.
    iou_thresh : float, default 0.5
        Threshold that counts a match as *correct* when IoU ≥ this value.
    ignore_index : int, default -100
        Label padding value to discard before decoding.

    Returns
    -------
    dict with keys:
        "mean_iou"  – average IoU of all matched pairs (0 if none).
        "recall"    – correct_matches / total_gt (0 if no GT boxes).
    """
    if pred_ids.ndim != 2 or gt_ids.ndim != 2:
        raise ValueError("pred_ids and gt_ids must be (B,S) tensors")

    device = pred_ids.device
    batch_size = pred_ids.size(0)

    total_gt = 0
    correct_matches = 0
    iou_sum = 0.0
    iou_count = 0

    # Iterate over samples; decoding on CPU for efficiency
    for b in range(batch_size):
        # ---------- decode prediction & GT strings ---------------------------
        pred_str = tokenizer.decode(
            pred_ids[b].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True
        )
        # Filter ignore_index before decoding GT
        gt_filtered = [int(t) for t in gt_ids[b].tolist() if t != ignore_index]
        gt_str = tokenizer.decode(
            gt_filtered, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )

        # ---------- extract detection snippets ------------------------------
        pred_snips = _extract_snippets(pred_str)
        gt_snips = _extract_snippets(gt_str)

        pred_boxes = _snippets_to_boxes(pred_snips).to(device)
        gt_boxes = _snippets_to_boxes(gt_snips).to(device)

        total_gt += gt_boxes.size(0)
        if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
            continue  # nothing to match in this sample

        matches = hungarian_match(pred_boxes, gt_boxes)
        if not matches:
            continue

        # Fetch IoUs for the matched pairs
        ious = iou_matrix_xywh(pred_boxes, gt_boxes)
        for pi, gi in matches:
            iou = float(ious[pi, gi])
            iou_sum += iou
            iou_count += 1
            if iou >= iou_thresh:
                correct_matches += 1

    mean_iou = iou_sum / iou_count if iou_count else 0.0
    recall = correct_matches / total_gt if total_gt else 0.0

    return {"mean_iou": mean_iou, "recall": recall} 