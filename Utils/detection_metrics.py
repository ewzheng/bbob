from __future__ import annotations

"""Object-detection evaluation helpers used during validation.

The functions here work purely on *token IDs* so they can be called from the
memory-efficient evaluation loop that keeps logits on the GPU only long enough
for an ``argmax``.  They decode the prediction / ground-truth sequences,
extract the ``<|bbob|>…</|bbob|>`` fragments, convert them to numeric xywh
boxes in the 0‥1 range, and finally run Hungarian matching to compute mean IoU
and recall.
"""

from typing import Dict, List, Tuple, Sequence

import torch

from Train.loss_helpers import (
    TAG_OPEN,
    TAG_CLOSE,
    parse_detection_string,
    iou_matrix_xywh,
)

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


def _split_snippet(det: str) -> Tuple[str, List[float]]:
    """Return *(label, xywh)* for one <bbob> snippet (label lower-cased)."""
    parts = det.split(":", 1)
    label = parts[0].strip().lower() if parts else ""
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
    total_pred = 0
    correct_matches = 0
    iou_sum = 0.0
    iou_count = 0

    # class presence stats (IoU-independent)
    class_total = 0
    class_correct = 0

    # ------------------------------------------------------------------
    # 1. Batch-decode all samples in one tokenizer call (much faster than
    #    decoding each sequence separately, especially for large batches).
    # ------------------------------------------------------------------
    pred_filtered: List[List[int]] = [
        [int(t) for t in row.tolist() if t != ignore_index] for row in pred_ids
    ]
    gt_filtered: List[List[int]] = [
        [int(t) for t in row.tolist() if t != ignore_index] for row in gt_ids
    ]

    # `batch_decode` accepts empty lists fine and returns "" for them.
    pred_strs: List[str] = tokenizer.batch_decode(
        pred_filtered, skip_special_tokens=False, clean_up_tokenization_spaces=True
    )
    gt_strs: List[str] = tokenizer.batch_decode(
        gt_filtered,   skip_special_tokens=False, clean_up_tokenization_spaces=True
    )

    # Iterate over samples; heavy work is now string-processing, not decoding
    for pred_str, gt_str in zip(pred_strs, gt_strs):
        # ---------- extract detection snippets ------------------------------
        pred_snips = _extract_snippets(pred_str)
        gt_snips = _extract_snippets(gt_str)

        pred_boxes, pred_labels = _snippets_to_boxes_labels(pred_snips)
        gt_boxes, gt_labels = _snippets_to_boxes_labels(gt_snips)

        total_gt += gt_boxes.size(0)
        total_pred += pred_boxes.size(0)
        if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
            continue  # nothing to match in this sample

        # ---------------- class accuracy -----------------------------
        class_total += len(gt_labels)
        class_correct += sum(1 for lab in gt_labels if lab in pred_labels)

        # -------- greedy label-aware matching --------------------------------
        used_pred: set[int] = set()
        used_gt: set[int] = set()

        if pred_boxes.numel() and gt_boxes.numel():
            ious = iou_matrix_xywh(pred_boxes, gt_boxes)

            # Build list of candidate pairs (pred, gt, iou) where labels match
            candidates: List[Tuple[int, int, float]] = []
            for pi, plab in enumerate(pred_labels):
                for gi, glab in enumerate(gt_labels):
                    if plab == glab:
                        iou_val = float(ious[pi, gi])
                        if iou_val > 0.0:
                            candidates.append((pi, gi, iou_val))

            # Sort by IoU descending for greedy assignment
            candidates.sort(key=lambda x: x[2], reverse=True)

            for pi, gi, iou_val in candidates:
                if pi in used_pred or gi in used_gt:
                    continue
                used_pred.add(pi)
                used_gt.add(gi)

                iou_sum += iou_val
                iou_count += 1
                if iou_val >= iou_thresh:
                    correct_matches += 1

    mean_iou = iou_sum / iou_count if iou_count else 0.0
    recall = correct_matches / total_gt if total_gt else 0.0
    precision = correct_matches / total_pred if total_pred else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    # Detection accuracy: TP / (TP + FP + FN)
    denom = total_pred + total_gt - correct_matches
    accuracy = correct_matches / denom if denom else 0.0

    class_accuracy = class_correct / class_total if class_total else 0.0

    return {
        "mean_iou": mean_iou,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        "class_accuracy": class_accuracy,
    }

# -----------------------------------------------------------------------------
# Experimental: object-level confidence & AP (Pix2Seq-style)
# -----------------------------------------------------------------------------

def _object_scores(
    pred_ids: torch.Tensor,   # (S,)
    logits: torch.Tensor,     # (S,V) – *pre-softmax* for the same sequence
    tokenizer,
    *,
    ignore_index: int = -100,
) -> List[Tuple[float, str, List[float]]]:
    """Return list of *(score, label, xywh)* for one image.

    Score = exp( mean log-prob(token) ) over the tokens inside a single
    <|bbob|> … </|bbob|> fragment (class + coords + closing tag).
    """

    # 1. Precompute log-prob of each sampled token
    logp_tokens = logits.log_softmax(dim=-1)  # (S,V)
    token_logp = logp_tokens.gather(1, pred_ids.unsqueeze(1)).squeeze(1)  # (S,)

    # 2. Walk through the sequence to find fragments
    bb_open_id = tokenizer.convert_tokens_to_ids(TAG_OPEN)
    bb_close_id = tokenizer.convert_tokens_to_ids(TAG_CLOSE)

    objects: List[Tuple[float, str, List[float]]] = []
    i = 0
    S = pred_ids.size(0)
    while i < S:
        if int(pred_ids[i]) != bb_open_id:
            i += 1
            continue
        # find closing tag index
        j = i + 1
        while j < S and int(pred_ids[j]) != bb_close_id:
            j += 1
        if j >= S:
            break  # unmatched

        # Tokens i .. j inclusive form one fragment
        frag_ids = pred_ids[i + 1 : j]  # exclude <bbob|>, include ':' etc.
        frag_logp = token_logp[i + 1 : j]

        # Score = exp(mean logp)
        if frag_logp.numel():
            score = float(torch.exp(frag_logp.mean()).clamp(0.0, 1.0))
        else:
            score = 0.0

        # Convert IDs → text then parse label / box
        frag_text = tokenizer.decode(frag_ids.tolist(), skip_special_tokens=False)
        label, xywh = _split_snippet(frag_text)
        objects.append((score, label, xywh))
        i = j + 1

    return objects


def average_precision(
    preds: List[Tuple[float, int]],  # (score, is_true_positive)
    total_gt: int,
) -> float:
    """Compute 11-point interpolated AP given TP/FP list sorted by score ↓."""
    if not preds:
        return 0.0

    # CRITICAL: Determine device for tensor operations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Cumulate TP/FP with correct device specification
    tps = torch.tensor([int(tp) for _, tp in preds], device=device, dtype=torch.long).cumsum(0)
    fps = torch.tensor([1 - int(tp) for _, tp in preds], device=device, dtype=torch.long).cumsum(0)
    recalls = tps / max(1, total_gt)
    precisions = tps / (tps + fps)

    # 11-point interpolation (recall 0.0 … 1.0 step 0.1)
    ap = 0.0
    for r in torch.linspace(0.0, 1.0, 11, device=device):
        precisions_r = precisions[recalls >= r]
        p = precisions_r.max() if len(precisions_r) else 0.0
        ap += p / 11
    return float(ap)


def detection_ap_batch(
    logits: torch.Tensor,      # (B,S,V)
    pred_ids: torch.Tensor,    # (B,S)
    gt_ids: torch.Tensor,      # (B,S)
    tokenizer,
    *,
    iou_thresh: float = 0.5,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """Return AP-style metrics using Pix2Seq confidence scores.

    Note: This is a lightweight approximation (11-point AP at IoU=τ) and
    does *not* replicate the full COCO metric suite.
    """

    total_gt_boxes = 0
    pred_score_tp: List[Tuple[float, int]] = []

    for b in range(pred_ids.size(0)):
        ids = pred_ids[b]
        lp = logits[b]
        gt = gt_ids[b]

        # predicted objects with scores
        objects = _object_scores(ids, lp, tokenizer, ignore_index=ignore_index)

        # CRITICAL: Ensure device consistency for tensor operations
        device = logits.device
        preds_boxes = torch.tensor([o[2] for o in objects], dtype=torch.float32, device=device)
        preds_labels = [o[1] for o in objects]
        preds_scores = [o[0] for o in objects]

        # ground truth objects
        gt_str = tokenizer.decode([int(t) for t in gt.tolist() if t != ignore_index], skip_special_tokens=False)
        gt_snips = _extract_snippets(gt_str)
        gt_boxes, gt_labels = _snippets_to_boxes_labels(gt_snips)

        total_gt_boxes += gt_boxes.size(0)

        matched_gt: set[int] = set()
        if preds_boxes.numel() and gt_boxes.numel():
            ious = iou_matrix_xywh(preds_boxes, gt_boxes)
            # Iterate predictions in *score* order
            order = sorted(range(len(objects)), key=lambda k: preds_scores[k], reverse=True)
            for idx in order:
                box = preds_boxes[idx]
                label = preds_labels[idx]
                score = preds_scores[idx]

                # find best matching GT with same label
                best_iou = 0.0
                best_gt = -1
                for gi, (glabel) in enumerate(gt_labels):
                    if gi in matched_gt or glabel != label:
                        continue
                    iou_val = float(ious[idx, gi])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gt = gi

                if best_iou >= iou_thresh:
                    matched_gt.add(best_gt)
                    pred_score_tp.append((score, 1))
                else:
                    pred_score_tp.append((score, 0))
        else:
            # no GT or no preds → all preds are FP
            for score in preds_scores:
                pred_score_tp.append((score, 0))

    # sort global list by score desc
    pred_score_tp.sort(key=lambda x: x[0], reverse=True)
    ap = average_precision(pred_score_tp, total_gt_boxes)

    return {"ap50": ap, "total_gt": total_gt_boxes, "total_pred": len(pred_score_tp)} 