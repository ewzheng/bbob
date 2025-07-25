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
    iou_matrix_xyxy,
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
    label = parts[1].strip().lower() if len(parts)==2 else ""
    xyxy, _ = parse_detection_string(det)  # already (x1,y1,x2,y2)

    return label, xyxy


def _snippets_to_boxes_labels(snippets: List[str]) -> Tuple[torch.Tensor, List[str]]:
    """Convert list of snippets to tensor boxes and parallel label list."""
    coords: List[List[float]] = []
    labels: List[str] = []
    for det in snippets:
        lab, xyxy = _split_snippet(det)
        coords.append(xyxy)
        labels.append(lab)

    if coords:
        return torch.tensor(coords, dtype=torch.float32), labels
    return torch.zeros((0, 4), dtype=torch.float32), labels


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def detection_metrics_batch(
    pred_ids: torch.Tensor,          # (B,S)
    gt_ids: torch.Tensor,            # (B,S)
    tokenizer,
    *,
    logits: torch.Tensor | None = None,  # (B,S,V) optional – enables noise recovery
    iou_thresh: float = 0.5,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """
    Compute mean IoU, recall, precision, F1, accuracy and class_accuracy.

    Minimal patch:
    - Explicit TP/FP/FN counters.
    - Handle empty-pred / empty-GT cases without silently skipping TP updates.
    - Accuracy defined as TP / (TP + FP + FN).
    - Returns TP/FP/FN for debugging.
    """
    if pred_ids.ndim != 2 or gt_ids.ndim != 2:
        raise ValueError("pred_ids and gt_ids must be (B,S) tensors")

    device = pred_ids.device
    batch_size = pred_ids.size(0)

    # For IoU stats
    iou_sum = 0.0
    iou_count = 0

    # Class presence stats (ignores IoU)
    class_total = 0
    class_correct = 0

    # Detection counts
    tp = fp = fn = 0
    total_gt = 0
    total_pred = 0

    # ------------------------------------------------------------------
    # 1. Decode predictions / GT
    # ------------------------------------------------------------------
    if logits is not None:
        if logits.shape[:2] != pred_ids.shape:
            raise ValueError("logits must match (B,S,⋅) shape of pred_ids")

        pred_boxes_labels: List[Tuple[torch.Tensor, List[str]]] = []
        for b in range(batch_size):
            objs = _object_scores(
                pred_ids[b], logits[b], tokenizer, ignore_index=ignore_index
            )
            if objs:
                boxes = torch.tensor([o[2] for o in objs], dtype=torch.float32, device=device)
                labels = [o[1] for o in objs]
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
                labels = []
            pred_boxes_labels.append((boxes, labels))

        gt_filtered: List[List[int]] = [
            [int(t) for t in row.tolist() if t != ignore_index] for row in gt_ids
        ]
        gt_strs: List[str] = tokenizer.batch_decode(
            gt_filtered, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )
        gt_boxes_labels: List[Tuple[torch.Tensor, List[str]]] = []
        for gt_str in gt_strs:
            snips = _extract_snippets(gt_str)
            boxes, labels = _snippets_to_boxes_labels(snips)
            gt_boxes_labels.append((boxes.to(device), labels))

    else:
        pred_filtered: List[List[int]] = [
            [int(t) for t in row.tolist() if t != ignore_index] for row in pred_ids
        ]
        gt_filtered: List[List[int]] = [
            [int(t) for t in row.tolist() if t != ignore_index] for row in gt_ids
        ]

        pred_strs: List[str] = tokenizer.batch_decode(
            pred_filtered, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )
        gt_strs: List[str] = tokenizer.batch_decode(
            gt_filtered,   skip_special_tokens=False, clean_up_tokenization_spaces=True
        )

        pred_boxes_labels = []
        for pred_str in pred_strs:
            snips = _extract_snippets(pred_str)
            boxes, labels = _snippets_to_boxes_labels(snips)
            pred_boxes_labels.append((boxes.to(device), labels))

        gt_boxes_labels = []
        for gt_str in gt_strs:
            snips = _extract_snippets(gt_str)
            boxes, labels = _snippets_to_boxes_labels(snips)
            gt_boxes_labels.append((boxes.to(device), labels))

    # ------------------------------------------------------------------
    # 2. Sample loop with explicit TP/FP/FN
    # ------------------------------------------------------------------
    for (pred_boxes, pred_labels), (gt_boxes, gt_labels) in zip(pred_boxes_labels, gt_boxes_labels):
        n_pred = pred_boxes.size(0)
        n_gt   = gt_boxes.size(0)
        total_pred += n_pred
        total_gt   += n_gt

        # Empty cases -> pure FP/FN
        if n_pred == 0 and n_gt == 0:
            continue
        if n_pred == 0:
            fn += n_gt
            continue
        if n_gt == 0:
            fp += n_pred
            continue

        # Class stats (IoU-independent)
        class_total += len(gt_labels)
        class_correct += sum(1 for lab in gt_labels if lab in pred_labels)

        # IoU matrix
        ious = iou_matrix_xyxy(pred_boxes, gt_boxes)

        # Build candidates (label-equal only)
        candidates: List[Tuple[int, int, float]] = []
        for pi, plab in enumerate(pred_labels):
            for gi, glab in enumerate(gt_labels):
                if plab == glab:
                    iou_val = float(ious[pi, gi])
                    # keep all pairs (even tiny IoU); we'll threshold later
                    candidates.append((pi, gi, iou_val))

        # Greedy assignment by IoU ↓
        candidates.sort(key=lambda x: x[2], reverse=True)
        used_pred: set[int] = set()
        used_gt: set[int] = set()

        for pi, gi, iou_val in candidates:
            if pi in used_pred or gi in used_gt:
                continue
            used_pred.add(pi)
            used_gt.add(gi)

            # track IoU stats
            iou_sum += iou_val
            iou_count += 1

            if iou_val >= iou_thresh:
                tp += 1
            else:
                # label matched but IoU too low → counts as both FP and FN
                fp += 1
                fn += 1

        # Remaining unmatched preds/GT
        fp += (n_pred - len(used_pred))
        fn += (n_gt   - len(used_gt))

    # ------------------------------------------------------------------
    # 3. Aggregate metrics
    # ------------------------------------------------------------------
    mean_iou = iou_sum / iou_count if iou_count else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy  = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    class_accuracy = class_correct / class_total if class_total else 0.0

    return {
        "mean_iou": mean_iou,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,          # TP / (TP+FP+FN)
        "class_accuracy": class_accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_gt": total_gt,
        "total_pred": total_pred,
    }


# -----------------------------------------------------------------------------
# Experimental: object-level confidence & AP (Pix2Seq-style)
# -----------------------------------------------------------------------------

def _recover_noise_label(pred_ids: torch.Tensor, logp_tokens: torch.Tensor, start_idx: int, end_idx: int, tokenizer) -> Tuple[str, float]:
    """Return *(label, score)* recovered from a noise fragment.

    Parameters
    ----------
    pred_ids : LongTensor (S,)
        Full predicted ID sequence for the image.
    logp_tokens : FloatTensor (S,V)
        Log-probabilities (after softmax+log) for the same sequence.
    start_idx : int
        Index of *first* label token (i.e. token right after <|bbob|>).  The
        function scans forward until it meets a token whose **text** contains
        a ':' which marks the end of the label.
    end_idx : int
        Index of the closing tag (`</|bbob|>`); used as safety bound.
    tokenizer : transformers.PreTrainedTokenizerBase

    Returns
    -------
    label : str
        Replacement class name (lower-cased).
    score : float
        Geometric-mean probability of the chosen label tokens.
    """
    bb_open_id, bb_close_id = tokenizer.convert_tokens_to_ids([TAG_OPEN, TAG_CLOSE])
    try:
        noise_id = tokenizer.convert_tokens_to_ids("noise")
    except Exception:
        noise_id = -1

    banned_ids = {
        bb_open_id,
        bb_close_id,
        noise_id,
        tokenizer.pad_token_id or -1,
    }

    label_tokens: list[int] = []
    label_logps: list[torch.Tensor] = []

    k = start_idx
    while k < end_idx:
        tok_id = int(pred_ids[k])
        tok_txt = tokenizer.convert_ids_to_tokens(tok_id)
        if ":" in tok_txt:
            break  # reached delimiter

        logp_row = logp_tokens[k]
        # top-k search for candidate replacement
        top_val, top_idx = torch.topk(logp_row, k=32)
        chosen_id = tok_id  # default
        # CRITICAL: Check if tok_id is valid before accessing logp_row
        if tok_id >= 0 and tok_id < logp_row.size(0):
            chosen_logp = logp_row[tok_id]
        else:
            # Invalid token ID, use a safe default
            chosen_logp = logp_row[0]  # Use first token as fallback
            chosen_id = 0
        
        for cand_id in top_idx.tolist():
            if cand_id in banned_ids or cand_id < 0 or cand_id >= logp_row.size(0):
                continue
            cand_txt = tokenizer.convert_ids_to_tokens(cand_id)
            stripped = cand_txt.strip().replace(".", "", 1)
            if not stripped or stripped.isnumeric():
                continue
            chosen_id = cand_id
            chosen_logp = logp_row[cand_id]
            break

        label_tokens.append(chosen_id)
        label_logps.append(chosen_logp)
        k += 1

    if not label_tokens:
        # CRITICAL: Check if noise_id is valid before accessing logp_tokens
        if noise_id >= 0 and noise_id < logp_tokens.size(-1):
            return "noise", float(torch.exp(logp_tokens[start_idx][noise_id]))
        else:
            return "noise", 0.0

    label = tokenizer.decode(label_tokens, skip_special_tokens=True).strip().lower()
    score = float(torch.exp(torch.stack(label_logps).mean()).clamp(0.0, 1.0))
    return label, score


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
    # ------------------------------------------------------------------
    # NOTE: ``pred_ids`` may contain *ignore_index* (usually ``-100``)
    # values introduced by loss masking / padding.  Using such negative
    # indices with ``gather`` raises "index out of bounds" errors.
    # We therefore replace ignore IDs with a safe token ID (pad token if
    # available else 0) *before* gathering, then zero-out their log-prob so
    # they do not contribute to any downstream score computations.
    # ------------------------------------------------------------------
    safe_pred_ids = pred_ids.clone()
    if (safe_pred_ids == ignore_index).any():
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        safe_pred_ids[safe_pred_ids == ignore_index] = pad_id

    # Final safety: clamp all indices into valid [0, V-1] range so gather can never
    # raise an out-of-bounds error even if some other negative / overflow value
    # slipped through (e.g. data corruption or unexpected mask value).
    vocab_size = logp_tokens.size(1)
    safe_pred_ids = safe_pred_ids.clamp(min=0, max=vocab_size - 1)

    token_logp = logp_tokens.gather(1, safe_pred_ids.unsqueeze(1)).squeeze(1)  # (S,)

    # For completeness, ensure ignore positions contribute zero probability.
    # This keeps tensor shapes unchanged while removing their influence.
    if (pred_ids == ignore_index).any():
        token_logp[pred_ids == ignore_index] = 0.0

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

        # ------------------------------------------------------------------
        # Altered inference (Pix2Seq): if the model emitted the placeholder
        # "noise" label we substitute it with the *most likely* real token at
        # the position of the first label token.  This lets an open-vocabulary
        # LLM name any class it recognises while still enjoying the sequence
        # augmentation benefits of noise fragments during training.
        # ------------------------------------------------------------------
        if label == "noise":
            label, score = _recover_noise_label(pred_ids, logp_tokens, i + 1, j, tokenizer)

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
            ious = iou_matrix_xyxy(preds_boxes, gt_boxes)
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