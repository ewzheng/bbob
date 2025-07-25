import random
import torch
import torch.nn.functional as F  # required for some helpers (retained just in case)

# Constants – keep consistent with Train.train_collate but avoid circular import
MIN_COORD = 1.0 / 1000.0  # 0.001

__all__ = [
    "_generate_noise_boxes_batch",
    "_generate_noise_boxes",
    "_build_and_process_sequence",
    "_truncate_at_fragment_boundary",
    "jitter_bboxes_norm",
]


def _generate_noise_boxes_batch(self, batch_gt_boxes, batch_label_strs, batch_noise_counts):
    """Vectorised Pix2Seq-style noise box generation for a *batch*.

    Parameters
    ----------
    self : BBOBCollator
        The collator instance – required for access to logger and other helpers.
    batch_gt_boxes : list[Tensor]
        List of GT box tensors (N_i,4) per sample.
    batch_label_strs : list[list[str]]
        Corresponding list of label-string lists per sample.
    batch_noise_counts : list[int]
        How many noise boxes to generate for each sample.

    Returns
    -------
    list[tuple[Tensor, list[str]]]
        Per-sample tuple of (noise_boxes, noise_labels).
    """

    batch_size = len(batch_gt_boxes)
    batch_results = [None] * batch_size

    all_gt_boxes = []
    box_to_sample_map = []
    sample_label_pools: dict[int, list[str]] = {}

    # ------------------------------------------------------------------
    # 1) Gather GT boxes across the batch for vectorised generation
    # ------------------------------------------------------------------
    for sample_idx, (gt_boxes, label_strs, num_noise) in enumerate(
        zip(batch_gt_boxes, batch_label_strs, batch_noise_counts)
    ):
        if num_noise <= 0:
            device = gt_boxes.device if gt_boxes.numel() > 0 else torch.device("cpu")
            batch_results[sample_idx] = (torch.empty((0, 4), device=device), [])
            continue

        sample_label_pools[sample_idx] = label_strs

        for box in gt_boxes:
            all_gt_boxes.append(box)
            box_to_sample_map.append(sample_idx)

    if not all_gt_boxes:
        # Fallback when no GT boxes in entire batch
        for sample_idx, num_noise in enumerate(batch_noise_counts):
            if batch_results[sample_idx] is None:
                device = (
                    batch_gt_boxes[sample_idx].device
                    if batch_gt_boxes[sample_idx].numel() > 0
                    else torch.device("cpu")
                )
                noise_boxes, noise_labels = _generate_noise_boxes(
                    self, num_noise, ["object"], torch.empty((0, 4), device=device)
                )
                batch_results[sample_idx] = (noise_boxes, noise_labels)
        return batch_results

    all_gt_boxes_tensor = torch.stack(all_gt_boxes)

    total_noise_needed = sum(num for num in batch_noise_counts if num > 0)
    if total_noise_needed == 0:
        return batch_results

    n_jittered = max(1, int(total_noise_needed * 0.4))
    n_shifted = max(1, int(total_noise_needed * 0.3))
    n_random = total_noise_needed - n_jittered - n_shifted

    all_noise_boxes = []
    all_noise_labels = []
    noise_sample_assignments = []

    # ------------- helper lambdas --------------------------------------
    rand_device = all_gt_boxes_tensor.device
    rand_u = lambda *size: torch.rand(*size, device=rand_device)

    # ------------------------------------------------------------------
    # 1) JITTERED BOXES
    # ------------------------------------------------------------------
    if len(all_gt_boxes) > 0 and n_jittered > 0:
        jitter_indices = torch.randint(0, len(all_gt_boxes), (n_jittered,), device=rand_device)
        boxes_to_jitter = all_gt_boxes_tensor[jitter_indices]
        x, y, w, h = boxes_to_jitter.T  # unpack columns

        jitter_x = (rand_u(n_jittered) * 2.0 - 1.0) * 0.4 * w
        jitter_y = (rand_u(n_jittered) * 2.0 - 1.0) * 0.4 * h
        jitter_w = (rand_u(n_jittered) * 2.0 - 1.0) * 0.3 * w
        jitter_h = (rand_u(n_jittered) * 2.0 - 1.0) * 0.3 * h

        new_x = (x + jitter_x).clamp_min(0.0)
        new_y = (y + jitter_y).clamp_min(0.0)
        new_w = (w + jitter_w).clamp_min(0.01)
        new_h = (h + jitter_h).clamp_min(0.01)

        new_x = torch.minimum(new_x, 1.0 - new_w)
        new_y = torch.minimum(new_y, 1.0 - new_h)
        new_w = torch.minimum(new_w, 1.0 - new_x)
        new_h = torch.minimum(new_h, 1.0 - new_y)

        jittered_boxes = torch.stack([new_x, new_y, new_w, new_h], dim=1)
        all_noise_boxes.append(jittered_boxes)

        for idx in jitter_indices.tolist():
            noise_sample_assignments.append(box_to_sample_map[idx])
            all_noise_labels.append("noise")

    # ------------------------------------------------------------------
    # 2) SHIFTED BOXES
    # ------------------------------------------------------------------
    if len(all_gt_boxes) > 0 and n_shifted > 0:
        shift_indices = torch.randint(0, len(all_gt_boxes), (n_shifted,), device=rand_device)
        boxes_to_shift = all_gt_boxes_tensor[shift_indices]
        w, h = boxes_to_shift[:, 2], boxes_to_shift[:, 3]

        cx = rand_u(n_shifted) * (1.0 - w) + w / 2
        cy = rand_u(n_shifted) * (1.0 - h) + h / 2

        new_x = torch.clamp(cx - w / 2, min=MIN_COORD)
        new_y = torch.clamp(cy - h / 2, min=MIN_COORD)

        shifted_boxes = torch.stack([new_x, new_y, w, h], dim=1)
        all_noise_boxes.append(shifted_boxes)

        for idx in shift_indices.tolist():
            noise_sample_assignments.append(box_to_sample_map[idx])
            all_noise_labels.append("noise")

    # ------------------------------------------------------------------
    # 3) RANDOM BOXES
    # ------------------------------------------------------------------
    if n_random > 0:
        x = rand_u(n_random) * 0.8
        y = rand_u(n_random) * 0.8
        max_w = torch.minimum(torch.full_like(x, 0.5), 1.0 - x)
        max_h = torch.minimum(torch.full_like(y, 0.5), 1.0 - y)
        w = rand_u(n_random) * max_w
        h = rand_u(n_random) * max_h

        random_boxes = torch.stack([x, y, w, h], dim=1)
        all_noise_boxes.append(random_boxes)

        # Assign to random samples that still need noise
        samples_needing_noise = [i for i, cnt in enumerate(batch_noise_counts) if cnt > 0]
        for _ in range(n_random):
            noise_sample_assignments.append(random.choice(samples_needing_noise))
            all_noise_labels.append("noise")

    # ------------------------------------------------------------------
    # 4) COLLATE & DISTRIBUTE BACK TO SAMPLES
    # ------------------------------------------------------------------
    if all_noise_boxes:
        combined_noise_boxes = torch.cat(all_noise_boxes, dim=0)
    else:
        combined_noise_boxes = torch.empty((0, 4), device=rand_device)

    sample_noise_lists = {
        idx: ([], []) for idx in range(batch_size) if batch_noise_counts[idx] > 0
    }

    for box, label, tgt in zip(
        combined_noise_boxes, all_noise_labels, noise_sample_assignments
    ):
        if tgt in sample_noise_lists:
            sample_noise_lists[tgt][0].append(box)
            sample_noise_lists[tgt][1].append(label)

    # Ensure exact noise count per sample
    for idx in range(batch_size):
        requested = batch_noise_counts[idx]
        if requested <= 0:
            if batch_results[idx] is None:
                device = (
                    batch_gt_boxes[idx].device
                    if batch_gt_boxes[idx].numel() > 0
                    else torch.device("cpu")
                )
                batch_results[idx] = (torch.empty((0, 4), device=device), [])
            continue

        boxes_list, labels_list = sample_noise_lists.get(idx, ([], []))
        if len(boxes_list) < requested:
            missing = requested - len(boxes_list)
            extra_boxes, extra_labels = _generate_noise_boxes(
                self,
                missing,
                batch_label_strs[idx] or ["noise"],
                batch_gt_boxes[idx],
            )
            boxes_list.extend(extra_boxes)
            labels_list.extend(extra_labels)

        if boxes_list:
            boxes_tensor = (
                torch.stack(boxes_list[:requested])
                if isinstance(boxes_list[0], torch.Tensor)
                else torch.as_tensor(boxes_list[:requested])
            )
            labels_final = labels_list[:requested]
        else:
            device = (
                batch_gt_boxes[idx].device
                if batch_gt_boxes[idx].numel() > 0
                else torch.device("cpu")
            )
            boxes_tensor = torch.empty((0, 4), device=device)
            labels_final = []

        batch_results[idx] = (boxes_tensor, labels_final)

    return batch_results


def _generate_noise_boxes(self, num_boxes, label_strs, gt_boxes):
    """Generate noise boxes for a *single* sample (Pix2Seq spec)."""
    if num_boxes <= 0:
        return torch.empty((0, 4)), []

    noise_boxes = []
    noise_labels = []

    n_jittered = max(1, int(num_boxes * 0.4))
    n_shifted = max(1, int(num_boxes * 0.3))
    n_random = num_boxes - n_jittered - n_shifted

    # Jittered
    if len(gt_boxes) > 0:
        for _ in range(n_jittered):
            gt_idx = random.randint(0, len(gt_boxes) - 1)
            x, y, w, h = gt_boxes[gt_idx].tolist()
            jit_x = random.uniform(-0.4, 0.4) * w
            jit_y = random.uniform(-0.4, 0.4) * h
            jit_w = random.uniform(-0.3, 0.3) * w
            jit_h = random.uniform(-0.3, 0.3) * h
            new_x = max(0.0, min(1.0 - w, x + jit_x))
            new_y = max(0.0, min(1.0 - h, y + jit_y))
            new_w = max(0.01, min(1.0 - new_x, w + jit_w))
            new_h = max(0.01, min(1.0 - new_y, h + jit_h))
            noise_boxes.append([new_x, new_y, new_w, new_h])
            noise_labels.append("noise")
    else:
        n_random += n_jittered
        n_jittered = 0

    # Shifted
    if len(gt_boxes) > 0:
        for _ in range(n_shifted):
            gt_idx = random.randint(0, len(gt_boxes) - 1)
            _, _, w, h = gt_boxes[gt_idx].tolist()
            cx = random.uniform(w / 2, 1.0 - w / 2)
            cy = random.uniform(h / 2, 1.0 - h / 2)
            new_x = cx - w / 2
            new_y = cy - h / 2
            noise_boxes.append([new_x, new_y, w, h])
            noise_labels.append("noise")
    else:
        n_random += n_shifted
        n_shifted = 0

    # Random boxes
    for _ in range(n_random):
        x = random.uniform(MIN_COORD, 0.8)
        y = random.uniform(MIN_COORD, 0.8)
        w = random.uniform(0.1, min(0.5, 1.0 - x))
        h = random.uniform(0.1, min(0.5, 1.0 - y))
        noise_boxes.append([x, y, w, h])
        noise_labels.append("noise")

    noise_boxes_tensor = torch.tensor(noise_boxes, dtype=torch.float32)
    return noise_boxes_tensor, noise_labels


def _build_and_process_sequence(self, boxes, labels, noise_mask, max_length, _unused=None):
    """Tokenise bbox + label fragments and build input/target sequences."""

    if not labels:
        empty = torch.empty(0, dtype=torch.long, device=boxes.device)
        return empty, empty

    fragments = []
    is_noise_list = []
    noise_mask_list = noise_mask.tolist() if noise_mask is not None else [False] * len(labels)

    for bb, lab, is_noise in zip(boxes.tolist(), labels, noise_mask_list):
        x, y, w, h = bb
        x2, y2 = x + w, y + h
        coord_txt = ", ".join(self.fmt_coord(v) for v in (x, y, x2, y2))
        fragments.append(f"<|bbob|>[{coord_txt}]: {lab}</|bbob|>")
        is_noise_list.append(bool(is_noise))

    fragments_pref = [fragments[0]] + [" " + f for f in fragments[1:]]
    tok_lists = self.tokenizer(fragments_pref, add_special_tokens=False).input_ids

    inp_flat, tgt_flat = [], []
    for toks, is_noise in zip(tok_lists, is_noise_list):
        inp_flat.extend(toks)
        if not is_noise:
            tgt_flat.extend(toks)
        else:
            tok_texts = self.tokenizer.convert_ids_to_tokens(toks)
            coord_start = next((i for i, t in enumerate(tok_texts) if "[" in t), len(toks))
            coord_end = next((i for i, t in enumerate(tok_texts) if "]" in t and i >= coord_start), len(toks)-1)

            # 1) keep tokens before '[' (opening bracket)
            tgt_flat.extend(toks[:coord_start])

            # 2) process tokens from '[' up to ']' inclusive
            for tid, ttxt in zip(toks[coord_start:coord_end+1], tok_texts[coord_start:coord_end+1]):
                if any(sym in ttxt for sym in "[],"):
                    tgt_flat.append(tid)  # keep bracket/comma delimiters
                else:
                    tgt_flat.append(-100)  # mask numeric coordinate value

            # 3) keep everything after ']' (colon + label tokens)
            tgt_flat.extend(toks[coord_end+1:])

    if max_length is not None and max_length > 0 and len(inp_flat) > max_length:
        inp_tensor = torch.tensor(inp_flat, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_flat, dtype=torch.long)
        inp_flat = _truncate_at_fragment_boundary(self, inp_tensor, max_length).tolist()
        tgt_flat = _truncate_at_fragment_boundary(self, tgt_tensor, max_length).tolist()

    min_len = min(len(inp_flat), len(tgt_flat))
    inp_flat, tgt_flat = inp_flat[:min_len], tgt_flat[:min_len]

    return (
        torch.tensor(inp_flat, dtype=torch.long, device=boxes.device),
        torch.tensor(tgt_flat, dtype=torch.long, device=boxes.device),
    )


def _truncate_at_fragment_boundary(self, token_ids, max_length):
    """Truncate token sequence so we never cut a <|bbob|>…</|bbob|> fragment mid-way."""

    if token_ids.size(0) <= max_length:
        return token_ids

    if self.close_id == -1 or self.close_id is None or self.open_id is None:
        return token_ids[:max_length]

    tokens = token_ids.tolist()
    last_complete_end = 0
    i = 0
    while i < min(len(tokens), max_length):
        if tokens[i] == self.open_id:
            depth = 1
            j = i + 1
            while j < len(tokens) and depth > 0:
                if tokens[j] == self.open_id:
                    depth += 1
                elif tokens[j] == self.close_id:
                    depth -= 1
                j += 1
            if j <= max_length:
                last_complete_end = j
                i = j
            else:
                break
        else:
            i += 1
    if last_complete_end > 0:
        return token_ids[:last_complete_end]
    else:
        return torch.empty(0, dtype=token_ids.dtype, device=token_ids.device)


def jitter_bboxes_norm(self, bboxes, dtype, jitter_ratio=0.05):
    """Vectorised jitter for normalised (x,y,w,h) bboxes in 0‥1 space."""
    
    if not isinstance(bboxes, torch.Tensor):
        bboxes = torch.as_tensor(bboxes, dtype=dtype)
    if bboxes.numel() == 0:
        return bboxes.to(dtype=dtype)

    device = bboxes.device
    bx = bboxes.to(dtype=dtype, device=device).clone()

    cxcy = bx[:, :2] + 0.5 * bx[:, 2:]
    wh = bx[:, 2:]

    trans_rng = (torch.rand_like(cxcy, device=device) * 2.0 - 1.0) * jitter_ratio
    cxcy = cxcy + trans_rng * wh

    scale_rng = (torch.rand_like(wh, device=device) * 2.0 - 1.0) * jitter_ratio
    wh = wh * (1.0 + scale_rng)

    xy = cxcy - 0.5 * wh
    xy = torch.clamp(xy, MIN_COORD, 1.0)

    w_nonneg = torch.clamp_min(wh[:, 0], 0.0)
    h_nonneg = torch.clamp_min(wh[:, 1], 0.0)

    w_clamped = torch.minimum(w_nonneg, 1.0 - xy[:, 0])
    h_clamped = torch.minimum(h_nonneg, 1.0 - xy[:, 1])
    wh = torch.stack([w_clamped, h_clamped], dim=1)

    return torch.cat([xy, wh], dim=1) 