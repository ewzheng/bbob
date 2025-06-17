import torch
import torch.nn.functional as F
import re

def _parse_boxes(logits, tokenizer):
    """Decode logits and extract detection strings.

    Args:
        logits (torch.Tensor): Model output logits with shape (batch_size, seq_len, vocab_size).
        tokenizer: Tokenizer that provides a `decode(ids, **kwargs)` method.

    Returns:
        List[List[str]]: A list with one entry per batch element, containing all detection strings
        (the raw content found between <detect> and </detect> tokens).
    """

    # Safety check: ensure we have the expected tensor dimensionality
    if logits.dim() != 3:
        raise ValueError(
            f"Expected `logits` to have shape (batch, seq_len, vocab_size) but got tensor with shape {logits.shape}"
        )

    # Greedy decoding – take the most probable token at each position
    token_ids = logits.argmax(dim=-1)  # (batch_size, seq_len)

    detections_batch = []

    for ids in token_ids:
        # Convert tensor row to python list for the tokenizer
        ids_list = ids.tolist()

        # Decode the sequence into text. We skip special tokens to avoid extraneous artifacts.
        decoded_text = tokenizer.decode(ids_list, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Use regex to find every substring wrapped by <detect> ... </detect>
        # The non-greedy qualifier (.*?) ensures we catch individual detections.
        matches = re.findall(r"<detect>(.*?)</detect>", decoded_text)

        # Strip surrounding whitespace from each detection
        detections = [m.strip() for m in matches]

        detections_batch.append(detections)

    return detections_batch

def compute_loss(model, inputs):
    loss = 0

    return loss