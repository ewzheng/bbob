from typing import Dict, List, Tuple

# Global mapping – lives in one process; each Dataloader worker will create its
# own mapping when `ensure_init` is called.  The IDs are deterministic (sorted
# by the original dataset label indices) so there is no mismatch across
# processes even though the dicts are not shared.
SEQ2ID: Dict[Tuple[int, ...], int] = {}
ID2SEQ: List[Tuple[int, ...]] = []
_INITIALISED: bool = False


def init_from_labels(label_lookup: Dict[int, str], tokenizer) -> None:
    """Initialise the mapping from dataset *label_lookup*.

    Parameters
    ----------
    label_lookup: dict[int, str]
        Mapping of canonical *dataset class index → human-readable phrase*.
    tokenizer
        Tokeniser used to turn phrases into token-id lists (must be the same
        everywhere: dataset preprocessing *and* loss computation).
    """
    global _INITIALISED
    if _INITIALISED:
        return

    SEQ2ID.clear()
    ID2SEQ.clear()

    for cls_idx in sorted(label_lookup):
        toks = tuple(tokenizer(label_lookup[cls_idx], add_special_tokens=False)["input_ids"])
        SEQ2ID[toks] = cls_idx
        if len(ID2SEQ) <= cls_idx:
            # pad ID2SEQ so indices line up with cls_idx
            ID2SEQ.extend([()] * (cls_idx + 1 - len(ID2SEQ)))
        ID2SEQ[cls_idx] = toks

    _INITIALISED = True


def get_id(token_ids: List[int]) -> int:
    """Return the integer ID for a class phrase (or −1 if unseen)."""
    return SEQ2ID.get(tuple(token_ids), -1) 