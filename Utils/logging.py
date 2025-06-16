import logging
import os
import sys

from transformers import TrainerCallback

def get_logger(dir, filename="training.log"):
    '''
    Get a logger for the given directory
    '''
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create a file handler
    logfile = os.path.join(dir, filename)
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(logfile) for h in logger.handlers):
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
     
        logger.addHandler(file_handler)

    # create a stream handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(stream_handler)

    return logger


def size_of_module(module):
    '''
    Get the size of a module in MB

    Parameters:
        - module: the module to get the size of

    Returns:
        - size: the size of the module in MB
    '''
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    return total / (1024 ** 2)  # MB


def model_size_breakdown(components, root_name="model"):
    """
    Get the size of a model or collection of components in MB.

    Parameters:
        - components: Either
            * a dict mapping names to modules (or lists/tuples of modules), or
            * a single ``torch.nn.Module`` (or list/tuple of modules).
        - root_name: Name to use when ``components`` is not a dict. Defaults to ``"model"``.

    Returns:
        - A formatted multiline string with the size breakdown in MB.
    """
    lines = ["Model size breakdown (MB):"]
    total = 0.0

    # Detect a BBOB model instance (or other similar wrapper) by attribute names
    if not isinstance(components, dict):
        if all(hasattr(components, attr) for attr in ("vision_tower", "projector", "language_model")):
            # Automatically build a dict with the desired parts
            items = {
                "vision_tower": getattr(components, "vision_tower"),
                "projector"   : getattr(components, "projector"),
                "base_model"  : getattr(components, "language_model"),
            }.items()
        else:
            # Fallbacks handled above if items already set; if not yet set, make single entry
            items = [(root_name, components)]
    else:
        items = components.items()

    for name, module in items:
        if module is None:
            continue

        if isinstance(module, (list, tuple)):
            size_mb = sum(size_of_module(m) for m in module)
        else:
            size_mb = size_of_module(module)

        total += size_mb
        lines.append(f"  {name:<15}: {size_mb:8.2f} MB")

    lines.append("-----------------------------")
    lines.append(f"  Total            : {total:8.2f} MB")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# HuggingFace / TRL Trainer logging callback
# --------------------------------------------------------------------------------------


class LoggingCallback(TrainerCallback):
    """Send Trainer logs to the given logger."""

    def __init__(self, logger: logging.Logger | None = None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)

    def on_log(self, args, state, control, logs=None, **kwargs):  
        if not logs:
            return

        prefix = f"[step {state.global_step}]"
        # Format numbers; show LR in scientific notation so values like 1e-5 are readable
        def _fmt(k, v):
            if not isinstance(v, (float, int)):
                return f"{k}={v}"

            if k == "learning_rate":
                # 6-sigfigs scientific notation, e.g. 1.999750e-05
                return f"{k}={v:.6e}"

            # default – fixed-point with 6 decimals; switch to sci-notation for very small/large
            if abs(v) < 1e-3 or abs(v) >= 1e4:
                return f"{k}={v:.4e}"
            return f"{k}={v:.6f}"

        joined = ", ".join(_fmt(k, v) for k, v in logs.items())
        self.logger.info("%s %s", prefix, joined)

    # Extra callback – compute embedding similarity for the current batch and
    # log it once per step. We use `on_train_batch_end` because HF passes both
    # `inputs` and `model` to that hook (they are *not* available in
    # `on_step_end`).
    def on_train_batch_end(self, args, state, control, **kwargs):
        model  = kwargs.get("model")
        inputs = kwargs.get("inputs")

        if model is None or inputs is None:
            return control

        try:
            from Train.train_common import compute_embedding_similarity
            import torch

            images = inputs.get("pixel_values", inputs.get("images"))
            input_ids = inputs.get("input_ids")

            if images is None or input_ids is None:
                return control

            with torch.no_grad():
                # Model‐specific helpers. Fall back gracefully if absent.
                if hasattr(model, "_prepare_visual_inputs"):
                    vision_feats = model._prepare_visual_inputs(images)
                else:
                    return control

                if hasattr(model, "_embed_tokens"):
                    text_embeds = model._embed_tokens(input_ids.to(model.device))
                else:
                    return control

            sim = compute_embedding_similarity(vision_feats, text_embeds)
            self.logger.info("[step %d] embedding_similarity=%.6f", state.global_step, sim)
        except Exception as e:
            # Fail silently so training is never interrupted by metrics
            self.logger.debug("Similarity computation failed: %s", e)

        return control



