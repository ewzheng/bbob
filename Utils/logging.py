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


def model_size_breakdown(components):
    """
    Get the size of a model in MB

    Parameters:
        - components: a dictionary of components and their sizes

    Returns:
        - size: the size of the model in MB
    """
    lines = ["Model size breakdown (MB):"]
    total = 0.0

    for name, module in components.items():
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
        joined = ", ".join(
            f"{k}={v:.6f}" if isinstance(v, (float, int)) else f"{k}={v}"
            for k, v in logs.items()
        )
        self.logger.info("%s %s", prefix, joined)



