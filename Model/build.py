import torch

# util packages
import os
import sys

# import modules
from Model.model import BBOB, BBOBConfig

def build_BBOB(
    model_path: str,
    bnb_config=None,
    *,
    memory_pct: float = 0.95,
    device_map: str | dict | None = "auto",
    load: bool = False,
):
    """
    Build or load a BBOB model.
    
    Args:
        model_path: Path to base model or checkpoint directory
        bnb_config: BitsAndBytes configuration
        memory_pct: Memory percentage for device mapping
        device_map: Device map for model distribution
        load: Whether to load from pretrained checkpoint
    """

    # informational print, initialize gpu
    print("Loading BBOB with " + model_path + "...\n") 
    n_gpus = torch.cuda.device_count()

    max_memory = None
    if memory_pct is not None:
        assert 0.0 < memory_pct <= 1.0, "memory_pct should be a 0‥1 fraction"
        max_memory = {
            idx: f"{int(torch.cuda.get_device_properties(idx).total_memory * memory_pct / 1024**2)}MB"
            for idx in range(n_gpus)
        }
        print(f"Capping GPU memory to {int(memory_pct*100)}% ⇒ {max_memory}")

    print("Present working Directory", os.getcwd())
    print(f"Number of GPUs detected: {n_gpus}")

    if load:
        # Loading from checkpoint - override config if needed
        return BBOB.from_pretrained(
            model_path,
            device_map=device_map,
            max_memory=max_memory,
            bnb_config=bnb_config,
        )
    else:
        # Creating new model
        config = BBOBConfig(
            base_model_name=model_path,
            max_memory=max_memory,
            bnb_config=bnb_config,
            device_map=device_map,
        )
        return BBOB(config=config)




