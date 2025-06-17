import torch

# util packages
import os
import sys

# import modules
from Model.model import BBOB, BBOBConfig

def build_BBOB(model_path, bnb_config=None, max_memory=None, load=False):
    """
    Build or load a BBOB model.
    
    Args:
        model_path: Path to base model or checkpoint directory
        bnb_config: BitsAndBytes configuration
        max_memory: Memory configuration for device mapping
        load: Whether to load from pretrained checkpoint
    """

    # informational print, initialize gpu
    print("Loading BBOB with " + model_path + "...\n") 
    n_gpus = torch.cuda.device_count()
    max_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    max_memory_gb = max_memory_bytes / (1024**3)
    print("Max Memory (GB)", max_memory_gb)
    
    # format max_memory correctly for transformers library
    usable_memory_mb = int((max_memory_bytes * 0.8) / (1024**2))
    max_memory = {0: f"{usable_memory_mb}MB"}

    print("Present working Directory",os.getcwd())
    print(f"Number of GPUs: {n_gpus}")
    print(f"Using max_memory: {max_memory}")

    if load:
        # Loading from checkpoint - override config if needed
        return BBOB.from_pretrained(
            model_path, 
            max_memory=max_memory, 
            bnb_config=bnb_config
        )
    else:
        # Creating new model
        config = BBOBConfig(
            base_model_name=model_path,
            max_memory=max_memory,
            bnb_config=bnb_config
        )
        return BBOB(config=config)




