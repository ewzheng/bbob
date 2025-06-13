import torch

# util packages
import os
import sys

# import modules
from .model import BBOB

def build_BBOB(model_path, bnb_config=None, load=False):
    '''
    Build a BBOB model
    '''

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
        return BBOB.from_pretrained(model_path, max_memory, bnb_config)
    
    return BBOB(model_path, max_memory, bnb_config)





