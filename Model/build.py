import torch

# util packages
import os
import sys

# import modules
from Model.model import BBOB

def build_BBOB(model_path, bnb_config=None, load=False):
    '''
    construct or load a bbob model instance.

    parameters:
        - model_path (str): hf repo or local ckpt dir of the base llm.
        - bnb_config (str|None): quantisation mode – {"8bit","4bit","bf16","fp16"}.
        - load (bool): when true, load from `model_path` via
          `BBOB.from_pretrained`; otherwise initialise new weights.

    returns: BBOB model ready for training/inference.
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





