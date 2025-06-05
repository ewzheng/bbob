import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import logging
from datetime import datetime
from torch.utils.data import DataLoader
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, AutoPeftModelForCausalLM
import pandas as pd
from trl import SFTTrainer
import matplotlib.pyplot as plt
import plotext as pltt
from tqdm.auto import tqdm
import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Model.model import BBOB
from train_common import load_and_prepare_dataset, load_model

def fine_tune(model, dataset, vision_tower, instruction, lora_rank, lora_alpha, lora_dropout, learning_rate, batch_size, gradient_accumulation_steps, max_steps, save_steps, logging_steps):
    
    
    return
    

def main():
    parser = argparse.ArgumentParser(description="Fine-tune BBOB with LoRA")
    parser.add_argument("-m", "--model", required=True, help="Model location/path")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset name from HuggingFace Hub")  
    parser.add_argument("-v", "--vision_tower", required=True, help="Vision tower model location/path")
    parser.add_argument("-i", "--instruction", required=True, help="Instruction text to add to dataset examples")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout (default: 0.1)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--max_steps", type=int, default=2048, help="Maximum training steps (default: 2048)")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps (default: 100)")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps (default: 10)")

    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    print(f"Loading dataset from: {args.dataset}")
    print(f"Max training steps: {args.max_steps}")
    print(f"Vision tower: {args.vision_tower}")
    print(f"Instruction: {args.instruction}")
    print(f"LoRA config: rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Training config: lr={args.learning_rate}, batch_size={args.batch_size}, grad_accum={args.gradient_accumulation_steps}")
    

    return

if __name__ == "__main__":
    main()