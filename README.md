# BBOB

BBOB is an end-to-end deep learning framework for composite vision-language modeling, featuring a modular architecture with a vision tower, projector, and language model. It supports training for detection, vision, and projector components.

## Installation

1. **Clone the repository**
2. **Set up the environment:**
   - Using Conda:  
     ```
     conda env create -f environment.yml
     conda activate bbobenv
     ```
   - Or install dependencies from `requirements.txt`:
     ```
     pip install -r requirements.txt
     ```

## Usage

### Train Detection
```sh
python train_detection.py -m <checkpoint_with_projector> -d <dataset_name> -i <instruction_text> [other args]
```

### Train Projector
```sh
python train_projector.py -m <base_llm_path> -d <dataset_name> -e <epochs> -i <instruction_text>
```

### Train Vision
```sh
python train_vision.py -m <checkpoint_with_projector> -d <dataset_name> -i <instruction_text> [other args]
```

## Directory Structure

- `Model/`: Model components (vision tower, projector, main model)
- `Train/`: Training utilities and scripts
- `Utils/`: Metrics and logging
- `Checkpoints/`: Saved model checkpoints
- `Labels/`: Dataset label configs

---

**Note:** This repository is partially AI-generated and may contain code or documentation produced by automated systems.

