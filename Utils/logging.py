import logging
import os
import sys
import torch
import torch.nn.functional as F

from transformers import TrainerCallback
from Train import compute_embedding_similarity

def get_logger(dir, filename="training.log"):   
    '''
    Get a logger for the given directory
    '''
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)      

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
        stream_handler.setLevel(logging.DEBUG)
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



class LoggingCallback(TrainerCallback):
    """Send Trainer logs to the given logger."""

    def __init__(self, logger: logging.Logger | None = None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize tracking variables
        self.total_input_tokens = 0
        self.embedding_similarities = []
        self.token_accuracies = []
        
        # Store current step inputs for sharing between methods
        self.current_step_inputs = None

    def _fmt(self, k, v):
        if not isinstance(v, (float, int)):
            return f"{k}={v}"

        if k == "learning_rate":
            # 6-sigfigs scientific notation, e.g. 1.999750e-05
            return f"{k}={v:.6e}"

        # default – fixed-point with 6 decimals; switch to sci-notation for very small/large
        if abs(v) < 1e-3 or abs(v) >= 1e4:
            return f"{k}={v:.4e}"
        return f"{k}={v:.6f}"

    def _count_tokens(self, input_ids):
        """Count the number of tokens in input_ids, excluding padding tokens."""
        if input_ids is None:
            return 0
        
        # Assuming pad_token_id is 0 or -100 (common conventions)
        # You may need to adjust this based on your tokenizer
        if hasattr(input_ids, 'ne'):  # tensor
            return (input_ids != 0).sum().item()
        else:  # assume it's already a count or list
            return len(input_ids) if isinstance(input_ids, (list, tuple)) else input_ids

    def _calculate_embedding_similarity(self, outputs, labels=None):
        """Calculate embedding similarity using the imported function."""
        try:
            return compute_embedding_similarity(outputs, labels)
        except Exception as e:
            self.logger.warning(f"Could not calculate embedding similarity: {e}")
            return None

    def _calculate_token_accuracy(self, outputs, labels):
        """Calculate token-level accuracy."""
        if labels is None or not hasattr(outputs, 'logits'):
            return None
            
        try:
            logits = outputs.logits
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Mask out padding tokens (assuming -100 is ignore index)
            mask = (labels != -100)
            
            if mask.sum() == 0:
                return None
                
            # Calculate accuracy only on non-masked tokens
            correct = (predictions == labels) & mask
            accuracy = correct.sum().float() / mask.sum().float()
            
            return accuracy.item()
        except Exception as e:
            self.logger.warning(f"Could not calculate token accuracy: {e}")
            return None

    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each training step to capture inputs."""
        print(f"on_step_begin kwargs: {list(kwargs.keys())}")
        
        # Store inputs for use in other callback methods
        inputs = kwargs.get("inputs")
        if inputs is not None:
            self.current_step_inputs = inputs
            print(f"Found inputs with keys: {list(inputs.keys())}")
            
            # Count input tokens when inputs are available
            if 'input_ids' in inputs:
                token_count = self._count_tokens(inputs['input_ids'])
                self.total_input_tokens += token_count
                print(f"Counted {token_count} tokens, total: {self.total_input_tokens}")
        else:
            print("No inputs in on_step_begin")

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step to update metrics."""
        print(f"on_step_end kwargs: {list(kwargs.keys())}")
        
        # Get inputs and outputs from kwargs, falling back to stored inputs
        inputs = kwargs.get("inputs") or self.current_step_inputs
        outputs = kwargs.get("outputs")
        
        print(f"Found inputs: {inputs is not None}, outputs: {outputs is not None}")
        if inputs is not None:
            print(f"Input keys: {list(inputs.keys()) if isinstance(inputs, dict) else 'not a dict'}")
        
        if outputs is not None and inputs is not None:
            # Calculate embedding similarity
            labels = inputs.get('labels') if inputs else None
            similarity = self._calculate_embedding_similarity(outputs, labels)
            if similarity is not None:
                self.embedding_similarities.append(similarity)
                print(f"Added similarity: {similarity}")
            
            # Calculate token accuracy
            if labels is not None:
                accuracy = self._calculate_token_accuracy(outputs, labels)
                if accuracy is not None:
                    self.token_accuracies.append(accuracy)
                    print(f"Added accuracy: {accuracy}")
        
        # Clear stored inputs after processing
        self.current_step_inputs = None

    def on_substep_end(self, args, state, control, **kwargs):
        """Called at the end of a substep during gradient accumulation."""
        print(f"on_substep_end kwargs: {list(kwargs.keys())}")
        
        # Also try to capture inputs during gradient accumulation steps
        inputs = kwargs.get("inputs")
        if inputs is not None and self.current_step_inputs is None:
            self.current_step_inputs = inputs
            print(f"Found inputs in substep with keys: {list(inputs.keys())}")
            
            # Count tokens for gradient accumulation steps too
            if 'input_ids' in inputs:
                token_count = self._count_tokens(inputs['input_ids'])
                self.total_input_tokens += token_count
                print(f"Counted {token_count} tokens in substep, total: {self.total_input_tokens}")
        else:
            print(f"No new inputs in substep (inputs available: {inputs is not None})")

    def on_log(self, args, state, control, logs=None, **kwargs):  
        if not logs:
            return

        # Add our custom metrics directly to logs
        logs['total_input_tokens'] = self.total_input_tokens
        
        # Add mean embedding similarity if we have data
        if self.embedding_similarities:
            mean_similarity = sum(self.embedding_similarities) / len(self.embedding_similarities)
            logs['mean_embedding_similarity'] = mean_similarity
        
        # Add mean token accuracy if we have data
        if self.token_accuracies:
            mean_accuracy = sum(self.token_accuracies) / len(self.token_accuracies)
            logs['mean_token_accuracy'] = mean_accuracy

        prefix = f"[step {state.global_step}]"

        joined = ", ".join(self._fmt(k, v) for k, v in logs.items())
        self.logger.info("%s %s", prefix, joined)