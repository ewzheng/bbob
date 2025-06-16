import logging
import os
import sys
import torch
import torch.nn.functional as F

from transformers import TrainerCallback
from Train import compute_embedding_similarity

def create_metrics_functions(model):
    """
    Factory function that creates compute_metrics and preprocess_logits_for_metrics
    with shared state for accumulating metrics across batches.
    
    This uses closures to avoid global variables while still allowing the two functions
    to share state. Call this once in your train() function to get both functions.
    
    Args:
        model: The model instance to access stored embeddings
    
    Returns:
        tuple: (compute_metrics_func, preprocess_logits_for_metrics_func)
    """
    # Local lists to accumulate metrics (avoiding global variables)
    token_accuracies = []
    top3_accuracies = []
    top5_accuracies = []
    prediction_target_similarities = []

    def preprocess_logits_for_metrics_impl(logits, labels, inputs=None):
        """
        Memory-efficient preprocessing to avoid OOM during evaluation.
        Processes logits immediately instead of accumulating them all in GPU memory.
        
        Now receives inputs parameter with original forward() inputs.
        """
        nonlocal token_accuracies, top3_accuracies, top5_accuracies
        nonlocal prediction_target_similarities
        
        try:
            # Convert logits to predictions immediately (much smaller memory footprint)
            pred_ids = torch.argmax(logits, dim=-1)
            
            # Calculate multiple accuracy metrics
            if labels is not None:
                mask = (labels != -100)
                if mask.sum() > 0:
                    # Exact token accuracy (top-1)
                    correct = (pred_ids == labels) & mask
                    accuracy = correct.sum().float() / mask.sum().float()
                    token_accuracies.append(accuracy.item())
                    
                    # Top-3 accuracy - much more forgiving
                    top3_preds = torch.topk(logits, k=3, dim=-1).indices  # [batch, seq, 3]
                    labels_expanded = labels.unsqueeze(-1).expand_as(top3_preds)  # [batch, seq, 3]
                    top3_correct = (top3_preds == labels_expanded).any(dim=-1) & mask
                    top3_acc = top3_correct.sum().float() / mask.sum().float()
                    top3_accuracies.append(top3_acc.item())
                    
                    # Top-5 accuracy - even more forgiving  
                    top5_preds = torch.topk(logits, k=5, dim=-1).indices  # [batch, seq, 5]
                    labels_expanded = labels.unsqueeze(-1).expand_as(top5_preds)  # [batch, seq, 5]
                    top5_correct = (top5_preds == labels_expanded).any(dim=-1) & mask
                    top5_acc = top5_correct.sum().float() / mask.sum().float()
                    top5_accuracies.append(top5_acc.item())
                    
                    # Prediction-target similarity (most important for projector training)
                    try:
                        batch_size, seq_len, vocab_size = logits.shape
                        probs = F.softmax(logits, dim=-1)
                        
                        # Convert target tokens to one-hot distributions
                        targets_onehot = F.one_hot(labels, num_classes=vocab_size).float()
                        
                        # Only measure on non-masked tokens
                        if mask.sum() > 0:
                            probs_masked = probs[mask]
                            targets_masked = targets_onehot[mask]
                            
                            # Cosine similarity between predicted probs and target distributions
                            pred_target_sims = F.cosine_similarity(probs_masked, targets_masked, dim=-1)
                            mean_pred_target_sim = pred_target_sims.mean().item()
                            prediction_target_similarities.append(mean_pred_target_sim)
                    except Exception as e:
                        print(f"Error calculating prediction-target similarity: {e}")                        
            
            # Return only predictions and labels (much smaller than full logits)
            return pred_ids, labels
            
        except Exception as e:
            print(f"Error in preprocess_logits_for_metrics: {e}")
            # Fallback: just return argmax predictions
            pred_ids = torch.argmax(logits, dim=-1)
            return pred_ids, labels

    def compute_metrics_impl(eval_pred):
        """
        Memory-efficient compute_metrics that works with preprocessed predictions.
        Now has access to inputs via eval_pred.inputs if include_for_metrics=["inputs"] is set.
        """
        nonlocal token_accuracies, top3_accuracies, top5_accuracies
        nonlocal prediction_target_similarities
        
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        
        # Access inputs if available
        inputs = getattr(eval_pred, 'inputs', None)
        if inputs is not None:
            # You can access original inputs here if needed for final metrics computation
            # images = inputs.get('images')
            # input_ids = inputs.get('input_ids')
            pass
        
        # Calculate final metrics from accumulated values
        mean_token_accuracy = sum(token_accuracies) / len(token_accuracies) if token_accuracies else 0.0
        mean_top3_accuracy = sum(top3_accuracies) / len(top3_accuracies) if top3_accuracies else 0.0
        mean_top5_accuracy = sum(top5_accuracies) / len(top5_accuracies) if top5_accuracies else 0.0
        mean_prediction_target_similarity = sum(prediction_target_similarities) / len(prediction_target_similarities) if prediction_target_similarities else 0.0
     
        # Clear accumulated metrics for next evaluation
        token_accuracies.clear()
        top3_accuracies.clear()
        top5_accuracies.clear()
        prediction_target_similarities.clear()
        
        return {
            "exact_token_accuracy": mean_token_accuracy,
            "top3_token_accuracy": mean_top3_accuracy,
            "top5_token_accuracy": mean_top5_accuracy,
            "prediction_target_similarity": mean_prediction_target_similarity,
        }
    
    return compute_metrics_impl, preprocess_logits_for_metrics_impl

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


# New helper to count parameters
def param_count_module(module):
    """
    Get the number of parameters of a module.

    Parameters:
        - module: the module to count parameters for.

    Returns:
        - count: the total number of parameters (int).
    """
    return sum(p.numel() for p in module.parameters())


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
    lines = ["Model size breakdown:"]
    total_mb = 0.0
    total_params = 0

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
            param_cnt = sum(param_count_module(m) for m in module)
        else:
            size_mb = size_of_module(module)
            param_cnt = param_count_module(module)

        total_mb += size_mb
        total_params += param_cnt

        # Format parameters in millions for readability, keep exact count too if desired
        param_millions = param_cnt / 1_000_000
        lines.append(f"  {name:<15}: {size_mb:8.2f} MB | {param_millions:8.2f}M params ({param_cnt:,})")

    lines.append("-----------------------------")
    total_param_millions = total_params / 1_000_000
    lines.append(f"  Total            : {total_mb:8.2f} MB | {total_param_millions:8.2f}M params ({total_params:,})")
    return "\n".join(lines)




class LoggingCallback(TrainerCallback):
    """Send Trainer logs to the given logger."""

    def __init__(self, logger: logging.Logger | None = None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        

    def _fmt(self, k, v):
        if not isinstance(v, (float, int)):
            return f"{k}={v}"

        if k == "learning_rate" or "grad" in k:
            # 6-sigfigs scientific notation, e.g. 1.999750e-05  
            return f"{k}={v:.6e}"
        if "token" in k or "accuracy" in k:
            return f"{k}={v:.2f}"
        if "epoch" in k:
            return f"{k}={v:.2f}"
        # default – fixed-point with 6 decimals; switch to sci-notation for very small
        if abs(v) < 1e-3:
            return f"{k}={v:.4e}"
        return f"{k}={v:.6f}"


    def on_log(self, args, state, control, logs=None, **kwargs):  
        if not logs:
            return
        
        prefix = f"[step {state.global_step}]"

        joined = ", ".join(self._fmt(k, v) for k, v in logs.items())
        self.logger.info("%s %s", prefix, joined)