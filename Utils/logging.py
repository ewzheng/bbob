import logging
import os
import sys
import torch
import torch.nn.functional as F

from transformers import TrainerCallback

from .detection_metrics import detection_metrics_batch

def create_metrics_functions(tokenizer, do_detection_metrics=False, logger=None):
    """
    Factory that returns `(compute_metrics, preprocess_logits_for_metrics)` with **shared**
    state.  Added basic detection metrics (parse-rate, mean IoU) on top of the existing
    language-model metrics.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizerBase
        Needed to decode label sequences for GT box extraction and to feed `_parse_boxes`.
    """
    # Local lists to accumulate metrics (avoiding global variables)
    token_accuracies = []
    top3_accuracies = []
    top5_accuracies = []
    # -- existing scalar accumulators -------------------------------------------
    # CRITICAL: Initialize as scalars to avoid device mismatch issues
    seq_correct_total = 0.0     # number of sequences predicted fully-correct
    seq_total         = 0       # total number of sequences evaluated

    pred_target_sim_sum   = 0.0  # summed cosine similarities
    pred_target_sim_count = 0    # number of token positions contributing

    # Detection-metric accumulators (filled only when do_detection_metrics=True)
    det_iou_vals: list[float]       = []
    det_recall_vals: list[float]    = []
    det_prec_vals: list[float]      = []
    det_f1_vals: list[float]        = []
    det_acc_vals: list[float]       = []
    det_recall25_vals: list[float]  = []
    det_iou25_vals: list[float]     = []
    det_class_acc_vals: list[float] = []

    def preprocess_logits_for_metrics_impl(logits, labels):
        """
        This function converts logits to predictions immediately and accumulates metrics
        in memory-efficient way. The labels should already be properly aligned by the model's
        forward method, so we don't need to adjust them here.
        """
        nonlocal token_accuracies, top3_accuracies, top5_accuracies
        nonlocal seq_correct_total, seq_total
        nonlocal pred_target_sim_sum, pred_target_sim_count
        nonlocal det_iou_vals, det_recall_vals, det_prec_vals, det_f1_vals, det_acc_vals, det_recall25_vals, det_iou25_vals, det_class_acc_vals
        
        try:
            # Ensure labels live on the same device as logits to avoid mismatches
            if labels is not None and labels.device != logits.device:
                labels = labels.to(logits.device)

            # CRITICAL: Check if labels need alignment with logits
            # During evaluation, HF Trainer may pass unaligned labels directly from dataloader
            # while logits come from model with visual token replacement applied
            if labels is not None and logits.shape[1] != labels.shape[1]:
                # Labels are shorter than logits, likely need visual token alignment
                # This happens when logits have visual tokens (64) but labels don't
                seq_diff = logits.shape[1] - labels.shape[1]
                if seq_diff == 63:  # 64 visual tokens - 1 placeholder = 63 difference
                    # Apply the same alignment as the model does internally:
                    # Remove first label (placeholder) and prepend 64 visual ignore tokens
                    batch_size = labels.shape[0]
                    device = logits.device  # Use logits device to ensure consistency
                    
                    # Skip first token (placeholder) from labels
                    labels_after = labels[:, 1:].to(device)  # Ensure on correct device
                    
                    # Create ignore labels for visual tokens
                    visual_ignore = torch.full((batch_size, 64), -100, dtype=labels.dtype, device=device)
                    
                    # Concatenate: visual_ignore + labels_after
                    labels = torch.cat([visual_ignore, labels_after], dim=1)

            # Convert logits to predictions immediately (much smaller memory footprint)
            pred_ids = torch.argmax(logits, dim=-1)
            
            # Calculate multiple accuracy metrics
            if labels is not None:
                mask = (labels != -100)
                mask_sum = mask.sum()
                
                if mask_sum > 0:
                    # Exact token accuracy (top-1)
                    correct = (pred_ids == labels) & mask
                    accuracy = correct.sum().float() / mask_sum.float()
                    # CRITICAL: Move to CPU before storing to avoid device issues
                    token_accuracies.append(accuracy.detach().cpu())
                    
                    # Top-k accuracies (k=3 & 5)
                    top3_preds = torch.topk(logits, k=3, dim=-1).indices   # [B, S, 3]
                    top3_correct = (top3_preds == labels.unsqueeze(-1)).any(dim=-1) & mask
                    top3_acc = top3_correct.sum().float() / mask_sum.float()
                    # CRITICAL: Move to CPU before storing to avoid device issues
                    top3_accuracies.append(top3_acc.detach().cpu())

                    top5_preds = torch.topk(logits, k=5, dim=-1).indices   # [B, S, 5]
                    top5_correct = (top5_preds == labels.unsqueeze(-1)).any(dim=-1) & mask
                    top5_acc = top5_correct.sum().float() / mask_sum.float()
                    # CRITICAL: Move to CPU before storing to avoid device issues
                    top5_accuracies.append(top5_acc.detach().cpu())
                    
                    # Sequence-level accuracy (vectorised)
                    seq_correct = ((pred_ids == labels) | ~mask).all(dim=1)  # (batch,)
                    # CRITICAL: Convert to scalar to avoid device issues
                    seq_correct_total += seq_correct.sum().item()
                    seq_total         += seq_correct.numel()
                    
                    # Prediction-target similarity (most important for projector training)
                    try:
                        batch_size, seq_len, vocab_size = logits.shape
                        probs = F.softmax(logits, dim=-1)

                        # Only operate on non-ignored target positions (labels != -100)
                        if mask_sum > 0:
                            labels_valid  = labels[mask]           # (N,)
                            probs_valid   = probs[mask]            # (N, V)

                            # gather probability mass assigned to the true token
                            p_target = probs_valid.gather(1, labels_valid.unsqueeze(1)).squeeze(1)  # (N,)

                            # cosine similarity with one-hot equals p_true / ||p||₂
                            l2_norm  = probs_valid.norm(dim=1).clamp(min=1e-12)
                            cos_sim  = p_target / l2_norm          # (N,)

                            # CRITICAL: Convert to scalar to avoid device issues
                            pred_target_sim_sum   += cos_sim.sum().item()
                            pred_target_sim_count += cos_sim.numel()

                            # Detection metrics - preserve device correctly
                            if do_detection_metrics:
                                try:
                                    # CRITICAL: Move tensors to CPU before detection metrics to avoid device issues
                                    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                                    pred_masked = pred_ids.masked_fill(~mask, pad_id).cpu()
                                    labels_cpu = labels.cpu()
                                    
                                    det_metrics = detection_metrics_batch(
                                        pred_masked,
                                        labels_cpu,
                                        tokenizer,
                                        logits=logits.detach().cpu(),
                                    )
                                    det_iou_vals.append(det_metrics.get("mean_iou", 0.0))
                                    det_recall_vals.append(det_metrics.get("recall", 0.0))
                                    det_prec_vals.append(det_metrics.get("precision", 0.0))
                                    det_f1_vals.append(det_metrics.get("f1", 0.0))
                                    det_acc_vals.append(det_metrics.get("accuracy", 0.0))
                                    det_class_acc_vals.append(det_metrics.get("class_accuracy", 0.0))

                                    # second threshold (IoU 0.25)
                                    det25 = detection_metrics_batch(
                                        pred_masked,  # Already on CPU
                                        labels_cpu,   # Already on CPU
                                        tokenizer,
                                        logits=logits.detach().cpu(),
                                        iou_thresh=0.25,
                                    )
                                    det_iou25_vals.append(det25.get("mean_iou", 0.0))
                                    det_recall25_vals.append(det25.get("recall", 0.0))
                                except Exception as e:
                                    if logger is not None:
                                        logger.warning(f"Per-batch detection metric failed: {e}")
                                    else:
                                        print(f"Warning: per-batch detection metric failed – {e}")
                    except Exception as e:
                        print(f"Error calculating prediction-target similarity: {e}")
            
            # Return only predictions; HF evaluation API will pass labels separately
            return pred_ids
            
        except Exception as e:
            if logger is not None:
                logger.error(f"Error in preprocess_logits_for_metrics: {e}")
            else:
                print(f"Error in preprocess_logits_for_metrics: {e}")
            # Fallback: just return argmax predictions
            pred_ids = torch.argmax(logits, dim=-1)
            return pred_ids

    def compute_metrics_impl(eval_pred):
        """
        Memory-efficient compute_metrics that works with preprocessed predictions.
        Now has access to inputs via eval_pred.inputs if include_for_metrics=["inputs"] is set.
        """
        nonlocal token_accuracies, top3_accuracies, top5_accuracies
        nonlocal seq_correct_total, seq_total
        nonlocal pred_target_sim_sum, pred_target_sim_count
        nonlocal det_iou_vals, det_recall_vals, det_prec_vals, det_f1_vals, det_acc_vals, det_recall25_vals, det_iou25_vals, det_class_acc_vals
        
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        
        # Calculate final metrics from accumulated values
        # CRITICAL: Only call .item() here at the end, not during training steps
        if token_accuracies:
            mean_token_accuracy = torch.stack(token_accuracies).mean().item()
        else:
            mean_token_accuracy = 0.0
            
        if top3_accuracies:
            mean_top3_accuracy = torch.stack(top3_accuracies).mean().item()
        else:
            mean_top3_accuracy = 0.0
            
        if top5_accuracies:
            mean_top5_accuracy = torch.stack(top5_accuracies).mean().item()
        else:
            mean_top5_accuracy = 0.0
            
        if seq_total > 0:
            mean_sequence_accuracy = seq_correct_total / seq_total
        else:
            mean_sequence_accuracy = 0.0
            
        if pred_target_sim_count > 0:
            mean_prediction_target_similarity = pred_target_sim_sum / pred_target_sim_count
        else:
            mean_prediction_target_similarity = 0.0
        
        # Clear accumulated metrics for next evaluation
        token_accuracies.clear()
        top3_accuracies.clear()
        top5_accuracies.clear()
        # CRITICAL: Reset scalar accumulators properly
        seq_correct_total = 0.0
        seq_total         = 0
        pred_target_sim_sum   = 0.0
        pred_target_sim_count = 0
        
        metrics_out = {
            "exact_token_accuracy": mean_token_accuracy,
            "top3_token_accuracy": mean_top3_accuracy,
            "top5_token_accuracy": mean_top5_accuracy,
            "sequence_accuracy": mean_sequence_accuracy,
            "prediction_target_similarity": mean_prediction_target_similarity,
        }

        # -------------------- aggregate detection metrics --------------------
        if do_detection_metrics:
            mean_iou     = sum(det_iou_vals)     / len(det_iou_vals)     if det_iou_vals else 0.0
            mean_recall  = sum(det_recall_vals)  / len(det_recall_vals)  if det_recall_vals else 0.0
            mean_prec    = sum(det_prec_vals)    / len(det_prec_vals)    if det_prec_vals else 0.0
            mean_f1      = sum(det_f1_vals)      / len(det_f1_vals)      if det_f1_vals else 0.0
            mean_acc     = sum(det_acc_vals)     / len(det_acc_vals)     if det_acc_vals else 0.0
            mean_recall25 = sum(det_recall25_vals) / len(det_recall25_vals) if det_recall25_vals else 0.0
            mean_iou25    = sum(det_iou25_vals)    / len(det_iou25_vals)    if det_iou25_vals else 0.0
            mean_class_acc= sum(det_class_acc_vals)/len(det_class_acc_vals) if det_class_acc_vals else 0.0

            metrics_out.update({
                "mean_iou": mean_iou,
                "recall": mean_recall,
                "precision": mean_prec,
                "f1": mean_f1,
                "accuracy": mean_acc,
                "recall_25": mean_recall25,
                "mean_iou_25": mean_iou25,
                "class_accuracy": mean_class_acc,
            })

            # Reset detection accumulators AFTER metrics have been computed
            det_iou_vals.clear()
            det_recall_vals.clear()
            det_prec_vals.clear()
            det_f1_vals.clear()
            det_acc_vals.clear()
            det_recall25_vals.clear()
            det_iou25_vals.clear()
            det_class_acc_vals.clear()

        return metrics_out
    
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