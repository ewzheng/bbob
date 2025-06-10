'''
File: model.py
Author: Elias Zheng and Claude
Description: This script contains the BBOB model class.
'''

import torch
import torch.nn as nn
import transformers
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb
import os
from .projector import Projector

class BBOB(nn.Module):
    """
    BBOB multimodal model combining vision encoder, projector, and language model
    """
    
    def __init__(self, model_path, vision_encoder, bnb_config, num_classes=80, num_queries=50):
        """
        Initialize BBOB model with specified components
        
        Parameters:
            - base_model: path or identifier for base language model
            - vision_encoder: path or identifier for vision encoder
            - bnb_config: quantization configuration for model loading
            - num_classes: number of classes for detection head (default 80 for COCO)
            - num_queries: number of object queries for DETR-style detection (default 100)
        """
        super(BBOB, self).__init__()
        # informational print, initialize gpu
        print("Loading BBOB with " + model_path + " and " + vision_encoder + "...\n") 
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

        if bnb_config == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif bnb_config == "4bit":
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        elif bnb_config == "bf16":
            bnb_config = BitsAndBytesConfig(load_in_bf16=True)
        elif bnb_config == "fp16":
            bnb_config = BitsAndBytesConfig(load_in_fp16=True)
        else:
            bnb_config = None

        # initialize components
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(model_path, max_memory=max_memory, quantization_config=bnb_config, device_map="auto", torch_dtype="auto")
        base_model_dtype = next(self.base_model.parameters()).dtype

        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.image_processor = transformers.AutoImageProcessor.from_pretrained(vision_encoder, use_fast=True, torch_dtype=base_model_dtype)
        self.vision_encoder = transformers.AutoModel.from_pretrained(vision_encoder)

        # get vision encoder hidden size (different attributes for different vision models)
        # always use dummy forward pass to get actual output dimensions
        print(f"Detecting vision encoder hidden size for {vision_encoder}...")
        dummy_input = torch.randn(1, 3, 256, 256).to(next(self.vision_encoder.parameters()).device)
        with torch.no_grad():
            dummy_output = self.vision_encoder(dummy_input)
            vision_features = dummy_output.last_hidden_state
            print(f"Vision encoder output shape: {vision_features.shape}")
            
            # handle different output formats
            if vision_features.dim() == 4:  # [batch, height, width, channels]
                # flatten spatial dimensions like the projector does
                vision_features = vision_features.flatten(2).transpose(1, 2)  # [batch, seq_len, channels]
                vision_hidden_size = vision_features.shape[-1]
            elif vision_features.dim() == 3:  # [batch, seq_len, channels]
                vision_hidden_size = vision_features.shape[-1]
            else:
                vision_hidden_size = vision_features.shape[1]
                
        print(f"Detected vision hidden size: {vision_hidden_size}")

        self.projector = Projector(vision_hidden_size, self.base_model.config.hidden_size)

        self.num_classes = num_classes
        self.num_queries = num_queries 
        self.query_embed = nn.Embedding(self.num_queries, self.base_model.config.hidden_size)
        self.detection_head = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.num_classes + 1)  # +1 for no-object class
        )
        self.bbox_head = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
        # Move heads and queries to same device as projector
        device = next(self.projector.parameters()).device
        self.detection_head = self.detection_head.to(device)
        self.bbox_head = self.bbox_head.to(device)
        self.query_embed = self.query_embed.to(device)
        
        # move components to GPU and match base model dtype
        if torch.cuda.is_available():
            self.vision_encoder = self.vision_encoder.to('cuda', dtype=base_model_dtype)
            print(f"Vision encoder loaded on: {next(self.vision_encoder.parameters()).device}, dtype: {next(self.vision_encoder.parameters()).dtype}")
            self.projector = self.projector.to('cuda', dtype=base_model_dtype)
            print(f"Projector loaded on: {next(self.projector.parameters()).device}, dtype: {next(self.projector.parameters()).dtype}")
        


    ''' Helpers for interacting with internal components'''
    def get_tokenizer(self):
        """
        Get the tokenizer for text processing
        
        Returns:
            - tokenizer: the base model tokenizer
        """
        return self.base_tokenizer
    
    def get_vision_tower(self):
        """
        Get the vision processing components
        
        Returns:
            - tuple: image processor and vision encoder
        """
        return self.image_processor, self.vision_encoder
    
    def freeze_projector(self):
        """
        Freeze projector parameters to prevent training
        """
        self.projector.freeze()
    
    def unfreeze_projector(self):
        """
        Unfreeze projector parameters to enable training
        """
        self.projector.unfreeze()

    def freeze_model(self):
        """
        Freeze base language model parameters
        """
        for weights in self.base_model.parameters():
            weights.requires_grad = False
    
    def unfreeze_model(self):
        """
        Unfreeze base language model parameters
        """
        for weights in self.base_model.parameters():
            weights.requires_grad = True
    
    def freeze_vision_tower(self):
        """
        Freeze vision encoder and image processor parameters
        """
        for weights in self.vision_encoder.parameters():
            weights.requires_grad = False
        # image processor typically has no trainable parameters

    def unfreeze_vision_tower(self):
        """
        Unfreeze vision encoder and image processor parameters
        """
        for weights in self.vision_encoder.parameters():
            weights.requires_grad = True
        # image processor typically has no trainable parameters

    def unfreeze_heads(self):
        for weights in self.bbox_head.parameters():
            weights.requires_grad = True
        for weights in self.detection_head.parameters():
            weights.requires_grad= True

    def freeze_heads(self):
        for weights in self.bbox_head.parameters():
            weights.requires_grad = False
        for weights in self.detection_head.parameters():
            weights.requires_grad= False

    def train(self):
        """
        Set all model components to training mode
        """
        self.base_model.train()
        self.vision_encoder.train()
        self.projector.train()

    def eval(self):
        """
        Set all model components to evaluation mode
        """
        self.base_model.eval()
        self.vision_encoder.eval()
        self.projector.eval()
        
    ''' Input processing helpers '''
    def _prepare_visual_inputs(self, vision_in):
        """
        Process raw images into visual tokens
        
        Parameters:
            - vision_in: raw images, image paths, or preprocessed tensors
            
        Returns:
            - visual_tokens: projected visual features ready for LLM
        """
        if vision_in is None:
            return None
            
        # handle different input types - strings, tensors, PIL images
        if isinstance(vision_in, (list, tuple)) and isinstance(vision_in[0], str):
            # image paths or URLs provided
            processed_images = self.image_processor(vision_in, return_tensors="pt")
            pixel_values = processed_images['pixel_values']
        else:
            # assume raw tensors or PIL images
            if not isinstance(vision_in, torch.Tensor):
                processed_images = self.image_processor(vision_in, return_tensors="pt")
                pixel_values = processed_images['pixel_values']
            else:
                pixel_values = vision_in
        
        # move to correct device - match vision encoder
        pixel_values = pixel_values.to(next(self.vision_encoder.parameters()).device)
        
        # extract visual features from vision encoder
        vision_outputs = self.vision_encoder(pixel_values)
        vision_features = vision_outputs.last_hidden_state  # mobilevit returns 4D tensor
        
        # project visual features to text embedding space
        visual_tokens = self.projector(vision_features)  # handles 4D -> 3D conversion internally
        
        return visual_tokens

    def _prepare_text_inputs(self, text_in):
        """
        Process text inputs into embeddings and attention masks
        
        Parameters:
            - text_in: text strings or pre-tokenized input_ids
            
        Returns:
            - text_embeddings: text converted to embedding space
            - attention_mask: mask for padded tokens
        """
        if text_in is None:
            return None, None
            
        # handle string inputs vs pre-tokenized
        if isinstance(text_in, str) or (isinstance(text_in, list) and isinstance(text_in[0], str)):
            # tokenize raw text strings
            text_inputs = self.base_tokenizer(
                text_in, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512  # prevent memory issues
            )
            input_ids = text_inputs['input_ids']
            attention_mask = text_inputs.get('attention_mask', None)
        else:
            # assume already tokenized input_ids
            input_ids = text_in
            attention_mask = None  # will be handled in kwargs
        
        # move to correct device - match base model
        input_ids = input_ids.to(next(self.base_model.parameters()).device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(next(self.base_model.parameters()).device)
        
        # convert token ids to embeddings
        text_embeddings = self.base_model.get_input_embeddings()(input_ids)
        
        return text_embeddings, attention_mask

    def _merge_multimodal_inputs(self, visual_tokens, text_embeddings, attention_mask):
        """
        Combine visual and text inputs for multimodal processing
        
        Parameters:
            - visual_tokens: projected visual features
            - text_embeddings: text embedding vectors  
            - attention_mask: mask for text tokens
            
        Returns:
            - combined_embeddings: concatenated visual + text embeddings
            - combined_attention_mask: extended attention mask for both modalities
        """
        
        # handle different input combinations
        if visual_tokens is not None and text_embeddings is not None:
            # multimodal case - concatenate visual tokens before text
            combined_embeddings = torch.cat([visual_tokens, text_embeddings], dim=1)
            
            # extend attention mask to cover visual tokens
            if attention_mask is not None:
                batch_size = attention_mask.shape[0]
                num_visual_tokens = visual_tokens.shape[1]
                # visual tokens always get attention (no masking)
                visual_attention = torch.ones(
                    batch_size, num_visual_tokens, 
                    device=attention_mask.device, 
                    dtype=attention_mask.dtype
                )
                combined_attention_mask = torch.cat([visual_attention, attention_mask], dim=1)
            else:
                combined_attention_mask = None
                
        elif visual_tokens is not None:
            # vision-only case
            combined_embeddings = visual_tokens
            combined_attention_mask = None  # no masking needed for visual tokens
            
        elif text_embeddings is not None:
            # text-only case  
            combined_embeddings = text_embeddings
            combined_attention_mask = attention_mask
            
        else:
            # error case - need at least one input modality
            raise ValueError("At least one of vision_features or input_ids must be provided")
        
        return combined_embeddings, combined_attention_mask


    ''' Primary API Functions'''

    def generate(self, vision_in=None, text_in=None, max_new_tokens=256, 
                 temperature=0.7, do_sample=True, **kwargs):
        """
        Generate text responses with optional image input
        Now, if both vision and text are provided, prepend detection outputs as structured text to the prompt.
        """
        was_base_training = self.base_model.training
        was_vision_training = self.vision_encoder.training

        self.base_model.eval()
        self.vision_encoder.eval()

        # process visual inputs through vision tower and projector
        visual_tokens = self._prepare_visual_inputs(vision_in)
        detection_prompt = ""
        if visual_tokens is not None:
            # Run detection/classification and bbox heads
            class_logits = self.detection_head(visual_tokens)
            box_preds = self.bbox_head(visual_tokens)
            # For each visual token, get top class and bbox
            top_classes = class_logits.argmax(dim=-1)  # [B, num_visual_tokens]
            # Assume batch size 1 for generation (can extend if needed)
            if top_classes.dim() == 2:
                top_classes = top_classes[0]
                box_preds = box_preds[0]
            # Map class indices to names if possible
            class_names = None
            if hasattr(self, 'class_map') and self.class_map is not None:
                inv_class_map = {v: k for k, v in self.class_map.items()}
                class_names = [inv_class_map.get(idx.item(), str(idx.item())) for idx in top_classes]
            else:
                class_names = [str(idx.item()) for idx in top_classes]
            # Format as 'class: [x1, y1, x2, y2]; ...'
            detection_strs = []
            for cls, box in zip(class_names, box_preds):
                x1, y1, x2, y2 = box.tolist()
                detection_strs.append(f"{cls}: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")
            detection_prompt = "; ".join(detection_strs)
            if detection_prompt:
                detection_prompt = detection_prompt + "\n"
        # process text inputs through tokenizer and embedding layer
        if text_in is not None:
            if isinstance(text_in, str):
                text_in = detection_prompt + text_in
            elif isinstance(text_in, list) and isinstance(text_in[0], str):
                text_in[0] = detection_prompt + text_in[0]
        else:
            text_in = detection_prompt
        text_embeddings, attention_mask = self._prepare_text_inputs(text_in)
        # combine visual and text modalities
        combined_embeddings, combined_attention_mask = self._merge_multimodal_inputs(
            visual_tokens, text_embeddings, attention_mask
        )
        # setup generation parameters with defaults
        generation_kwargs = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'do_sample': do_sample,
            'pad_token_id': self.base_tokenizer.pad_token_id,
            'eos_token_id': self.base_tokenizer.eos_token_id,
            **kwargs
        }
        # generate tokens without computing gradients
        with torch.no_grad():
            outputs = self.base_model.generate(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention_mask,
                **generation_kwargs
            )
        # convert token ids back to readable text
        generated_text = self.base_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # restore model state
        if was_base_training:
            self.base_model.train()
        if was_vision_training:
            self.vision_encoder.train()
        return generated_text

    def forward(self, vision_features=None, input_ids=None, attention_mask=None, labels=None, detection=False, target_labels=None, **kwargs):
        """
        Main forward pass for multimodal model using preprocessed inputs
        
        Parameters:
            - vision_features: precomputed vision encoder features [batch, seq_len, hidden_dim]
            - input_ids: preprocessed text token ids [batch, seq_len]
            - attention_mask: preprocessed attention mask [batch, seq_len]
            - labels: ground truth labels for loss computation
            - detection: whether to return detection outputs
            - target_labels: ground truth labels for dynamic class assignment
            
        Returns:
            - model outputs including logits and loss
        """
        
        # project vision features to text embedding space
        if vision_features is not None:
            vision_features = vision_features.to(next(self.projector.parameters()).device)
            visual_tokens = self.projector(vision_features)
        else:
            visual_tokens = None
        
        # convert token ids to text embeddings
        if input_ids is not None:
            input_ids = input_ids.to(next(self.base_model.parameters()).device)
            text_embeddings = self.base_model.get_input_embeddings()(input_ids)
        else:
            text_embeddings = None
        
        # move attention mask to correct device
        if attention_mask is not None:
            attention_mask = attention_mask.to(next(self.base_model.parameters()).device)
        
        # combine visual and text modalities
        combined_embeddings, combined_attention_mask = self._merge_multimodal_inputs(
            visual_tokens, text_embeddings, attention_mask
        )
        
        # forward through base language model
        outputs = self.base_model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            labels=labels,
            **kwargs
        )
        
        # Detection/classification output
        if detection:
            # Use DETR-style queries for detection heads
            batch_size = input_ids.shape[0] if input_ids is not None else (visual_tokens.shape[0] if visual_tokens is not None else 1)
            queries = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, num_queries, hidden_dim]
            class_logits = self.detection_head(queries)  # [B, num_queries, num_classes+1]
            box_preds = self.bbox_head(queries)          # [B, num_queries, 4]
            # Debug print for first batch of each epoch
            if not hasattr(self, '_debug_printed') or not self._debug_printed:
                print(f"[DEBUG] class_logits mean: {class_logits.mean().item():.4f}, std: {class_logits.std().item():.4f}")
                print(f"[DEBUG] box_preds mean: {box_preds.mean().item():.4f}, std: {box_preds.std().item():.4f}")
                print(f"[DEBUG] First 5 box_preds: {box_preds.view(-1, 4)[:5]}")
                print(f"[DEBUG] box_preds min: {box_preds.min().item():.4f}, max: {box_preds.max().item():.4f}")
                self._debug_printed = True
            return {
                "class_logits": class_logits,
                "box_preds": box_preds,
                "outputs": outputs
            }
        return outputs
    
    def save_pretrained(self, save_directory, save_base_model=False, save_vision_encoder=False):
        """
        Save model components to directory
        
        Parameters:
            - save_directory: path to save model files
            - save_base_model: whether to save the base LLM (usually unnecessary)
            - save_vision_encoder: whether to save vision encoder (usually unnecessary)
        """
        import json
        import os
        
        # create save directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # save model configuration
        config = {
            'model_type': 'BBOB',
            'base_model': self.base_model.config.name_or_path if hasattr(self.base_model.config, 'name_or_path') else 'unknown',
            'vision_encoder': self.vision_encoder.config.name_or_path if hasattr(self.vision_encoder.config, 'name_or_path') else 'unknown', 
            'projector_config': {
                'input_dim': self.projector.net[0].in_features,
                'output_dim': self.projector.net[0].out_features,
                'hidden_dim': self.projector.net[2].out_features
            },
            'vision_hidden_size': self.projector.net[0].in_features,  # get actual input size from projector
            'text_hidden_size': self.base_model.config.hidden_size
        }
        
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # always save the projector - this is our custom component
        projector_path = os.path.join(save_directory, 'projector.pth')
        torch.save(self.projector.state_dict(), projector_path)
        print(f"Saved projector to {projector_path}")

        # save detection and bbox heads
        detection_head_path = os.path.join(save_directory, 'detection_head.pth')
        torch.save(self.detection_head.state_dict(), detection_head_path)
        print(f"Saved detection head to {detection_head_path}")
        bbox_head_path = os.path.join(save_directory, 'bbox_head.pth')
        torch.save(self.bbox_head.state_dict(), bbox_head_path)
        print(f"Saved bbox head to {bbox_head_path}")
        
        # optionally save base model (usually unnecessary as it's a standard model)
        if save_base_model:
            base_model_path = os.path.join(save_directory, 'base_model')
            self.base_model.save_pretrained(base_model_path)
            self.base_tokenizer.save_pretrained(base_model_path)
            print(f"Saved base model to {base_model_path}")
        
        # optionally save vision encoder (usually unnecessary as it's a standard model)
        if save_vision_encoder:
            vision_path = os.path.join(save_directory, 'vision_encoder')
            self.vision_encoder.save_pretrained(vision_path)
            self.image_processor.save_pretrained(vision_path)
            print(f"Saved vision encoder to {vision_path}")
        
        # Save LoRA adapters if present
        try:
            from peft import PeftModel
            if isinstance(self.base_model, PeftModel):
                lora_adapter_path = os.path.join(save_directory, 'lora_adapter')
                os.makedirs(lora_adapter_path, exist_ok=True)
                self.base_model.save_pretrained(lora_adapter_path)
                print(f"Saved LoRA adapter(s) to {lora_adapter_path}")
        except ImportError:
            pass
        
        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, model_path, base_model=None, vision_encoder=None, 
                       load_base_model=False, load_vision_encoder=False, num_classes=80):
        """
        Load model from saved directory
        
        Parameters:
            - model_path: path to saved model directory
            - base_model: base model name/path (if not loading from saved)
            - vision_encoder: vision encoder name/path (if not loading from saved)
            - load_base_model: whether to load base model from saved files
            - load_vision_encoder: whether to load vision encoder from saved files
            - num_classes: number of classes for detection head (default 80 for COCO)
            
        Returns:
            - loaded BBOB model instance
        """
        import json
        import os
        
        # load configuration
        config_path = os.path.join(model_path, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # determine which models to use
        if load_base_model:
            # load base model from saved files
            base_model_to_use = os.path.join(model_path, 'base_model')
        else:
            # use provided base model or fallback to config
            base_model_to_use = base_model or config.get('base_model', 'microsoft/DialoGPT-medium')
            
        if load_vision_encoder:
            # load vision encoder from saved files  
            vision_encoder_to_use = os.path.join(model_path, 'vision_encoder')
        else:
            # use provided vision encoder or fallback to config
            vision_encoder_to_use = vision_encoder or config.get('vision_encoder', 'apple/mobilevit-small')
        
        # create model instance with specified components
        print(f"Loading BBOB from {model_path}")
        print(f"Base model: {base_model_to_use}")
        print(f"Vision encoder: {vision_encoder_to_use}")
        
        model = cls(
            base_model=base_model_to_use,
            vision_encoder=vision_encoder_to_use,
            bnb_config=None,  # TODO: add bnb_config parameter if needed
            num_classes=num_classes
        )
        
        # load projector weights
        projector_path = os.path.join(model_path, 'projector.pth')
        if os.path.exists(projector_path):
            projector_state = torch.load(projector_path, map_location='cpu')
            model.projector.load_state_dict(projector_state)
            print(f"Loaded projector from {projector_path}")
        else:
            print(f"Warning: No projector weights found at {projector_path}")

        # load detection head weights if present
        detection_head_path = os.path.join(model_path, 'detection_head.pth')
        if os.path.exists(detection_head_path):
            detection_head_state = torch.load(detection_head_path, map_location='cpu')
            model.detection_head.load_state_dict(detection_head_state)
            print(f"Loaded detection head from {detection_head_path}")
        else:
            print(f"Warning: No detection head weights found at {detection_head_path}")

        # load bbox head weights if present
        bbox_head_path = os.path.join(model_path, 'bbox_head.pth')
        if os.path.exists(bbox_head_path):
            bbox_head_state = torch.load(bbox_head_path, map_location='cpu')
            model.bbox_head.load_state_dict(bbox_head_state)
            print(f"Loaded bbox head from {bbox_head_path}")
        else:
            print(f"Warning: No bbox head weights found at {bbox_head_path}")
        
        return model

    def save_projector_only(self, save_path):
        """
        Save only the projector weights for quick fine-tuning
        
        Parameters:
            - save_path: path to save projector weights (.pth file)
        """
        torch.save(self.projector.state_dict(), save_path)
        print(f"Projector saved to {save_path}")
        
    def load_projector_weights(self, weights_path):
        """
        Load projector weights from file
        
        Parameters:
            - weights_path: path to projector weights file
        """
        projector_state = torch.load(weights_path, map_location='cpu')
        self.projector.load_state_dict(projector_state)
        print(f"Loaded projector weights from {weights_path}")
        
    def set_num_classes(self, num_classes):
        """Update the number of classes for the detection head."""
        self.num_classes = num_classes
        self.detection_head = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_classes)
        )
        # Move detection head to same device as projector
        device = next(self.projector.parameters()).device
        self.detection_head = self.detection_head.to(device)
        
    def load_lora_adapter(self, adapter_path, adapter_name="default"):
        """
        Load a LoRA adapter into the base model using PEFT.
        If the base model is already a PeftModel (i.e., already has a LoRA adapter),
        this will add the new adapter with the given name. Otherwise, it wraps the base model.

        Parameters:
            - adapter_path: path to the directory containing the LoRA adapter (e.g., 'Tuning/2025-06-07_02-36-15/lora_adapter')
            - adapter_name: name to assign to this adapter (default: "default")

        Usage:
            model = BBOB.from_pretrained(...)
            model.load_lora_adapter('path/to/lora_adapter', adapter_name="my_adapter")
        """
        try:
            from peft import PeftModel, PeftConfig
        except ImportError:
            raise ImportError("peft is required to load LoRA adapters. Please install with 'pip install peft'.")
        if isinstance(self.base_model, PeftModel):
            # Already a PeftModel, add new adapter (multi-adapter support)
            config = PeftConfig.from_pretrained(adapter_path)
            self.base_model.add_adapter(adapter_name, config)
            self.base_model.load_adapter(adapter_path, adapter_name)
            print(f"Added LoRA adapter '{adapter_name}' from {adapter_path}")
        else:
            # Not a PeftModel, wrap with first adapter
            self.base_model = PeftModel.from_pretrained(self.base_model, adapter_path, adapter_name=adapter_name)
            print(f"Loaded LoRA adapter '{adapter_name}' from {adapter_path}")
        
    def get_model_size(self):
        """
        Return the size (in MB) of all major model components as a string.
        """
        import sys
        def size_of_module(module):
            total = 0
            for p in module.parameters():
                total += p.numel() * p.element_size()
            return total / (1024 ** 2)  # MB

        lines = []
        lines.append("Model size breakdown (MB):")
        lines.append(f"  data type:       {next(self.base_model.parameters()).dtype}")
        lines.append(f"  base_model:      {size_of_module(self.base_model):.2f} MB")
        lines.append(f"  vision_encoder:  {size_of_module(self.vision_encoder):.2f} MB")
        lines.append(f"  projector:       {size_of_module(self.projector):.2f} MB")
        lines.append(f"  detection_head:  {size_of_module(self.detection_head):.2f} MB")
        lines.append(f"  bbox_head:       {size_of_module(self.bbox_head):.2f} MB")
        # LoRA adapters (if present)
        try:
            from peft import PeftModel
            if isinstance(self.base_model, PeftModel):
                lines.append("  LoRA adapters:")
                for name, adapter in self.base_model.peft_config.items():
                    adapter_size = 0
                    for n, p in self.base_model.named_parameters():
                        if n.startswith(f"peft.{name}"):
                            adapter_size += p.numel() * p.element_size()
                    lines.append(f"    {name}: {adapter_size / (1024 ** 2):.2f} MB")
        except ImportError:
            pass
        lines.append("-----------------------------")
        total = (size_of_module(self.base_model) + size_of_module(self.vision_encoder) +
                 size_of_module(self.projector) + size_of_module(self.detection_head) +
                 size_of_module(self.bbox_head))
        lines.append(f"  Total (excluding LoRA): {total:.2f} MB")
        return "\n".join(lines)
        