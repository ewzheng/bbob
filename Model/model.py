"""BBOB model and config."""

import torch
import torch.nn as nn
import transformers
from transformers import BitsAndBytesConfig, AutoModel, AutoImageProcessor
import bitsandbytes as bnb
import os
from .projector import Projector
from peft import PeftModel, PeftConfig
from safetensors.torch import save_file 
from .vision_tower import VisionTower
import json
import safetensors.torch as st
from transformers import PreTrainedModel

# Constants for special tokens
IGNORE_INDEX = -100
# Default number of visual tokens when not specified in config
DEFAULT_OUTPUT_TOKENS = 169

from transformers import PretrainedConfig


class BBOBConfig(PretrainedConfig):
    """
    configuration container for the bbob model.
    """

    model_type = "bbob"

    def __init__(
        self,
        vision_hidden_size: int | None = None,
        text_hidden_size: int | None = None,
        base_model_name: str | None = None,
        max_memory: dict | None = None,
        bnb_config: str | dict | None = None,
        output_tokens: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Default placeholder values when none provided (HF needs non-None)
        self.vision_hidden_size = vision_hidden_size if vision_hidden_size is not None else 1
        self.text_hidden_size   = text_hidden_size   if text_hidden_size   is not None else 1
        self.base_model_name    = base_model_name    if base_model_name    is not None else "unknown"
        self.max_memory         = max_memory
        self.bnb_config         = bnb_config
        # store projector token count (defaults handled later)
        self.output_tokens      = output_tokens
        
        # paths for extra components (filled during save_pretrained)
        self.projector_path: str | None = kwargs.get("projector_path")
        self.vision_tower_path: str | None = kwargs.get("vision_tower_path")
        self.language_model_path: str | None = kwargs.get("language_model_path")

class BBOB(PreTrainedModel):
    """
    vision-language model composed of:
    • mobilevit-v2 backbone (frozen)
    • learnable projector to text space
    • hf causal-lm language head.
    """

    config_class = BBOBConfig

    def __init__(self, config: BBOBConfig | None = None, **kwargs):
        """Initialize model."""

        if config is None:
            # Create default config if none provided
            config = BBOBConfig(**kwargs)
        
        super().__init__(config)
        
        # Extract configuration values
        output_tokens = getattr(config, "output_tokens", None) or DEFAULT_OUTPUT_TOKENS
        model_path = config.base_model_name
        max_memory = config.max_memory
        bnb_config = config.bnb_config

        # Process quantisation config once via helper
        bnb_config = self._resolve_bnb_config(bnb_config)

        # ensure we don't clash with PreTrainedModel.base_model property
        self.base_model_prefix = "language_model"

        self.language_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            max_memory=max_memory,
            quantization_config=bnb_config,
            device_map="auto" if max_memory is not None else None,
            torch_dtype="auto",
        )

        base_model_dtype = next(self.language_model.parameters()).dtype
        base_model_device = next(self.language_model.parameters()).device

        self._dtype = base_model_dtype
        self._device = base_model_device

        # store tokenizer (AutoTokenizer does not accept torch_dtype)
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        emb_layer = self._find_input_embedding(self.language_model)
        self._embedding_layer = emb_layer  # cache for forward()
        text_hidden_size = emb_layer.weight.shape[1]

        self.vision_tower = VisionTower(dtype=self._dtype, device=self._device)
        self.image_processor = self.vision_tower.image_processor
        self.vision_encoder = self.vision_tower.model

        vision_hidden_size = self.vision_tower.hidden_size
        print(f"Using VisionTower with hidden_size={vision_hidden_size}")

        self.projector = Projector(
            vision_hidden_size,
            text_hidden_size,
            dtype=base_model_dtype,
            device=base_model_device,
            output_tokens=output_tokens,
        )

        # ensure projector on same device/dtype as base model weights
        self.projector.to(base_model_device, dtype=base_model_dtype)
        print(f"Model device: {self._device}, dtype: {self._dtype}")
        print(f"Vision Tower device: {self.vision_tower.device}, dtype: {self.vision_tower.dtype}")
        print(f"Projector device: {next(self.projector.parameters()).device}, dtype: {next(self.projector.parameters()).dtype}")

        # Update config with actual values
        self.config.vision_hidden_size = vision_hidden_size
        self.config.text_hidden_size = text_hidden_size
        # persist projector token count
        self.config.output_tokens = self.projector.output_tokens
            

    ''' Helpers for interacting with internal components'''
    def process_image(self, images):
        '''
        process images to be used in the model.
        '''
        return self.vision_tower.process_image(images)
    
    def get_tokenizer(self):    
        '''
        returns: hf tokenizer tied to the base language model.
        '''
        return self.base_tokenizer
    
    def get_image_processor(self):
        '''
        returns: mobilevit image processor instance.
        '''
        return self.image_processor
    
    def freeze_projector(self):
        """
        Freeze projector parameters to prevent training
        """
        self.projector.eval()
        self.projector.freeze()
    
    def unfreeze_projector(self):
        """
        Unfreeze projector parameters to enable training
        """
        self.projector.train()
        self.projector.unfreeze()

    def freeze_vision_tower(self):
        """Freeze vision tower parameters."""
        self.vision_tower.freeze()

    def unfreeze_vision_tower(self):
        """Unfreeze vision tower parameters."""
        self.vision_tower.unfreeze()

    def freeze_model(self):
        """Freeze language model parameters."""
        self.language_model.eval()
        for p in self.language_model.parameters():
            p.requires_grad = False
    
    def unfreeze_model(self):
        """Unfreeze language model parameters."""
        self.language_model.train()
        for p in self.language_model.parameters():
            p.requires_grad = True

    def train(self):
        """
        Set all model components to training mode
        """ 
        self.language_model.train()
        self.projector.train()

    def eval(self):
        """
        Set all model components to evaluation mode
        """
        self.language_model.eval()
        self.projector.eval()

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device
    
    @property
    def vis_length(self):
        return self.projector.output_tokens

    '''
    API Functions
    '''

    def _replace_image_tokens(self, text_embeds, visual_embeds, attention_mask):
        """
        Replace image placeholder tokens with visual embeddings using position-based approach.
        
        OPTIMIZED: Uses vectorized operations instead of Python loops for much better performance.
        
        Since the collator always inserts the placeholder at position 0, we can use a position-based
        approach instead of token matching. This avoids conflicts with legitimate EOS tokens that
        appear elsewhere in the sequence (e.g., at the end of sequences).
        
        Flow:
        1. Collator creates: [PLACEHOLDER, instruction_tokens...]
        2. This method transforms: [PLACEHOLDER, instruction_tokens...] 
           → [visual_emb1, visual_emb2, ..., visual_emb64, instruction_tokens...]
        
        parameters:
            - input_ids (tensor): `(b, t)` input token ids.
            - text_embeds (tensor): `(b, t, d)` text embeddings.
            - visual_embeds (tensor|None): `(b, v, d)` visual embeddings or none.
            - attention_mask (tensor|None): `(b, t)` attention mask.
            
        returns: tuple(tensor, tensor) -> combined_embeds, combined_mask.
        """
        if visual_embeds is None:
            return text_embeds, attention_mask
            
        batch_size, visual_tokens, embed_dim = visual_embeds.shape
        device = visual_embeds.device
        
        # Vectorized approach: process entire batch at once
        # Skip first token (placeholder) from text embeddings
        text_after = text_embeds[:, 1:]  # (batch_size, text_tokens-1, embed_dim)
        
        # Concatenate visual embeddings with remaining text embeddings
        combined_embeds = torch.cat([visual_embeds, text_after], dim=1)
        
        # Handle attention mask vectorized
        if attention_mask is not None:
            mask_after = attention_mask[:, 1:]  # Skip first token
            visual_mask = torch.ones(batch_size, visual_tokens, dtype=torch.long, device=device)
            combined_mask = torch.cat([visual_mask, mask_after], dim=1)
        else:
            seq_len = visual_tokens + text_embeds.shape[1] - 1
            combined_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        
        return combined_embeds, combined_mask

    def _prepare_labels_for_replacement(self, input_ids, labels, visual_tokens):
        """
        Adjust labels to account for image token replacement using position-based approach.
        
        OPTIMIZED: Uses vectorized operations instead of Python loops for much better performance.
        
        Since we know the placeholder is always at position 0, we can use a position-based
        approach instead of token matching.
        
        parameters:
            - input_ids (tensor): `(b, t)` input token ids.
            - labels (tensor): `(b, t)` original labels.
            - visual_tokens (int): number of visual tokens per image.
            
        returns: tensor - adjusted labels.
        """
        if labels is None:
            return None
            
        batch_size, text_tokens = labels.shape
        device = labels.device
        
        # Vectorized approach: process entire batch at once
        # Skip first token (placeholder) from labels
        labels_after = labels[:, 1:]  # (batch_size, text_tokens-1)
        
        # Create ignore labels for visual tokens for entire batch
        visual_ignore = torch.full((batch_size, visual_tokens), IGNORE_INDEX, dtype=labels.dtype, device=device)
        
        # Concatenate: visual_ignore + labels_after
        combined_labels = torch.cat([visual_ignore, labels_after], dim=1)
        
        return combined_labels

    def _prepare_visual_inputs(self, images):
        '''
        convert raw images to projected visual tokens.

        parameters:
            - images (list|tensor|None): raw images or pre-norm tensor.

        returns: tensor `(b, v, d)` or *none* when `images` is none.
        '''
        if images is None:
            return None

        # Step-1: obtain pixel values
        # If the caller already provides a BCHW tensor we assume it is
        # correctly normalised for MobileViT and skip the image processor.
        if isinstance(images, torch.Tensor):
            pixel_values = images.to(device=self.device, dtype=self.dtype)
        else:
            pixel_values = self.vision_tower.process_image(images)

        # Step-2: spatial feature map from the frozen vision backbone
        with torch.no_grad():
            feats = self.vision_tower(pixel_values)  # (B, C, H, W)

        # Step-3: project to language embedding space producing visual tokens (B, H*W, D)
        visual_embeds = self.projector(feats)  # (B, H*W, D)
        
        return visual_embeds

    def _embed_tokens(self, input_ids):
        """Return embeddings for `input_ids`."""
        return self._embedding_layer(input_ids)

    @staticmethod
    def _find_input_embedding(model):
        """Locate token-embedding layer in the language model."""
        # 1) Official accessor
        if hasattr(model, "get_input_embeddings"):
            try:
                emb = model.get_input_embeddings()
                if emb is not None:
                    return emb
            except NotImplementedError:
                pass

        # 2) Common attribute names
        for name in ("embed_tokens", "wte", "tok_embeddings"):
            if hasattr(model, name):
                return getattr(model, name)

        # 3) Heuristic search in parameters – large vocab dimension
        state_dict = model.state_dict()
        for key, weight in state_dict.items():
            if weight.ndim == 2 and max(weight.shape) > 10000:  # vocab dim likely large
                return torch.nn.Embedding.from_pretrained(weight, freeze=True)

        raise RuntimeError("Cannot locate input embedding layer in base model")

    def forward(
        self,
        input_ids=None,
        input_embeds=None,
        attention_mask=None,
        position_ids=None,
        images=None,
        labels=None,
        **kwargs,
    ):
        '''
        multimodal causal-lm pass using image token replacement.

        parameters:
            - input_ids (tensor|None): token ids (b, t) with IMAGE_TOKEN_INDEX placeholders.
            - input_embeds (tensor|None): pre-computed embeddings.
            - attention_mask (tensor|None): mask.
            - position_ids (tensor|None): position ids.
            - images (list|tensor|None): raw or processed images.
            - labels (tensor|None): lm labels.

        returns: transformers.CausalLMOutput.
        '''

        visual_embeds = self._prepare_visual_inputs(images)

        if input_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or input_embeds must be provided.")
            text_embeds = self._embed_tokens(input_ids)
        else:
            text_embeds = input_embeds
            
        # Replace image tokens with visual embeddings (position-based approach)
        inputs_embeds, combined_mask = self._replace_image_tokens(
            text_embeds, visual_embeds, attention_mask
        )

        # Labels are already aligned by the collator, so use them as-is
        # The collator accounts for visual token replacement when creating labels

        # Handle position_ids for multimodal inputs
        if position_ids is not None and visual_embeds is not None:
            # Position IDs need to be adjusted for the new sequence length
            # after image token replacement
            batch_size = inputs_embeds.shape[0]
            seq_length = inputs_embeds.shape[1]
            
            # Optimized position ID generation
            position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Validate sequence length alignment between inputs and labels
        if labels is not None and inputs_embeds is not None:
            if labels.size(1) != inputs_embeds.size(1):
                raise ValueError(
                    f"Sequence length mismatch: labels has {labels.size(1)} tokens, "
                    f"but inputs_embeds has {inputs_embeds.size(1)} tokens. "
                    f"This suggests a bug in the data collator or model preprocessing."
                )

        lm_outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            position_ids=position_ids,
            labels=labels,
            **kwargs,
        )

        return lm_outputs

    # ------------------------------------------------------------------
    # Generation helper -------------------------------------------------
    # ------------------------------------------------------------------

    def generate(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        images=None,
        **generate_kwargs,
    ):
        """Wrapper around ``self.language_model.generate`` that supports images.

        Parameters
        ----------
        input_ids : LongTensor (B, T)
            Prompt token IDs with IMAGE_TOKEN_INDEX placeholders.  Mandatory.
        attention_mask : LongTensor (B, T) | None
            Standard HF attention mask for *input_ids*.
        images : list|Tensor|None
            Raw image(s) or pre-normalised tensor consumed by the vision tower.
        **generate_kwargs : Any
            Additional keyword arguments forwarded to ``generate``.
        """

        if input_ids is None:
            raise ValueError("generate requires `input_ids` argument")

        # Ensure input_ids are on the correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # 1) Prepare visual embeddings (may be None)
        visual_embeds = self._prepare_visual_inputs(images)

        # 2) Text embeddings from the prompt IDs
        text_embeds = self._embed_tokens(input_ids)

        # 3) Replace image tokens with visual embeddings (position-based approach)
        inputs_embeds, combined_mask = self._replace_image_tokens(
            text_embeds, visual_embeds, attention_mask
        )

        # 4) Call base language model's generate; we provide *inputs_embeds*
        #    so tokens are not re-embedded inside the LM.
        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            **generate_kwargs,
        )

    def save_pretrained(self, output_dir: str, save_full_model: bool = True, **kwargs):
        """Save the complete model for end-to-end training.

        Args:
            output_dir: Directory to save the model
            save_full_model: If True, saves all model components. If False, saves only trainable parts.
        """

        os.makedirs(output_dir, exist_ok=True)

        # --- 1) Config -----------------------------------------------------------------
        self.config.projector_path = "projector.safetensors"
        if save_full_model:
            self.config.language_model_path = "language_model"
            self.config.vision_tower_path = "vision_tower"
        self.config.save_pretrained(output_dir)

        # --- 2) Projector (always save) ------------------------------------------------
        proj_path = os.path.join(output_dir, self.config.projector_path)
        st.save_file(self.projector.state_dict(), proj_path)

        if save_full_model:
            # --- 3) Language Model -----------------------------------------------------
            lm_dir = os.path.join(output_dir, "language_model")
            os.makedirs(lm_dir, exist_ok=True)
            
            # Handle both regular models and PEFT models
            if hasattr(self.language_model, 'save_pretrained'):
                self.language_model.save_pretrained(lm_dir, **kwargs)
            else:
                # Fallback for models without save_pretrained
                torch.save(self.language_model.state_dict(), os.path.join(lm_dir, "pytorch_model.bin"))
            
            # Save tokenizer alongside language model
            if hasattr(self, 'base_tokenizer'):
                self.base_tokenizer.save_pretrained(lm_dir)

          # --- 3) vision tower (rarely trainable) ---------------------------------------
            if any(p.requires_grad for p in self.vision_tower.model.parameters()):
                vt_dir = os.path.join(output_dir, "vision_tower")
                os.makedirs(vt_dir, exist_ok=True)
                self.vision_tower.model.save_pretrained(vt_dir)
            
                # Save image processor
                if hasattr(self.vision_tower, 'image_processor'):
                    self.vision_tower.image_processor.save_pretrained(vt_dir)
        else:
            # Legacy mode: only save trainable components
            if any(p.requires_grad for p in self.vision_tower.model.parameters()):
                vt_dir = os.path.join(output_dir, "vision_tower")
                os.makedirs(vt_dir, exist_ok=True)
                self.vision_tower.model.save_pretrained(vt_dir)

    @classmethod
    def from_pretrained(cls, ckpt_dir: str, **kwargs):
        """Load model from `ckpt_dir`."""

        # 1) Load BBOBConfig first
        config = BBOBConfig.from_pretrained(ckpt_dir)
        
        # 2) Check if we have a full model checkpoint or just projector
        try:
            language_model_path = getattr(config, 'language_model_path', 'language_model')
            full_path = os.path.join(ckpt_dir, language_model_path)
            has_full_model = os.path.exists(full_path) and os.path.isdir(full_path)
        except (AttributeError, TypeError, OSError) as e:
            print(f"Warning: Could not check for full model checkpoint: {e}")
            has_full_model = False
            
        if has_full_model:
            # Loading full model from checkpoint
            # Override config values with any provided kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
            # Create model with original base_model_name for component initialization
            obj: "BBOB" = cls.__new__(cls)
            obj.config = config
            PreTrainedModel.__init__(obj, config)
            
            # Initialize components manually
            obj._init_from_checkpoint(ckpt_dir, config)
            
        else:
            # Legacy loading: projector-only, reconstruct from HF Hub
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            obj: "BBOB" = cls(config=config)
            
            # Load projector weights
            proj_rel = getattr(config, "projector_path", "projector.safetensors")
            proj_path = os.path.join(ckpt_dir, proj_rel)
            if os.path.isfile(proj_path):
                output_tokens = getattr(config, "output_tokens", None)
                obj.projector = Projector.from_pretrained(
                    proj_path,
                    indim=obj.vision_tower.hidden_size,
                    outdim=obj._embedding_layer.weight.shape[1] if hasattr(obj, '_embedding_layer') else obj.projector.outdim,
                    dtype=obj._dtype,
                    device=obj._device,
                    output_tokens=output_tokens or DEFAULT_OUTPUT_TOKENS,
                )

            # Load vision tower if available
            vt_dir = os.path.join(ckpt_dir, "vision_tower")
            if os.path.isdir(vt_dir):
                try:
                    vision_model = AutoModel.from_pretrained(vt_dir, torch_dtype=obj._dtype)
                    image_processor = AutoImageProcessor.from_pretrained(vt_dir)

                    # Initialise a fresh VisionTower then swap in loaded components
                    obj.vision_tower = VisionTower(dtype=obj._dtype, device=obj._device)
                    obj.vision_tower.model = vision_model
                    obj.vision_tower.image_processor = image_processor

                    # Recompute hidden size from the loaded model config
                    cfg = vision_model.config
                    if hasattr(cfg, "hidden_size") and cfg.hidden_size is not None:
                        obj.vision_tower._hidden_size = cfg.hidden_size
                    elif hasattr(cfg, "hidden_sizes") and len(cfg.hidden_sizes) > 0:
                        obj.vision_tower._hidden_size = cfg.hidden_sizes[-1]
                    elif hasattr(cfg, "neck_hidden_sizes") and len(cfg.neck_hidden_sizes) > 0:
                        obj.vision_tower._hidden_size = cfg.neck_hidden_sizes[-1]

                    # Refresh cached helper attributes to point to the new tower
                    obj.image_processor = image_processor
                    obj.vision_encoder = vision_model
                except Exception:
                    # Fallback: initialize new and load state dict
                    obj.vision_tower = VisionTower(dtype=obj._dtype, device=obj._device)
                    sd = torch.load(os.path.join(vt_dir, "pytorch_model.bin"), map_location=obj._device)
                    obj.vision_tower.model.load_state_dict(sd)
                    # Refresh cached attributes in fallback path as well
                    obj.image_processor = obj.vision_tower.image_processor
                    obj.vision_encoder = obj.vision_tower.model
        return obj
    
    def _init_from_checkpoint(self, ckpt_dir: str, config: BBOBConfig):
        """Internal helper to restore all components."""
        
        # Process bnb_config
        bnb_config = self._resolve_bnb_config(config.bnb_config)

        # Load language model from checkpoint
        lm_dir = os.path.join(ckpt_dir, getattr(config, 'language_model_path', 'language_model'))
        self.language_model = transformers.AutoModelForCausalLM.from_pretrained(
            lm_dir,
            max_memory=config.max_memory,
            quantization_config=bnb_config,
            device_map="auto" if config.max_memory is not None else None,
            torch_dtype="auto",
        )

        base_model_dtype = next(self.language_model.parameters()).dtype
        base_model_device = next(self.language_model.parameters()).device
        self._dtype = base_model_dtype
        self._device = base_model_device

        # Load tokenizer (torch_dtype not supported for tokenizer)
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(lm_dir)
        emb_layer = self._find_input_embedding(self.language_model)
        self._embedding_layer = emb_layer

        # Load vision tower
        vt_dir = os.path.join(ckpt_dir, getattr(config, 'vision_tower_path', 'vision_tower'))
        if os.path.isdir(vt_dir):
            try:
                vision_model = AutoModel.from_pretrained(vt_dir, torch_dtype=self._dtype)
                image_processor = AutoImageProcessor.from_pretrained(vt_dir)

                # Initialise a fresh VisionTower then swap in loaded components
                self.vision_tower = VisionTower(dtype=self._dtype, device=self._device)
                self.vision_tower.model = vision_model
                self.vision_tower.image_processor = image_processor

                # Recompute hidden size from the loaded model config
                cfg = vision_model.config
                if hasattr(cfg, "hidden_size") and cfg.hidden_size is not None:
                    self.vision_tower._hidden_size = cfg.hidden_size
                elif hasattr(cfg, "hidden_sizes") and len(cfg.hidden_sizes) > 0:
                    self.vision_tower._hidden_size = cfg.hidden_sizes[-1]
                elif hasattr(cfg, "neck_hidden_sizes") and len(cfg.neck_hidden_sizes) > 0:
                    self.vision_tower._hidden_size = cfg.neck_hidden_sizes[-1]

                # Refresh cached helper attributes to point to the new tower
                self.image_processor = image_processor
                self.vision_encoder = vision_model
            except Exception:
                # Fallback: initialize new and load state dict
                self.vision_tower = VisionTower(dtype=self._dtype, device=self._device)
                sd = torch.load(os.path.join(vt_dir, "pytorch_model.bin"), map_location=self._device)
                self.vision_tower.model.load_state_dict(sd)
                # Refresh cached attributes in fallback path as well
                self.image_processor = self.vision_tower.image_processor
                self.vision_encoder = self.vision_tower.model
        else:
            # Initialize new vision tower
            self.vision_tower = VisionTower(dtype=self._dtype, device=self._device)

        self.image_processor = self.vision_tower.image_processor
        self.vision_encoder = self.vision_tower.model

        # Initialize projector
        vision_hidden_size = self.vision_tower.hidden_size
        text_hidden_size = emb_layer.weight.shape[1]
        output_tokens = getattr(config, "output_tokens", None) or DEFAULT_OUTPUT_TOKENS
        self.projector = Projector(
            vision_hidden_size,
            text_hidden_size,
            dtype=base_model_dtype,
            device=base_model_device,
            output_tokens=output_tokens,
        )

        # Load projector weights
        proj_path = os.path.join(ckpt_dir, getattr(config, "projector_path", "projector.safetensors"))
        if os.path.isfile(proj_path):
            output_tokens_cfg = getattr(config, "output_tokens", None)
            self.projector = Projector.from_pretrained(
                proj_path,
                indim=vision_hidden_size,
                outdim=text_hidden_size,
                dtype=base_model_dtype,
                device=base_model_device,
                output_tokens=output_tokens_cfg or DEFAULT_OUTPUT_TOKENS,
            )

        # Set base model prefix
        self.base_model_prefix = "language_model"

    @staticmethod
    def _resolve_bnb_config(cfg):
        """Convert user-friendly bnb_config spec into BitsAndBytesConfig or None."""
        if cfg == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        if cfg == "4bit":
            return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        if cfg == "bf16":
            return BitsAndBytesConfig(load_in_bf16=True)
        if cfg == "fp16":
            return BitsAndBytesConfig(load_in_fp16=True)
        if isinstance(cfg, dict):
            return BitsAndBytesConfig(**cfg)
        return None