"""BBOB multimodal model and config."""

import torch
import torch.nn as nn
import transformers
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb
import os
from .projector import Projector
from peft import PeftModel, PeftConfig
from safetensors.torch import save_file 
from .vision_tower import VisionTower
import json
import safetensors.torch as st
from transformers import PreTrainedModel

# -----------------------------------------------------------------------------
# Configuration class – moved here for convenience
# -----------------------------------------------------------------------------

from transformers import PretrainedConfig


class BBOBConfig(PretrainedConfig):
    """Config for the BBOB model."""

    model_type = "bbob"

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        base_model_name: str,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.base_model_name = base_model_name
        # paths for extra components (filled during save_pretrained)
        self.projector_path: str | None = kwargs.get("projector_path")
        self.vision_tower_path: str | None = kwargs.get("vision_tower_path")

class BBOB(PreTrainedModel):
    """Vision–language model with MobileViT tower and projector."""

    config_class = BBOBConfig

    def __init__(self, model_path=None, max_memory=None, bnb_config=None, config: BBOBConfig | None = None, **kwargs):
        """
        Initialize BBOB model with specified components
        
        Parameters:
            - base_model: path or identifier for base language model
            - vision_encoder: path or identifier for vision encoder
            - bnb_config: quantization configuration for model loading
        """

        if config is not None:
            self.config = config
            if model_path is None:
                model_path = config.base_model_name
        else:
            self.config = None

        super().__init__(self.config if self.config is not None else BBOBConfig(vision_hidden_size=1, text_hidden_size=1, base_model_name=str(model_path)))

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

        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            max_memory=max_memory,
            quantization_config=bnb_config,
            device_map="auto" if max_memory is not None else None,
            torch_dtype="auto",
        )

        base_model_dtype = next(self.base_model.parameters()).dtype
        base_model_device = next(self.base_model.parameters()).device

        self._dtype = base_model_dtype
        self._device = base_model_device

        # store tokenizer
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, torch_dtype=base_model_dtype)
        cfg = self.base_model.config
        text_hidden_size = None
        for attr in ("hidden_size", "n_embd", "d_model"):
            if hasattr(cfg, attr) and getattr(cfg, attr) is not None:
                text_hidden_size = getattr(cfg, attr)
                break

        if text_hidden_size is None:
            # fall back to dummy forward pass
            with torch.no_grad():
                tok_id = self.base_tokenizer.eos_token_id or 0
                dummy = torch.tensor([[tok_id]], device=base_model_device)
                h = self.base_model(dummy).last_hidden_state
                text_hidden_size = h.shape[-1]

        self.vision_tower = VisionTower(dtype=self._dtype, device=self._device)
        self.image_processor = self.vision_tower.image_processor
        self.vision_encoder = self.vision_tower.model

        vision_hidden_size = self.vision_tower.hidden_size
        print(f"Using VisionTower with hidden_size={vision_hidden_size}")

        self.projector = Projector(vision_hidden_size, text_hidden_size, dtype=base_model_dtype, device=base_model_device)

        # ensure projector on same device/dtype as base model weights
        self.projector.to(base_model_device, dtype=base_model_dtype)
        print(f"Model device: {self._device}, dtype: {self._dtype}")
        print(f"Vision Tower device: {self.vision_tower.device}, dtype: {self.vision_tower.dtype}")
        print(f"Projector device: {next(self.projector.parameters()).device}, dtype: {next(self.projector.parameters()).dtype}")

        if self.config is None:
            self.config = BBOBConfig(
                vision_hidden_size=vision_hidden_size,
                text_hidden_size=text_hidden_size,
                base_model_name=str(model_path),
            )
            

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
        return self.image_processor, self.vision_tower
    
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
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_model(self):
        """
        Unfreeze base language model parameters
        """
        self.base_model.train()
        for param in self.base_model.parameters():
            param.requires_grad = True

    def train(self):
        """
        Set all model components to training mode
        """ 
        self.base_model.train()
        self.projector.train()

    def eval(self):
        """
        Set all model components to evaluation mode
        """
        self.base_model.eval()
        self.projector.eval()

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    '''
    API Functions
    '''

    def _merge_multimodal_inputs(self, visual_embeds, text_embeds, attention_mask):
        """
        Merge visual-token embeddings in front of text-token embeddings.

        Parameters:
            - visual_embeds: torch.Tensor shape (B, V, D) or *None*
            - text_embeds:  torch.Tensor shape (B, T, D)
            - attention_mask: torch.Tensor shape (B, T) or *None*

        Returns:
            - combined_embeds: torch.Tensor shape (B, V+T, D)
            - combined_mask:   torch.Tensor shape (B, V+T)
        """

        if visual_embeds is None:
            return text_embeds, attention_mask

        # create a visual attention mask (all 1s since every visual token is valid)
        bsz, v_len, _ = visual_embeds.size()
        device = visual_embeds.device
        visual_mask = torch.ones((bsz, v_len), dtype=torch.long, device=device)

        if attention_mask is None:
            attention_mask = torch.ones((bsz, text_embeds.size(1)), dtype=torch.long, device=device)

        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        return combined_embeds, combined_mask

    def _prepare_visual_inputs(self, images):
        """
        Turn a batch of raw images into language-space visual tokens.

        Parameters:
            - images: list | torch.Tensor – raw images accepted by the
              `VisionTower.image_processor` (e.g. PIL.Image, numpy array, etc.)

        Returns:
            - torch.Tensor | *None*, shape (B, V, D) where *V = H×W*, the
              number of spatial locations after flattening.
        """
        if images is None:
            return None

        # Step-1: pixel values
        pixel_values = self.vision_tower.process_image(images)

        # Step-2: spatial feature map from the frozen vision backbone
        with torch.no_grad():
            feats = self.vision_tower(pixel_values)  # (B, C, H, W)

        # Step-3: project to language embedding space producing visual tokens (B, H*W, D)
        visual_embeds = self.projector(feats)  # (B, H*W, D)
        return visual_embeds

    def forward(self, input_ids=None, input_embeds=None, attention_mask=None, images=None, labels=None, **kwargs):
        """
        Multimodal forward pass.

        Parameters:
            - input_ids:      torch.LongTensor shape (B, T) – token ids. Optional if *input_embeds* is given.
            - input_embeds:   torch.FloatTensor shape (B, T, D) – pre-computed embeddings.
            - attention_mask: torch.LongTensor shape (B, T)
            - images:         list | torch.Tensor – raw images (optional)
            - labels:         torch.LongTensor shape (B, T) – language modelling labels

        Returns:
            - transformers.modeling_outputs.CausalLMOutput (or subclass)
        """

        visual_embeds = self._prepare_visual_inputs(images)

        if input_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or input_embeds must be provided.")
            text_embeds = self.base_model.get_input_embeddings()(input_ids)
        else:
            text_embeds = input_embeds

        inputs_embeds, combined_mask = self._merge_multimodal_inputs(visual_embeds, text_embeds, attention_mask)

        if labels is not None and visual_embeds is not None:
            pad = torch.full((labels.size(0), visual_embeds.size(1)), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([pad, labels], dim=1)

        # Pass through language model
        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            labels=labels,
            **kwargs,
        )

    def save_pretrained(self, output_dir: str, **kwargs):
        """Leverage PreTrainedModel.save_pretrained then add projector/tower."""
        # Update config with relative paths before saving
        self.config.projector_path = "projector.safetensors"
        self.config.vision_tower_path = "vision_tower"

        super().save_pretrained(output_dir, **kwargs)

        # Save extra pieces the vanilla save does not cover (optional)
        proj_path = os.path.join(output_dir, "projector.safetensors")
        self.projector.save_pretrained(proj_path)

        vt_dir = os.path.join(output_dir, "vision_tower")
        os.makedirs(vt_dir, exist_ok=True)
        try:
            self.vision_tower.model.save_pretrained(vt_dir)
        except Exception:
            torch.save(self.vision_tower.model.state_dict(), os.path.join(vt_dir, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, model_path: str, *model_args, **kwargs):
        """Load via the standard HF mechanism then restore projector & tower."""

        obj: "BBOB" = super().from_pretrained(model_path, *model_args, **kwargs)

        # restore projector using path from config if available
        proj_rel = getattr(obj.config, "projector_path", "projector.safetensors")
        proj_path = os.path.join(model_path, proj_rel)
        if os.path.isfile(proj_path):
            state = st.load_file(proj_path, device=obj.projector.device)
            obj.projector.load_state_dict(state)

        # restore vision tower if directory exists
        vt_rel = getattr(obj.config, "vision_tower_path", "vision_tower")
        vt_dir = os.path.join(model_path, vt_rel)
        if os.path.isdir(vt_dir):
            try:
                obj.vision_tower.model = obj.vision_tower.model.from_pretrained(vt_dir, torch_dtype=obj.vision_tower.dtype)
            except Exception:
                sd = torch.load(os.path.join(vt_dir, "pytorch_model.bin"), map_location=obj.vision_tower.device)
                obj.vision_tower.model.load_state_dict(sd)

        return obj
        
        
        
        
        