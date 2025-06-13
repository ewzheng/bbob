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
from peft import PeftModel, PeftConfig
from safetensors.torch import save_file 
from .vision_tower import VisionTower

class BBOB(nn.Module):
    """
    BBOB multimodal model combining vision encoder, projector, and language model
    """
    
    def __init__(self, model_path, max_memory=None, bnb_config=None):
        """
        Initialize BBOB model with specified components
        
        Parameters:
            - base_model: path or identifier for base language model
            - vision_encoder: path or identifier for vision encoder
            - bnb_config: quantization configuration for model loading
        """
        super(BBOB, self).__init__()
     

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

        self.vision_tower = VisionTower(dtype=self._dtype, device=self._device)
        self.image_processor = self.vision_tower.image_processor
        self.vision_encoder = self.vision_tower.model

        vision_hidden_size = self.vision_tower.hidden_size
        print(f"Using VisionTower with hidden_size={vision_hidden_size}")

        self.projector = Projector(vision_hidden_size, self.base_model.config.hidden_size, dtype=base_model_dtype, device=base_model_device)

        # ensure projector on same device/dtype as base model weights
        self.projector.to(base_model_device, dtype=base_model_dtype)
        print(f"Projector device: {next(self.projector.parameters()).device}, dtype: {next(self.projector.parameters()).dtype}")


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

    def save_pretrained(self, output_dir, save_vision_tower: bool = False):
        """
        Save the entire BBOB checkpoint to *output_dir*.

        Parameters:
            - output_dir: str – target directory where the checkpoint is written.
            - save_vision_tower: bool – if *True*, also persist the (potentially
              fine-tuned) MobileViT-V2 backbone.  Set to *False* to shrink disk
              footprint when the standard pretrained backbone is sufficient.

        Creates the following structure:
            • `config.json`           – minimal metadata to reload via
              :py:meth:`from_pretrained`.
            • `language_model/`       – sub-directory with base LLM + tokenizer
              saved via `transformers.PreTrainedModel.save_pretrained()`.
            • `projector.safetensors` – binary weights for the projector MLP.
            • `vision_tower/`         – (optional) directory with vision-tower
              weights.
        """
        import json, os
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save language model
        lm_dir = os.path.join(output_dir, "language_model")
        self.base_model.save_pretrained(lm_dir)
        self.base_tokenizer.save_pretrained(lm_dir)

        # 2. Save projector
        proj_path = os.path.join(output_dir, "projector.safetensors")
        self.projector.save_pretrained(proj_path)

        # 3. Optionally save vision tower (mostly for fine-tuned backbones)
        if save_vision_tower and hasattr(self, "vision_tower"):
            vt_dir = os.path.join(output_dir, "vision_tower")
            os.makedirs(vt_dir, exist_ok=True)
            try:
                # If the underlying MobileViTV2 model implements save_pretrained, use that.
                self.vision_tower.model.save_pretrained(vt_dir)
            except Exception:
                # Fallback – state_dict
                torch.save(self.vision_tower.model.state_dict(), os.path.join(vt_dir, "pytorch_model.bin"))

        # 4. Tiny JSON config so `from_pretrained` knows what to load.
        config = {
            "model_type": "BBOB",
            "base_model": lm_dir,
            "vision_tower": (vt_dir if save_vision_tower else None),
            "projector": proj_path,
            "vision_hidden_size": self.projector.indim,
            "text_hidden_size": self.projector.outdim,
        }
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, model_path, max_memory=None, bnb_config=None, map_location=None):
        """
        Restore a BBOB checkpoint that was previously saved with
        :py:meth:`save_pretrained`.

        Parameters:
            - model_path: str – path to the root directory containing the
              checkpoint.
            - max_memory: dict | None – optional `transformers` style GPU memory
              map (same semantics as the main constructor).
            - bnb_config: BitsAndBytesConfig | str | None – optional quantisation
              config (same semantics as the main constructor).
            - map_location: torch.device | str | None – device mapping override
              when loading raw `state_dict`s.

        Returns:
            - BBOB – fully initialised model, ready for inference or further
              fine-tuning.
        """
        import json, os
        # read config
        cfg_file = os.path.join(model_path, "config.json")
        if not os.path.isfile(cfg_file):
            raise FileNotFoundError(f"Missing config.json in {model_path}")
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # instantiate skeleton
        obj = cls(cfg["base_model"], max_memory=max_memory, bnb_config=bnb_config)

        # load projector
        proj_path = cfg.get("projector")
        if proj_path and os.path.isfile(proj_path):
            import safetensors.torch as st
            state = st.load_file(proj_path, device=obj.projector.device)
            obj.projector.load_state_dict(state)
        else:
            raise FileNotFoundError("Projector weights not found during from_pretrained")

        # load vision tower if present and path exists
        vt_dir = cfg.get("vision_tower")
        if vt_dir and os.path.isdir(vt_dir):
            try:
                obj.vision_tower.model = obj.vision_tower.model.from_pretrained(vt_dir, torch_dtype=obj.vision_tower.dtype)
            except Exception:
                # fallback – load_state_dict
                state_dict = torch.load(os.path.join(vt_dir, "pytorch_model.bin"), map_location=map_location or obj.vision_tower.device)
                obj.vision_tower.model.load_state_dict(state_dict)

        return obj
        
        
        
        
        