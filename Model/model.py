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
    """
    configuration container for the bbob model.
    """

    model_type = "bbob"

    def __init__(
        self,
        vision_hidden_size: int | None = None,
        text_hidden_size: int | None = None,
        base_model_name: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Use placeholder values when none are provided – this allows
        # `BBOBConfig()` to be instantiated with zero args, which
        # Hugging-Face does internally when it compares against a default
        # config during `save_pretrained()`.
        self.vision_hidden_size = vision_hidden_size if vision_hidden_size is not None else 1
        self.text_hidden_size   = text_hidden_size   if text_hidden_size   is not None else 1
        self.base_model_name    = base_model_name    if base_model_name    is not None else "unknown"
        # paths for extra components (filled during save_pretrained)
        self.projector_path: str | None = kwargs.get("projector_path")
        self.vision_tower_path: str | None = kwargs.get("vision_tower_path")

class BBOB(PreTrainedModel):
    """
    vision-language model composed of:
    • mobilevit-v2 backbone (frozen)
    • learnable projector to text space
    • hf causal-lm language head.
    """

    config_class = BBOBConfig

    def __init__(self, model_path=None, max_memory=None, bnb_config=None, config: BBOBConfig | None = None, **kwargs):
        '''
        constructor.

        parameters:
            - model_path (str|Path|None): base llm repo or ckpt dir.
            - max_memory (dict|None): per-gpu memory map for hf `device_map`.
            - bnb_config (str|BitsAndBytesConfig|None): "8bit"/"4bit"/"bf16"/"fp16".
            - config (BBOBConfig|None): pre-built config when loading.
        '''

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

        # store tokenizer
        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, torch_dtype=base_model_dtype)
        emb_layer = self._find_input_embedding(self.language_model)
        self._embedding_layer = emb_layer  # cache for forward()
        text_hidden_size = emb_layer.weight.shape[1]

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
        self.projector.freeze()
    
    def unfreeze_projector(self):
        """
        Unfreeze projector parameters to enable training
        """
        self.projector.unfreeze()

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

    '''
    API Functions
    '''

    def _merge_multimodal_inputs(self, visual_embeds, text_embeds, attention_mask):
        '''
        prepend visual tokens to text tokens.

        parameters:
            - visual_embeds (tensor|None): `(b, v, d)` or none.
            - text_embeds (tensor): `(b, t, d)`.
            - attention_mask (tensor|None): `(b, t)`.

        returns: tuple(tensor, tensor) -> combined_embeds, combined_mask.
        '''

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
        """Return input embeddings even if get_input_embeddings is not implemented."""
        return self._embedding_layer(input_ids)

    @staticmethod
    def _find_input_embedding(model):
        """Locate the token embedding layer without calling forward."""
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

    def forward(self, input_ids=None, input_embeds=None, attention_mask=None, images=None, labels=None, **kwargs):
        '''
        multimodal causal-lm pass.

        parameters:
            - input_ids (tensor|None): token ids `(b, t)`.
            - input_embeds (tensor|None): pre-computed embeddings.
            - attention_mask (tensor|None): mask.
            - images (list|tensor|None): raw or processed images.
            - labels (tensor|None): lm labels.

        returns: `transformers.CausalLMOutput`.
        '''

        visual_embeds = self._prepare_visual_inputs(images)

        if input_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or input_embeds must be provided.")
            text_embeds = self._embed_tokens(input_ids)
        else:
            text_embeds = input_embeds

        inputs_embeds, combined_mask = self._merge_multimodal_inputs(visual_embeds, text_embeds, attention_mask)

        if labels is not None and visual_embeds is not None:
            # If labels already include placeholders for visual tokens, skip adding again
            expected_len = text_embeds.size(1) + visual_embeds.size(1)
            if labels.size(1) < expected_len:
                pad = torch.full((labels.size(0), visual_embeds.size(1)), -100, dtype=labels.dtype, device=labels.device)
                labels = torch.cat([pad, labels], dim=1)

        # Pass through language model
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            labels=labels,
            **kwargs,
        )

    def save_pretrained(self, output_dir: str, **kwargs):
        """Persist only the **small** pieces we actually fine-tune.

        We do **not** store the full language-model weights ─ they are unchanged
        from the public checkpoint referenced by ``config.base_model_name`` and
        would massively bloat every projector checkpoint.  Instead we save:

        1. ``config.json`` – updated so it knows where to find the base model
           and the projector file.
        2. ``projector.safetensors`` – a few-MB state-dict with the learned
           projection layer.
        3. (optional) a fine-tuned vision-tower dir if the user ever unfreezes
           those parameters.
        """

        os.makedirs(output_dir, exist_ok=True)

        # --- 1) config -----------------------------------------------------------------
        self.config.projector_path = "projector.safetensors"
        self.config.save_pretrained(output_dir)

        # --- 2) projector --------------------------------------------------------------
        proj_path = os.path.join(output_dir, self.config.projector_path)
        st.save_file(self.projector.state_dict(), proj_path)

        # --- 3) vision tower (rarely trainable) ---------------------------------------
        if any(p.requires_grad for p in self.vision_tower.model.parameters()):
            vt_dir = os.path.join(output_dir, "vision_tower")
            os.makedirs(vt_dir, exist_ok=True)
            self.vision_tower.model.save_pretrained(vt_dir)

    @classmethod
    def from_pretrained(cls, ckpt_dir: str, *model_args, **kwargs):
        """Rebuild a BBOB instance from a lightweight checkpoint saved above."""

        # 1) Load BBOBConfig first
        cfg = BBOBConfig.from_pretrained(ckpt_dir)

        # 2) Re-instantiate the full model; the heavy base LM comes from HF Hub
        obj: "BBOB" = cls(model_path=cfg.base_model_name, config=cfg, *model_args, **kwargs)

        # 3) Projector weights
        proj_rel = getattr(cfg, "projector_path", "projector.safetensors")
        proj_path = os.path.join(ckpt_dir, proj_rel)
        if os.path.isfile(proj_path):
            obj.projector.load_state_dict(st.load_file(proj_path, device=obj.projector.device))

        # 4) Vision tower (optional)
        vt_dir = os.path.join(ckpt_dir, "vision_tower")
        if os.path.isdir(vt_dir):
            try:
                obj.vision_tower.model = obj.vision_tower.model.from_pretrained(vt_dir, torch_dtype=obj.vision_tower.dtype)
            except Exception:
                sd = torch.load(os.path.join(vt_dir, "pytorch_model.bin"), map_location=obj.vision_tower.device)
                obj.vision_tower.model.load_state_dict(sd)

        return obj
        
        
        
        
        