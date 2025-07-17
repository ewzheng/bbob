import torch
import torch.nn as nn
from transformers import MobileViTV2Model, MobileViTImageProcessor, CLIPVisionModel, CLIPImageProcessor
import math

#ENCODER = "apple/mobilevitv2-2.0-imagenet1k-256"
ENCODER = "openai/clip-vit-base-patch16"

import math
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor

ENCODER = "openai/clip-vit-base-patch16"


class VisionTower(nn.Module):
    """
    Frozen CLIP vision encoder that returns either the CLS token or a (B,C,H,W)
    spatial map.  Works for square **and** rectangular inputs.
    """
    def __init__(
        self,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
        encoder_name: str | None = None,
        return_spatial: bool = True,
    ):
        super().__init__()
        encoder_name = encoder_name or ENCODER

        # backbone + processor -------------------------------------------------
        self.model = CLIPVisionModel.from_pretrained(encoder_name, torch_dtype=dtype)
        self.image_processor = CLIPImageProcessor.from_pretrained(
            encoder_name,
            do_center_crop=False,                 # keep full FOV
            size={"shortest_edge": 512}
        )

        self._dtype   = dtype
        self._device  = torch.device(device)
        self._hidden_size = self.model.config.hidden_size

        self.model.to(self._device).eval()        # keep frozen for Stage-1/2
        self.return_spatial = return_spatial

    # ---------------------------------------------------------------- utils
    def freeze(self):   self.model.requires_grad_(False)
    def unfreeze(self): self.model.requires_grad_(True)

    # ---------------------------------------------------------------- processing
    def process_image(self, images):
        px = self.image_processor(images=images, return_tensors="pt").pixel_values
        return px.to(self.device, dtype=self.dtype)

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        outs = self.model(
            pixel_values=images.to(self.device, dtype=self.dtype),
            output_hidden_states=False,
            interpolate_pos_encoding=True,     # <-- key line
        )
        tokens = outs.last_hidden_state         # (B, 1+N, C)

        if not self.return_spatial:
            return tokens[:, 0]                 # CLS only (B,C)

        # -------- reshape patch tokens to (B,C,H_grid,W_grid) -------------
        patch_tokens = tokens[:, 1:]            # drop CLS
        B, N, C = patch_tokens.shape

        h_grid = images.shape[-2] // 16         # rows  (16-px patch)
        w_grid = images.shape[-1] // 16         # cols
        if h_grid * w_grid != N:                # sanity-guard
            raise ValueError(
                f"Mismatch: tokens={N}, expected {h_grid*w_grid} "
                f"from {h_grid}×{w_grid} patch grid."
            )

        spatial = (
            patch_tokens
            .permute(0, 2, 1)                   # (B,C,N)
            .reshape(B, C, h_grid, w_grid)      # (B,C,H,W)
            .contiguous()
        )
        return spatial

    # ---------------------------------------------------------------- props
    @property
    def dtype(self):       return self._dtype
    @property
    def device(self):      return self._device
    @property
    def config(self):      return self.model.config
    @property
    def hidden_size(self): return self._hidden_size


"""
class VisionTower(nn.Module):
    def __init__(self, dtype, device):
        '''
        construct a frozen mobilevit-v2 vision tower.

        parameters:
            - dtype (torch.dtype): weight / activation dtype.
            - device (str | torch.device): target device.

        returns: instance ready for `.process_image()` / `.forward()`.
        '''
        super().__init__()

        self.model = MobileViTV2Model.from_pretrained(ENCODER, torch_dtype=dtype)
        self.image_processor = MobileViTImageProcessor.from_pretrained(ENCODER, do_center_crop=False)
        self._dtype = dtype
        self._device = torch.device(device)
        
        self.model.to(self._device)
        self.model.eval()

        # derive final feature dimension across MobileViT variants
        cfg = self.model.config
        if hasattr(cfg, "hidden_size") and cfg.hidden_size is not None:
            self._hidden_size = cfg.hidden_size
        elif hasattr(cfg, "hidden_sizes") and len(cfg.hidden_sizes) > 0:
            self._hidden_size = cfg.hidden_sizes[-1]
        elif hasattr(cfg, "neck_hidden_sizes") and len(cfg.neck_hidden_sizes) > 0:
            self._hidden_size = cfg.neck_hidden_sizes[-1]
        else:
            # last resort: run a dummy forward pass to infer channel dim
            with torch.no_grad():
                img_size = getattr(cfg, "image_size", 512)
                dummy = torch.zeros(1, 3, img_size, img_size, device=self._device, dtype=self._dtype)
                c = self.model(dummy, output_hidden_states=False).last_hidden_state.shape[1]
                self._hidden_size = c

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True  

    def process_image(self, images):
        '''
        convert raw images to mobilevit pixel tensors.

        pre: `images` must be list[pil.Image] or equivalent format
        accepted by `MobileViTImageProcessor`.

        parameters:
            - images (list[Any]): list of input images.

        returns: torch.tensor of shape `(b, 3, h, w)` on tower.device.
        '''
        return self.image_processor(images, return_tensors="pt").pixel_values.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def forward(self, images):
        '''
        run the backbone and return the spatial feature map.

        pre: `images` is already normalised float tensor produced by
        `process_image` or the gpu collate path.

        parameters:
            - images (torch.Tensor): pixel tensor `(b, 3, h, w)`.

        returns: torch.tensor `(b, c, h', w')` where `c = hidden_size`.
        Returns:
            - features: tensor of shape (B, C, H, W) ready for the projector
        '''

        seq_feats = self.model(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=False
        ).last_hidden_state  # (B, C, H, W)

        return seq_feats

    @property
    def dtype(self):
        return self._dtype
    
    @property
    def device(self):
        return self._device
    
    @property
    def config(self):
        return self.model.config
        
    @property
    def hidden_size(self):
        return self._hidden_size
"""