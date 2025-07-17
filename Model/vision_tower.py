import torch
import torch.nn as nn
from transformers import MobileViTV2Model, MobileViTImageProcessor, CLIPVisionModel, CLIPImageProcessor
import math

#ENCODER = "apple/mobilevitv2-2.0-imagenet1k-256"
ENCODER = "openai/clip-vit-base-patch16"

class VisionTower(nn.Module):
    """
    Frozen CLIP vision encoder wrapped in the same style as the MobileViT VisionTower.
    By default it returns a spatial feature map (B, C, H, W).
    """
    def __init__(
        self,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
        encoder_name: str | None = None,
        return_spatial: bool = True,        # ← now True by default
    ):
        super().__init__()
        encoder_name = encoder_name or ENCODER

        # backbone + processor -------------------------------------------------
        self.model = CLIPVisionModel.from_pretrained(encoder_name, torch_dtype=dtype)
        self.image_processor = CLIPImageProcessor.from_pretrained(encoder_name, do_center_crop=False, size={"shortest_edge": 512})
        self._dtype  = dtype
        self._device = torch.device(device)

        self.model.to(self._device).eval()
        self._hidden_size = self.model.config.hidden_size   # single number in CLIP

        self.return_spatial = return_spatial                # CLS vs (B,C,H,W)

    # --------------------------------------------------------------------- utils
    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.model.parameters():
            p.requires_grad = True

    # ---------------------------------------------------------------- processing
    def process_image(self, images):
        """
        images : list[PIL.Image] or ndarray – anything CLIPImageProcessor accepts.
        Returns pixel tensor (B,3,H,W) on the tower’s device/dtype.
        """
        px = self.image_processor(images=images, return_tensors="pt").pixel_values
        return px.to(self.device, dtype=self.dtype)

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        """
        images : (B,3,H,W) tensor already on the correct device/dtype.
        Returns
            * CLS embedding  → (B,hidden_size)  if self.return_spatial is False
            * Spatial map    → (B,C,h,w)        if self.return_spatial is True
        """
        outs   = self.model(
            images.to(self.device, dtype=self.dtype),
            output_hidden_states=False,
            interpolate_pos_encoding=True
        )
        tokens = outs.last_hidden_state       # (B, 1 + N, C)

        if not self.return_spatial:           # just the pooled CLS token
            return tokens[:, 0]               # (B, C)

        # reshape patch tokens back to 2-D grid -----------------------------
        patch_tokens = tokens[:, 1:, :]       # (B, N, C)
        B, N, C      = patch_tokens.shape
        side         = int(math.sqrt(N))      # assumes square grid
        if side * side != N:
            raise ValueError(
                f"Cannot reshape {N} patches into a square – "
                "set return_spatial=False or handle non-square grids manually."
            )
        spatial = (
            patch_tokens
            .permute(0, 2, 1)                # (B, C, N)
            .reshape(B, C, side, side)       # (B, C, H, W)
            .contiguous()
        )
        return spatial

    # ---------------------------------------------------------------- props
    @property
    def dtype(self):         return self._dtype
    @property
    def device(self):        return self._device
    @property
    def config(self):        return self.model.config
    @property
    def hidden_size(self):   return self._hidden_size

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