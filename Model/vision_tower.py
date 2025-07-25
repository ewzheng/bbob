import torch
import torch.nn as nn
from transformers import MobileViTV2Model, MobileViTImageProcessor
import PIL.Image as Image
import numpy as np

DEFAULT_TARGET_SIZE = (256, 256)  # (W,H) matches MobileViT 256

def _letterbox_image(image: Image.Image, target_size=DEFAULT_TARGET_SIZE):
    """Resize *image* to fit in *target_size* with unchanged aspect ratio.

    Returns (new_image, scale, pad_w, pad_h). Matches Train.train_common logic
    but avoids the heavy OpenCV dependency.
    """
    iw, ih = image.size
    w, h = target_size
    scale = min(w / iw, h / ih)
    nw = max(1, int(iw * scale))
    nh = max(1, int(ih * scale))
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', target_size, (0, 0, 0))
    pad_w = (w - nw) // 2
    pad_h = (h - nh) // 2
    new_image.paste(image, (pad_w, pad_h))
    return new_image, scale, pad_w, pad_h

ENCODER = "apple/mobilevitv2-2.0-imagenet1k-256"

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
                img_size = getattr(cfg, "image_size", 256)
                dummy = torch.zeros(1, 3, img_size, img_size, device=self._device, dtype=self._dtype)
                c = self.model(dummy, output_hidden_states=False).last_hidden_state.shape[1]
                self._hidden_size = c

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True  

    def process_image(self, images, *, target_size: tuple[int, int] | None = None):
        """Convert *images* to pixel tensors ready for the backbone.

        This centralises *all* image-preprocessing logic so every caller
        (training collator, evaluation scripts, inference endpoints) sees
        the exact same transform:

        1. Convert to RGB ``PIL.Image``
        2. Letter-box resize to ``target_size`` (defaults to 256×256)
        3. Pass through the MobileViT image-processor with **resize disabled**

        Accepts either a single image or an iterable; returns a stacked
        ``torch.float32`` tensor on the tower device.
        """

        if target_size is None:
            target_size = DEFAULT_TARGET_SIZE

        # Allow single image input
        single = False
        if not isinstance(images, (list, tuple)):
            images = [images]
            single = True

        proc_imgs = []
        for img in images:
            # to PIL
            if not isinstance(img, Image.Image):
                if isinstance(img, torch.Tensor):
                    if img.dim() == 3 and img.shape[0] in (1, 3):
                        img = img.permute(1, 2, 0).cpu().numpy()
                    img = Image.fromarray(img)
                elif isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_lb, _, _, _ = _letterbox_image(img, target_size)
            proc_imgs.append(img_lb)

        px = self.image_processor(
            proc_imgs,
            return_tensors="pt",
            do_center_crop=False,
            do_resize=False,
        )["pixel_values"].to(self.device, dtype=self.dtype)

        return px[0] if single else px

    @torch.no_grad()
    def forward(self, images):
        '''
        run the backbone and return the spatial feature map.

        pre: `images` is already normalised float tensor produced by
        `process_image` or the gpu collate path.

        parameters:
            - images (torch.Tensor): pixel tensor `(b, 3, h, w)`.

        returns: list[torch.Tensor] where each tensor has shape `(b, c_i, h_i, w_i)`.
        The list contains the feature maps produced after every layer/stage of
        MobileViT-V2 (as exposed via ``hidden_states``).  These multi-scale
        maps allow the projector to pool information from shallow and deep
        layers alike.
        '''

        # Run the backbone **with** hidden-state capture
        outputs = self.model(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,   # ← capture all intermediate maps
            return_dict=True,
        )

        # HF returns a tuple – convert to list for convenience
        all_feats = list(outputs.hidden_states)  # tuple[(B,C_i,H_i,W_i), ...]

        return all_feats

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
