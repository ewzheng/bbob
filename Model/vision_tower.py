import torch
import torch.nn as nn
from transformers import MobileViTV2Model, MobileViTImageProcessor

ENCODER = "timm/mobilevitv2_200.cvnets_in22k_ft_in1k_384"

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
