import torch
import torch.nn as nn
from transformer import MobileViTV2Model, MobileViTV2Config, MobileViTImageProcessor

class VisionTower(nn.Module):
    def __init__(self, dtype, device):
        '''
        Initialize vision tower with MobileViTV2 model and image processor

        Parameters:
            - dtype: data type
            - device: device to move the vision tower to
        '''
        super().__init__()

        self.model = MobileViTV2Model.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256", torch_dtype=dtype)
        self.image_processor = MobileViTImageProcessor()

        self._dtype = dtype
        self._device = torch.device(device)

        # move to device and set to eval mode since this shit is never getting trained
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self._device)
        self.model.eval()

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
        return self.model.config.hidden_size

    def process_image(self, images):
        '''
        Process images into pixel values

        Pre: images should be formatted correctly prior to running through the image processor

        Parameters:
            - images: list of images

        Returns:
            - pixel_values: tensor of pixel values
        '''
        return self.image_processor(images, return_tensors="pt").pixel_values.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def forward(self, images):
        '''
        Pre: images are the processed pixel values outputted by the image processor

        Parameters:
            - images: tensor of pixel values

        Returns:
            - features: tensor of shape (B, C, H, W) ready for the projector
        '''

        seq_feats = self.model(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=False
        ).last_hidden_state  # (B, C, H, W)

        return seq_feats
        