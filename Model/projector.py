'''
File: projector.py
Author: Elias Zheng and Claude
Description: This script contains the projector class.
'''

import torch 
import torch.nn as nn
from safetensors.torch import save_file

class Projector(nn.Module):
    """
    Two layer multi-layer perception projector
    """

    def __init__(self, indim, outdim, dtype, device):
        """
        Initialize projector with input and output dimensions
        
        Parameters:
            - indim: input dimension from vision encoder
            - outdim: output dimension matching text embedding space
            - dtype: data type
            - device: device to move the projector to
        """
        super().__init__()
        # two layer MLP: visiondim > textdim, GELU activation
        self.net = nn.Sequential(
            nn.Linear(indim, outdim, dtype=dtype),
            nn.GELU(),
            nn.Linear(outdim, outdim, dtype=dtype)
        )

        # learnable row and column embeddings
        max_h = 32  # maximum grid height supported
        max_w = 32  # maximum grid width supported
        self.row_embedding = nn.Parameter(torch.empty(max_h, indim, dtype=dtype))
        self.col_embedding = nn.Parameter(torch.empty(max_w, indim, dtype=dtype))
        nn.init.trunc_normal_(self.row_embedding, std=0.02)
        nn.init.trunc_normal_(self.col_embedding, std=0.02)

        self._indim = indim
        self._outdim = outdim
        self._dtype = dtype
        self._device = torch.device(device)

        self.net.to(self._device)
        self.row_embedding = self.row_embedding.to(self._device)
        self.col_embedding = self.col_embedding.to(self._device)

    '''
    Utils, getters, and setters
    '''
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def device(self):
        return self._device
    
    @property
    def indim(self):
        return self._indim
    
    @property
    def outdim(self):
        return self._outdim

    def freeze(self):
        """
        Freeze all projector parameters to prevent training
        """
        for p in self.net.parameters():
            p.requires_grad = False
        self.row_embedding.requires_grad = False
        self.col_embedding.requires_grad = False
    
    def unfreeze(self):
        """
        Unfreeze all projector parameters to enable training
        """
        for p in self.net.parameters():
            p.requires_grad = True
        self.row_embedding.requires_grad = True
        self.col_embedding.requires_grad = True

    '''
    API Functions
    '''

    def forward(self, vision_in):
        """
        Projector forward pass

        Parameters:
            - vision_in: torch.Tensor format is (B, C, H, W)
        
        Returns: 
            - torch.Tensor, vision_in projected to outdim
        """

        if vision_in is None: return None

        # vision_in: (B, C, H, W)
        B, C, H, W = vision_in.shape

        if H > self.row_embedding.size(0) or W > self.col_embedding.size(0):
            raise ValueError(f"Input feature map size {(H, W)} exceeds max supported {(self.row_embedding.size(0), self.col_embedding.size(0))}")

        pos = (
            self.row_embedding[:H].unsqueeze(1) +  # (H, 1, C)
            self.col_embedding[:W].unsqueeze(0)    # (1, W, C)
        )  # (H, W, C)
        pos = pos.reshape(H * W, C).unsqueeze(0).expand(B, -1, -1)  # (B, H*W, C)

        vision_in = vision_in.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # add spatial features
        vision_in = vision_in + pos

        return self.net(vision_in)
    
    ''' Saving methods, here if needed '''

    @classmethod
    def from_pretrained(cls, model_path, indim, outdim, dtype, device):
        """
        Load projector from pretrained checkpoint
        """
        projector = cls(indim, outdim, dtype, device)
        projector.load_state_dict(torch.load(model_path, map_location=device))
    
        return projector
    
    def save_pretrained(self, path):
        """Save projector weights to *path* (a .pt or .bin file)."""
        save_file(self.state_dict(), path)



    
    