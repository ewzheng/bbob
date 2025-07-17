'''
File: projector.py
Author: Elias Zheng and Claude
Description: This script contains the projector class.
'''

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
import torch.nn.functional as F
import math

class Projector(nn.Module):
    '''
    two-layer perceptron projector.

    pre: `indim` must equal the channel dim of the vision tower.

    parameters:
        - indim (int): input dim from vision encoder.
        - outdim (int): target dim (text embedding size).
        - dtype (torch.dtype): dtype for weights.
        - device (str | torch.device): where to allocate parameters.

    returns: instance ready for `.forward()`.
    '''

    def __init__(self, indim, outdim, dtype, device, output_tokens=1024):
        '''
        ctor.

        parameters:
            - indim (int): see class doc.
            - outdim (int): see class doc.
            - dtype (torch.dtype): torch dtype.
            - device (str | torch.device): allocation device.
            - output_tokens (int): number of output tokens (default 64)
        '''
        super().__init__()
        # two layer MLP: visiondim > textdim, GELU activation
        self.net = nn.Sequential(
            nn.Linear(indim, outdim, dtype=dtype),
            nn.GELU(),
            nn.Linear(outdim, outdim, dtype=dtype),
        )

        # flexible token pooling - automatically handles any input token count
        self.output_tokens = output_tokens
        self.output_spatial = (int(output_tokens ** 0.5), int(output_tokens ** 0.5))

        # learnable row and column embeddings (for output spatial size)
        # max_h = 32  # maximum grid height supported
        # max_w = 32  # maximum grid width supported
        # self.row_embedding = nn.Parameter(torch.empty(max_h, outdim, dtype=dtype, device=device))
        # self.col_embedding = nn.Parameter(torch.empty(max_w, outdim, dtype=dtype, device=device))
        # nn.init.trunc_normal_(self.row_embedding, std=0.02)
        # nn.init.trunc_normal_(self.col_embedding, std=0.02)

        self._indim = indim
        self._outdim = outdim
        self._dtype = dtype
        self._device = torch.device(device)

        # move sub-modules to target device
        self.net.to(self._device)

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
    
    @property
    def hiddendim(self):
        return self._hiddendim

    def _build_2d_sincos_embedding(self, h, w, dim, device):
        """
        Build 2D sinusoidal positional embedding as in ViT.
        Returns tensor of shape (h*w, dim)
        """
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=device),
            torch.arange(w, dtype=torch.float32, device=device),
            indexing='ij'
        )
        pos = torch.stack([grid_y, grid_x], dim=-1)  # (h, w, 2)
        pos = pos.reshape(-1, 2)  # (h*w, 2)
        dim_half = dim // 2
        omega = torch.arange(dim_half, dtype=torch.float32, device=device) / dim_half
        omega = 1.0 / (10000 ** omega)  # (dim_half,)
        out = []
        for i in range(2):  # y, x
            p = pos[:, i].unsqueeze(1)  # (h*w, 1)
            out.append(torch.sin(p * omega))
            out.append(torch.cos(p * omega))
        emb = torch.cat(out, dim=1)  # (h*w, dim)
        if emb.shape[1] > dim:
            emb = emb[:, :dim]
        elif emb.shape[1] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[1]))
        return emb.unsqueeze(0)  # (1, h*w, dim)

    def freeze(self):
        """
        Freeze all projector parameters to prevent training
        """
        for p in self.net.parameters():
            p.requires_grad = False
        # self.row_embedding.requires_grad = False
        # self.col_embedding.requires_grad = False
    
    def unfreeze(self):
        """
        Unfreeze all projector parameters to enable training
        """
        for p in self.net.parameters():
            p.requires_grad = True
        # self.row_embedding.requires_grad = True
        # self.col_embedding.requires_grad = True



    '''
    API Functions
    '''

    def forward(self, vision_in):
        '''
        forward pass.

        parameters:
            - vision_in (torch.Tensor): shape `(b, c, h, w)`.

        returns: torch.tensor shape `(b, output_tokens, outdim)`.
        '''

        if vision_in is None: return None

        # vision_in: (B, C, H, W)
        B, C, H, W = vision_in.shape

        # ------------------------------------------------------------------
        # Adaptive pooling: always convert feature map to fixed spatial size
        # (self.output_spatial) so the projector outputs exactly
        # `self.output_tokens` tokens regardless of the input resolution.
        # This preserves edge information because we *never* crop – we only
        # down-/up-sample the feature map via avg+max pooling combo.
        # ------------------------------------------------------------------
        if (H, W) != self.output_spatial:
            # Combine average + max pooling (empirically better than avg only)
            # 1. Keep input as BCHW
            avg_pool = F.adaptive_avg_pool2d(vision_in, self.output_spatial)  # (B,C,h*,w*)
            max_pool = F.adaptive_max_pool2d(vision_in, self.output_spatial)  # (B,C,h*,w*)
            vision_in = avg_pool + max_pool                                   # (B,C,h*,w*)
            H, W = self.output_spatial

        # Flatten for projection: (B, C, H, W) => (B, H*W, C)
        vision_in = vision_in.flatten(2).transpose(1, 2)  # (B, N, C) where N = H*W == self.output_tokens

        # Project to text space first (deep path)
        projected = self.net(vision_in)              # (B, N, D)

        # Add sinusoidal positional embeddings
        pos = self._build_2d_sincos_embedding(H, W, self._outdim, vision_in.device).expand(B, -1, -1)  # (B, N, D)

        return projected + pos
    
    ''' Saving methods, here if needed '''

    @classmethod
    def from_pretrained(cls, model_path, indim, outdim, dtype, device, output_tokens=128):
        """Load projector weights from a *.safetensors* or legacy *.pt* file.

        Parameters
        ----------
        model_path : str
            Path to the checkpoint file.
        indim, outdim : int
            Dimension parameters (must match those used during training).
        dtype : torch.dtype
        device : str | torch.device
            Target device for the loaded projector.
        output_tokens : int
            Number of output tokens (default 64).
        """

        projector = cls(indim, outdim, dtype, device, output_tokens=output_tokens)

        try:
            if model_path.endswith(".safetensors"):
                # Always load safetensors to CPU first for compatibility
                state_dict = load_file(model_path)
            else:
                # Fallback – legacy checkpoints saved with torch.save
                state_dict = torch.load(model_path, map_location="cpu")
            
            # Load state dict and let PyTorch handle device placement
            projector.load_state_dict(state_dict)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load projector weights from '{model_path}': {e}")

        return projector
    
    def save_pretrained(self, path):
        """Save projector weights to *path* (a .pt or .bin file)."""
        save_file(self.state_dict(), path)



    
    