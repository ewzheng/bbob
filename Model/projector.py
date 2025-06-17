'''
File: projector.py
Author: Elias Zheng and Claude
Description: This script contains the projector class.
'''

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

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

    def __init__(self, indim, outdim, dtype, device):
        '''
        ctor.

        parameters:
            - indim (int): see class doc.
            - outdim (int): see class doc.
            - dtype (torch.dtype): torch dtype.
            - device (str | torch.device): allocation device.
        '''
        super().__init__()
        # two layer MLP: visiondim > textdim, GELU activation
        self._hiddendim = max(indim, outdim*2)
        self.net = nn.Sequential(
            nn.Linear(indim, self._hiddendim, dtype=dtype),
            nn.GELU(),
            nn.Linear(self._hiddendim, outdim, dtype=dtype),
            nn.GELU(),
            nn.Linear(outdim, outdim, dtype=dtype),
        )

        # skip connection (1×1 projection) – bias not needed due to subsequent LN
        self.skip = nn.Linear(indim, outdim, bias=False, dtype=dtype)

        # final normalisation applied *after* deep+skip fusion
        self.norm_out = nn.LayerNorm(outdim, eps=1e-5, dtype=dtype)

        # learnable row and column embeddings
        max_h = 32  # maximum grid height supported
        max_w = 32  # maximum grid width supported
        self.row_embedding = nn.Parameter(torch.empty(max_h, outdim, dtype=dtype, device=device))
        self.col_embedding = nn.Parameter(torch.empty(max_w, outdim, dtype=dtype, device=device))
        nn.init.trunc_normal_(self.row_embedding, std=0.02)
        nn.init.trunc_normal_(self.col_embedding, std=0.02)

        self._indim = indim
        self._outdim = outdim
        self._dtype = dtype
        self._device = torch.device(device)

        # move sub-modules to target device
        self.net.to(self._device)
        self.skip.to(self._device)
        self.norm_out.to(self._device)

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

    def freeze(self):
        """
        Freeze all projector parameters to prevent training
        """
        for p in self.net.parameters():
            p.requires_grad = False
        for p in self.skip.parameters():
            p.requires_grad = False
        for p in self.norm_out.parameters():
            p.requires_grad = False
        self.row_embedding.requires_grad = False
        self.col_embedding.requires_grad = False
    
    def unfreeze(self):
        """
        Unfreeze all projector parameters to enable training
        """
        for p in self.net.parameters():
            p.requires_grad = True
        for p in self.skip.parameters():
            p.requires_grad = True
        for p in self.norm_out.parameters():
            p.requires_grad = True
        self.row_embedding.requires_grad = True
        self.col_embedding.requires_grad = True

    '''
    API Functions
    '''

    def forward(self, vision_in):
        '''
        forward pass.

        parameters:
            - vision_in (torch.Tensor): shape `(b, c, h, w)`.

        returns: torch.tensor shape `(b, h*w, outdim)`.
        '''

        if vision_in is None: return None

        # vision_in: (B, C, H, W)
        B, C, H, W = vision_in.shape

        if H > self.row_embedding.size(0) or W > self.col_embedding.size(0):
            raise ValueError(f"Input feature map size {(H, W)} exceeds max supported {(self.row_embedding.size(0), self.col_embedding.size(0))}")

        # Flatten for projection: (B, C, H, W) > (B, H*W, C)
        vision_in = vision_in.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Project to text space first (deep path)
        projected = self.net(vision_in)              # (B, N, D)

        # Residual skip path
        skip_out = self.skip(vision_in)              # (B, N, D)

        # Fuse
        fused = self.norm_out(projected + skip_out)  # (B, N, D)

        # Add spatial embeddings back after projection to restore spatial information
        pos = (
            self.row_embedding[:H].unsqueeze(1) +  # (H, 1, outdim)
            self.col_embedding[:W].unsqueeze(0)    # (1, W, outdim)
        )  # (H, W, outdim)
        pos = pos.reshape(H * W, self._outdim).unsqueeze(0).expand(B, -1, -1)  # (B, H*W, outdim)

        # add spatial features after projection
        return fused + pos
    
    ''' Saving methods, here if needed '''

    def from_pretrained(cls, model_path, indim, outdim, dtype, device):
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
        """

        projector = cls(indim, outdim, dtype, device)

        try:
            if model_path.endswith(".safetensors"):
                state_dict = load_file(model_path, device=device)
            else:
                # Fallback – legacy checkpoints saved with torch.save
                state_dict = torch.load(model_path, map_location=device)

            projector.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load projector weights from '{model_path}': {e}")

        return projector
    
    def save_pretrained(self, path):
        """Save projector weights to *path* (a .pt or .bin file)."""
        save_file(self.state_dict(), path)



    
    