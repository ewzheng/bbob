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

        # ENHANCED: Split spatial embedding into two parts
        max_h = 32  # maximum grid height supported
        max_w = 32  # maximum grid width supported
        spatial_dim = outdim // 2  # Half for learned embeddings
        coord_dim = outdim - spatial_dim  # Half for coordinate embeddings
        
        # Learned row and column embeddings (reduced dimension)
        self.row_embedding = nn.Parameter(torch.empty(max_h, spatial_dim, dtype=dtype, device=device))
        self.col_embedding = nn.Parameter(torch.empty(max_w, spatial_dim, dtype=dtype, device=device))
        
        # NEW: Coordinate embedding network
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, coord_dim, dtype=dtype),  # 2D coordinates -> embedding
            nn.GELU(),
            nn.Linear(coord_dim, coord_dim, dtype=dtype)
        )
        
        nn.init.trunc_normal_(self.row_embedding, std=0.02)
        nn.init.trunc_normal_(self.col_embedding, std=0.02)
        
        self._spatial_dim = spatial_dim
        self._coord_dim = coord_dim

        self._indim = indim
        self._outdim = outdim
        self._dtype = dtype
        self._device = torch.device(device)

        # move sub-modules to target device
        self.net.to(self._device)
        self.skip.to(self._device)
        self.norm_out.to(self._device)
        self.coord_mlp.to(self._device)

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
        for p in self.coord_mlp.parameters():
            p.requires_grad = False
    
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
        for p in self.coord_mlp.parameters():
            p.requires_grad = True

    def _create_coordinate_features(self, H, W, B):
        """Create normalized coordinate features for each spatial location"""
        # Create normalized coordinates [0, 1]
        y_coords = torch.linspace(0, 1, H, device=self._device, dtype=self._dtype)
        x_coords = torch.linspace(0, 1, W, device=self._device, dtype=self._dtype)
        
        # Create meshgrid
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Stack coordinates: (H, W, 2)
        coords = torch.stack([x_grid, y_grid], dim=-1)
        
        # Reshape to (H*W, 2) and expand for batch
        coords = coords.reshape(H*W, 2).unsqueeze(0).expand(B, -1, -1)
        
        return coords

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

        # ENHANCED: Create two types of spatial embeddings
        
        # 1. Learned positional embeddings (reduced dimension)
        learned_pos = (
            self.row_embedding[:H].unsqueeze(1) +  # (H, 1, spatial_dim)
            self.col_embedding[:W].unsqueeze(0)    # (1, W, spatial_dim)
        )  # (H, W, spatial_dim)
        learned_pos = learned_pos.reshape(H*W, self._spatial_dim).unsqueeze(0).expand(B, -1, -1)
        
        # 2. Coordinate embeddings (new!)
        coords = self._create_coordinate_features(H, W, B)  # (B, H*W, 2)
        coord_features = self.coord_mlp(coords)  # (B, H*W, coord_dim)
        
        # Concatenate both types of spatial information
        spatial_features = torch.cat([learned_pos, coord_features], dim=-1)  # (B, H*W, outdim)
        
        # Add spatial features to fused features
        return fused + spatial_features
    
    ''' Saving methods, here if needed '''

    @classmethod
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



    
    