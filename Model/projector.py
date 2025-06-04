import torch 
import torch.nn as nn

class Projector(nn.Module):
    """
    Two layer multi-layer perception projector
    """

    def __init__(self, indim, outdim):
        """
        Initialize projector with input and output dimensions
        
        Parameters:
            - indim: input dimension from vision encoder
            - outdim: output dimension matching text embedding space
        """
        super().__init__()
        # two layer MLP: visiondim > textdim, GELU activation
        self.net =nn.Sequential(
            nn.Linear(indim, outdim),
            nn.GELU(),
            nn.Linear(outdim, outdim)
        )
        self.indim = indim
        self.outdim = outdim

    def freeze(self):
        """
        Freeze all projector parameters to prevent training
        """
        for weights in self.net.parameters():
            weights.requires_grad = False
    
    def unfreeze(self):
        """
        Unfreeze all projector parameters to enable training
        """
        for weights in self.net.parameters():
            weights.requires_grad = True

    def train(self):
        """
        Set projector to training mode
        """
        self.net.train()
    
    def eval(self):
        """
        Set projector to evaluation mode
        """
        self.net.eval()
    
    def forward(self, vision_in):
        """
        Projector forward pass

        Parameters:
            - vision_in: torch.Tensor, the output of the vision encoder
        
        Returns: 
            - torch.Tensor, vision_in projected to outdim
        """

        if vision_in is None: return None

        #  mobilevit does weird shit, need to flatten to 2D
        if vision_in.dim() == 4:  
                    # Flatten spatial dimensions to create visual tokens
                    vision_in = vision_in.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        return self.net(vision_in)

