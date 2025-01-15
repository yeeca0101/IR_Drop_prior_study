import torch
from torch import nn
from torch.nn import functional as F

'''
fishbone 

x
'''

class MLPMixerChannelInvariantBlock(nn.Module):
    def __init__(self, in_channels, patch_size=4, hidden_dim=256):
        super(MLPMixerChannelInvariantBlock, self).__init__()
        self.patch_size = patch_size
        self.tokens_mlp_dim = 2 * hidden_dim
        self.channels_mlp_dim = 2 * hidden_dim
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        
        self.token_mix = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.tokens_mlp_dim),
            nn.GELU(),
            nn.Linear(self.tokens_mlp_dim, hidden_dim)
        )
        
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.channels_mlp_dim),
            nn.GELU(),
            nn.Linear(self.channels_mlp_dim, hidden_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # (B, hidden_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, hidden_dim) where N = H' * W'
        
        # Token mixing
        y = self.token_mix(x)
        x = x + y
        
        # Channel mixing
        y = self.channel_mix(x).transpose(1, 2)  # (B, hidden_dim, N) after transpose
        x = x.transpose(1, 2) + y  # Make x also (B, hidden_dim, N) for addition
        
        # Reshape back to image
        x = x.transpose(1, 2).reshape(B, -1, H // self.patch_size, W // self.patch_size)
        return x
    
# 
class BasicBlock(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self,x):
        return x
    
# down block



