import torch
import torch.nn as nn


class DepthwiseChannelInvariantConv(nn.Module):
    def __init__(self, in_channels, patch_size=4):
        super(DepthwiseChannelInvariantConv, self).__init__()
        self.patch_size = patch_size
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, padding=1)
        self.token_mix = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * 2),
            nn.GELU(),
            nn.Linear(in_channels * 2, in_channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.depthwise_conv(x)
        
        # Token mixing (flatten and apply mix)
        x_flat = x.flatten(2).transpose(1, 2)
        x = x + self.token_mix(x_flat).transpose(1, 2).view(B, C, H, W)
        
        return x

class PointwiseChannelInvariantConv(nn.Module):
    def __init__(self, in_channels, patch_size=4):
        super(PointwiseChannelInvariantConv, self).__init__()
        self.patch_size = patch_size
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * 2),
            nn.GELU(),
            nn.Linear(in_channels * 2, in_channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.pointwise_conv(x)
        
        # Channel mixing
        x_flat = x.flatten(2).transpose(1, 2)
        x = x + self.channel_mix(x_flat).transpose(1, 2).view(B, C, H, W)
        
        return x

class SelfAttentionChannelInvariantConv(nn.Module):
    def __init__(self, in_channels, patch_size=4):
        super(SelfAttentionChannelInvariantConv, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = in_channels * patch_size * patch_size
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=3)
        self.token_mix = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, self.embed_dim)
        x = x.permute(1, 0, 2)
        
        # Attention + Token mixing
        attn_out, _ = self.attention(x, x, x)
        x = x + self.token_mix(attn_out)
        
        # Reshape back
        x = x.permute(1, 2, 0).reshape(B, C, H, W)
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, patch_size=4):
        super(DepthwiseSeparableConv, self).__init__()
        self.patch_size = patch_size
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * 2),
            nn.GELU(),
            nn.Linear(in_channels * 2, in_channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.depthwise(x)
        x = self.pointwise(x)
        
        # Channel mixing
        x_flat = x.flatten(2).transpose(1, 2)
        x = x + self.channel_mix(x_flat).transpose(1, 2).view(B, C, H, W)
        
        return x

class ChannelInvariantBlock(nn.Module):
    def __init__(self, in_channels, patch_size=4):
        super(ChannelInvariantBlock, self).__init__()
        self.patch_size = patch_size
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.token_mix = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * 2),
            nn.GELU(),
            nn.Linear(in_channels * 2, in_channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.batch_norm(x)
        x = self.conv1x1(x)
        
        # Token mixing
        x_flat = x.flatten(2).transpose(1, 2)
        x = x + self.token_mix(x_flat).transpose(1, 2).view(B, C, H, W)
        
        return x
    


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
    
    

if __name__ == '__main__':
    # Reduce the dummy input batch size to minimize memory usage
    dummy_input = torch.randn(2, 3, 32, 32)

    # Instantiate each block
    depthwise_channel_invariant_conv = DepthwiseChannelInvariantConv(in_channels=3)
    pointwise_channel_invariant_conv = PointwiseChannelInvariantConv(in_channels=3)
    self_attention_channel_invariant_conv = SelfAttentionChannelInvariantConv(in_channels=3)
    depthwise_separable_conv = DepthwiseSeparableConv(in_channels=3)
    channel_invariant_block = ChannelInvariantBlock(in_channels=3)

    # Test each block with the dummy input
    try:
        print("Testing DepthwiseChannelInvariantConv...")
        output_depthwise = depthwise_channel_invariant_conv(dummy_input)
        print("Output shape:", output_depthwise.shape)
    except Exception as e:
        print("Error in DepthwiseChannelInvariantConv:", e)

    try:
        print("\nTesting PointwiseChannelInvariantConv...")
        output_pointwise = pointwise_channel_invariant_conv(dummy_input)
        print("Output shape:", output_pointwise.shape)
    except Exception as e:
        print("Error in PointwiseChannelInvariantConv:", e)

    try:
        print("\nTesting SelfAttentionChannelInvariantConv...")
        output_self_attention = self_attention_channel_invariant_conv(dummy_input)
        print("Output shape:", output_self_attention.shape)
    except Exception as e:
        print("Error in SelfAttentionChannelInvariantConv:", e)

    try:
        print("\nTesting DepthwiseSeparableConv...")
        output_depthwise_separable = depthwise_separable_conv(dummy_input)
        print("Output shape:", output_depthwise_separable.shape)
    except Exception as e:
        print("Error in DepthwiseSeparableConv:", e)

    try:
        print("\nTesting ChannelInvariantBlock...")
        output_channel_invariant = channel_invariant_block(dummy_input)
        print("Output shape:", output_channel_invariant.shape)
    except Exception as e:
        print("Error in ChannelInvariantBlock:", e)
