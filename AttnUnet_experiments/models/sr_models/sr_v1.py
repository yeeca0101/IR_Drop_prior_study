import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#############################################
# Necessary Building Blocks for SR Model
#############################################

class SwishT(nn.Module):
    """Swish activation: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.

    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    def __init__(self, p, block_size=7):
        super(DropBlock2D, self).__init__()

        self.drop_prob = p
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)
            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            # place mask on input device
            mask = mask.to(x.device)
            # compute block mask
            block_mask = self._compute_block_mask(mask)
            # apply block mask
            out = x * block_mask[:, None, :, :]
            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)

    
class AttentionGate(nn.Module):
    """
    A simple attention gate that receives two inputs: a “skip” feature map (from a higher resolution branch)
    and a “gating” signal (from the lower resolution branch). It computes an attention coefficient to
    weight the skip connection.
    """
    def __init__(self, in_ch_x, in_ch_g, out_ch, concat=True):
        super().__init__()
        self.concat = concat
        self.W_x = nn.Conv2d(in_ch_x, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_g = nn.Conv2d(in_ch_g, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(out_ch, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        # x: skip connection (high resolution)
        # g: gating signal (low resolution)
        theta_x = self.W_x(x)
        phi_g = self.W_g(g)
        # If spatial sizes differ, upsample the gating signal features
        if theta_x.shape[2:] != phi_g.shape[2:]:
            phi_g = F.interpolate(phi_g, size=theta_x.shape[2:], mode='bilinear', align_corners=True)
        f = self.relu(theta_x + phi_g)
        psi_f = self.psi(f)
        alpha = self.sigmoid(psi_f)
        return x * alpha

class UpBlock(nn.Module):
    """
    An upsampling block that upsamples the gating signal by a factor of 2,
    concatenates it with the (attention gated) skip connection, applies optional dropout,
    and then performs two convolutional layers.
    """
    def __init__(self, in_ch_x, in_ch_g, out_ch, dropout_m, dropout_p, act):
        """
        in_ch_x: number of channels in the skip connection (after attention gating)
        in_ch_g: number of channels in the gating signal (before upsampling)
        out_ch: number of channels after this block
        dropout_m: dropout module (or Identity if no dropout is desired)
        dropout_p: dropout probability (for documentation; already set in dropout_m)
        act: activation module (e.g., SwishT)
        """
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dropout = dropout_m  # dropout module (can be Identity)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch_x + in_ch_g, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            act,
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            act
        )

    def forward(self, x, g):
        # g: gating signal, x: skip connection feature (after attention gating)
        g_up = self.upsample(g)
        # Concatenate along channel dimension
        x_cat = torch.cat([x, g_up], dim=1)
        x_cat = self.dropout(x_cat)
        return self.conv(x_cat)

#############################################
# Super–Resolution Model using only Upsampling (Decoder) Parts
#############################################

class SRModelV1(nn.Module):
    """
    A Super–Resolution (SR) model that uses only the upsampling (decoder) portion of the original network.
    
    This model first extracts features from the low resolution (LR) image via a simple feature extractor.
    It then uses a series of UpBlocks (with attention gating) to progressively increase the resolution.
    
    To “simulate” skip connections (as in the original U–Net decoder) we build skip features by upsampling
    the LR input to the target resolution at that level and processing it with a convolution.
    
    Args:
        in_ch (int): Number of input channels (e.g., 3 for RGB).
        out_ch (int): Number of output channels.
        upscale_factor (int): Overall upscaling factor (must be a power of 2). For example, 4 means 4× SR.
        dropout_name (str): Either 'nn.dropout' or 'dropblock'. (Default 'nn.dropout')
        dropout_p (float): Dropout probability. (Default 0.0 means no dropout)
        act (nn.Module): Activation function. (Default: SwishT)
    """
    def __init__(self, in_ch=1, out_ch=1, upscale_factor=4, 
                 dropout_name='nn.dropout', dropout_p=0.1, act=SwishT()):
        super().__init__()
        # We assume upscale_factor is a power of 2.
        self.upscale_factor = upscale_factor
        num_up_blocks = int(math.log2(upscale_factor))
        self.num_up_blocks = num_up_blocks

        base_channels = 64  # You can adjust the number of feature channels

        # A simple feature extractor to “lift” the LR image into a feature space.
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_ch, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            act,
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            act
        )

        # Define a dictionary for dropout modules.
        self.drop_out = {
            'nn.dropout': nn.Dropout2d,
            'dropblock': DropBlock2D
        }

        # Create a ModuleList for the skip–connection extractors.
        # For each upsampling level, we “simulate” a skip connection by upsampling the input image
        # to the corresponding scale and processing it with a convolution.
        self.skip_extractors = nn.ModuleList()
        self.attn_gates = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(num_up_blocks):
            # For block i, the skip connection is built from the input image upsampled by 2^(i+1)
            self.skip_extractors.append(
                nn.Conv2d(in_ch, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
            )
            # Attention gate: both inputs have base_channels channels.
            self.attn_gates.append(
                AttentionGate(in_ch_x=base_channels, in_ch_g=base_channels, out_ch=base_channels, concat=True)
            )
            # Create dropout module (if dropout_p > 0, otherwise use Identity)
            if dropout_p > 0:
                dropout_module = self.drop_out[dropout_name](dropout_p)
            else:
                dropout_module = nn.Identity()
            # UpBlock: It will upsample the gating signal and combine it with the skip connection.
            self.up_blocks.append(
                UpBlock(in_ch_x=base_channels, in_ch_g=base_channels, out_ch=base_channels, 
                        dropout_m=dropout_module, dropout_p=dropout_p, act=act)
            )

        # Final head: a convolution that maps the final feature map to the desired output channels.
        self.head = nn.Conv2d(base_channels, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x,target_hw):
        """
        Args:
            x: low resolution input image of shape (B, in_ch, H, W)
        Returns:
            out: high resolution output image of shape (B, out_ch, upscale_factor*H, upscale_factor*W)
        """
        # Extract base features from the low resolution input.
        feat = self.feature_extractor(x)  # shape: (B, base_channels, H, W)
        current_feat = feat

        # For each upsampling block, build the skip connection from the input and fuse with the current features.
        for i in range(self.num_up_blocks):
            scale = 2 ** (i + 1)  # e.g., 2, 4, 8, ... (here upscale_factor=4 gives scales 2 and 4)
            # Upsample the input image to the corresponding resolution.
            skip = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=True)
            # Process the upsampled input with the corresponding skip extractor.
            skip = self.skip_extractors[i](skip)
            # Apply the attention gate: the skip connection (high–res) is weighted by the current (low–res) features.
            attn = self.attn_gates[i](skip, current_feat)
            # Use the UpBlock to upsample the current features and fuse with the attention–weighted skip connection.
            current_feat = self.up_blocks[i](attn, current_feat)
        
        # The head produces the final high resolution output.
        out = self.head(current_feat)
        out = F.interpolate(out,size=target_hw,align_corners=False,mode='bicubic')
        return {'x_recon':out}

#############################################
# Testing the SR Model
#############################################

if __name__ == '__main__':
    # Example: create a 4× Super–Resolution model.
    # For instance, an LR image of size 64×64 will be upscaled to 256×256.
    model = SRModelV1(in_ch=1, out_ch=1, upscale_factor=4, dropout_name='dropblock', dropout_p=0.1).cuda()
    # Create a dummy low resolution input (e.g., 64×64)
    lr_inp = torch.randn((2, 1, 64, 64)).cuda()
    sr_out = model(lr_inp,lr_inp.shape[-2:])['x_recon']
    print(f"Input shape: {lr_inp.shape}")
    print(f"Output shape: {sr_out.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
