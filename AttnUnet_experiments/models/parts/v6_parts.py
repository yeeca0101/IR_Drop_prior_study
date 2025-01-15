'''
1. double conv -> resnet style
2. act relu -> choice
3. Maxpool
4. 
'''
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class SwishT(nn.Module):
    def __init__(self, beta_init=1.0, alpha=0.1,requires_grad=True):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  
        self.alpha = alpha  # Could also be made learnable if desired

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x) + self.alpha * torch.tanh(x)
    
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

class PreConv(nn.Module):
    def __init__(self, in_ch,out_ch,stride=1,padding=1,act=nn.ReLU()) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=stride,padding=padding,bias=False)
        self.bn = nn.GroupNorm(1, out_ch)
        self.act = act

    def forward(self,x):
        x = self.conv(x)
        x = self.act(self.bn(x))

        return x
    

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_m, dropout_p, act = nn.ReLU()) -> None:
        super().__init__()
        self.dropout = dropout_m(p=dropout_p)
        self.act = act

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, out_ch),  # Changed from LayerNorm
            self.act,
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, out_ch),  # Changed from LayerNorm
            self.dropout
        )

        self.residual_conv = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv_block(x)
        out += residual  
        return self.act(out)  


class DownBlock(nn.Module):
    def __init__(self, in_ch,out_ch,dropout_m,dropout_p,act=nn.ReLU()) -> None:
        super().__init__()
        self.conv = DoubleConv(in_ch,out_ch,dropout_m,dropout_p,act=act)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        x1 = self.conv(x)
        x2 = self.pool(x1)
        return x2

class AttentionGate(nn.Module):
    def __init__(self,in_ch_x,in_ch_g,out_ch,concat=True) -> None:
        super().__init__()
        self.concat = concat
        self.act = nn.ReLU(inplace=True)
        if concat:
            self.w_x_g = nn.Conv2d(in_ch_x+in_ch_g,out_ch,kernel_size=1,stride=1,padding=0,bias=False)
        else:
            self.w_x = nn.Conv2d(in_ch_x,out_ch,kernel_size=1,stride=1,padding=0,bias=False)
            self.w_g = nn.Conv2d(in_ch_g,out_ch,kernel_size=1,stride=1,padding=0,bias=False)
        
        self.attn = nn.Conv2d(out_ch,out_ch,kernel_size=1,padding=0,bias=False)

    def forward(self,x,g):
        res = x
        if self.concat:
            xg = torch.cat([x,g],dim=1) # B (x_c + g_c) H W
            xg = self.w_x_g(xg)
        else:
            xg = self.w_x(x) + self.w_g(g)
        
        xg = self.act(xg)
        attn = torch.sigmoid(self.attn(xg))

        out = res*attn
        return out

class UpBlock(nn.Module):
    def __init__(self,in_ch_x,in_ch_g,out_ch,dropout_m,dropout_p,act=nn.ReLU()):
        super(UpBlock,self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        # self.conv1 = nn.Conv2d(in_ch_x+in_ch_g,out_ch,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv = DoubleConv(in_ch_x+in_ch_g,out_ch,dropout_m,dropout_p,act=act)

    def forward(self,attn,x):
        x = torch.cat([attn,x],dim=1)
        x = self.up(x)
        x = self.conv(x)
    
        return x
    

if __name__ == '__main__':
    # m = AttnUnetV2(12,1,dropout_name='dropblock').to('cuda:0')
    # inp = torch.randn((1,12,512,512)).to('cuda:0')
    # print(m(inp).shape)
    # total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    # print(f"Total trainable parameters: {total_params:,}")
    pass
