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
    def __init__(self, beta_init=1.5, alpha=0.1,requires_grad=False):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  
        self.alpha = alpha  # Could also be made learnable if desired

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x) + self.alpha * torch.tanh(x)

class SwishT_B(nn.Module):
    def __init__(self, beta_init=1.5, alpha=0.1,requires_grad=False):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  
        self.alpha = alpha  

    def forward(self, x):
        return torch.sigmoid(self.beta*x)*(x+2*self.alpha)-self.alpha

class SwishT_C(nn.Module):
    def __init__(self, beta_init=1.5, alpha=0.1,requires_grad=False):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]),requires_grad=requires_grad)  
        self.alpha = alpha  

    def forward(self, x):
        return torch.sigmoid(self.beta*x)*(x+2*self.alpha/self.beta)-self.alpha/self.beta
    
    
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
    def __init__(self, in_ch,out_ch,stride=1,padding=1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=stride,padding=padding,bias=False)
        self.bn = nn.GroupNorm(1, out_ch)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)

        return x

class ResNeXtBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cardinality=8, dropout_m=nn.Dropout2d, dropout_p=0.0, act=SwishT_C()):
        super().__init__()
        # mid_ch를 out_ch의 절반으로 계산한 후, cardinality로 나누어 떨어지도록 조정
        mid_ch = (out_ch // 2 // cardinality) * cardinality
        if mid_ch == 0:
            mid_ch = cardinality

        # 1x1 컨볼루션: 차원 축소
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(1, mid_ch)
        
        self.act = act
        
        # 3x3 컨볼루션: 그룹 컨볼루션 (mid_ch가 cardinality로 나누어 떨어짐)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, groups=cardinality, bias=False)
        self.gn2 = nn.GroupNorm(1, mid_ch)
        
        # 1x1 컨볼루션: 차원 복원
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.gn3 = nn.GroupNorm(1, out_ch)
        
        # dropout (dropout_p > 0인 경우)
        self.dropout = dropout_m(p=dropout_p) if dropout_p > 0 else nn.Identity()
        
        # 입력과 출력의 채널 수가 다르면 shortcut 경로에 1x1 컨볼루션 적용
        self.residual_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.residual_conv(x)
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.act(out)
        
        out = self.conv3(out)
        out = self.gn3(out)
        out = self.dropout(out)
        
        out += residual
        out = self.act(out)
        
        return out

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_m, dropout_p, act=SwishT_C(), cardinality=8):
        super().__init__()
        # 기존 ResNeXtBlock 대신 ResNeXtBlock 사용
        self.conv = ResNeXtBlock(in_ch, out_ch, cardinality=cardinality, dropout_m=dropout_m, dropout_p=dropout_p, act=act)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
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
    def __init__(self, in_ch_x, in_ch_g, out_ch, dropout_m, dropout_p, act=SwishT_C(), cardinality=8):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        # UpBlock에서는 입력 채널은 in_ch_x + in_ch_g입니다.
        self.conv = ResNeXtBlock(in_ch_x + in_ch_g, out_ch, cardinality=cardinality, dropout_m=dropout_m, dropout_p=dropout_p, act=act)

    def forward(self, attn, x):
        x = torch.cat([attn, x], dim=1)
        x = self.up(x)
        x = self.conv(x)
        return x
    

if __name__ == '__main__':
    # 예시: 입력 채널 12, 출력 채널 24, dropout 확률 0.1
    block = DownBlock(in_ch=12, out_ch=24, dropout_m=nn.Dropout2d, dropout_p=0.1, act=SwishT_C(), cardinality=8)
    x = torch.randn((1, 12, 64, 64))
    y = block(x)
    print("Output shape:", y.shape)
