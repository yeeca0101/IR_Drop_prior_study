'''
add Dropout, DropBlock, ...
    DropBlock : 
'''
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


 

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

class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=int(nr_steps))

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.head_dim = in_channels // num_heads
        
        # Q, K, V projections for multi-head attention
        self.query = nn.Conv2d(in_channels, in_channels, 1,groups=num_heads,bias=False)
        self.key = nn.Conv2d(in_channels, in_channels, 1,groups=num_heads,bias=False)
        self.value = nn.Conv2d(in_channels, in_channels, 1,groups=num_heads,bias=False)
        
        # Output projection
        self.proj = nn.Conv2d(in_channels, in_channels, 1,bias=False)
        
        # Normalization and residual
        self.norm = nn.GroupNorm(1, in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def reshape_to_heads(self,x,B,H,W):
        return x.view(B, self.num_heads, self.head_dim, H*W)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate Q, K, V
        q = self.query(x)  # B, C, H, W
        k = self.key(x)    # B, C, H, W
        v = self.value(x)  # B, C, H, W
        
        # Reshape to multi-head format
        
        q = self.reshape_to_heads(q,B,H,W)  # B, num_heads, head_dim, HW
        k = self.reshape_to_heads(k,B,H,W)  # B, num_heads, head_dim, HW
        v = self.reshape_to_heads(v,B,H,W)  # B, num_heads, head_dim, HW
        
        # Compute attention scores for each head
        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)  # B, num_heads, head_dim, head_dim
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)  # B, num_heads, head_dim, HW
        
        # Reshape back to original format
        out = out.view(B, C, H, W)
        
        # Output projection
        out = self.proj(out)
        
        # Residual connection with learnable weight
        out = x + self.gamma * out
        
        # Normalization
        out = self.norm(out)
        
        return out

class PreConvWithChannelAttention(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, padding=0, num_heads=4) -> None:
        super().__init__()
        # Channel attention before any spatial operations
        self.channel_attn = ChannelAttention(in_ch, num_heads=num_heads)
        
        # Original PreConv layers
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, 
                            kernel_size=2, stride=stride, padding=padding, bias=False)
        self.bn = nn.GroupNorm(1, out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # Apply channel attention first
        x = self.channel_attn(x)
        
        # Then apply spatial operations
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class PreConv(nn.Module):
    def __init__(self, in_ch,out_ch,stride=2,padding=0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=2,stride=stride,padding=padding,bias=False)
        self.bn = nn.GroupNorm(1, out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv(x)
        x = self.act(self.bn(x))

        return x
    

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_m, dropout_p) -> None:
        super().__init__()
        self.dropout = dropout_m(p=dropout_p)

        self.dconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, out_ch),  # Changed from LayerNor
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, out_ch),  # Changed from LayerNor
            nn.ReLU(inplace=True),
            self.dropout
        )

    def forward(self, x):
        out = self.dconv(x)
        return out

class DownBlock(nn.Module):
    def __init__(self, in_ch,out_ch,dropout_m,dropout_p) -> None:
        super().__init__()
        self.conv = DoubleConv(in_ch,out_ch,dropout_m,dropout_p)
        self.pool = nn.AvgPool2d(kernel_size=2,stride=2)

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
    def __init__(self,in_ch_x,in_ch_g,out_ch,dropout_m,dropout_p):
        super(UpBlock,self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        # self.conv1 = nn.Conv2d(in_ch_x+in_ch_g,out_ch,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv = DoubleConv(in_ch_x+in_ch_g,out_ch,dropout_m,dropout_p)

    def forward(self,attn,x):
        x = torch.cat([attn,x],dim=1)
        x = self.up(x)
        x = self.conv(x)
    
        return x
    

class AttnUnetV2(nn.Module):
    def __init__(self,in_ch=12,out_ch=1,dropout_name='',dropout_p=0.5) -> None:
        super().__init__()

        in_channels = [12,32,64,128,256,512]
        self.concat = True
        self.drop_out = {
            'nn.dropout': nn.Dropout2d,
            'dropblock': DropBlock2D
        }

        self.preconv = PreConvWithChannelAttention(in_ch, out_ch=in_channels[0])
        self.d1 = DownBlock(in_ch=in_channels[0],out_ch=in_channels[1],dropout_m=self.drop_out[dropout_name],dropout_p=dropout_p)
        self.d2 = DownBlock(in_ch=in_channels[1],out_ch=in_channels[2],dropout_m=self.drop_out[dropout_name],dropout_p=dropout_p)
        self.d3 = DownBlock(in_ch=in_channels[2],out_ch=in_channels[3],dropout_m=self.drop_out[dropout_name],dropout_p=dropout_p)
        self.d4 = DownBlock(in_ch=in_channels[3],out_ch=in_channels[4],dropout_m=self.drop_out[dropout_name],dropout_p=dropout_p)

        self.bottleneck = DoubleConv(in_ch=in_channels[4],out_ch=in_channels[5],dropout_m=self.drop_out[dropout_name],dropout_p=dropout_p)

        self.attn1 = AttentionGate(in_ch_x=in_channels[4],in_ch_g=in_channels[5],out_ch=in_channels[4],concat=self.concat)
        self.attn2 = AttentionGate(in_ch_x=in_channels[3],in_ch_g=in_channels[4],out_ch=in_channels[3],concat=self.concat)
        self.attn3 = AttentionGate(in_ch_x=in_channels[2],in_ch_g=in_channels[3],out_ch=in_channels[2],concat=self.concat)
        self.attn4 = AttentionGate(in_ch_x=in_channels[1],in_ch_g=in_channels[2],out_ch=in_channels[1],concat=self.concat)

        self.up1 = UpBlock(in_ch_x=in_channels[4],in_ch_g=in_channels[5],out_ch=in_channels[4],dropout_m=self.drop_out[dropout_name],dropout_p=dropout_p)
        self.up2 = UpBlock(in_ch_x=in_channels[3],in_ch_g=in_channels[4],out_ch=in_channels[3],dropout_m=self.drop_out[dropout_name],dropout_p=dropout_p)
        self.up3 = UpBlock(in_ch_x=in_channels[2],in_ch_g=in_channels[3],out_ch=in_channels[2],dropout_m=self.drop_out[dropout_name],dropout_p=dropout_p)
        self.up4 = UpBlock(in_ch_x=in_channels[1],in_ch_g=in_channels[2],out_ch=in_channels[1],dropout_m=self.drop_out[dropout_name],dropout_p=dropout_p)

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels[1],in_channels[1],kernel_size=1,stride=1,padding=0,bias=False),
            ASPP(in_channels[1],out_ch,1)
        )

    def forward(self,x):
        x = self.preconv(x)
        x1 = self.d1(x)    # B 32 128 128
        x2 = self.d2(x1)   # B 64 64 64
        x3 = self.d3(x2)   # B 128 32 32
        x4 = self.d4(x3)   # B 256 16 16 

        x5= self.bottleneck(x4) # g B 512 16 16 
        
        attn1 = self.attn1(x4,x5)   # B 256 16 16
        up1 = self.up1(attn1,x5)    # B 256 32 32

        attn2 = self.attn2(x3,up1)  # B 128 32 32
        up2 = self.up2(attn2,up1)   # B 128 64 64

        attn3 = self.attn3(x2,up2)  # B 64 128 128
        up3 = self.up3(attn3,up2)   # B 64 128 128

        attn4 = self.attn4(x1,up3)  # B 32 256 256
        up4 = self.up4(attn4,up3)   # B 32 256 256
        return self.head(up4) # B 1 512 512

if __name__ == '__main__':
    m = AttnUnetV2(12,1,dropout_name='dropblock').to('cuda:0')
    inp = torch.randn((1,12,512,512)).to('cuda:0')
    print(m(inp).shape)
    total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")