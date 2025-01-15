import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Inception Block
# Inception Block with dynamic input channels
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        # 각 branch에 맞는 채널 수 설정
        self.branch1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=1),
            nn.Conv2d(48, 64, kernel_size=5, padding=2)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.Conv2d(96, 96, kernel_size=3, padding=1)
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 32, kernel_size=1)
        )
        
        # 채널 정렬을 위해 마지막 출력 채널 수 변경
        self.out_channels = 64 + 64 + 96 + 32

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)

# Global Attention (Transformer Block)
class GlobalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(GlobalAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

# Local Attention (CBAM)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=25, padding=12, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        return x

# U-Net with Inception and Attention Mechanisms
# U-Net with Inception and Attention Mechanisms
class Inception_A_Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Inception_A_Unet, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.encoder2 = InceptionA(32)  # 입력 채널 32
        self.encoder3 = InceptionA(self.encoder2.out_channels)  # 입력 채널은 encoder2의 출력 채널 수
        self.encoder4 = InceptionA(self.encoder3.out_channels)  # 동일한 패턴으로 채널 조정

        self.global_attention = GlobalAttention(self.encoder4.out_channels)

        self.decoder1 = InceptionA(self.encoder4.out_channels)
        self.decoder2 = InceptionA(self.decoder1.out_channels)
        self.decoder3 = InceptionA(self.decoder2.out_channels)
        self.decoder4 = InceptionA(self.decoder3.out_channels)

        # Add Conv layers to match the number of channels for skip connections
        self.conv1x1_e1 = nn.Conv2d(32, self.decoder3.out_channels, kernel_size=1)

        self.local_attention1 = CBAM(self.encoder4.out_channels)
        self.local_attention2 = CBAM(self.decoder1.out_channels)

        self.conv_final = nn.Conv2d(self.decoder4.out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Global Attention
        global_att = rearrange(e4, 'b c h w -> b (h w) c')
        global_att = self.global_attention(global_att)
        global_att = rearrange(global_att, 'b (h w) c -> b c h w', h=e4.shape[2])

        # Decoding with Local Attention
        d1 = self.decoder1(global_att)
        d1 = self.local_attention1(d1 + e3)
        d2 = self.decoder2(d1)
        d2 = self.local_attention2(d2 + e2)
        d3 = self.decoder3(d2)

        # Apply 1x1 conv to match channels before adding e1
        e1_resized = self.conv1x1_e1(e1)
        d4 = self.decoder4(d3 + e1_resized)

        # Final output
        out = self.conv_final(d4)
        return out



# Example usage
model = Inception_A_Unet(in_channels=3)
x = torch.randn(1,3, 256, 256)  # Example input
output = model(x)
print(output.shape)
