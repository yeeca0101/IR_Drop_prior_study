import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# 기본 ResMLP 구성요소
# ----------------------------

# No norm layer
class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        return self.alpha * x + self.beta

# MLP on channels
class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# ResMLP block: 선형 패치 간 연산 + 채널별 MLP 연산  
class ResMLP_BLocks(nn.Module):
    def __init__(self, nb_patches, dim, layerscale_init):
        """
        nb_patches: 토큰(패치) 수 (예, H*W)
        dim: 채널 수
        layerscale_init: LayerScale 초기값
        """
        super().__init__()
        self.affine_1 = Affine(dim)
        self.affine_2 = Affine(dim)
        # 패치 차원(linear layer를 통해 토큰 간 상호작용)
        self.linear_patches = nn.Linear(nb_patches, nb_patches)  
        # 채널 MLP
        self.mlp_channels = Mlp(dim)
        self.layerscale_1 = nn.Parameter(layerscale_init * torch.ones((dim)))
        self.layerscale_2 = nn.Parameter(layerscale_init * torch.ones((dim)))
        
    def forward(self, x):
        # x: (B, nb_patches, dim)
        res_1 = self.linear_patches(self.affine_1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.layerscale_1 * res_1
        res_2 = self.mlp_channels(self.affine_2(x))
        x = x + self.layerscale_2 * res_2
        return x

# 2D 입력(특징 맵)을 ResMLP 블록에 적용하기 위한 래퍼  
class ResMLPBlock2D(nn.Module):
    def __init__(self, dim, H, W, layerscale_init):
        """
        dim: 채널 수  
        H, W: 입력 특징 맵의 높이와 너비  
        layerscale_init: LayerScale 초기값
        """
        super().__init__()
        nb_patches = H * W
        self.block = ResMLP_BLocks(nb_patches, dim, layerscale_init)
        self.H = H
        self.W = W
        self.dim = dim
        
    def forward(self, x):
        # x: (B, dim, H, W)
        B, C, H, W = x.shape
        # (B, H*W, dim)
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.block(x_flat)
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        return x

# ----------------------------
# ResMLP 기반 UNet (Segmentation)
# ----------------------------
class ResMLP_UNet(nn.Module):
    def __init__(self, in_ch=3, base_dim=64, layerscale_init=1e-5, num_classes=1):
        """
        in_ch: 입력 채널 수  
        base_dim: 첫 단계의 기본 채널 수  
        layerscale_init: ResMLP의 layerscale 초기값  
        num_classes: 최종 출력 채널(클래스) 수
        """
        super().__init__()
        # Encoder
        # Level 1: 256x256, 채널: base_dim
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,base_dim),
            nn.ReLU(inplace=True),
            # 주의: 256×256 해상도에서의 토큰 수가 매우 크므로, 실제 모델에서는 사용량을 고려하세요.
            ResMLPBlock2D(base_dim, H=256, W=256, layerscale_init=layerscale_init)
        )
        self.down1 = nn.Conv2d(base_dim, base_dim, kernel_size=3, stride=2, padding=1)  # 256 -> 128
        
        # Level 2: 128x128, 채널: base_dim*2
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim*2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,base_dim*2),
            nn.ReLU(inplace=True),
            ResMLPBlock2D(base_dim*2, H=128, W=128, layerscale_init=layerscale_init)
        )
        self.down2 = nn.Conv2d(base_dim*2, base_dim*2, kernel_size=3, stride=2, padding=1)  # 128 -> 64
        
        # Level 3: 64x64, 채널: base_dim*4
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_dim*2, base_dim*4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,base_dim*4),
            nn.ReLU(inplace=True),
            ResMLPBlock2D(base_dim*4, H=64, W=64, layerscale_init=layerscale_init)
        )
        self.down3 = nn.Conv2d(base_dim*4, base_dim*4, kernel_size=3, stride=2, padding=1)  # 64 -> 32
        
        # Level 4: 32x32, 채널: base_dim*8
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_dim*4, base_dim*8, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,base_dim*8),
            nn.ReLU(inplace=True),
            ResMLPBlock2D(base_dim*8, H=32, W=32, layerscale_init=layerscale_init)
        )
        self.down4 = nn.Conv2d(base_dim*8, base_dim*8, kernel_size=3, stride=2, padding=1)  # 32 -> 16
        
        # Bottleneck: 16x16, 채널: base_dim*16
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_dim*8, base_dim*16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,base_dim*16),
            nn.ReLU(inplace=True),
            ResMLPBlock2D(base_dim*16, H=16, W=16, layerscale_init=layerscale_init)
        )
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_dim*16, base_dim*8, kernel_size=2, stride=2)  # 16->32
        self.dec4 = nn.Sequential(
            nn.Conv2d(base_dim*16, base_dim*8, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,base_dim*8),
            nn.ReLU(inplace=True),
            ResMLPBlock2D(base_dim*8, H=32, W=32, layerscale_init=layerscale_init)
        )
        
        self.up3 = nn.ConvTranspose2d(base_dim*8, base_dim*4, kernel_size=2, stride=2)  # 32->64
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_dim*8, base_dim*4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,base_dim*4),
            nn.ReLU(inplace=True),
            ResMLPBlock2D(base_dim*4, H=64, W=64, layerscale_init=layerscale_init)
        )
        
        self.up2 = nn.ConvTranspose2d(base_dim*4, base_dim*2, kernel_size=2, stride=2)  # 64->128
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_dim*4, base_dim*2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,base_dim*2),
            nn.ReLU(inplace=True),
            ResMLPBlock2D(base_dim*2, H=128, W=128, layerscale_init=layerscale_init)
        )
        
        self.up1 = nn.ConvTranspose2d(base_dim*2, base_dim, kernel_size=2, stride=2)  # 128->256
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_dim*2, base_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,base_dim),
            nn.ReLU(inplace=True),
            ResMLPBlock2D(base_dim, H=256, W=256, layerscale_init=layerscale_init)
        )
        
        self.out_conv = nn.Conv2d(base_dim, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)           # (B, base_dim, 256, 256)
        down1 = self.down1(enc1)       # (B, base_dim, 128, 128)
        
        enc2 = self.enc2(down1)        # (B, base_dim*2, 128, 128)
        down2 = self.down2(enc2)       # (B, base_dim*2, 64, 64)
        
        enc3 = self.enc3(down2)        # (B, base_dim*4, 64, 64)
        down3 = self.down3(enc3)       # (B, base_dim*4, 32, 32)
        
        enc4 = self.enc4(down3)        # (B, base_dim*8, 32, 32)
        down4 = self.down4(enc4)       # (B, base_dim*8, 16, 16)
        
        bottleneck = self.bottleneck(down4)  # (B, base_dim*16, 16, 16)
        
        # Decoder
        up4 = self.up4(bottleneck)     # (B, base_dim*8, 32, 32)
        cat4 = torch.cat([up4, enc4], dim=1)  # (B, base_dim*16, 32, 32)
        dec4 = self.dec4(cat4)         # (B, base_dim*8, 32, 32)
        
        up3 = self.up3(dec4)           # (B, base_dim*4, 64, 64)
        cat3 = torch.cat([up3, enc3], dim=1)  # (B, base_dim*8, 64, 64)
        dec3 = self.dec3(cat3)         # (B, base_dim*4, 64, 64)
        
        up2 = self.up2(dec3)           # (B, base_dim*2, 128, 128)
        cat2 = torch.cat([up2, enc2], dim=1)  # (B, base_dim*4, 128, 128)
        dec2 = self.dec2(cat2)         # (B, base_dim*2, 128, 128)
        
        up1 = self.up1(dec2)           # (B, base_dim, 256, 256)
        cat1 = torch.cat([up1, enc1], dim=1)  # (B, base_dim*2, 256, 256)
        dec1 = self.dec1(cat1)         # (B, base_dim, 256, 256)
        
        out = self.out_conv(dec1)      # (B, num_classes, 256, 256)
        return out

# ----------------------------
# 테스트 코드
# ----------------------------
if __name__ == '__main__':
    # 임의 입력 텐서 (배치 크기 2, 채널 3, 256×256)
    B, C, H, W = 2, 25, 256, 256
    dummy_input = torch.randn(B, C, H, W)
    
    # 모델 생성
    model = ResMLP_UNet(in_ch=25, base_dim=32, layerscale_init=1e-5, num_classes=1)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 예상 출력: (2, 2, 256, 256)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
