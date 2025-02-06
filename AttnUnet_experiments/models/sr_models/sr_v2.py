import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#############################################
# Channel Attention Layer (from RCAN)
#############################################
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        Args:
            channel (int): 입력 채널 수.
            reduction (int): 채널 축소 비율.
        """
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

#############################################
# Residual Dense Block with Channel Attention (RDB)
#############################################
class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=5):
        """
        Args:
            in_channels (int): 입력 특징 수.
            growth_rate (int): 각 레이어마다 증가하는 특징 수.
            num_layers (int): RDB 내의 컨볼루션 계층 수.
        """
        super(ResidualDenseBlock, self).__init__()
        self.num_layers = num_layers
        modules = []
        curr_channels = in_channels
        for i in range(num_layers):
            modules.append(
                nn.Conv2d(curr_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=True)
            )
            curr_channels += growth_rate
        self.convs = nn.ModuleList(modules)
        # Local feature fusion: 1x1 conv to fuse concatenated features back to in_channels.
        self.lff = nn.Conv2d(in_channels + num_layers * growth_rate, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        # 채널 어텐션 적용
        self.ca = CALayer(in_channels, reduction=16)
        self.res_scale = 0.2

    def forward(self, x):
        inputs = x
        concat_features = x
        for conv in self.convs:
            out = F.relu(conv(concat_features), inplace=True)
            concat_features = torch.cat([concat_features, out], dim=1)
        fused = self.lff(concat_features)
        fused = self.ca(fused)
        return inputs + fused * self.res_scale

#############################################
# Residual in Residual Dense Block (RRDB)
#############################################
class RRDB(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=5):
        """
        3개의 RDB를 중첩하여 깊은 특성 학습을 수행합니다.
        """
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(in_channels, growth_rate, num_layers)
        self.rdb2 = ResidualDenseBlock(in_channels, growth_rate, num_layers)
        self.rdb3 = ResidualDenseBlock(in_channels, growth_rate, num_layers)
        self.res_scale = 0.2

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * self.res_scale

#############################################
# Advanced Super-Resolution Model (SRModelV2)
#############################################
class SRModelV2(nn.Module):
    """
    SRModelV2는 최신 super–resolution 기법(Residual in Residual Dense Block + Channel Attention)을
    적용한 모델입니다.
    
    Args:
        in_ch (int): 입력 채널 수.
        out_ch (int): 출력 채널 수.
        upscale_factor (int): 대략적인 업스케일 배율 (PixelShuffle 기반으로 내부에서 사용됨).
                              최종 해상도는 forward() 시 target_hw에 맞게 보간됨.
        num_features (int): 내부 특징 차원 (보통 64 또는 128).
        num_rrdb (int): RRDB 블록의 개수.
        growth_rate (int): RDB 내에서의 growth rate.
    """
    def __init__(self, in_ch=1, out_ch=1, upscale_factor=4, num_features=64, num_rrdb=8, growth_rate=32):
        super(SRModelV2, self).__init__()
        self.upscale_factor = upscale_factor

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_ch, num_features, kernel_size=3, stride=1, padding=1, bias=True)

        # Deep feature extraction: 여러 개의 RRDB 블록
        rrdb_blocks = []
        for _ in range(num_rrdb):
            rrdb_blocks.append(RRDB(num_features, growth_rate, num_layers=5))
        self.RRDB_trunk = nn.Sequential(*rrdb_blocks)
        
        # Trunk conv: 깊은 특징과 shallow 특징의 잔차 결합을 위한 conv.
        self.trunk_conv = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=True)
        
        # Upsampling via PixelShuffle
        # (upscale_factor가 반드시 2의 거듭제곱이 아닐 경우, 최종 보간으로 target_hw에 맞춤)
        self.upconv = nn.Sequential(
            nn.Conv2d(num_features, num_features * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=True)
        )
        
        # Reconstruction head: 원하는 출력 채널 수로 매핑
        self.conv_last = nn.Conv2d(num_features, out_ch, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, target_hw):
        """
        Args:
            x (Tensor): low resolution input, shape (B, in_ch, H, W)
            target_hw (tuple): 최종 출력 해상도 (height, width)
        Returns:
            dict: {'x_recon': high resolution output tensor of shape (B, out_ch, target_hw[0], target_hw[1])}
        """
        # Shallow feature extraction
        shallow_feat = self.conv_first(x)
        
        # Deep feature extraction
        trunk = self.RRDB_trunk(shallow_feat)
        trunk = self.trunk_conv(trunk)
        feat = shallow_feat + trunk  # Global residual connection
        
        # Upsampling
        up_feat = self.upconv(feat)
        out = self.conv_last(up_feat)
        
        # 최종 해상도에 맞게 보간 (target_hw가 upscale_factor와 정확히 일치하지 않을 경우 대응)
        out = F.interpolate(out, size=target_hw, mode='bicubic', align_corners=False)
        return {'x_recon': out}

#############################################
# Testing the Advanced SR Model (SRModelV2)
#############################################
if __name__ == '__main__':
    # 예시: 입력 채널 1, 출력 채널 1, 업스케일 배율 4 (내부적으로 PixelShuffle 사용)
    # 실제 최종 해상도는 forward()의 target_hw에 의해 결정됨.
    model = SRModelV2(in_ch=1, out_ch=1, upscale_factor=4, num_features=64, num_rrdb=8, growth_rate=32).cuda()
    
    # Dummy low resolution 입력 (예: 64×64)
    lr_inp = torch.randn((2, 1, 64, 64)).cuda()
    
    # 최종 target 해상도: 예를 들어 256×256 (또는 원하는 해상도)
    target_hw = (256, 256)
    
    # Forward pass
    sr_out = model(lr_inp, target_hw)['x_recon']
    
    print(f"Input shape: {lr_inp.shape}")
    print(f"Output shape: {sr_out.shape}")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
