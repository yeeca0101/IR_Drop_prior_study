import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    간단한 convolution block: Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, norm_layer=nn.BatchNorm2d, activation=nn.ReLU):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            norm_layer(out_channels),
            activation(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class IRDropHead(nn.Module):
    """
    IR Drop Prediction Head for image-to-image prediction.
    
    백본의 multi-scale feature maps (예: [f1, f2, f3, f4], 
    f1은 가장 높은 해상도, f4는 가장 낮은 해상도)를 입력받아,
    ConvTranspose2d를 이용해 단계별 upsample하고, 
    각 단계에서 skip connection을 통해 더 높은 해상도의 feature와 결합합니다.
    최종적으로 ConvBlock을 거쳐 원본 해상도의 예측값을 출력합니다.
    
    Args:
        in_channels_list (list[int]): 백본 각 단계의 채널 수. 예) [C1, C2, C3, C4].
        mid_channels (int): 각 upsample 후 ConvBlock에서 사용할 중간 채널 수.
        out_channels (int): 최종 upsample 후의 채널 수 (예측 채널 수, regression이면 보통 1).
    """
    def __init__(self, in_channels_list, mid_channels, out_channels):
        super(IRDropHead, self).__init__()
        # in_channels_list: [f1, f2, f3, f4] (f1 highest resolution)
        # f4 -> f3: upsample f4 to f3 resolution
        self.up1 = nn.ConvTranspose2d(in_channels_list[3], mid_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(mid_channels + in_channels_list[2], mid_channels)
        
        # f3 -> f2: upsample to f2 resolution
        self.up2 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2)
        self.conv2 = ConvBlock(mid_channels + in_channels_list[1], mid_channels)
        
        # f2 -> f1: upsample to f1 resolution
        self.up3 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2)
        self.conv3 = ConvBlock(mid_channels + in_channels_list[0], mid_channels)
        
        # f1 resolution is usually 1/4 of the input.
        # To reach original resolution, upsample by factor of 4.
        self.up4 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=4, stride=4, padding=0)
        self.final_conv = nn.Sequential(
            ConvBlock(mid_channels, mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        )
        
    def forward(self, features):
        # features: [f1, f2, f3, f4] (f1 highest resolution)
        f1, f2, f3, f4 = features
        # Stage 1: f4 -> upsample to f3 resolution, concatenate with f3
        x = self.up1(f4)
        x = torch.cat([x, f3], dim=1)
        x = self.conv1(x)
        # Stage 2: upsample to f2 resolution, concat with f2
        x = self.up2(x)
        x = torch.cat([x, f2], dim=1)
        x = self.conv2(x)
        # Stage 3: upsample to f1 resolution, concat with f1
        x = self.up3(x)
        x = torch.cat([x, f1], dim=1)
        x = self.conv3(x)
        # Stage 4: upsample to original input resolution
        x = self.up4(x)
        x = self.final_conv(x)
        
        outputs = {}
        outputs['dictionary_loss'] = 0.
        outputs['commitment_loss'] = 0.
        outputs['x_recon'] = x

        return outputs

def build_head(cfg, backbone_channels):
    """
    cfg에 따라 head 모듈 생성.
    
    현재 cfg["type"]이 "IRDropHead"이면, 위에서 정의한 IRDropHead를 생성합니다.
    
    Args:
        cfg (dict): head 구성 딕셔너리. 예)
            {
              "type": "IRDropHead",
              "mid_channels": 256,
              "out_channels": 1,   # 예: regression이면 1, segmentation이면 num_classes 등
            }
        backbone_channels (list[int]): 백본 각 단계의 채널 수, 예) [C1, C2, C3, C4].
        
    Returns:
        nn.Module: 구성된 head 모듈.
    """
    head_type = cfg.get("type", "FPNHead")
    if head_type == "IRDropHead":
        mid_channels = cfg.get("mid_channels", 256)
        out_channels = cfg.get("out_channels", 1)
        return IRDropHead(in_channels_list=backbone_channels, mid_channels=mid_channels, out_channels=out_channels)
    elif head_type == "FPNHead":
        # 기존 FPNHead (세그멘테이션 등) 구현 – 필요 시 유지
        out_channels = cfg.get("out_channels", 256)
        num_classes = cfg.get("num_classes", 21)
        dropout = cfg.get("dropout", 0.1)
        return FPNHead(in_channels_list=backbone_channels, out_channels=out_channels,
                       num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unsupported head type: {head_type}")
        
# 기존 FPNHead 예제 (필요에 따라 유지)
class FPNHead(nn.Module):
    def __init__(self, in_channels_list, out_channels, num_classes, dropout=0.1):
        super(FPNHead, self).__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(out_channels, num_classes, kernel_size=1)
    def forward(self, features):
        lateral_feats = [l_conv(f) for f, l_conv in zip(features, self.lateral_convs)]
        out_feats = [None] * len(lateral_feats)
        out_feats[-1] = lateral_feats[-1]
        for i in range(len(lateral_feats) - 2, -1, -1):
            size = lateral_feats[i].shape[-2:]
            upsampled = F.interpolate(out_feats[i + 1], size=size, mode='bilinear', align_corners=False)
            out_feats[i] = lateral_feats[i] + upsampled
        out_feats = [fpn_conv(f) for fpn_conv, f in zip(self.fpn_convs, out_feats)]
        target_size = out_feats[0].shape[-2:]
        fusion = out_feats[0]
        for f in out_feats[1:]:
            fusion = fusion + F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
        fusion = self.dropout(fusion)
        seg_out = self.classifier(fusion)

        return seg_out
