import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResNeXtBottleneck(nn.Module):
    """
    ResNeXt bottleneck block.
    기존 Residual Block에 grouped convolution을 적용하여 
    더 다양한 표현을 효율적으로 학습할 수 있도록 합니다.
    """
    def __init__(self, in_channels, out_channels, cardinality=32, bottleneck_width=4, stride=1):
        super(ResNeXtBottleneck, self).__init__()
        D = int(math.floor(out_channels * (bottleneck_width / 64)))
        C = cardinality
        mid_channels = D * C
        
        self.conv_reduce = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(mid_channels)
        
        self.conv_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, 
                                   padding=1, groups=C, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        
        self.conv_expand = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv_reduce(x)
        out = self.bn_reduce(out)
        out = self.relu(out)
        
        out = self.conv_conv(out)
        out = self.bn(out)
        out = self.relu(out)
        
        out = self.conv_expand(out)
        out = self.bn_expand(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class AdvancedIRDropHead(nn.Module):
    """
    최신 논문의 기법을 일부 반영하여, ResNeXt 기반 bottleneck block을 이용한
    multi-scale feature fusion head입니다.
    
    입력은 백본의 multi-scale feature map 리스트 [f1, f2, f3, f4]로,
    f1은 가장 높은 해상도입니다.
    
    각 단계에서는 ConvTranspose2d를 통해 upsample하고, skip connection으로 
    해당 resolution의 feature를 결합한 뒤 ResNeXt block으로 feature를 정제합니다.
    마지막에 추가 upsample과 1x1 conv를 통해 최종 예측을 산출합니다.
    
    Args:
        in_channels_list (list[int]): 백본의 각 단계의 채널 수. 예: [C1, C2, C3, C4].
        mid_channels (int): upsample 후와 ResNeXt block 내부에서 사용할 채널 수.
        out_channels (int): 최종 예측 채널 수 (regression이면 보통 1).
        cardinality (int): ResNeXt block에서 사용할 그룹 수.
        bottleneck_width (int): ResNeXt block의 bottleneck width.
    """
    def __init__(self, in_channels_list, mid_channels, out_channels, cardinality=32, bottleneck_width=4):
        super(AdvancedIRDropHead, self).__init__()
        
        # Stage 1: f4 -> f3 해상도로 upsample 후, f3과 concat
        self.up1 = nn.ConvTranspose2d(in_channels_list[3], mid_channels, kernel_size=2, stride=2)
        self.block1 = ResNeXtBottleneck(mid_channels + in_channels_list[2], mid_channels, 
                                        cardinality=cardinality, bottleneck_width=bottleneck_width)
        
        # Stage 2: f3 -> f2 해상도로 upsample 후, f2과 concat
        self.up2 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2)
        self.block2 = ResNeXtBottleneck(mid_channels + in_channels_list[1], mid_channels, 
                                        cardinality=cardinality, bottleneck_width=bottleneck_width)
        
        # Stage 3: f2 -> f1 해상도로 upsample 후, f1과 concat
        self.up3 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2)
        self.block3 = ResNeXtBottleneck(mid_channels + in_channels_list[0], mid_channels, 
                                        cardinality=cardinality, bottleneck_width=bottleneck_width)
        
        # Stage 4: f1 해상도에서 원본 해상도로 upsample
        # (예: f1이 입력의 1/4 해상도라면 factor 4 upsample)
        self.up4 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=4, stride=4, padding=0)
        self.final_block = ResNeXtBottleneck(mid_channels, mid_channels, 
                                             cardinality=cardinality, bottleneck_width=bottleneck_width)
        self.final_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
    
    def forward(self, features):
        # features: [f1, f2, f3, f4] (f1: highest resolution)
        f1, f2, f3, f4 = features
        
        # Stage 1
        x = self.up1(f4)
        x = torch.cat([x, f3], dim=1)
        x = self.block1(x)
        
        # Stage 2
        x = self.up2(x)
        x = torch.cat([x, f2], dim=1)
        x = self.block2(x)
        
        # Stage 3
        x = self.up3(x)
        x = torch.cat([x, f1], dim=1)
        x = self.block3(x)
        
        # Stage 4
        x = self.up4(x)
        x = self.final_block(x)
        x = self.final_conv(x)
        
        outputs = {
            'dictionary_loss': 0.,  # 추가적인 loss term이 필요한 경우 수정
            'commitment_loss': 0.,
            'x_recon': x
        }
        return outputs
#############################################################################
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

        cardinality = cfg.get("cardinality", 32)
        bottleneck_width = cfg.get("bottleneck_width", 4)
        return AdvancedIRDropHead(in_channels_list=backbone_channels,
                                  mid_channels=mid_channels,
                                  out_channels=out_channels,
                                  cardinality=cardinality,
                                  bottleneck_width=bottleneck_width)
        # return IRDropHead(in_channels_list=backbone_channels, mid_channels=mid_channels, out_channels=out_channels)
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
