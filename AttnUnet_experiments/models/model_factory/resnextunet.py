import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################
# 1. LayerNorm2d (직접 구현)
################################################################################
class LayerNorm2d(nn.Module):
    """
    (N, C, H, W) 텐서에 대해 채널별 LayerNorm을 수행하는 예시 구현입니다.
    """
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x shape: (N, C, H, W)
        mean = x.mean(dim=[2, 3], keepdim=True)   # HW 차원에 대한 평균
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)  # HW 차원에 대한 분산

        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

################################################################################
# 2. DropBlock2D
################################################################################
class DropBlock2D(nn.Module):
    r"""
    DropBlock: A regularization method for convolutional networks
    https://arxiv.org/abs/1810.12890
    """
    def __init__(self, p, block_size=7):
        super(DropBlock2D, self).__init__()
        self.drop_prob = p
        self.block_size = block_size

    def forward(self, x):
        # 학습 모드가 아니거나 drop_prob=0이면 그대로 반환
        if (not self.training) or self.drop_prob == 0.:
            return x
        else:
            gamma = self._compute_gamma(x)
            # (N, H, W)
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float().to(x.device)
            block_mask = self._compute_block_mask(mask)
            # x * block_mask(4D) = x * (N,1,H,W)
            out = x * block_mask[:, None, :, :]
            # scale
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_block_mask(self, mask):
        # block_size x block_size 영역을 커버
        block_mask = F.max_pool2d(
            input=mask[:, None, :, :],
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=self.block_size // 2
        )
        # 짝수 block_size의 경우 패딩으로 생기는 끝단 처리
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)  # (N, H, W)
        return block_mask

    def _compute_gamma(self, x):
        # 논문에서 사용된 gamma
        return self.drop_prob / (self.block_size ** 2)

################################################################################
# 3. ResNeXt 논문 방식 Bottleneck 블록
################################################################################
class ResNeXtBottleneck(nn.Module):
    """
    ResNeXt 논문(https://arxiv.org/abs/1611.05431)에 나오는 Aggregated Residual Transformations.
    공식 구현 로직에 따라,
      - expansion = 4
      - width = int(planes * (base_width / 64.0)) * cardinality
        (TorchVision 등에서도 동일 로직)
    최종 out_channels = planes * expansion

    구성: 
        1x1 Conv -> 3x3 group conv -> 1x1 Conv
        + shortcut
        + activation
    """
    expansion = 4

    def __init__(
        self,
        in_channels,     # 이 블록에 들어오는 채널 수
        planes,          # 블록의 'base' 채널 수 (이후 expansion 고려)
        cardinality=32,  # 그룹 수
        base_width=4,    # base_width (논문에서 4가 기본)
        stride=1,
        act_layer=nn.GELU,
        norm_layer=LayerNorm2d,
        drop_layer=None
    ):
        super().__init__()

        self.cardinality = cardinality
        self.base_width = base_width
        self.planes = planes
        self.expansion = ResNeXtBottleneck.expansion

        # 논문(및 torchvision ResNeXt)에서 사용하는 공식
        width = int(planes * (base_width / 64.0)) * cardinality
        out_channels = planes * self.expansion

        # conv1: 1x1
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = norm_layer(width)
        self.act = act_layer()  # conv 뒤 활성함수 모두 동일

        # conv2: 3x3 group conv
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,  # 여기서 group=cardinality
            bias=False
        )
        self.norm2 = norm_layer(width)

        # conv3: 1x1
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = norm_layer(out_channels)

        # shortcut: in_channels != out_channels or stride != 1 일 때
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels)
            )

        # drop_layer (예: DropBlock2D 등)
        self.drop = drop_layer if drop_layer is not None else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        if self.drop:
            out = self.drop(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)
        if self.drop:
            out = self.drop(out)

        out = self.conv3(out)
        out = self.norm3(out)
        if self.drop:
            out = self.drop(out)

        out += self.shortcut(identity)
        out = self.act(out)

        return out

################################################################################
# 4. ResNeXt를 기반으로 하는 간단한 U-Net
#    (각 stage마다 bottleneck block 2개씩, 혹은 1개씩 등 다양한 설계 가능)
################################################################################
class ResNeXtUNet(nn.Module):
    """
    - 일반적인 U-Net 레이아웃:
        [Enc]
          Block -> Down(2x)
          Block -> Down(2x)
          ...
        [Center]
        [Dec]
          Up(2x) + Skip + Block
          Up(2x) + Skip + Block
          ...
    - 각 Block으로 ResNeXtBottleneck을 사용 (갯수는 상황에 맞춰 조정)
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base_channels=64,
        num_enc_stages=4,
        cardinality=32,
        base_width=4,
        act_layer=nn.GELU,
        norm_layer=LayerNorm2d,
        drop_layer=lambda: DropBlock2D(p=0.2),
    ):
        super().__init__()

        # --------------------------------------------------------------------------------
        # 1) 인코더 (Down)
        # --------------------------------------------------------------------------------
        # enc1
        self.enc1_block1 = ResNeXtBottleneck(
            in_channels,
            planes=base_channels,
            cardinality=cardinality,
            base_width=base_width,
            stride=1,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_layer=drop_layer()
        )
        # 논문에선 stage마다 2~3개의 블록을 쌓지만, 여기선 예시로 2개 정도
        self.enc1_block2 = ResNeXtBottleneck(
            in_channels=base_channels * ResNeXtBottleneck.expansion,  # 이전 block의 out_channels
            planes=base_channels,
            cardinality=cardinality,
            base_width=base_width,
            stride=1,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_layer=drop_layer()
        )
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # enc2
        self.enc2_block1 = ResNeXtBottleneck(
            in_channels=base_channels * ResNeXtBottleneck.expansion,
            planes=base_channels * 2,  # 두 번째 스테이지는 채널 2배
            cardinality=cardinality,
            base_width=base_width,
            stride=1,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_layer=drop_layer()
        )
        self.enc2_block2 = ResNeXtBottleneck(
            in_channels=(base_channels * 2) * ResNeXtBottleneck.expansion,
            planes=base_channels * 2,
            cardinality=cardinality,
            base_width=base_width,
            stride=1,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_layer=drop_layer()
        )
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # enc3
        self.enc3_block1 = ResNeXtBottleneck(
            in_channels=(base_channels * 2) * ResNeXtBottleneck.expansion,
            planes=base_channels * 4,
            cardinality=cardinality,
            base_width=base_width,
            stride=1,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_layer=drop_layer()
        )
        self.enc3_block2 = ResNeXtBottleneck(
            in_channels=(base_channels * 4) * ResNeXtBottleneck.expansion,
            planes=base_channels * 4,
            cardinality=cardinality,
            base_width=base_width,
            stride=1,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_layer=drop_layer()
        )
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # enc4
        self.enc4_block1 = ResNeXtBottleneck(
            in_channels=(base_channels * 4) * ResNeXtBottleneck.expansion,
            planes=base_channels * 8,
            cardinality=cardinality,
            base_width=base_width,
            stride=1,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_layer=drop_layer()
        )
        self.enc4_block2 = ResNeXtBottleneck(
            in_channels=(base_channels * 8) * ResNeXtBottleneck.expansion,
            planes=base_channels * 8,
            cardinality=cardinality,
            base_width=base_width,
            stride=1,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_layer=drop_layer()
        )
        self.down4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --------------------------------------------------------------------------------
        # 2) Center
        # --------------------------------------------------------------------------------
        self.center_block1 = ResNeXtBottleneck(
            in_channels=(base_channels * 8) * ResNeXtBottleneck.expansion,
            planes=base_channels * 8,
            cardinality=cardinality,
            base_width=base_width,
            stride=1,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_layer=drop_layer()
        )
        self.center_block2 = ResNeXtBottleneck(
            in_channels=(base_channels * 8) * ResNeXtBottleneck.expansion,
            planes=base_channels * 8,
            cardinality=cardinality,
            base_width=base_width,
            stride=1,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_layer=drop_layer()
        )

        # --------------------------------------------------------------------------------
        # 3) 디코더(Up) - TransposedConv 사용
        # --------------------------------------------------------------------------------
        # up4
        self.up4 = nn.ConvTranspose2d(
            (base_channels * 8) * ResNeXtBottleneck.expansion,
            (base_channels * 8) * ResNeXtBottleneck.expansion,
            kernel_size=2,
            stride=2
        )
        self.dec4_block1 = ResNeXtBottleneck(
            in_channels=((base_channels * 8) * ResNeXtBottleneck.expansion +  # up 결과
                         (base_channels * 8) * ResNeXtBottleneck.expansion),  # skip
            planes=base_channels * 8,
            cardinality=cardinality,
            base_width=base_width,
            stride=1,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_layer=drop_layer()
        )
        self.dec4_block2 = ResNeXtBottleneck(
            in_channels=(base_channels * 8) * ResNeXtBottleneck.expansion,
            planes=base_channels * 8,
            cardinality=cardinality,
            base_width=base_width,
            stride=1,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_layer=drop_layer()
        )

        # up3
        self.up3 = nn.ConvTranspose2d(
            (base_channels * 8) * ResNeXtBottleneck.expansion,
            (base_channels * 4) * ResNeXtBottleneck.expansion,
            kernel_size=2,
            stride=2
        )
        self.dec3_block1 = ResNeXtBottleneck(
            in_channels=((base_channels * 4) * ResNeXtBottleneck.expansion +  # up 결과
                         (base_channels * 4) * ResNeXtBottleneck.expansion),  # skip
            planes=base_channels * 4,
            cardinality=cardinality,
            base_width=base_width,
            stride=1,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_layer=drop_layer()
        )
        self.dec3_block2 = ResNeXtBottleneck(
            in_channels=(base_channels * 4) * ResNeXtBottleneck.expansion,
            planes=base_channels * 4,
            cardinality=cardinality,
            base_width=base_width,
            stride=1,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_layer=drop_layer()
        )

        # up2
        self.up2 = nn.ConvTranspose2d(
            (base_channels * 4) * ResNeXtBottleneck.expansion,
            (base_channels * 2) * ResNeXtBottleneck.expansion,
            kernel_size=2,
            stride=2
        )
        self.dec2_block1 = ResNeXtBottleneck(
            in_channels=((base_channels * 2) * ResNeXtBottleneck.expansion +
                         (base_channels * 2) * ResNeXtBottleneck.expansion),
            planes=base_channels * 2,
            cardinality=cardinality,
            base_width=base_width,
            stride=1,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_layer=drop_layer()
        )
        self.dec2_block2 = ResNeXtBottleneck(
            in_channels=(base_channels * 2) * ResNeXtBottleneck.expansion,
            planes=base_channels * 2,
            cardinality=cardinality,
            base_width=base_width,
            stride=1,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_layer=drop_layer()
        )

        # up1
        self.up1 = nn.ConvTranspose2d(
            (base_channels * 2) * ResNeXtBottleneck.expansion,
            (base_channels * 1) * ResNeXtBottleneck.expansion,
            kernel_size=2,
            stride=2
        )
        self.dec1_block1 = ResNeXtBottleneck(
            in_channels=((base_channels * 1) * ResNeXtBottleneck.expansion +
                         (base_channels * 1) * ResNeXtBottleneck.expansion),
            planes=base_channels * 1,
            cardinality=cardinality,
            base_width=base_width,
            stride=1,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_layer=drop_layer()
        )
        self.dec1_block2 = ResNeXtBottleneck(
            in_channels=(base_channels * 1) * ResNeXtBottleneck.expansion,
            planes=base_channels,
            cardinality=cardinality,
            base_width=base_width,
            stride=1,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_layer=drop_layer()
        )

        # --------------------------------------------------------------------------------
        # 4) 최종 출력
        # --------------------------------------------------------------------------------
        self.out_conv = nn.Conv2d(base_channels * ResNeXtBottleneck.expansion, out_channels, kernel_size=1)

    def forward(self, x):
        # -------------------------
        # Encoder
        # -------------------------
        x1 = self.enc1_block1(x)
        x1 = self.enc1_block2(x1)
        d1 = self.down1(x1)

        x2 = self.enc2_block1(d1)
        x2 = self.enc2_block2(x2)
        d2 = self.down2(x2)

        x3 = self.enc3_block1(d2)
        x3 = self.enc3_block2(x3)
        d3 = self.down3(x3)

        x4 = self.enc4_block1(d3)
        x4 = self.enc4_block2(x4)
        d4 = self.down4(x4)

        # -------------------------
        # Center
        # -------------------------
        c = self.center_block1(d4)
        c = self.center_block2(c)

        # -------------------------
        # Decoder
        # -------------------------
        u4 = self.up4(c)
        u4 = torch.cat([u4, x4], dim=1)
        u4 = self.dec4_block1(u4)
        u4 = self.dec4_block2(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.dec3_block1(u3)
        u3 = self.dec3_block2(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.dec2_block1(u2)
        u2 = self.dec2_block2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.dec1_block1(u1)
        u1 = self.dec1_block2(u1)

        out = self.out_conv(u1)
        out_dict = {
            'x_recon':out
        }
        return out_dict

################################################################################
# 5. Model Registry (small, base, large 예시)
################################################################################
MODEL_CONFIGS = {
    "small": dict(
        in_channels=3,
        out_channels=1,
        base_channels=32,   # U-Net 첫 블록에서의 base_planes
        cardinality=8,      # 그룹 수
        base_width=4,       # ResNeXt 논문 기본=4
    ),
    "base": dict(
        in_channels=3,
        out_channels=1,
        base_channels=64,
        cardinality=32,     # 논문에서 많이 쓰는 32x4d
        base_width=4,
    ),
    "large": dict(
        in_channels=3,
        out_channels=1,
        base_channels=64,
        cardinality=32,
        base_width=8,       # base_width 늘림 (예시)
    ),
}

def build_resnext_unet(model_name: str,
                       act_layer=nn.GELU,
                       norm_layer=LayerNorm2d,
                       drop_layer=lambda: DropBlock2D(p=0.1),
                       in_ch=3,
                       out_ch=1,
                       ):
    """
    model_name (str): "small", "base", "large"
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_name '{model_name}'. Choose from {list(MODEL_CONFIGS.keys())}")

    cfg = MODEL_CONFIGS[model_name]
    model = ResNeXtUNet(
        in_channels=in_ch,
        out_channels=out_ch,
        base_channels=cfg["base_channels"],
        cardinality=cfg["cardinality"],
        base_width=cfg["base_width"],
        act_layer=act_layer,
        norm_layer=norm_layer,
        drop_layer=drop_layer
    )
    return model

################################################################################
# 6. 테스트(main) - 모델별 파라미터 수, 더미 입력 결과
################################################################################
if __name__ == "__main__":
    model_names = ["small", "base", "large"]
    for name in model_names:
        print("====================================")
        print(f"Building ResNeXt-UNet model: {name}")
        net = build_resnext_unet(name)

        # 파라미터 수
        param_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Total trainable params: {param_count:,}")

        # 더미 입력 후 결과 확인
        dummy = torch.randn(1, 3, 256, 256)  # 배치=1, 채널=3, H=256, W=256
        out = net(dummy)
        print(f"Input shape:  {dummy.shape}")
        print(f"Output shape: {out.shape}")
