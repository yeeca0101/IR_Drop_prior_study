import math
import copy
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    efficientformerv2_s0_config,
    efficientformerv2_s1_config,
    efficientformerv2_s2_config,
    efficientformerv2_l_config,
    efficientformerv2_s0_seg_config,
    efficientformerv2_s1_seg_config,
    efficientformerv2_s2_seg_config,
    efficientformerv2_l_seg_config,
    efficientformerv2_s0_irdrop_config,
)
from .head import build_head  # head 모듈 생성용 함수

########################################
# Helper Functions & Layers
########################################

def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(l, u)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

########################################
# Model Modules (EfficientFormer 구성 요소)
########################################

class Attention4D(nn.Module):
    def __init__(self, dim=384, key_dim=32, num_heads=8,
                 attn_ratio=4, resolution=7,
                 act_layer=nn.ReLU, stride=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads
        if stride is not None:
            self.resolution = math.ceil(resolution / stride)
            self.stride_conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim),
                nn.BatchNorm2d(dim),
            )
            self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=False)
        else:
            self.resolution = resolution
            self.stride_conv = None
            self.upsample = None
        self.N = self.resolution ** 2
        self.d = int(attn_ratio * key_dim)
        self.dh = self.d * num_heads
        self.attn_ratio = attn_ratio
        self.q = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * key_dim, 1),
            nn.BatchNorm2d(self.num_heads * key_dim),
        )
        self.k = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * key_dim, 1),
            nn.BatchNorm2d(self.num_heads * key_dim),
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.d, 1),
            nn.BatchNorm2d(self.num_heads * self.d),
        )
        self.v_local = nn.Sequential(
            nn.Conv2d(self.num_heads * self.d, self.num_heads * self.d,
                      kernel_size=3, stride=1, padding=1, groups=self.num_heads * self.d),
            nn.BatchNorm2d(self.num_heads * self.d),
        )
        self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1)
        self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1)
        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2d(self.dh, dim, 1),
            nn.BatchNorm2d(dim),
        )
        # 상대적 거리 기반 bias
        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.register_buffer('attention_biases', torch.zeros(num_heads, self.resolution * self.resolution))
        self.register_buffer('attention_bias_idxs', torch.ones(self.resolution * self.resolution, self.resolution * self.resolution).long())
        self.attention_biases_seg = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs_seg', torch.LongTensor(idxs).view(self.N, self.N))
        self.ab = None

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases_seg[:, self.attention_bias_idxs_seg]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.stride_conv is not None:
            x = self.stride_conv(x)
            H //= 2
            W //= 2
        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, H * W).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, H * W).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, H * W).permute(0, 1, 3, 2)
        attn = (q @ k) * self.scale
        bias = self.attention_biases_seg[:, self.attention_bias_idxs_seg] if self.training else self.ab
        bias = F.interpolate(bias.unsqueeze(0), size=(attn.size(-2), attn.size(-1)), mode='bicubic', align_corners=False)
        attn = attn + bias
        attn = self.talking_head1(attn)
        attn = attn.softmax(dim=-1)
        attn = self.talking_head2(attn)
        x_out = (attn @ v)
        x_out = x_out.transpose(2, 3).reshape(B, self.dh, H, W) + v_local
        if self.upsample is not None:
            x_out = self.upsample(x_out)
        x_out = self.proj(x_out)
        return x_out

class stem(nn.Module):
    def __init__(self, in_chs, out_chs, act_layer=nn.ReLU):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_chs // 2),
            act_layer(),
            nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_chs),
            act_layer(),
        )
    def forward(self, x):
        return self.stem(x)

class LGQuery(nn.Module):
    def __init__(self, in_dim, out_dim, resolution1, resolution2):
        super().__init__()
        self.pool = nn.AvgPool2d(1, 2, 0)
        self.local = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
        )
    def forward(self, x):
        local_q = self.local(x)
        pool_q = self.pool(x)
        return self.proj(local_q + pool_q)

class Attention4DDownsample(nn.Module):
    def __init__(self, dim=384, key_dim=16, num_heads=8,
                 attn_ratio=4, resolution=7,
                 out_dim=None, act_layer=nn.ReLU):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads
        self.resolution = resolution
        self.d = int(attn_ratio * key_dim)
        self.dh = self.d * num_heads
        self.attn_ratio = attn_ratio
        self.out_dim = out_dim if out_dim is not None else dim
        self.resolution2 = math.ceil(self.resolution / 2)
        self.q = LGQuery(dim, self.num_heads * key_dim, self.resolution, self.resolution2)
        self.k = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * key_dim, 1),
            nn.BatchNorm2d(self.num_heads * key_dim),
        )
        self.v = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.d, 1),
            nn.BatchNorm2d(self.num_heads * self.d),
        )
        self.v_local = nn.Sequential(
            nn.Conv2d(self.num_heads * self.d, self.num_heads * self.d,
                      kernel_size=3, stride=2, padding=1, groups=self.num_heads * self.d),
            nn.BatchNorm2d(self.num_heads * self.d),
        )
        self.proj = nn.Sequential(
            act_layer(),
            nn.Conv2d(self.dh, self.out_dim, 1),
            nn.BatchNorm2d(self.out_dim),
        )
        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        points_ = list(itertools.product(range(self.resolution2), range(self.resolution2)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * math.ceil(self.resolution / self.resolution2) - p2[0] + (size - 1) / 2),
                    abs(p1[1] * math.ceil(self.resolution / self.resolution2) - p2[1] + (size - 1) / 2)
                )
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.register_buffer('attention_biases', torch.zeros(num_heads, 196))
        self.register_buffer('attention_bias_idxs', torch.ones(self.resolution2 * self.resolution2, 196).long())
        self.attention_biases_seg = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs_seg', torch.LongTensor(idxs).view(len(points_), N))
        self.ab = None
    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases_seg[:, self.attention_bias_idxs_seg]
    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, (H * W) // 4).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, H * W).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, H * W).permute(0, 1, 3, 2)
        attn = (q @ k) * self.scale
        bias = self.attention_biases_seg[:, self.attention_bias_idxs_seg] if self.training else self.ab
        bias = F.interpolate(bias.unsqueeze(0), size=(attn.size(-2), attn.size(-1)), mode='bicubic', align_corners=False)
        attn = attn + bias
        attn = attn.softmax(dim=-1)
        x_out = (attn @ v).transpose(2, 3)
        x_out = x_out.reshape(B, self.dh, H // 2, W // 2) + v_local
        x_out = self.proj(x_out)
        return x_out

class Embedding(nn.Module):
    def __init__(self, patch_size=3, stride=2, padding=1,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d,
                 light=False, asub=False, resolution=None, act_layer=nn.GELU,
                 attn_block=Attention4DDownsample):
        super().__init__()
        self.light = light
        self.asub = asub
        if self.light:
            self.new_proj = nn.Sequential(
                nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=2, padding=1, groups=in_chans),
                nn.BatchNorm2d(in_chans),
                nn.Hardswish(),
                nn.Conv2d(in_chans, embed_dim, kernel_size=1),
                nn.BatchNorm2d(embed_dim),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=2),
                nn.BatchNorm2d(embed_dim),
            )
        elif self.asub:
            self.attn = attn_block(dim=in_chans, out_dim=embed_dim,
                                   resolution=resolution, act_layer=act_layer)
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
            self.bn = norm_layer(embed_dim) if norm_layer else nn.Identity()
        else:
            patch_size = to_2tuple(patch_size)
            stride = to_2tuple(stride)
            padding = to_2tuple(padding)
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
            self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        if self.light:
            return self.new_proj(x) + self.skip(x)
        elif self.asub:
            out_conv = self.conv(x)
            out_conv = self.bn(out_conv)
            return self.attn(x) + out_conv
        else:
            x = self.proj(x)
            return self.norm(x)

class Mlp(nn.Module):
    """
    MLP with 1x1 convolutions.
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., mid_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(out_features)
        if self.mid_conv:
            self.mid = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)
            self.mid_norm = nn.BatchNorm2d(hidden_features)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        if self.mid_conv:
            x_mid = self.mid(x)
            x_mid = self.mid_norm(x_mid)
            x = self.act(x_mid)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.drop(x)
        return x

class AttnFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 resolution=7, stride=None):
        super().__init__()
        self.token_mixer = Attention4D(dim, resolution=resolution, act_layer=act_layer, stride=stride)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, mid_conv=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim, 1, 1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim, 1, 1), requires_grad=True)
    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(x))
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x

class FFN(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, mid_conv=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim, 1, 1), requires_grad=True)
    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.mlp(x))
        return x

def meta_blocks(dim, index, layers,
                pool_size=3, mlp_ratio=4.,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                drop_rate=0., drop_path_rate=0.,
                use_layer_scale=True, layer_scale_init_value=1e-5,
                vit_num=1, resolution=7, e_ratios=None):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        this_mlp_ratio = e_ratios[str(index)][block_idx]
        if index >= 2 and block_idx > layers[index] - 1 - vit_num:
            stride = 2 if index == 2 else None
            blocks.append(AttnFFN(
                dim, mlp_ratio=this_mlp_ratio,
                act_layer=act_layer, norm_layer=norm_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                resolution=resolution,
                stride=stride,
            ))
        else:
            blocks.append(FFN(
                dim, pool_size=pool_size, mlp_ratio=this_mlp_ratio,
                act_layer=act_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
    return nn.Sequential(*blocks)

class EfficientFormer(nn.Module):
    def __init__(self, layers, embed_dims, mlp_ratios=4, downsamples=None,
                 pool_size=3, norm_layer=nn.BatchNorm2d, act_layer=nn.GELU,
                 num_classes=1000, down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0., use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False, vit_num=0, distillation=True, resolution=512, e_ratios=None,
                 head_cfg=None,in_ch=3, **kwargs):
        """
        head_cfg: segmentation head 구성 딕셔너리. fork_feat=True인 경우, 주어지면 head 모듈을 build하여
                  forward 시 feature maps에 적용합니다.
        """
        super().__init__()
        self.fork_feat = fork_feat
        self.num_classes = num_classes if not fork_feat else None
        self.dist = distillation
        self.in_ch = in_ch
        self.patch_embed = stem(in_ch, embed_dims[0], act_layer=act_layer)
        network = []
        for i in range(len(layers)):
            stage = meta_blocks(embed_dims[i], i, layers,
                                pool_size=pool_size, mlp_ratio=mlp_ratios,
                                act_layer=act_layer, norm_layer=norm_layer,
                                drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                                use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                resolution=math.ceil(resolution / (2 ** (i + 2))),
                                vit_num=vit_num,
                                e_ratios=e_ratios)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            asub = True if i >= 2 else False
            network.append(
                Embedding(
                    patch_size=down_patch_size, stride=down_stride, padding=down_pad,
                    in_chans=embed_dims[i], embed_dim=embed_dims[i + 1],
                    resolution=math.ceil(resolution / (2 ** (i + 2))),
                    asub=asub,
                    act_layer=act_layer, norm_layer=norm_layer,
                )
            )
        self.network = nn.ModuleList(network)
        if self.fork_feat:
            self.out_indices = [0, 2, 4, 6]
        else:
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
            if self.dist:
                self.dist_head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        # head_cfg 적용: segmentation task의 경우, fork_feat=True이면 head 모듈을 build
        self.head_module = None
        if self.fork_feat and head_cfg is not None:
            self.head_module = build_head(head_cfg, embed_dims)
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                outs.append(x)
        return outs if self.fork_feat else x

    def forward(self, x):
        x = self.patch_embed(x)
        if self.fork_feat:
            features = self.forward_tokens(x)
            if self.head_module is not None:
                return self.head_module(features)
            return features
        else:
            x = self.forward_tokens(x)
            x = self.norm(x)
            x = x.flatten(2).mean(-1)
            if self.dist:
                cls_out = (self.head(x) + self.dist_head(x)) / 2 if not self.training else (self.head(x), self.dist_head(x))
            else:
                cls_out = self.head(x)
            return cls_out

########################################
# Test Functions
########################################

def test_model(model, input_size=(1, 3, 224, 224), device="cpu"):
    model.to(device)
    model.eval()
    x = torch.randn(input_size).to(device)
    with torch.no_grad():
        out = model(x)
    num_params = sum(p.numel() for p in model.parameters())
    if isinstance(out, list):
        out_shapes = [o.shape for o in out]
    else:
        out_shapes = out.shape
    print(f"Model: {model.__class__.__name__}, Params: {num_params:,}, Input: {x.shape}, Output: {out_shapes}")

def test_all_models(device="cpu"):
    print("===== Testing EfficientFormer Variants (Classification) =====")
    # print("EfficientFormer V2 S0")
    # model_s0 = EfficientFormer(resolution=224, num_classes=1000, **efficientformerv2_s0_config)
    # test_model(model_s0, device=device)
    # print("EfficientFormer V2 S1")
    # model_s1 = EfficientFormer(resolution=224, num_classes=1000, **efficientformerv2_s1_config)
    # test_model(model_s1, device=device)
    # print("EfficientFormer V2 S2")
    # model_s2 = EfficientFormer(resolution=224, num_classes=1000, **efficientformerv2_s2_config)
    # test_model(model_s2, device=device)
    # print("EfficientFormer V2 L")
    # model_l = EfficientFormer(resolution=224, num_classes=1000, **efficientformerv2_l_config)
    # test_model(model_l, device=device)
    
    # print("===== Testing EfficientFormer Variants (Segmentation with FPN Head) =====")
    # print("EfficientFormer V2 S0 (Segmentation)")
    # model_s0_seg = EfficientFormer(resolution=224, num_classes=0, **efficientformerv2_s0_seg_config)
    # test_model(model_s0_seg, device=device)
    # print("EfficientFormer V2 S1 (Segmentation)")
    # model_s1_seg = EfficientFormer(resolution=224, num_classes=0, **efficientformerv2_s1_seg_config)
    # test_model(model_s1_seg, device=device)
    # print("EfficientFormer V2 S2 (Segmentation)")
    # model_s2_seg = EfficientFormer(resolution=224, num_classes=0, **efficientformerv2_s2_seg_config)
    # test_model(model_s2_seg, device=device)
    # print("EfficientFormer V2 L (Segmentation)")
    # model_l_seg = EfficientFormer(resolution=224, num_classes=0, **efficientformerv2_l_seg_config)
    # test_model(model_l_seg, device=device)


    print("EfficientFormer V2 S0 (IR)")
    model_l_seg = EfficientFormer(resolution=256, num_classes=1, **efficientformerv2_s0_irdrop_config)
    test_model(model_l_seg, device=device, input_size=(1,25,256,256))


if __name__ == '__main__':
    test_all_models(device="cuda:0")
