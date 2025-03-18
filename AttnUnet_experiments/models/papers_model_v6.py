import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from .parts.vqvae import VectorQuantizer  # V6 parts
from .parts.v6_parts import *
from .parts.exp_parts import VectorQuantiser # 6_1 parts

class AttnUnetV6(nn.Module):
    def __init__(self, in_ch=12, out_ch=1, dropout_name='', dropout_p=0.5,
                  num_embeddings=64, decay=0.99, epsilon=1e-5,use_ema=False,act=SwishT()):
        super().__init__()

        in_channels = [12, 32, 64, 128, 256, 512]
        self.concat = True
        self.drop_out = {
            'nn.dropout': nn.Dropout2d,
            'dropblock': DropBlock2D
        }
        self.use_ema = use_ema
        self.act= act

        self.preconv = PreConv(in_ch, out_ch=in_channels[0],act=self.act)
        self.d1 = DownBlock(in_ch=in_channels[0], out_ch=in_channels[1], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)
        self.d2 = DownBlock(in_ch=in_channels[1], out_ch=in_channels[2], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)
        self.d3 = DownBlock(in_ch=in_channels[2], out_ch=in_channels[3], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)
        self.d4 = DownBlock(in_ch=in_channels[3], out_ch=in_channels[4], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)

        # Bottleneck
        self.bottleneck = DoubleConv(in_ch=in_channels[4], out_ch=in_channels[5], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)

        # VectorQuantizer 추가
        self.vq = VectorQuantizer(
            embedding_dim=in_channels[5],  # bottleneck의 채널 크기와 일치해야 함
            num_embeddings=num_embeddings, 
            use_ema=use_ema, 
            decay=decay, 
            epsilon=epsilon
        )

        self.attn1 = AttentionGate(in_ch_x=in_channels[4], in_ch_g=in_channels[5], out_ch=in_channels[4], concat=self.concat)
        self.attn2 = AttentionGate(in_ch_x=in_channels[3], in_ch_g=in_channels[4], out_ch=in_channels[3], concat=self.concat)
        self.attn3 = AttentionGate(in_ch_x=in_channels[2], in_ch_g=in_channels[3], out_ch=in_channels[2], concat=self.concat)
        self.attn4 = AttentionGate(in_ch_x=in_channels[1], in_ch_g=in_channels[2], out_ch=in_channels[1], concat=self.concat)

        self.up1 = UpBlock(in_ch_x=in_channels[4], in_ch_g=in_channels[5], out_ch=in_channels[4], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)
        self.up2 = UpBlock(in_ch_x=in_channels[3], in_ch_g=in_channels[4], out_ch=in_channels[3], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)
        self.up3 = UpBlock(in_ch_x=in_channels[2], in_ch_g=in_channels[3], out_ch=in_channels[2], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)
        self.up4 = UpBlock(in_ch_x=in_channels[1], in_ch_g=in_channels[2], out_ch=in_channels[1], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)

        self.head = nn.Sequential(
            nn.Conv2d(in_channels[1], out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.preconv(x)
        x1 = self.d1(x)    
        x2 = self.d2(x1)   
        x3 = self.d3(x2)   
        x4 = self.d4(x3)   

        x5 = self.bottleneck(x4)  # Bottleneck

        # VectorQuantizer 추가
        z_quantized, dict_loss, commit_loss, _ = self.vq(x5)  # 벡터 양자화

        attn1 = self.attn1(x4, z_quantized)   
        up1 = self.up1(attn1, z_quantized)    
        attn2 = self.attn2(x3, up1)  
        up2 = self.up2(attn2, up1)   

        attn3 = self.attn3(x2, up2)  
        up3 = self.up3(attn3, up2)   

        attn4 = self.attn4(x1, up3)  
        up4 = self.up4(attn4, up3)   

        output = self.head(up4)
        return {
            'x_recon': output, 
            'dictionary_loss': dict_loss, 
            'commitment_loss': commit_loss
        }

class AttnUnetV6_1(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, dropout_name='', dropout_p=0.5,
                  num_embeddings=512, decay=0.99, epsilon=1e-5,use_ema=False,act=SwishT()):
        super().__init__()

        in_channels = [12, 32, 64, 128, 256, 512]
        self.concat = True
        self.drop_out = {
            'nn.dropout': nn.Dropout2d,
            'dropblock': DropBlock2D
        }
        self.use_ema = use_ema
        self.act= act

        self.preconv = PreConv(in_ch, out_ch=in_channels[0],act=self.act)
        self.d1 = DownBlock(in_ch=in_channels[0], out_ch=in_channels[1], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)
        self.d2 = DownBlock(in_ch=in_channels[1], out_ch=in_channels[2], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)
        self.d3 = DownBlock(in_ch=in_channels[2], out_ch=in_channels[3], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)
        self.d4 = DownBlock(in_ch=in_channels[3], out_ch=in_channels[4], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)

        # Bottleneck
        self.bottleneck = DoubleConv(in_ch=in_channels[4], out_ch=in_channels[5], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)

        # VectorQuantizer 추가
        self.vq = VectorQuantiser(num_embeddings, in_channels[5], 0.25, distance='l2', 
                                       anchor='probrandom', first_batch=False, contras_loss=True)


        self.attn1 = AttentionGate(in_ch_x=in_channels[4], in_ch_g=in_channels[5], out_ch=in_channels[4], concat=self.concat)
        self.attn2 = AttentionGate(in_ch_x=in_channels[3], in_ch_g=in_channels[4], out_ch=in_channels[3], concat=self.concat)
        self.attn3 = AttentionGate(in_ch_x=in_channels[2], in_ch_g=in_channels[3], out_ch=in_channels[2], concat=self.concat)
        self.attn4 = AttentionGate(in_ch_x=in_channels[1], in_ch_g=in_channels[2], out_ch=in_channels[1], concat=self.concat)

        self.up1 = UpBlock(in_ch_x=in_channels[4], in_ch_g=in_channels[5], out_ch=in_channels[4], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)
        self.up2 = UpBlock(in_ch_x=in_channels[3], in_ch_g=in_channels[4], out_ch=in_channels[3], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)
        self.up3 = UpBlock(in_ch_x=in_channels[2], in_ch_g=in_channels[3], out_ch=in_channels[2], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)
        self.up4 = UpBlock(in_ch_x=in_channels[1], in_ch_g=in_channels[2], out_ch=in_channels[1], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)

        self.head = nn.Sequential(
            nn.Conv2d(in_channels[1], out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.preconv(x)
        x1 = self.d1(x)    
        x2 = self.d2(x1)   
        x3 = self.d3(x2)   
        x4 = self.d4(x3)   

        x5 = self.bottleneck(x4)  # Bottleneck

        # VectorQuantizer 추가
        z_quantized, dict_loss, (commit_loss, encodings, _) =self.vq(x5)

        attn1 = self.attn1(x4, z_quantized)   
        up1 = self.up1(attn1, z_quantized)    
        attn2 = self.attn2(x3, up1)  
        up2 = self.up2(attn2, up1)   

        attn3 = self.attn3(x2, up2)  
        up3 = self.up3(attn3, up2)   

        attn4 = self.attn4(x1, up3)  
        up4 = self.up4(attn4, up3)   

        output = self.head(up4)
        return {
            'x_recon': output, 
            'dictionary_loss': dict_loss, 
            'commitment_loss': 0
        }
    
class AttnUnetV6_2(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, dropout_name='', dropout_p=0.5,
                num_embeddings=512, act=SwishT(),distance='l2',**kwargs):
        super().__init__()

        in_channels = [12, 32, 64, 128, 256, 512]
        self.concat = True
        self.drop_out = {
            'nn.dropout': nn.Dropout2d,
            'dropblock': DropBlock2D
        }
        self.act= act

        self.preconv = PreConv(in_ch, out_ch=in_channels[0],act=self.act)
        self.d1 = DownBlock(in_ch=in_channels[0], out_ch=in_channels[1], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)
        self.d2 = DownBlock(in_ch=in_channels[1], out_ch=in_channels[2], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)
        self.d3 = DownBlock(in_ch=in_channels[2], out_ch=in_channels[3], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)
        self.d4 = DownBlock(in_ch=in_channels[3], out_ch=in_channels[4], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)

        # Bottleneck
        self.bottleneck = DoubleConv(in_ch=in_channels[4], out_ch=in_channels[5], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)

        # VectorQuantizer 추가
        self.vq = VectorQuantiser(num_embeddings, in_channels[5], 0.25, distance=distance, 
                                    anchor='probrandom', first_batch=False, contras_loss=True)


        self.attn1 = AttentionGate(in_ch_x=in_channels[4], in_ch_g=in_channels[5], out_ch=in_channels[4], concat=self.concat)
        self.attn2 = AttentionGate(in_ch_x=in_channels[3], in_ch_g=in_channels[4], out_ch=in_channels[3], concat=self.concat)
        self.attn3 = AttentionGate(in_ch_x=in_channels[2], in_ch_g=in_channels[3], out_ch=in_channels[2], concat=self.concat)
        self.attn4 = AttentionGate(in_ch_x=in_channels[1], in_ch_g=in_channels[2], out_ch=in_channels[1], concat=self.concat)

        self.up1 = UpBlock(in_ch_x=in_channels[4], in_ch_g=in_channels[5], out_ch=in_channels[4], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)
        self.up2 = UpBlock(in_ch_x=in_channels[3], in_ch_g=in_channels[4], out_ch=in_channels[3], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)
        self.up3 = UpBlock(in_ch_x=in_channels[2], in_ch_g=in_channels[3], out_ch=in_channels[2], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)
        self.up4 = UpBlock(in_ch_x=in_channels[1], in_ch_g=in_channels[2], out_ch=in_channels[1], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p,act=self.act)


        # multi task head
        self.head_out = nn.Conv2d(in_channels[1], 1, kernel_size=1, stride=1, padding=0)
        self.head_dice = nn.Conv2d(in_channels[1], 1, kernel_size=1, stride=1, padding=0)
        
        
    def forward(self, x):
        x = self.preconv(x)
        x1 = self.d1(x)    
        x2 = self.d2(x1)   
        x3 = self.d3(x2)   
        x4 = self.d4(x3)   

        x5 = self.bottleneck(x4)  # Bottleneck

        # VectorQuantizer 추가
        z_quantized, dict_loss, (commit_loss, encodings, _) =self.vq(x5)

        attn1 = self.attn1(x4, z_quantized)   
        up1 = self.up1(attn1, z_quantized)    
        attn2 = self.attn2(x3, up1)  
        up2 = self.up2(attn2, up1)   

        attn3 = self.attn3(x2, up2)  
        up3 = self.up3(attn3, up2)   

        attn4 = self.attn4(x1, up3)  
        up4 = self.up4(attn4, up3)   

        output = self.head_out(up4) 
        loc = self.head_dice(up4)

        return {
            'x_recon': output, 
            'dictionary_loss': dict_loss, 
            'commitment_loss': 0,
            'loc':loc,
        }
import torch
import torch.nn as nn

# ------------------------------------
# 가정: 아래 모듈들은 이미 정의되어 있거나 import되어 있다고 가정합니다.
# PreConv, DownBlock, DoubleConv, UpBlock, AttentionGate, VectorQuantiser, 
# DropBlock2D, SwishT 등...
# ------------------------------------

class AttnUnetV6_3(nn.Module):
    """
    AttnUnetV6_3
    - 기존 AttnUnetV6_2 구조에 추가적인 RefineBlock을 적용하여
      최종 단계에서 회귀( x_recon )와 분류( loc )를 좀 더 정교하게 예측하는 버전.
    """
    def __init__(self, 
                 in_ch=3, 
                 out_ch=1, 
                 dropout_name='', 
                 dropout_p=0.5,
                 num_embeddings=512, 
                 act=SwishT(),
                 distance='l2',
                 **kwargs):
        super().__init__()

        # 채널 정의 (기존과 동일)
        in_channels = [12, 32, 64, 128, 256, 512]
        self.concat = True

        # 드롭아웃 타입 매핑
        self.drop_out = {
            'nn.dropout': nn.Dropout2d,
            'dropblock': DropBlock2D
        }
        self.act = act

        # 1) Encoder
        self.preconv = PreConv(in_ch, out_ch=in_channels[0], act=self.act)
        self.d1 = DownBlock(in_ch=in_channels[0], out_ch=in_channels[1], 
                            dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p, act=self.act)
        self.d2 = DownBlock(in_ch=in_channels[1], out_ch=in_channels[2], 
                            dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p, act=self.act)
        self.d3 = DownBlock(in_ch=in_channels[2], out_ch=in_channels[3], 
                            dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p, act=self.act)
        self.d4 = DownBlock(in_ch=in_channels[3], out_ch=in_channels[4], 
                            dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p, act=self.act)

        # 2) Bottleneck
        self.bottleneck = DoubleConv(
            in_ch=in_channels[4], 
            out_ch=in_channels[5], 
            dropout_m=self.drop_out[dropout_name], 
            dropout_p=dropout_p,
            act=self.act
        )

        # 3) VectorQuantizer
        self.vq = VectorQuantiser(
            num_embeddings, 
            in_channels[5], 
            commitment_cost=0.25, 
            distance=distance,
            anchor='probrandom', 
            first_batch=False, 
            contras_loss=True
        )

        # 4) Attention Gates
        self.attn1 = AttentionGate(
            in_ch_x=in_channels[4], 
            in_ch_g=in_channels[5], 
            out_ch=in_channels[4], 
            concat=self.concat
        )
        self.attn2 = AttentionGate(
            in_ch_x=in_channels[3], 
            in_ch_g=in_channels[4], 
            out_ch=in_channels[3], 
            concat=self.concat
        )
        self.attn3 = AttentionGate(
            in_ch_x=in_channels[2], 
            in_ch_g=in_channels[3], 
            out_ch=in_channels[2], 
            concat=self.concat
        )
        self.attn4 = AttentionGate(
            in_ch_x=in_channels[1], 
            in_ch_g=in_channels[2], 
            out_ch=in_channels[1], 
            concat=self.concat
        )

        # 5) Decoder (Up Blocks)
        self.up1 = UpBlock(in_ch_x=in_channels[4], in_ch_g=in_channels[5], 
                           out_ch=in_channels[4], 
                           dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p, act=self.act)
        self.up2 = UpBlock(in_ch_x=in_channels[3], in_ch_g=in_channels[4], 
                           out_ch=in_channels[3], 
                           dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p, act=self.act)
        self.up3 = UpBlock(in_ch_x=in_channels[2], in_ch_g=in_channels[3], 
                           out_ch=in_channels[2], 
                           dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p, act=self.act)
        self.up4 = UpBlock(in_ch_x=in_channels[1], in_ch_g=in_channels[2], 
                           out_ch=in_channels[1], 
                           dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p, act=self.act)

        # -------------------------------------------------------------
        # (새로운 부분) 최종 업샘플 결과(up4)를 한 번 더 정제(Refine)하는 블록 추가
        # -------------------------------------------------------------
        self.refine_reg = DoubleConv(
            in_ch=in_channels[1], 
            out_ch=in_channels[1], 
            dropout_m=self.drop_out[dropout_name], 
            dropout_p=dropout_p, 
            act=self.act
        )
        self.refine_cls = DoubleConv(
            in_ch=in_channels[1], 
            out_ch=in_channels[1], 
            dropout_m=self.drop_out[dropout_name], 
            dropout_p=dropout_p, 
            act=self.act
        )

        # (멀티태스킹) 최종 헤드
        # 회귀(regression)와 분류(classification) 각각 별도 Conv1x1
        self.head_reg = nn.Conv2d(in_channels[1], out_ch, kernel_size=1, stride=1, padding=0)
        self.head_cls = nn.Conv2d(in_channels[1], 1, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        # 1) Encoder
        x0 = self.preconv(x)
        x1 = self.d1(x0)   # 1/2
        x2 = self.d2(x1)   # 1/4
        x3 = self.d3(x2)   # 1/8
        x4 = self.d4(x3)   # 1/16

        # 2) Bottleneck
        x5 = self.bottleneck(x4)

        # 3) VectorQuantizer
        z_quantized, dict_loss, (commit_loss, encodings, _) = self.vq(x5)

        # 4) Decoder + Attention
        attn1 = self.attn1(x4, z_quantized)
        up1   = self.up1(attn1, z_quantized)

        attn2 = self.attn2(x3, up1)
        up2   = self.up2(attn2, up1)

        attn3 = self.attn3(x2, up2)
        up3   = self.up3(attn3, up2)

        attn4 = self.attn4(x1, up3)
        up4   = self.up4(attn4, up3)  # 최종 Decoder Feature

        # --------------------------------------------------
        # (새로운 부분) Refine Block을 통해 회귀/분류 분기를 각각 정제
        # --------------------------------------------------
        # 회귀용 정제
        reg_feat = self.refine_reg(up4)
        reg_out = self.head_reg(reg_feat)

        # 분류용 정제
        cls_feat = self.refine_cls(up4)
        cls_out = self.head_cls(cls_feat)

        return {
            'x_recon': reg_out,         # 최종 회귀 결과
            'dictionary_loss': dict_loss,
            'commitment_loss': commit_loss, 
            'loc': cls_out,             # 최종 분류 결과 (ex. 상위 10% 마스크 예측 등에 활용)
        }



if __name__ == '__main__':
    model = AttnUnetV6_2(
        in_ch=3, 
        out_ch=1, 
        dropout_name='dropblock', 
        num_embeddings=512
    ).cuda()
    inp = torch.randn((2, 3, 512, 512)).cuda()
    output = model(inp)
    print(f"Output shape: {output['x_recon'].shape}")
    print(f"loc shape: {output['loc'].shape}")
    # print(f"error shape: {output['error'].shape}") # 6_2 legacy
    print(f"Dictionary Loss: {output['dictionary_loss']}")
    print(f"Commitment Loss: {output['commitment_loss']}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
