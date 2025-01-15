import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from .parts.vqvae import VectorQuantizer  # vqvae.py의 VectorQuantizer만 가져오기
from .parts.v5_parts import *
from .parts.exp_parts import VectorQuantiser

class AttnUnetV5(nn.Module):
    def __init__(self, in_ch=12, out_ch=1, dropout_name='', dropout_p=0.5,
                  num_embeddings=512, decay=0.99, epsilon=1e-5,use_ema=False):
        super().__init__()

        in_channels = [12, 32, 64, 128, 256, 512]
        self.concat = True
        self.drop_out = {
            'nn.dropout': nn.Dropout2d,
            'dropblock': DropBlock2D
        }
        self.use_ema = use_ema
        self.preconv = PreConv(in_ch, out_ch=in_channels[0])
        self.d1 = DownBlock(in_ch=in_channels[0], out_ch=in_channels[1], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)
        self.d2 = DownBlock(in_ch=in_channels[1], out_ch=in_channels[2], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)
        self.d3 = DownBlock(in_ch=in_channels[2], out_ch=in_channels[3], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)
        self.d4 = DownBlock(in_ch=in_channels[3], out_ch=in_channels[4], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)

        # Bottleneck
        self.bottleneck = DoubleConv(in_ch=in_channels[4], out_ch=in_channels[5], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)

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

        self.up1 = UpBlock(in_ch_x=in_channels[4], in_ch_g=in_channels[5], out_ch=in_channels[4], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)
        self.up2 = UpBlock(in_ch_x=in_channels[3], in_ch_g=in_channels[4], out_ch=in_channels[3], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)
        self.up3 = UpBlock(in_ch_x=in_channels[2], in_ch_g=in_channels[3], out_ch=in_channels[2], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)
        self.up4 = UpBlock(in_ch_x=in_channels[1], in_ch_g=in_channels[2], out_ch=in_channels[1], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2),
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


class AttnUnetV5_1(nn.Module):
    def __init__(self, in_ch=12, out_ch=1, dropout_name='', dropout_p=0.5,
                  num_embeddings=64, decay=0.99, epsilon=1e-5,use_ema=False):
        super().__init__()

        in_channels = [12, 32, 64, 128, 256, 512]
        self.concat = True
        self.drop_out = {
            'nn.dropout': nn.Dropout2d,
            'dropblock': DropBlock2D
        }
        self.use_ema = use_ema
        self.preconv = PreConv(in_ch, out_ch=in_channels[0])
        self.d1 = DownBlock(in_ch=in_channels[0], out_ch=in_channels[1], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)
        self.d2 = DownBlock(in_ch=in_channels[1], out_ch=in_channels[2], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)
        self.d3 = DownBlock(in_ch=in_channels[2], out_ch=in_channels[3], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)
        self.d4 = DownBlock(in_ch=in_channels[3], out_ch=in_channels[4], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)

        # Bottleneck
        self.bottleneck = DoubleConv(in_ch=in_channels[4], out_ch=in_channels[5], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)

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

        self.up1 = UpBlock(in_ch_x=in_channels[4], in_ch_g=in_channels[5], out_ch=in_channels[4], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)
        self.up2 = UpBlock(in_ch_x=in_channels[3], in_ch_g=in_channels[4], out_ch=in_channels[3], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)
        self.up3 = UpBlock(in_ch_x=in_channels[2], in_ch_g=in_channels[3], out_ch=in_channels[2], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)
        self.up4 = UpBlock(in_ch_x=in_channels[1], in_ch_g=in_channels[2], out_ch=in_channels[1], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2),
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

        attn1 = self.attn1(x4, z_quantized + x5)   
        up1 = self.up1(attn1, z_quantized + x5)    
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


class AttnUnetV5_2(nn.Module):
    def __init__(self, in_ch=12, out_ch=1, dropout_name='', dropout_p=0.5,
                  num_embeddings=512, decay=0.99, epsilon=1e-5,use_ema=False):
        super().__init__()

        in_channels = [12, 32, 64, 128, 256, 512]
        self.concat = True
        self.drop_out = {
            'nn.dropout': nn.Dropout2d,
            'dropblock': DropBlock2D
        }
        self.use_ema = use_ema
        self.preconv = PreConv(in_ch, out_ch=in_channels[0])
        self.d1 = DownBlock(in_ch=in_channels[0], out_ch=in_channels[1], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)
        self.d2 = DownBlock(in_ch=in_channels[1], out_ch=in_channels[2], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)
        self.d3 = DownBlock(in_ch=in_channels[2], out_ch=in_channels[3], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)
        self.d4 = DownBlock(in_ch=in_channels[3], out_ch=in_channels[4], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)

        # Bottleneck
        self.bottleneck = DoubleConv(in_ch=in_channels[4], out_ch=in_channels[5], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)

        # VectorQuantizer 추가
        self.vq = VectorQuantiser(num_embeddings, in_channels[5], 0.25, distance='l2', 
                                       anchor='probrandom', first_batch=False, contras_loss=True)

        self.attn1 = AttentionGate(in_ch_x=in_channels[4], in_ch_g=in_channels[5], out_ch=in_channels[4], concat=self.concat)
        self.attn2 = AttentionGate(in_ch_x=in_channels[3], in_ch_g=in_channels[4], out_ch=in_channels[3], concat=self.concat)
        self.attn3 = AttentionGate(in_ch_x=in_channels[2], in_ch_g=in_channels[3], out_ch=in_channels[2], concat=self.concat)
        self.attn4 = AttentionGate(in_ch_x=in_channels[1], in_ch_g=in_channels[2], out_ch=in_channels[1], concat=self.concat)

        self.up1 = UpBlock(in_ch_x=in_channels[4], in_ch_g=in_channels[5], out_ch=in_channels[4], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)
        self.up2 = UpBlock(in_ch_x=in_channels[3], in_ch_g=in_channels[4], out_ch=in_channels[3], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)
        self.up3 = UpBlock(in_ch_x=in_channels[2], in_ch_g=in_channels[3], out_ch=in_channels[2], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)
        self.up4 = UpBlock(in_ch_x=in_channels[1], in_ch_g=in_channels[2], out_ch=in_channels[1], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p)

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2),
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

        attn1 = self.attn1(x4, z_quantized + x5)   
        up1 = self.up1(attn1, z_quantized + x5)    
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
            'commitment_loss': 0.
        }
    

if __name__ == '__main__':
    model = AttnUnetV5_2(
        in_ch=3, 
        out_ch=1, 
        dropout_name='dropblock', 
        num_embeddings=512
    )
    inp = torch.randn((1, 3, 512, 512))
    output = model(inp)
    print(f"Output shape: {output['x_recon'].shape}")
    print(f"Dictionary Loss: {output['dictionary_loss']}")
    print(f"Commitment Loss: {output['commitment_loss']}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
