import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from .parts.v7_parts import *
from .parts.exp_parts import VectorQuantiser

class AttnUnetV7(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, dropout_name='', dropout_p=0.5,
                 num_embeddings=512, act=SwishT(), distance='l2', **kwargs):
        super().__init__()
        in_channels = [64, 128, 256, 512, 1024, 1024]
        self.concat = True
        self.drop_out = {
            'nn.dropout': nn.Dropout2d,
            'dropblock': DropBlock2D
        }
        self.act = act

        self.preconv = PreConv(in_ch, out_ch=in_channels[0],stride=2,padding=3,act=self.act)
        self.d1 = DownBlock(in_ch=in_channels[0], out_ch=in_channels[1], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p, act=self.act)
        self.d2 = DownBlock(in_ch=in_channels[1], out_ch=in_channels[2], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p, act=self.act)
        self.d3 = DownBlock(in_ch=in_channels[2], out_ch=in_channels[3], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p, act=self.act)
        self.d4 = DownBlock(in_ch=in_channels[3], out_ch=in_channels[4], dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p, act=self.act)

        # Bottleneck
        self.bottleneck = ResNeXtBottleneck(in_ch=in_channels[4], out_ch=in_channels[5],
                                       cardinality=8, dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p, act=self.act)

        # VectorQuantiser 추가
        self.vq = VectorQuantiser(num_embeddings, in_channels[5], 0.25, distance=distance, 
                                  anchor='probrandom', first_batch=False, contras_loss=True)

        self.attn1 = AttentionGate(in_ch_x=in_channels[4], in_ch_g=in_channels[5], out_ch=in_channels[4], concat=self.concat)
        self.attn2 = AttentionGate(in_ch_x=in_channels[3], in_ch_g=in_channels[4], out_ch=in_channels[3], concat=self.concat)
        self.attn3 = AttentionGate(in_ch_x=in_channels[2], in_ch_g=in_channels[3], out_ch=in_channels[2], concat=self.concat)
        self.attn4 = AttentionGate(in_ch_x=in_channels[1], in_ch_g=in_channels[2], out_ch=in_channels[1], concat=self.concat)

        self.up1 = UpBlock(in_ch_x=in_channels[4], in_ch_g=in_channels[5], out_ch=in_channels[4],
                           dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p, act=self.act, cardinality=8)
        self.up2 = UpBlock(in_ch_x=in_channels[3], in_ch_g=in_channels[4], out_ch=in_channels[3],
                           dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p, act=self.act, cardinality=8)
        self.up3 = UpBlock(in_ch_x=in_channels[2], in_ch_g=in_channels[3], out_ch=in_channels[2],
                           dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p, act=self.act, cardinality=8)
        self.up4 = UpBlock(in_ch_x=in_channels[1], in_ch_g=in_channels[2], out_ch=in_channels[1],
                           dropout_m=self.drop_out[dropout_name], dropout_p=dropout_p, act=self.act, cardinality=8)

        self.head = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels[1], out_channels=in_channels[1], kernel_size=2, stride=2,bias=False),
            nn.GroupNorm(1, in_channels[1]),
            self.act,
            nn.Conv2d(in_channels[1], out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        )


    def forward(self, x):
        x = self.preconv(x)
        x1 = self.d1(x)    
        x2 = self.d2(x1)   
        x3 = self.d3(x2)   
        x4 = self.d4(x3)   

        x5 = self.bottleneck(x4)  # Bottleneck

        # VectorQuantizer 처리
        z_quantized, dict_loss, (commit_loss, encodings, _) =self.vq(x5)

        attn1 = self.attn1(x5, z_quantized)   
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

if __name__ == '__main__':
    model = AttnUnetV7(
        in_ch=25, 
        out_ch=1, 
        dropout_name='dropblock', 
        num_embeddings=512
    ).cuda()
    inp = torch.randn((2, 25, 256, 256)).cuda()
    output = model(inp)
    print(f"Output shape: {output['x_recon'].shape}")
    # print(f"error shape: {output['error'].shape}") # 6_2 legacy
    print(f"Dictionary Loss: {output['dictionary_loss']}")
    print(f"Commitment Loss: {output['commitment_loss']}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
