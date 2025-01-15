'''
v3 : add input channel shuffling 
v4 : MLPMixerBlock, ASPP

'''
from .parts.v4_parts import *

def random_channel_shuffle(x):
    channels = x.size(1)
    fixed_channels = x[:, :3, :, :]
    
    remaining_channels = x[:, 3:, :, :]
    try:
        remaining_indices = torch.randperm(channels - 3)
    except:
        print(x.size(1))
    shuffled_remaining = remaining_channels[:, remaining_indices, :, :]
    
    return torch.cat([fixed_channels, shuffled_remaining], dim=1)

class PreConv(nn.Module):
    def __init__(self, in_ch,out_ch,stride=2,padding=0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=2,stride=stride,padding=padding,bias=False)
        self.bn = nn.GroupNorm(1,out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv(x)
        x = self.act(self.bn(x))

        return x


########################################################################################### 

    
class AttnUnetV4(nn.Module):
    def __init__(self,in_ch=12,out_ch=1,dropout_name='',dropout_p=0.5,
                 num_head=4) -> None:
        super().__init__()

        in_channels = [12,32,64,128,256,512]
        self.concat = True
        self.drop_out = {
            'nn.dropout': nn.Dropout2d,
            'dropblock': DropBlock2D
        }
        self.preconv = PreConvWithChannelAttention(in_ch, out_ch=in_channels[0],dropout_m=self.drop_out[dropout_name],dropout_p=dropout_p,num_heads=num_head)

        self.d1 = DownBlock(in_ch=in_channels[0],out_ch=in_channels[1],dropout_m=self.drop_out[dropout_name],dropout_p=dropout_p)
        self.d2 = DownBlock(in_ch=in_channels[1],out_ch=in_channels[2],dropout_m=self.drop_out[dropout_name],dropout_p=dropout_p)
        self.d3 = DownBlock(in_ch=in_channels[2],out_ch=in_channels[3],dropout_m=self.drop_out[dropout_name],dropout_p=dropout_p)
        self.d4 = DownBlock(in_ch=in_channels[3],out_ch=in_channels[4],dropout_m=self.drop_out[dropout_name],dropout_p=dropout_p)

        self.bottleneck = DoubleConv(in_ch=in_channels[4],out_ch=in_channels[5],dropout_m=self.drop_out[dropout_name],dropout_p=dropout_p)
        # self.bottleneck = DoubleConv(in_channels=in_channels[4],out_channels=in_channels[5])

        self.attn1 = AttentionGate(in_ch_x=in_channels[4],in_ch_g=in_channels[5],out_ch=in_channels[4],concat=self.concat)
        self.attn2 = AttentionGate(in_ch_x=in_channels[3],in_ch_g=in_channels[4],out_ch=in_channels[3],concat=self.concat)
        self.attn3 = AttentionGate(in_ch_x=in_channels[2],in_ch_g=in_channels[3],out_ch=in_channels[2],concat=self.concat)
        self.attn4 = AttentionGate(in_ch_x=in_channels[1],in_ch_g=in_channels[2],out_ch=in_channels[1],concat=self.concat)

        self.up1 = UpBlock(in_ch_x=in_channels[4],in_ch_g=in_channels[5],out_ch=in_channels[4],dropout_m=self.drop_out[dropout_name],dropout_p=dropout_p)
        self.up2 = UpBlock(in_ch_x=in_channels[3],in_ch_g=in_channels[4],out_ch=in_channels[3],dropout_m=self.drop_out[dropout_name],dropout_p=dropout_p)
        self.up3 = UpBlock(in_ch_x=in_channels[2],in_ch_g=in_channels[3],out_ch=in_channels[2],dropout_m=self.drop_out[dropout_name],dropout_p=dropout_p)
        self.up4 = UpBlock(in_ch_x=in_channels[1],in_ch_g=in_channels[2],out_ch=in_channels[1],dropout_m=self.drop_out[dropout_name],dropout_p=dropout_p)

        self.head = nn.Sequential(
            nn.Conv2d(in_channels[1],out_ch,kernel_size=1,stride=1,padding=0,bias=False),
        )

    def forward(self,x):
        # if self.training:
        #     x = random_channel_shuffle(x)

        x = self.preconv(x)     # B 32 128 128
        x1 = self.d1(x)         # B 32 64 64
        x2 = self.d2(x1)        # B 64 32 32
        x3 = self.d3(x2)        # B 128 16 16 
        x4 = self.d4(x3)        # B 256 8 8 

        x5= self.bottleneck(x4) # g B 512 8 8 

        attn1 = self.attn1(x4,x5)   # B 256 16 16
        up1 = self.up1(attn1,x5)    # B 256 32 32

        attn2 = self.attn2(x3,up1)  # B 128 32 32
        up2 = self.up2(attn2,up1)   # B 128 64 64

        attn3 = self.attn3(x2,up2)  # B 64 128 128
        up3 = self.up3(attn3,up2)   # B 64 128 128

        attn4 = self.attn4(x1,up3)  # B 32 256 256
        up4 = self.up4(attn4,up3)   # B 32 256 256

        return self.head(up4) # B 1 512 512

if __name__ == '__main__':
    m = AttnUnetV4(12,1,dropout_name='dropblock').to('cuda:0')
    inp = torch.randn((1,12,512,512)).to('cuda:0')
    print(m(inp).shape)
    total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")