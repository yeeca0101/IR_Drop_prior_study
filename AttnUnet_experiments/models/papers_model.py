import torch
from torch import nn
from torch.nn import functional as F


class PreConv(nn.Module):
    def __init__(self, in_ch,out_ch,stride=2,padding=0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=2,stride=stride,padding=padding,bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv(x)
        x = self.act(self.bn(x))

        return x


class DoubleConv(nn.Module):
    def __init__(self, in_ch,out_ch) -> None:
        super().__init__()
        self.dconv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        out = self.dconv(x)
        return out

class DownBlock(nn.Module):
    def __init__(self, in_ch,out_ch) -> None:
        super().__init__()
        self.conv = DoubleConv(in_ch,out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        x1 = self.conv(x)
        x2 = self.pool(x1)
        return x2

class AttentionGate(nn.Module):
    def __init__(self,in_ch_x,in_ch_g,out_ch,concat=True) -> None:
        super().__init__()
        self.concat = concat
        self.act = nn.ReLU(inplace=True)
        if concat:
            self.w_x_g = nn.Conv2d(in_ch_x+in_ch_g,out_ch,kernel_size=1,stride=1,padding=0,bias=True)
        else:
            self.w_x = nn.Conv2d(in_ch_x,out_ch,kernel_size=1,stride=1,padding=0,bias=False)
            self.w_g = nn.Conv2d(in_ch_g,out_ch,kernel_size=1,stride=1,padding=0,bias=True)
        
        self.attn = nn.Conv2d(out_ch,out_ch,kernel_size=1,padding=0,bias=True)

    def forward(self,x,g):
        res = x
        if self.concat:
            xg = torch.cat([x,g],dim=1) # B (x_c + g_c) H W
            xg = self.w_x_g(xg)
        else:
            xg = self.w_x(x) + self.w_g(g)
        
        xg = self.act(xg)
        attn = torch.sigmoid(self.attn(xg))

        out = res*attn
        return out

class UpBlock(nn.Module):
    def __init__(self,in_ch_x,in_ch_g,out_ch):
        super(UpBlock,self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        # self.conv1 = nn.Conv2d(in_ch_x+in_ch_g,out_ch,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv = DoubleConv(in_ch_x+in_ch_g,out_ch)

    def forward(self,attn,x):
        x = torch.cat([attn,x],dim=1)
        x = self.up(x)
        x = self.conv(x)
    
        return x
    

class AttnUnetBase(nn.Module):
    def __init__(self,in_ch=12,out_ch=1) -> None:
        super().__init__()

        in_channels = [12,32,64,128,256,512]
        self.concat = True
        self.preconv = PreConv(in_ch,out_ch=in_channels[0])
        self.d1 = DownBlock(in_ch=in_channels[0],out_ch=in_channels[1])
        self.d2 = DownBlock(in_ch=in_channels[1],out_ch=in_channels[2])
        self.d3 = DownBlock(in_ch=in_channels[2],out_ch=in_channels[3])
        self.d4 = DownBlock(in_ch=in_channels[3],out_ch=in_channels[4])

        self.bottleneck = DoubleConv(in_ch=in_channels[4],out_ch=in_channels[5])

        self.attn1 = AttentionGate(in_ch_x=in_channels[4],in_ch_g=in_channels[5],out_ch=in_channels[4],concat=self.concat)
        self.attn2 = AttentionGate(in_ch_x=in_channels[3],in_ch_g=in_channels[4],out_ch=in_channels[3],concat=self.concat)
        self.attn3 = AttentionGate(in_ch_x=in_channels[2],in_ch_g=in_channels[3],out_ch=in_channels[2],concat=self.concat)
        self.attn4 = AttentionGate(in_ch_x=in_channels[1],in_ch_g=in_channels[2],out_ch=in_channels[1],concat=self.concat)

        self.up1 = UpBlock(in_ch_x=in_channels[4],in_ch_g=in_channels[5],out_ch=in_channels[4])
        self.up2 = UpBlock(in_ch_x=in_channels[3],in_ch_g=in_channels[4],out_ch=in_channels[3])
        self.up3 = UpBlock(in_ch_x=in_channels[2],in_ch_g=in_channels[3],out_ch=in_channels[2])
        self.up4 = UpBlock(in_ch_x=in_channels[1],in_ch_g=in_channels[2],out_ch=in_channels[1])

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels[1],out_ch,kernel_size=1,stride=1,padding=0,bias=True)
        )

    def forward(self,x):
        x = self.preconv(x)
        x1 = self.d1(x)    # B 32 128 128
        x2 = self.d2(x1)   # B 64 64 64
        x3 = self.d3(x2)   # B 128 32 32
        x4 = self.d4(x3)   # B 256 16 16 

        x5= self.bottleneck(x4) # g B 512 16 16 
        
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
    m = mokup(3,12)
    inp = torch.randn((1,3,512,512))
    print(m(inp).shape)
    total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")