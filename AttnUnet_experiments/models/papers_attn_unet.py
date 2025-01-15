import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2,bias=False)
        
    def forward(self, x):
        x = self.up(x)
        return x

class AttUNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super(AttUNet, self).__init__()
        
        self.preconv = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1,bias=False)
        
        self.enc1 = ConvBlock(64, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        
        self.bottleneck = ConvBlock(512, 1024)
        
        self.up4 = UpConvBlock(1024, 512)
        self.dec4 = ConvBlock(1024, 512)
        self.up3 = UpConvBlock(512, 256)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = UpConvBlock(256, 128)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = UpConvBlock(128, 64)
        self.dec1 = ConvBlock(128, 64)
        
        self.att4 = AttentionGate(512, 512, 256)
        self.att3 = AttentionGate(256, 256, 128)
        self.att2 = AttentionGate(128, 128, 64)
        self.att1 = AttentionGate(64, 64, 32)
        
        self.final = nn.Conv2d(64, output_channels, kernel_size=1)
        
    def forward(self, x):
        x1 = self.enc1(self.preconv(x))
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2))
        x4 = self.enc4(F.max_pool2d(x3, 2))
        
        x5 = self.bottleneck(F.max_pool2d(x4, 2))
        
        up4 = self.up4(x5)
        x4 = self.att4(g=up4, x=x4)
        d4 = self.dec4(self.crop_and_concat(up4, x4))
        
        up3 = self.up3(d4)
        x3 = self.att3(g=up3, x=x3)
        d3 = self.dec3(self.crop_and_concat(up3, x3))
        
        up2 = self.up2(d3)
        x2 = self.att2(g=up2, x=x2)
        d2 = self.dec2(self.crop_and_concat(up2, x2))
        
        up1 = self.up1(d2)
        x1 = self.att1(g=up1, x=x1)
        d1 = self.dec1(self.crop_and_concat(up1, x1))
        
        out = self.final(d1)
        return out

    def crop_and_concat(self, upsampled, bypass):
        if bypass.size()[2] > upsampled.size()[2]:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        elif upsampled.size()[2] > bypass.size()[2]:
            c = (upsampled.size()[2] - bypass.size()[2]) // 2
            upsampled = F.pad(upsampled, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)