import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # 인코더 부분
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.enc4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # 바텀 레이어
        self.bottom = self.conv_block(512, 1024)

        # 디코더 부분
        self.upconv4 = self.up_conv(1024, 512)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = self.up_conv(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = self.up_conv(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = self.up_conv(128, 64)
        self.dec1 = self.conv_block(128, 64)

        # 출력 레이어
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        return block

    def up_conv(self, in_channels, out_channels):
        up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        return up

    def forward(self, x):
        # 인코더
        e1 = self.enc1(x)
        e1p = self.pool1(e1)

        e2 = self.enc2(e1p)
        e2p = self.pool2(e2)

        e3 = self.enc3(e2p)
        e3p = self.pool3(e3)

        e4 = self.enc4(e3p)
        e4p = self.pool4(e4)

        # 바텀 레이어
        b = self.bottom(e4p)

        # 디코더
        d4 = self.upconv4(b)
        # 필요한 경우 e4를 크기에 맞게 크롭
        if d4.size() != e4.size():
            diffY = e4.size()[2] - d4.size()[2]
            diffX = e4.size()[3] - d4.size()[3]
            d4 = nn.functional.pad(d4, [diffX // 2, diffX - diffX // 2,
                                        diffY // 2, diffY - diffY // 2])
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        if d3.size() != e3.size():
            diffY = e3.size()[2] - d3.size()[2]
            diffX = e3.size()[3] - d3.size()[3]
            d3 = nn.functional.pad(d3, [diffX // 2, diffX - diffX // 2,
                                        diffY // 2, diffY - diffY // 2])
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        if d2.size() != e2.size():
            diffY = e2.size()[2] - d2.size()[2]
            diffX = e2.size()[3] - d2.size()[3]
            d2 = nn.functional.pad(d2, [diffX // 2, diffX - diffX // 2,
                                        diffY // 2, diffY - diffY // 2])
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        if d1.size() != e1.size():
            diffY = e1.size()[2] - d1.size()[2]
            diffX = e1.size()[3] - d1.size()[3]
            d1 = nn.functional.pad(d1, [diffX // 2, diffX - diffX // 2,
                                        diffY // 2, diffY - diffY // 2])
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)

        # 출력
        out = self.final_conv(d1)

        return out

# model = UNet()
# inp = torch.randn((1,3,32,32))
# print(model(inp).shape)