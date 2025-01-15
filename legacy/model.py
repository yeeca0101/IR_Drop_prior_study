import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=5, padding=2)
        
        self.max1 = nn.MaxPool2d(2, padding=1)
        self.max2 = nn.MaxPool2d(2, padding=1)
        self.max3 = nn.MaxPool2d(2, padding=1)
        
    def forward(self, x):
        x0 = torch.relu(self.conv1(x))
        x1 = self.max1(x0)
        x1 = torch.relu(self.conv2(x1))
        x2 = self.max2(x1)
        x2 = torch.relu(self.conv3(x2))
        x3 = self.max3(x2)
        return x0, x1, x2, x3

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv0 = nn.ConvTranspose2d(16, 16, kernel_size=7, padding=3)
        self.conv1 = nn.ConvTranspose2d(16, 32, kernel_size=7, padding=3)
        self.conv2 = nn.ConvTranspose2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1)
        
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.upsample3 = nn.Upsample(scale_factor=2)
    
    def forward(self, vals):
        x1 = torch.relu(self.conv0(vals[3]))
        x1 = self.upsample1(x1)
        print(x1.shape,vals[2].shape)
        x1 = torch.cat([x1, vals[2]], dim=1)
        
        x2 = torch.relu(self.conv1(x1))
        x2 = self.upsample2(x2)
        x2 = torch.cat([x2, vals[1]], dim=1)
        
        x3 = torch.relu(self.conv2(x2))
        x3 = self.upsample3(x3)
        x3 = torch.cat([x3, vals[0]], dim=1)
        
        x4 = torch.relu(self.conv3(x3))
        return x4

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        vals = self.encoder(x)
        x = self.decoder(vals)
        return x
    
def test_autoencoder():
    # 임의의 입력 데이터 생성
    batch_size = 1
    height, width = 256, 256  # 데이터셋의 크기와 맞춰줌
    input_data = torch.randn(batch_size, 3, height, width)  # (batch_size, channels, height, width)

    # Autoencoder 모델 생성
    model = AutoEncoder()

    # 모델의 출력 생성
    output_data = model(input_data)

    # 결과 출력
    print("Input shape: ", input_data.shape)  # (batch_size, 3, 820, 821)
    print("Output shape: ", output_data.shape)  # (batch_size, 1, 820, 821)

    # 모델의 파라미터 수 확인
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")


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

def test_attnunet():
    # 임의의 입력 데이터 생성
    batch_size = 1
    height, width = 256, 256  # 데이터셋의 크기와 맞춰줌
    input_data = torch.randn(batch_size, 3, height, width)  # (batch_size, channels, height, width)

    # Autoencoder 모델 생성
    model = AttUNet()
    model.cuda()
    # 모델의 출력 생성
    output_data = model(input_data.cuda())

    # 결과 출력
    print("Input shape: ", input_data.shape)  # (batch_size, 3, 820, 821)
    print("Output shape: ", output_data.shape)  # (batch_size, 1, 820, 821)

    # 모델의 파라미터 수 확인
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

# 테스트 함수 실행
# test_autoencoder()
# test_attnunet()