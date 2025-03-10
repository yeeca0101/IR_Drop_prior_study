import torch
from torch import nn

class AffineCoupling(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        """
        입력 텐서를 채널 기준으로 두 부분으로 나눈 후, 
        한쪽(x1)으로 scale(s)와 shift(t)를 예측하여 x2에 affine 변환을 수행.
        in_channels는 짝수여야 합니다.
        """
        super(AffineCoupling, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels // 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        h = self.net(x1)
        s, t = h.chunk(2, dim=1)
        s = torch.tanh(s)
        if not reverse:
            y2 = x2 * torch.exp(s) + t
        else:
            y2 = (x2 - t) * torch.exp(-s)
        return torch.cat([x1, y2], dim=1)

class Permutation(nn.Module):
    def __init__(self, num_channels):
        """
        채널 순서를 랜덤하게 섞는 layer. inverse 시 원래 순서로 복원.
        """
        super(Permutation, self).__init__()
        perm = torch.randperm(num_channels)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", torch.argsort(perm))
    
    def forward(self, x, reverse=False):
        if not reverse:
            return x[:, self.perm, ...]
        else:
            return x[:, self.inv_perm, ...]

class InvertibleNormalization(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=4):
        """
        RealNVP 스타일의 INN 모델.
        여러 coupling layer와 permutation layer를 쌓고, 마지막에 sigmoid를 적용하여
        출력이 [0,1] 범위가 되도록 강제합니다.
        """
        super(InvertibleNormalization, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(Permutation(in_channels))
            layers.append(AffineCoupling(in_channels, hidden_channels))
        self.layers = nn.ModuleList(layers)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        z = x
        for layer in self.layers:
            z = layer(z, reverse=False)
        # [0,1] 범위 강제
        z = self.sigmoid(z)
        return z
    
    def inverse(self, z):
        # 역변환을 위해 sigmoid의 역함수 (logit)를 적용합니다.
        eps = 1e-6
        z_clamped = torch.clamp(z, eps, 1 - eps)
        z_inv = torch.log(z_clamped / (1 - z_clamped))
        x = z_inv
        for layer in reversed(self.layers):
            x = layer(x, reverse=True)
        return x

def example_usage():
    # 예제 입력 생성: 배치 크기 1, 채널 4 (반드시 짝수), 높이와 너비는 8
    batch_size = 1
    in_channels = 4
    height = 8
    width = 8
    x = torch.randn(batch_size, in_channels, height, width)
    print("입력 텐서 (x):")
    print(x.max(), x.min(), x.mean())
    
    # InvertibleNormalization 모델 생성 (hidden_channels는 8로 설정)
    hidden_channels = 8
    model = InvertibleNormalization(in_channels=in_channels, hidden_channels=hidden_channels)
    
    # Forward pass 실행
    z = model(x)
    print("\nForward pass 결과 (z):")
    print(z.max(), z.min(), z.mean())

    # Inverse pass 실행하여 원본 입력 복원
    x_recovered = model.inverse(z)
    print("\nInverse pass 결과 (복원된 x):")
    print(x_recovered.max().item(), x_recovered.min().item(), x_recovered.mean().item())
    
    # 원본 x와 복원된 x의 차이 계산
    difference = torch.abs(x - x_recovered)
    print("\n원본 x와 복원된 x의 차이:")
    print(difference.mean().item())

if __name__ == "__main__":
    example_usage()
