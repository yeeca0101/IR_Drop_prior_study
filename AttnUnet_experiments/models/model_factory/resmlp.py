import torch
import torch.nn as nn

# 임시 Patch_projector 정의 (입력 이미지를 패치로 나눈 뒤, 임베딩)
class Patch_projector(nn.Module):
    def __init__(self, in_ch=3,patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, nb_patches, embed_dim)
        return x

# No norm layer
class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        return self.alpha * x + self.beta

# MLP on channels
class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# ResMLP blocks: a linear layer between patches + an MLP to process channels independently
class ResMLP_BLocks(nn.Module):
    def __init__(self, nb_patches, dim, layerscale_init):
        super().__init__()
        self.affine_1 = Affine(dim)
        self.affine_2 = Affine(dim)
        self.linear_patches = nn.Linear(nb_patches, nb_patches)  # Linear layer on patches
        self.mlp_channels = Mlp(dim)  # MLP on channels
        self.layerscale_1 = nn.Parameter(layerscale_init * torch.ones((dim)))  # LayerScale parameters
        self.layerscale_2 = nn.Parameter(layerscale_init * torch.ones((dim)))
        
    def forward(self, x):
        res_1 = self.linear_patches(self.affine_1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.layerscale_1 * res_1
        res_2 = self.mlp_channels(self.affine_2(x))
        x = x + self.layerscale_2 * res_2
        return x

# ResMLP model: stacking the full network
class ResMLP_models(nn.Module):
    def __init__(self, in_ch, dim, depth, nb_patches, layerscale_init, num_classes):
        super().__init__()
        self.patch_projector = Patch_projector(in_ch=in_ch, patch_size=16, embed_dim=dim)
        self.blocks = nn.ModuleList([
            ResMLP_BLocks(nb_patches, dim, layerscale_init)
            for _ in range(depth)
        ])
        self.affine = Affine(dim)
        self.linear_classifier = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_projector(x)
        for blk in self.blocks:
            print(x.shape)
            x = blk(x)
        x = self.affine(x)
        print(x.shape)
        x = x.mean(dim=1).reshape(B, -1)  # average pooling
        return self.linear_classifier(x)

# 테스트 코드
if __name__ == '__main__':
    # 임의 입력 텐서 (배치 크기 2, 채널 3, 높이 224, 너비 224)
    B, C, H, W = 2, 25, 256, 256
    dummy_input = torch.randn(B, C, H, W)
    
    # 모델 파라미터 설정
    dim = 768
    depth = 12
    layerscale_init = 1e-5
    num_classes = 2
    # nb_patches 계산: (H // patch_size) * (W // patch_size)
    nb_patches = (H // 16) * (W // 16)
    
    # 모델 생성
    model = ResMLP_models(C,dim, depth, nb_patches, layerscale_init, num_classes)
    
    # 모델 forward pass 테스트
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 예상 출력: (2, 1000)
