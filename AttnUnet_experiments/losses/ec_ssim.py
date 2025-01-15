import torch
import torch.nn.functional as F

class ECSSIMLoss(torch.nn.Module):
    def __init__(self, C1=0.01, C2=0.03):
        """
        간소화된 SSIM(0, y - y^)를 계산하기 위한 PyTorch 클래스
        :param C1: 안정화 상수 1
        :param C2: 안정화 상수 2
        """
        super(ECSSIMLoss, self).__init__()
        self.C1 = C1
        self.C2 = C2

    def forward(self, y, pred):
        """
        SSIM(0, y - y^) 계산
        :param y: Ground truth tensor
        :param pred: Predicted tensor
        :return: Simplified SSIM value
        """
        
        # Residual (y - pred)
        residual = y - pred
        # batch_size, channels, height, width = residual.shape
        # residual = residual.view(batch_size * channels, -1)  # (B*C, H*W)

        # Mean and variance of residual
        mean_residual = residual.mean(dim=(-2, -1), keepdim=True)  # 평균
        var_residual = residual.var(dim=(-2, -1), keepdim=True, unbiased=False)  # 분산

        # Simplified SSIM calculation
        numerator = self.C1 * self.C2
        denominator = (mean_residual ** 2 + self.C1) * (var_residual + self.C2)

        ssim_value = numerator / (denominator)
        return 1 - ssim_value.mean()

# 예제 코드
if __name__ == '__main__':
    ssim_loss_fn = ECSSIMLoss()

    # Suppose pred and target are torch tensors with shape (batch_size, channels, height, width)
    pred = torch.randn(32, 1, 512, 512, requires_grad=True).to('cuda:3')  # requires_grad 활성화
    target = torch.randn(32, 1, 512, 512).to('cuda:3')

    def loss_test(pred, target, loss_fn):
        loss = loss_fn(pred, target)
        loss.backward()  # 역전파
        print("Loss:", loss.item())

    loss_test(pred, target, ssim_loss_fn)
