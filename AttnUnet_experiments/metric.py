import torch
import torch.nn as nn
import numpy as np

from loss import SSIMLoss
class IRDropMetrics(nn.Module):
    def __init__(self, top_percent=0.9, loss_with_logit=True, post_min_max=False):
        super(IRDropMetrics, self).__init__()
        self.top_percent = top_percent
        self.smooth = 1.
        self.q = top_percent
        self.loss_with_logit = loss_with_logit
        self.post_min_max = post_min_max

        # SSIM 계산을 위한 모듈 (binary segmentation이므로 channel=1, data_range=1.0 사용)
        self.ssim_loss = SSIMLoss(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)

    @torch.no_grad()
    def forward(self, pred, target):
        # --- SSIM 계산 ---
        # SSIM은 (B, C, H, W) 형태에서 계산되어야 하므로, 원본 텐서를 사용합니다.
        # 타겟 텐서도 (B, C, H, W) 형태여야 합니다.
        target_for_ssim = target if target.ndim == 4 else target.unsqueeze(1)
        ssim_val = 1 - self.ssim_loss(pred, target_for_ssim)


        B, C, H, W = pred.shape
        assert C == 1, "This implementation is for binary segmentation"

        # 평탄화하여 (B, N) 형태로 변경
        pred = pred.view(B, -1)
        target = target.view(B, -1)

        # MAE 계산
        mae = torch.mean(torch.abs(pred - target))

        # F1 점수를 위한 threshold 계산
        pred_threshold = torch.quantile(pred.float(), self.q, dim=1, keepdim=True)
        target_threshold = torch.quantile(target.float(), self.q, dim=1, keepdim=True)
        pred_bin = (pred > pred_threshold).float()
        target_bin = (target > target_threshold).float()

        # Dice 계수 (F1) 계산
        intersection = (pred_bin * target_bin).sum(dim=1)
        union = pred_bin.sum(dim=1) + target_bin.sum(dim=1)
        f1 = (2. * intersection + self.smooth) / (union + self.smooth)

        return {"mae": mae.item(), "f1": f1.mean().item(), "ssim": ssim_val.item()}
    
    @torch.no_grad()
    def compute_metrics(self, pred, target):
        if self.post_min_max: pred = min_max_norm(pred)
        return self.forward(pred, target)

class ValMetric(nn.Module):
    def __init__(self, top_percent=0.1):
        super(ValMetric, self).__init__()
        self.ir_drop_metrics = IRDropMetrics(top_percent)

    def forward(self, pred, target):
        metrics = self.ir_drop_metrics(pred, target)
        # MAE와 F1 점수의 조화 평균을 반환
        harmonic_mean = 2 / (1/metrics['mae'] + 1/metrics['f1'])
        return harmonic_mean

def min_max_norm(x):
    return (x-x.min())/(x.max()-x.min())

# 사용 예시
if __name__ == "__main__":
    # 가상의 예측값과 실제값 생성
    pred = torch.randn((4,1,32,32),requires_grad=True)
    target  =torch.randint(0, 2, (4, 32, 32)).float()

    # 메트릭 계산
    ir_metrics = IRDropMetrics()
    val_metric = ValMetric()

    print("개별 메트릭:", ir_metrics.compute_metrics(pred, target))
    print("검증 메트릭:", val_metric(pred, target))