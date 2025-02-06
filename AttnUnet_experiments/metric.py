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
        self.ssim_loss = SSIMLoss(
            data_range=1.0,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            channel=1,
            spatial_dims=2,
            K=(0.01, 0.03),
            nonnegative_ssim=False,
        )

    @torch.no_grad()
    def forward(self, pred, target):
        # --- SSIM 계산 ---
        # SSIM은 (B, C, H, W) 형태에서 계산되어야 하므로, 원본 텐서를 사용합니다.
        # 만약 loss_with_logit가 True면 pred는 logits이므로, SSIM 계산을 위해 sigmoid 적용
        if self.loss_with_logit:
            pred_prob = torch.sigmoid(pred)
        else:
            pred_prob = pred

        # 타겟 텐서도 (B, C, H, W) 형태여야 합니다.
        target_for_ssim = target if target.ndim == 4 else target.unsqueeze(1)
        # SSIMLoss의 forward()는 1 - ssim을 반환하므로, 1 - (1 - ssim) = ssim 을 얻으려면:
        ssim_val = 1 - self.ssim_loss(pred_prob, target_for_ssim)
        # ssim_val가 스칼라인 경우 item()으로 추출 (배치 평균된 값)
        ssim_val = ssim_val.item() if ssim_val.dim() == 0 else ssim_val.mean().item()

        # --- MAE 및 F1 계산 ---
        # 기존 코드에서는 logits/확률에 따라 sigmoid 적용 여부가 결정됩니다.
        if not self.loss_with_logit:
            # 만약 loss_with_logit가 False면 pred는 이미 logits이 아닌 확률이므로 sigmoid 적용
            pred = torch.sigmoid(pred)
        B, C, H, W = pred.shape
        assert C == 1, "This implementation is for binary segmentation"

        # 채널 차원 제거하여 (B, H, W)로 변환
        pred = pred.squeeze(1)
        if len(target.shape) == 4:
            target = target.squeeze(1)

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

        return {"mae": mae.item(), "f1": f1.mean().item(), "ssim": ssim_val}
    
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