import torch
import torch.nn as nn
import numpy as np

class IRDropMetrics(nn.Module):
    def __init__(self, top_percent=0.9,loss_with_logit=True,post_min_max=False):
        super(IRDropMetrics, self).__init__()
        self.top_percent = top_percent
        self.smooth = 1.
        self.q = top_percent
        self.loss_with_logit=loss_with_logit
        self.post_min_max = post_min_max

    @torch.no_grad()
    def forward(self, pred, target):
        # mae = torch.mean(torch.abs(pred - target))
        if not self.loss_with_logit:
            pred = torch.sigmoid(pred)

        B, C, H, W = pred.shape
        assert C == 1, "This implementation is for binary segmentation"
        # Squeeze the channel dimension
        pred = pred.squeeze(1)  # (B, H, W)
        if len(target.shape) == 4:
            target = target.squeeze(1)  # (B, H, W)

        pred = pred.view(B, -1)
        target = target.view(B, -1)

        # MAE 계산
        mae = torch.mean(torch.abs(pred - target))
        # F1 점수 계산
        pred_threshold = torch.quantile(pred.float(), self.q, dim=1, keepdim=True)
        target_threshold = torch.quantile(target.float(), self.q, dim=1, keepdim=True)
        pred = (pred>pred_threshold).float()
        target = (target>target_threshold).float()

        # Compute Dice coefficient
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        f1 = (2. * intersection + self.smooth) / (union + self.smooth)

        return {"mae": mae.item(), "f1": f1.mean().item()}

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