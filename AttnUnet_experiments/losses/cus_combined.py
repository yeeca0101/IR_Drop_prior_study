import torch
import torch.nn as nn
import torch.nn.functional as F

class IRDropLoss(nn.Module):
    def __init__(self, 
                 weight_tversky=1.0, 
                 weight_mae=1.0, 
                 weight_ssim=0.5, 
                 weight_max=0.1,
                 tversky_alpha=0.3, 
                 tversky_beta=0.7, 
                 smooth=1e-6, 
                 lse_alpha=50,
                 window_size=11,
                 window_sigma=1.5,
                 data_range=1.0):
        """
        복합 손실 함수 클래스
        
        Args:
            weight_tversky (float): Tversky loss 항의 가중치 (Dice loss 대체).
            weight_mae (float): L1 (MAE) loss 항의 가중치.
            weight_ssim (float): SSIM loss 항의 가중치 (1 - SSIM 값을 사용).
            weight_max (float): smooth max penalty 항의 가중치.
            tversky_alpha (float): Tversky loss의 FP 가중치.
            tversky_beta (float): Tversky loss의 FN 가중치.
            smooth (float): 수치적 안정성을 위한 작은 값.
            lse_alpha (float): log-sum-exp 연산의 스케일 인자 (alpha가 클수록 max에 가까워짐).
            window_size (int): SSIM 계산에 사용할 윈도우 크기.
            window_sigma (float): SSIM 계산에 사용할 윈도우의 sigma.
            data_range (float): 입력 데이터의 범위 (대부분 1.0으로 normalize됨).
        """
        super(IRDropLoss, self).__init__()
        self.weight_tversky = weight_tversky
        self.weight_mae = weight_mae
        self.weight_ssim = weight_ssim
        self.weight_max = weight_max
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.smooth = smooth
        self.lse_alpha = lse_alpha
        self.window_size = window_size
        self.window_sigma = window_sigma
        self.data_range = data_range

    def forward(self, pred, target):
        """
        pred, target: (B, 1, H, W) 텐서
        """
        # 1. Tversky (Dice) Loss
        tversky_loss = self.tversky_loss(pred, target)
        
        # 2. L1 (MAE) Loss
        mae_loss = F.l1_loss(pred, target)
        
        # 3. SSIM Loss (여기서는 단일 scale SSIM 사용, 필요시 Multi-Scale SSIM으로 확장 가능)

        
        # 4. Smooth Max Loss: log-sum-exp로 예측 map의 최대값이 1에 가깝도록 유도
        softmax_max = self.soft_max(pred)
        max_loss = (softmax_max - 1) ** 2
        
        # 각 항을 가중합하여 최종 loss 계산
        loss = (self.weight_tversky * tversky_loss +
                self.weight_mae * mae_loss +
                self.weight_max * max_loss)
        
        return loss

    def tversky_loss(self, pred, target):
        """
        Tversky loss: Dice loss의 일반화 형태.
        """
        B = pred.size(0)
        # 예측과 타겟을 flatten
        pred_flat = pred.view(B, -1)
        target_flat = target.view(B, -1)
        
        # True Positives, False Positives, False Negatives 계산
        TP = (pred_flat * target_flat).sum(dim=1)
        FP = ((1 - target_flat) * pred_flat).sum(dim=1)
        FN = (target_flat * (1 - pred_flat)).sum(dim=1)
        
        tversky = (TP + self.smooth) / (TP + self.tversky_alpha * FP + self.tversky_beta * FN + self.smooth)
        return 1 - tversky.mean()

    def soft_max(self, pred):
        """
        log-sum-exp로 부드러운 최대값 계산.
        """
        B = pred.size(0)
        pred_flat = pred.view(B, -1)
        # log-sum-exp 연산: alpha가 클수록 실제 max에 가까워짐
        soft_max = torch.log(torch.sum(torch.exp(self.lse_alpha * pred_flat), dim=1, keepdim=True)) / self.lse_alpha
        # 배치 내 평균값을 반환 (또는 각 샘플 별로 개별 loss를 줄 수도 있음)
        return soft_max.mean()
    
    