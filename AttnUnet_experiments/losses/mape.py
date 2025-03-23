import torch
import torch.nn as nn
from torch.nn import functional as F

class IRMAPE(nn.Module):
    def __init__(self, th=0.9, epsilon=1e-8):
        super().__init__()
        self.th = th
        self.epsilon = epsilon

    def forward(self, pred, target):
        # 0이 아닌 픽셀에 대해서만 임계값 계산
        non_zero_mask = target != 0
        if non_zero_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        non_zero_target = target[non_zero_mask]
        thresholds = torch.quantile(non_zero_target, self.th,keepdim=True)
        
        # 임계값 이상이고 0이 아닌 픽셀에 대해 마스크 생성
        mask = (target >= thresholds) & non_zero_mask
        
        # 마스크가 적용된 영역에 대해서만 오차 계산
        pred_masked = pred[mask]
        target_masked = target[mask]
        
        absolute_percentage_error = torch.abs((target_masked - pred_masked) / (torch.abs(target_masked) + self.epsilon))
        
        valid_count = mask.sum()
        
        if valid_count == 0:
            return torch.tensor(0.0, device=pred.device)
        
        return torch.sum(absolute_percentage_error) / valid_count


    
class WeightedMAPELoss(nn.Module):
    def __init__(self, 
                 epsilon=1e-6, 
                 ignore_index=None, 
                 threshold_mode="max",   # "max" 또는 "quantile"
                 threshold_factor=0.9,   # max 버전일 경우: target의 max에 곱해지는 factor
                 weight_factor=2.0       # threshold보다 큰 영역에 곱할 배율 (1.0이면 변화없음)
                ):
        super(WeightedMAPELoss, self).__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self.threshold_mode = threshold_mode
        self.threshold_factor = threshold_factor
        self.weight_factor = weight_factor

    def forward(self, pred, target):
        """
        pred, target: (B, C, H, W) 형태이며, C==1라고 가정합니다.
        ignore_index가 설정된 경우 해당 위치는 loss 계산에서 제외합니다.
        """
        # 채널 차원이 1이면 squeeze
        if pred.shape[1] == 1:
            pred = pred.squeeze(1)  # (B, H, W)
        if target.shape[1] == 1:
            target = target.squeeze(1)  # (B, H, W)

        B = target.shape[0]
        # target을 (B, H*W)로 flatten하여 샘플별 threshold 계산에 사용
        target_flat = target.view(B, -1)
        
        # 각 샘플별 threshold 계산 (ignore_index 고려)
        if self.threshold_mode == "max":
            if self.ignore_index is not None:
                # 유효한 값만 고려하여 최대값 계산: 무시할 값은 매우 작은 수(-inf)로 대체
                valid_mask = (target_flat != self.ignore_index)
                target_valid = target_flat.clone()
                target_valid[~valid_mask] = -float('inf')
                th, _ = target_valid.max(dim=1, keepdim=True)
            else:
                th, _ = target_flat.max(dim=1, keepdim=True)
            th = th * self.threshold_factor
        elif self.threshold_mode == "quantile":
            # 각 샘플별 quantile threshold 계산 (ignore_index 제외)
            th_list = []
            for i in range(B):
                sample_vals = target_flat[i]
                if self.ignore_index is not None:
                    sample_vals = sample_vals[sample_vals != self.ignore_index]
                if sample_vals.numel() > 0:
                    th_list.append(torch.quantile(sample_vals, self.threshold_factor))
                else:
                    th_list.append(torch.tensor(0.0, device=target.device))
            th = torch.stack(th_list).view(B, 1)
        else:
            raise ValueError("Invalid threshold_mode. Use 'max' or 'quantile'.")

        # threshold를 (B, 1, 1)로 reshape하여 각 샘플의 모든 픽셀에 적용
        th_reshaped = th.view(B, 1, 1)
        # target이 threshold보다 큰 경우에 weight_factor 적용:
        # 기본 weight는 1.0이지만, weight_factor가 1.0보다 크면 해당 영역에 더 큰 loss penalty를 줍니다.
        weight_mask = (target > th_reshaped).float() * (self.weight_factor - 1.0) + 1.0

        # MAPE 계산 (절대 백분율 오차)
        error = torch.abs((target - pred) / (target + self.epsilon))
        weighted_error = error * weight_mask

        # ignore_index가 설정된 경우 해당 위치는 계산에서 제외
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            loss = (weighted_error * valid_mask).sum() / (valid_mask.sum() + self.epsilon)
        else:
            loss = weighted_error.mean()

        return loss

def test_irmape():
    irmape_loss = IRMAPE(th=0.8)

    # 기본 테스트
    pred = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    target = torch.tensor([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    loss = irmape_loss(pred, target)
    print(f"Basic test loss: {loss.item()}")
    assert 0 < loss.item() < 1, "Basic loss calculation failed"

    # 예측값이 0일 때 테스트
    pred = torch.zeros((2, 3))
    target = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    loss = irmape_loss(pred, target)
    print(f"Zero prediction test loss: {loss.item()}")
    assert loss.item() > 0, "Loss should be positive for zero predictions"

    # 예측값과 타겟값이 동일할 때 테스트
    pred = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    target = pred.clone()
    loss = irmape_loss(pred, target)
    print(f"Equal prediction and target test loss: {loss.item()}")
    assert loss.item() == 0, "Loss should be zero when pred equals target"

    # 임계값 효과 테스트
    pred = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    target = torch.tensor([[0.1, 0.2, 10.0], [0.4, 0.5, 10.0]])
    loss = irmape_loss(pred, target)
    print(f"Threshold effect test loss: {loss.item()}")
    assert loss.item() > 0, "Threshold should affect the loss calculation"

    # 모든 값이 임계값 미만일 때 테스트
    irmape_loss_all_below = IRMAPE(th=1.0)
    pred = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    target = torch.tensor([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    loss = irmape_loss_all_below(pred, target)
    print(f"All below threshold test loss: {loss.item()}")
    assert torch.isinf(loss), "Loss should be inf when all values are below threshold"

    print("All tests passed successfully!")

if __name__ == "__main__":
    test_irmape()