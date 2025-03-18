import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        assert C == 1, "This implementation is for binary segmentation"
        
        pred = torch.sigmoid(pred) # 이거 쓰면 예제에서 0.655가 최대 coeff
        
        # Flatten pred and target
        pred = pred.view(B, -1)
        target = target.view(B, -1)
        
        # Compute Dice coefficient
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice = 1. - (2. * intersection + self.smooth) / (union + self.smooth)
        return dice.mean()
    
    def coefficient(self,pred,target):
        return 1. -self.forward(pred,target)

class DiceLossforSeg(nn.Module):
    def __init__(self, smooth=1e-6):  # Fixed: 1e-6 instead of 1-6
        super(DiceLossforSeg, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        assert C == 1, "This implementation is for binary segmentation"
        
        pred = torch.sigmoid(pred)
        
        # Process target: reduce channel dim and scale by 0.9
        target = (target > target.max(dim=1,keepdim=True)[0]*0.9).float()   # [0] gets values, [1] gets indices
        
        # Flatten tensors
        pred = pred.view(B, -1)
        target = target.view(B, -1)
        # Calculate dice score
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice = 1. - (2. * intersection + self.smooth) / (union + self.smooth)
        return dice.mean()
    
    def coefficient(self, pred, target):
        return 1. - self.forward(pred, target)
    

class QuantileDiceLoss(nn.Module):
    def __init__(self, q=0.9, alpha=0.5, smooth=1e-5):
        super().__init__()
        self.q = q
        self.alpha = alpha  # scale loss 가중치
        self.smooth = smooth

    def forward(self, pred, target):
        # 분위수 계산 (미분 가능한 근사)
        target_qt = self.diff_quantile(target, self.q)  # (B,1)
        pred_qt = self.diff_quantile(pred, self.q)      # (B,1)
        
        # 이진 마스크 생성
        target_bin = (target > target_qt).float()
        pred_bin = (pred > pred_qt).float()
        
        # Dice Loss
        intersection = (pred_bin * target_bin).sum(dim=(1,2,3))
        union = pred_bin.sum(dim=(1,2,3)) + target_bin.sum(dim=(1,2,3))
        dice_loss = 1 - (2*intersection + self.smooth)/(union + self.smooth)
        
        # 스케일 정합 Loss
        scale_loss = F.l1_loss(pred_qt, target_qt)  # 예측분위수와 타겟분위수 정합
        
        return dice_loss.mean() + self.alpha * scale_loss

    def diff_quantile(self, x, q):
        """미분 가능한 분위수 근사"""
        n = x.shape[1]
        sorted_x = torch.sort(x, dim=1)[0]  # (B,C,H,W) → 각 채널별 정렬
        index = torch.tensor(q * (n - 1))
        lower = torch.floor(index).long()
        upper = torch.ceil(index).long()
        
        # 선형 보간
        return sorted_x[:,lower] + (index - lower)*(sorted_x[:,upper] - sorted_x[:,lower])


class CustomDiceLoss(nn.Module):
    def __init__(self, q=0.9,smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.q = q

    def forward(self, pred, target):
        # 타겟 동적 임계값 계산 (B,1)
        threshold = target.max(dim=1, keepdim=True)[0] * self.q
        
        # 미분 가능한 이진화
        pred_probs = torch.sigmoid(pred)  # [0,1] 범위 변환
        target_bin = (target > threshold).float()
        
        # 초점 영역 마스크 생성
        focus_mask = (target >= threshold * 0.8).float()  # 임계값 주변 확장
        
        # 가중치 적용 Dice 계산
        intersection = (pred_probs * target_bin * focus_mask).sum(dim=(1,2,3))
        union = (pred_probs + target_bin).sum(dim=(1,2,3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
    
class IRDiceLoss(nn.Module):
    def __init__(self, ignore_index=None, smooth=1e-6,q=0.9):
        super(IRDiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.q = q

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        assert C == 1, "This implementation is for binary segmentation"

        # Squeeze the channel dimension
        pred = pred.squeeze(1)  # (B, H, W)
        if len(target.shape) == 4:
            target = target.squeeze(1)  # (B, H, W)

        # Create mask for ignored index
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float()
        else:
            mask = torch.ones_like(pred)

        # Apply sigmoid to predictions
        pred = torch.sigmoid(pred) 
        target = torch.sigmoid(target)

        # Flatten pred and target for mask calculation
        pred_flat = pred.view(B, -1)  # (B, H*W)
        target_flat = target.view(B, -1)  # (B, H*W)

        # Compute top 10% threshold
        # pred_threshold = torch.quantile(pred_flat, self.q, dim=1, keepdim=True)
        target_threshold = torch.quantile(target_flat, self.q, dim=1, keepdim=True)

        # Create masks for top 10% values
        pred_top_10_mask = (pred_flat >= target_threshold).float()
        target_top_10_mask = (target_flat >= target_threshold).float()

        # Combine the masks with the ignore index mask
        mask = mask.view(B, -1)  # Convert mask to match shape of pred_flat and target_flat
        pred_top_10_mask = pred_top_10_mask * mask  # Remove in-place operation
        target_top_10_mask = target_top_10_mask * mask  # Remove in-place operation

        # Reshape back to original shape and apply masks
        pred = pred * pred_top_10_mask.view(B, H, W)  # Remove in-place operation
        target = target * target_top_10_mask.view(B, H, W)  # Remove in-place operation

        # Flatten pred and target for Dice coefficient calculation
        pred = pred.view(B, -1)
        target = target.view(B, -1)

        # Compute Dice coefficient
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1. - dice
    
        pred_max = torch.max(pred,dim=1,keepdim=True)[0]
        target_max = torch.max(target,dim=1,keepdim=True)[0]
        scale_loss = F.l1_loss(target_threshold+pred_max, target_threshold+target_max)  # 예측분위수와 타겟분위수 정합

        return loss.mean() + scale_loss


    def dice_coefficient(self, pred, target):
        return 1. - self.forward(pred, target)

   
class AvgQIRDiceLoss(nn.Module):
    def __init__(self, q_values=[0.9, 0.95, 0.99], ignore_index=None, smooth=1.0):
        super(AvgQIRDiceLoss, self).__init__()
        self.q_values = q_values
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        assert C == 1, "This implementation is for binary segmentation"
        
        pred = pred.squeeze(1)
        if len(target.shape) == 4:
            target = target.squeeze(1)

        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float()
        else:
            mask = torch.ones_like(pred)

        pred_flat = pred.view(B, -1)
        target_flat = target.view(B, -1)
        mask_flat = mask.view(B, -1)

        losses = []
        for q in self.q_values:
            pred_threshold = torch.quantile(pred_flat, q, dim=1, keepdim=True)
            target_threshold = torch.quantile(target_flat, q, dim=1, keepdim=True)

            pred_top_q_mask = (pred_flat >= pred_threshold).float() * mask_flat
            target_top_q_mask = (target_flat >= target_threshold).float() * mask_flat

            pred_masked = pred_flat * pred_top_q_mask
            target_masked = target_flat * target_top_q_mask

            intersection = (pred_masked * target_masked).sum(dim=1)
            union = pred_masked.sum(dim=1) + target_masked.sum(dim=1)

            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            losses.append(1. - dice)

        return torch.stack(losses).mean()

class ProdQIRDiceLoss(nn.Module):
    def __init__(self, q_values=[0.9, 0.95, 0.99], ignore_index=None, smooth=1.0):
        super(ProdQIRDiceLoss, self).__init__()
        self.q_values = q_values
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        assert C == 1, "This implementation is for binary segmentation"
        
        pred = pred.squeeze(1)
        if len(target.shape) == 4:
            target = target.squeeze(1)

        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float()
        else:
            mask = torch.ones_like(pred)

        pred_flat = pred.view(B, -1)
        target_flat = target.view(B, -1)
        mask_flat = mask.view(B, -1)

        loss_product = torch.ones(B).to(pred.device)
        for q in self.q_values:
            pred_threshold = torch.quantile(pred_flat, q, dim=1, keepdim=True)
            target_threshold = torch.quantile(target_flat, q, dim=1, keepdim=True)

            pred_top_q_mask = (pred_flat >= pred_threshold).float() * mask_flat
            target_top_q_mask = (target_flat >= target_threshold).float() * mask_flat

            pred_masked = pred_flat * pred_top_q_mask
            target_masked = target_flat * target_top_q_mask

            intersection = (pred_masked * target_masked).sum(dim=1)
            union = pred_masked.sum(dim=1) + target_masked.sum(dim=1)

            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            loss = 1. - dice

            loss_product *= loss  # 곱을 취함

        return loss_product.mean()


import torch
import torch.nn.functional as F

class ModifiedDiceLoss(nn.Module):
    def __init__(self, q=0.9, smooth=1e-5, alpha=0.5):
        super().__init__()
        self.q = q
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        assert C == 1, "This implementation is for binary segmentation"

        # Squeeze the channel dimension
        pred = pred.squeeze(1)  # (B, H, W)
        target = target.squeeze(1) if target.dim() == 4 else target  # (B, H, W)

        # Compute dynamic threshold
        target_threshold = target.max(dim=1, keepdim=True)[0] * 0.9  # (B, 1, 1)

        # Compute std from pred (you could use target instead if preferred)
        pred_std = pred.std(dim=(1,2), keepdim=True)  # (B, 1, 1)

        # Normalize pred
        pred_norm = (pred - target_threshold) / (pred_std + 1e-8)
        pred_norm = torch.sigmoid(pred_norm)

        # Create binary masks
        target_mask = (target > target_threshold).float()
        pred_mask = (pred_norm > 0.5).float()

        # Combine masks where they agree
        pred_final = torch.where(pred_mask == target_mask, target_mask, pred_norm)

        # Flatten for Dice calculation
        pred_flat = pred_final.view(B, -1)
        target_flat = target_mask.view(B, -1)

        # Compute Dice coefficient
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1. - dice

        # Compute scale loss
        pred_quantile = torch.quantile(pred_flat, self.q, dim=1)
        target_quantile = torch.quantile(target_flat, self.q, dim=1)
        scale_loss = F.l1_loss(pred_quantile, target_quantile)

        # Combine losses
        total_loss = dice_loss.mean() + self.alpha * scale_loss

        return total_loss


if __name__ == '__main__':
    # # Create dummy prediction and target tensors (Batch size: 2, Channel: 1, Height: 4, Width: 4)
    # pred = torch.rand(2, 1, 4, 4)  # Random predictions (not passed through sigmoid)
    # target = torch.rand(2, 1, 4, 4)  # Random target values

    # pred = F.relu(pred)
    # target = F.relu(target)
    # # Define multiple q values
    # q_values = [0.9, 0.95, 0.99]

    # # Instantiate the loss functions
    # avg_loss_fn = AvgQIRDiceLoss(q_values=q_values)
    # prod_loss_fn = ProdQIRDiceLoss(q_values=q_values)

    # # Compute the losses
    # avg_loss = avg_loss_fn(pred, pred)
    # prod_loss = prod_loss_fn(pred, pred)

    # print(f"AvgQIRDiceLoss: {avg_loss.item():.4f}")
    # print(f"ProdQIRDiceLoss: {prod_loss.item():.4f}")
    
    # # 예제 코드
    loss_fn = DiceLossforSeg()

    # img1 = torch.randint(0, 2, (32, 1, 512, 512)).float()
    # img2 = torch.randint(0, 2, (32, 1, 512, 512)).float()

    img1 = torch.randn((1, 1, 4, 4)).float()
    img2 = torch.randn((1, 1, 4, 4)).float()

    # fn = torch.sigmoid
    # fn = F.relu
    # img1 = fn(img1*3) 
    # img2 = fn(img2)

    loss = loss_fn(F.relu(img2), F.relu(img2))
    print(loss)
    # print(loss_fn.dice_coefficient(img1,img2))
