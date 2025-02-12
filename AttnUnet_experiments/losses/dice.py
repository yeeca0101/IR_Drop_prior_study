import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=None, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        assert C == 1, "This implementation is for binary segmentation"
        
        # Squeeze the channel dimension
        pred = pred.squeeze(1)
        if len(target.shape) == 4:
            target = target.squeeze(1)
        
        # Create mask for ignored index
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float()
        else:
            mask = torch.ones_like(pred)
        
        # Apply sigmoid to predictions
        # pred = torch.sigmoid(pred) # 이거 쓰면 예제에서 0.655가 최대 coeff
        
        # Multiply pred and target by mask
        pred = pred * mask
        target = target * mask
        
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


class IRDiceLoss(nn.Module):
    def __init__(self, ignore_index=None, smooth=1.,q=0.9):
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
        # pred = torch.sigmoid(pred) not used. ref LossSelect class 

        # Flatten pred and target for mask calculation
        pred_flat = pred.view(B, -1)  # (B, H*W)
        target_flat = target.view(B, -1)  # (B, H*W)

        # Compute top 10% threshold
        pred_threshold = torch.quantile(pred_flat, self.q, dim=1, keepdim=True)
        target_threshold = torch.quantile(target_flat, self.q, dim=1, keepdim=True)

        # Create masks for top 10% values
        pred_top_10_mask = (pred_flat >= pred_threshold).float()
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
        return loss.mean()

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


if __name__ == '__main__':
    # Create dummy prediction and target tensors (Batch size: 2, Channel: 1, Height: 4, Width: 4)
    pred = torch.rand(2, 1, 4, 4)  # Random predictions (not passed through sigmoid)
    target = torch.rand(2, 1, 4, 4)  # Random target values

    pred = F.relu(pred)
    target = F.relu(target)
    # Define multiple q values
    q_values = [0.9, 0.95, 0.99]

    # Instantiate the loss functions
    avg_loss_fn = AvgQIRDiceLoss(q_values=q_values)
    prod_loss_fn = ProdQIRDiceLoss(q_values=q_values)

    # Compute the losses
    avg_loss = avg_loss_fn(pred, pred)
    prod_loss = prod_loss_fn(pred, pred)

    print(f"AvgQIRDiceLoss: {avg_loss.item():.4f}")
    print(f"ProdQIRDiceLoss: {prod_loss.item():.4f}")

# # # 예제 코드
# loss_fn = IRDiceLoss()

# img1 = torch.randint(0, 2, (32, 1, 512, 512)).float()
# img2 = torch.randint(0, 2, (32, 1, 512, 512)).float()

# fn = torch.sigmoid
# # fn = F.relu
# img1 = fn(img1)
# img2 = fn(img2)

# loss = loss_fn(img1, img2)
# print(loss)
# print(loss_fn.dice_coefficient(img1,img2))