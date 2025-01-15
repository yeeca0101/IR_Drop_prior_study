import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def restoration_distribution_loss(pred, target, temperature=1.0, eps=1e-6):
    
    pred_dist = F.log_softmax(pred.view(pred.size(0), -1) / temperature + eps, dim=1)
    target_dist = F.softmax(target.view(target.size(0), -1) / temperature + eps, dim=1)
    
    kl_div_loss = F.kl_div(
        pred_dist,
        target_dist,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    return kl_div_loss

def kl_divergence(p, q):
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    return torch.sum(p * torch.log(p / q), dim=-1)

def soft_histogram(x, bins=256, min=0.0, max=1.0, sigma=0.01):
    bin_centers = torch.linspace(min, max, bins, device=x.device)
    x = x.view(x.size(0), x.size(1), -1, 1)  # (batch, channel, pixels, 1)
    diff = x - bin_centers.view(1, 1, 1, -1)
    weights = torch.exp(-0.5 * (diff / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    histogram = weights.sum(dim=2)  # Sum over pixels
    histogram = histogram / (histogram.sum(dim=-1, keepdim=True) + 1e-10)
    return histogram

class PixelDistributionLoss(nn.Module):
    def __init__(self, loss_type='kl', bins=256, sigma=0.01):
        super(PixelDistributionLoss, self).__init__()
        self.loss_type = loss_type
        self.bins = bins
        self.sigma = sigma
        
    def forward(self, target, prediction):
        if self.loss_type == 'kl_restoration':
            return restoration_distribution_loss(prediction,target)
        target_hist = soft_histogram(target, bins=self.bins, sigma=self.sigma)
        pred_hist = soft_histogram(prediction, bins=self.bins, sigma=self.sigma)

        if self.loss_type == 'kl':
            return kl_divergence(target_hist, pred_hist).mean()
        elif self.loss_type == 'js':
            m = 0.5 * (target_hist + pred_hist)
            return (0.5 * kl_divergence(target_hist, m) + 0.5 * kl_divergence(pred_hist, m)).mean()
        elif self.loss_type == 'wasserstein':
            return torch.abs(target_hist - pred_hist).sum(dim=-1).mean()
        elif self.loss_type == 'correlation':
            target_mean = target_hist.mean(dim=-1, keepdim=True)
            pred_mean = pred_hist.mean(dim=-1, keepdim=True)
            numerator = ((target_hist - target_mean) * (pred_hist - pred_mean)).sum(dim=-1)
            denominator = torch.sqrt(((target_hist - target_mean) ** 2).sum(dim=-1) * ((pred_hist - pred_mean) ** 2).sum(dim=-1))
            correlation = numerator / (denominator + 1e-10)
            return (1 - correlation).mean()
        elif self.loss_type == 'histogram_matching':
            return F.mse_loss(target_hist, pred_hist)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

def test(target, prediction):
    test_loss_list = ['kl', 'js', 'wasserstein', 'correlation', 'histogram_matching']
    for fn in test_loss_list:    
        loss_fn = PixelDistributionLoss(loss_type=fn)
        if fn =='kl':
            loss = F.kl_div(prediction.log(), target, reduction='mean')
        else:
            loss = loss_fn(target, prediction)
        try:
            loss.backward()
            print(f"{fn}  Loss: {loss.item()} (Backward pass successful)")
        except RuntimeError as e:
            print(f"{fn}  Loss: {loss.item()} (Backward pass failed: {str(e)})")


def generate_same_distribution(hw, batch_size=2, channels=1):
    """Generate the same distribution for target and prediction."""
    target = torch.rand((batch_size, channels, hw, hw), requires_grad=False)
    prediction = target.clone().detach() + torch.normal(0, 0.01, target.size())
    prediction.requires_grad = True
    return target, prediction

def generate_different_distribution(hw, batch_size=2, channels=1):
    """Generate different distributions for target and prediction."""
    # Gaussian distribution (mean=0.5, std=0.1) for target
    target = torch.normal(0.5, 0.1, (batch_size, channels, hw, hw), requires_grad=False)
    target = torch.clamp(target, 0.0, 1.0)  # Clamp to [0, 1] range

    # Uniform distribution [0, 1) for prediction
    prediction = torch.rand((batch_size, channels, hw, hw), requires_grad=True)
    
    return target, prediction

def test_distributions(hw=1024, batch_size=2, channels=1):
    """Test loss function with same and different distributions."""
    print("Testing same distribution:")
    target, prediction = generate_same_distribution(hw, batch_size, channels)
    test(target, prediction)

    print("\nTesting different distribution:")
    target, prediction = generate_different_distribution(hw, batch_size, channels)
    test(target, prediction)

# Example usage
if __name__ == "__main__":
    # hw = 1024
    # target = torch.rand((2, 1, hw, hw), requires_grad=False)
    # prediction = torch.rand((2, 1, hw, hw), requires_grad=True) 
    
    # test(target, prediction)
    test_distributions(hw=512)  # Reduce size for faster testing
