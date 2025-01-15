import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedMSEF1Loss(nn.Module):
    def __init__(self, alpha=0.5, threshold_percent=0.90):
        """
        Initialize the combined loss function.
        
        :param alpha: Weight for balancing MSE and F1 loss (0 <= alpha <= 1).
                      alpha = 0.5 means equal weight for both.
        :param threshold_percent: Percentile threshold to classify top 10% IR drop as hotspots.
        """
        super(CombinedMSEF1Loss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.threshold_percent = threshold_percent

    def forward(self, predictions, targets):
        """
        Compute the combined MSE + F1 loss.
        
        :param predictions: Tensor of predicted IR drop values (batch_size, height, width)
        :param targets: Tensor of ground truth IR drop values (batch_size, height, width)
        
        :return: Combined loss
        """
        # Calculate MSE loss
        mse_loss = self.mse_loss(predictions, targets)
        
        # Compute F1 score (as a loss: 1 - F1 score)
        f1_loss = self.compute_f1_loss(predictions, targets)
        
        # Combined loss = alpha * MSE + (1 - alpha) * F1 Loss
        combined_loss = self.alpha * mse_loss + (1 - self.alpha) * f1_loss
        
        return combined_loss

    def compute_f1_loss(self, predictions, targets):
        """
        Compute the F1 loss: 1 - F1 score, because we want to minimize the loss.
        
        :param predictions: Tensor of predicted IR drop values (batch_size, height, width)
        :param targets: Tensor of ground truth IR drop values (batch_size, height, width)
        
        :return: F1 loss (1 - F1 score)
        """
        # Flatten the predictions and targets for easier computation
        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate the IR drop threshold value (90th percentile) in the ground truth
        threshold_value = torch.quantile(targets_flat, self.threshold_percent)

        # Classify hotspots in predictions and targets
        predicted_hotspots = predictions_flat >= threshold_value
        actual_hotspots = targets_flat >= threshold_value
        
        # Calculate TP, FP, FN
        tp = torch.sum(predicted_hotspots & actual_hotspots).item()  # True Positives
        fp = torch.sum(predicted_hotspots & ~actual_hotspots).item() # False Positives
        fn = torch.sum(~predicted_hotspots & actual_hotspots).item() # False Negatives

        # Precision and Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # F1 Score
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        # Return F1 loss (1 - F1 score) since we want to minimize the loss
        f1_loss = 1 - f1_score
        return f1_loss

def gradient_loss(predicted, target):
    # Check the input shapes (B,C,H,W)
    
    # Compute gradient in the x direction
    predicted_dx = torch.abs(predicted[:, :, 1:, :] - predicted[:, :, :-1, :])
    target_dx = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    
    # Compute gradient in the y direction
    predicted_dy = torch.abs(predicted[:, :, :, 1:] - predicted[:, :, :, :-1])
    target_dy = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    
    # Pad the gradients to match the original size
    predicted_dx = torch.nn.functional.pad(predicted_dx, (0, 0, 1, 0))  # Pad height (x-direction)
    target_dx = torch.nn.functional.pad(target_dx, (0, 0, 1, 0))
    
    predicted_dy = torch.nn.functional.pad(predicted_dy, (1, 0, 0, 0))  # Pad width (y-direction)
    target_dy = torch.nn.functional.pad(target_dy, (1, 0, 0, 0))
    
    # Compute the gradient loss (MSE of the gradients)
    return torch.mean((predicted_dx - target_dx)**2 + (predicted_dy - target_dy)**2)

def weighted_mse_loss(predicted, target, threshold=0.9):
    # Create weight map where larger IR drop values get higher weight
    mask = (target > torch.quantile(target, threshold)).float()
    return torch.mean(mask * (predicted - target) ** 2)

def sobel_operator(x):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32,device=x.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32,device=x.device).unsqueeze(0).unsqueeze(0)
    grad_x = F.conv2d(x, sobel_x)
    grad_y = F.conv2d(x, sobel_y)
    return grad_x, grad_y

def edge_loss(predicted, target):
    predicted_edges_x, predicted_edges_y = sobel_operator(predicted)
    target_edges_x, target_edges_y = sobel_operator(target)
    return torch.mean((predicted_edges_x - target_edges_x) ** 2 + (predicted_edges_y - target_edges_y) ** 2)


def combined_loss(predicted, target, alpha=0.5, beta=0.3, gamma=0.2):
    mse = nn.MSELoss()(predicted, target)
    grad_loss = gradient_loss(predicted, target)
    hotspot_loss = weighted_mse_loss(predicted, target)
    
    # You can also add losses that consider how the current map and pdn_density affect IR drop
    return mse + alpha * grad_loss + beta * hotspot_loss + gamma * edge_loss(predicted, target)


# Example usage in a training loop
if __name__ == "__main__":
    # Example predictions and targets
    predictions = torch.tensor([[0.02, 0.05, 0.07], [0.08, 0.10, 0.12], [0.15, 0.20, 0.25]], dtype=torch.float32)
    targets = torch.tensor([[0.01, 0.05, 0.09], [0.15, 0.12, 0.18], [0.17, 3.21, 0.30]], dtype=torch.float32)

    # Create the combined loss function
    combined_loss_fn = CombinedMSEF1Loss(alpha=0.5, threshold_percent=0.80)

    # Compute the combined loss
    loss = combined_loss_fn(predictions, targets)
    print(f"Combined MSE + F1 Loss: {loss:.4f}")
