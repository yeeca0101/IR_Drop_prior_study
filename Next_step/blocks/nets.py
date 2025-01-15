import torch
import torch.nn as nn

from block import *

# Define a CNN model using DepthwiseChannelInvariantConv
class DepthwiseChannelInvariantCNN(nn.Module):
    def __init__(self, in_channels, num_classes=10, hidden_dim=256):
        super(DepthwiseChannelInvariantCNN, self).__init__()
        self.block1 = DepthwiseChannelInvariantConv(in_channels)
        self.block2 = DepthwiseChannelInvariantConv(in_channels)
        self.hidden_layer = nn.Linear(in_channels, hidden_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.hidden_layer(x)
        x = self.fc(x)
        return x


# Define a CNN model using PointwiseChannelInvariantConv
class PointwiseChannelInvariantCNN(nn.Module):
    def __init__(self, in_channels, num_classes=10, hidden_dim=256):
        super(PointwiseChannelInvariantCNN, self).__init__()
        self.block1 = PointwiseChannelInvariantConv(in_channels)
        self.block2 = PointwiseChannelInvariantConv(in_channels)
        self.hidden_layer = nn.Linear(in_channels, hidden_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.hidden_layer(x)
        x = self.fc(x)
        return x


# Define a CNN model using SelfAttentionChannelInvariantConv
class SelfAttentionChannelInvariantCNN(nn.Module):
    def __init__(self, in_channels, num_classes=10, hidden_dim=256):
        super(SelfAttentionChannelInvariantCNN, self).__init__()
        self.block1 = SelfAttentionChannelInvariantConv(in_channels)
        self.block2 = SelfAttentionChannelInvariantConv(in_channels)
        self.hidden_layer = nn.Linear(in_channels, hidden_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.hidden_layer(x)
        x = self.fc(x)
        return x


# Define a CNN model using DepthwiseSeparableConv
class DepthwiseSeparableCNN(nn.Module):
    def __init__(self, in_channels, num_classes=10, hidden_dim=256):
        super(DepthwiseSeparableCNN, self).__init__()
        self.block1 = DepthwiseSeparableConv(in_channels)
        self.block2 = DepthwiseSeparableConv(in_channels)
        self.hidden_layer = nn.Linear(in_channels, hidden_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.hidden_layer(x)
        x = self.fc(x)
        return x


# Define a CNN model using ChannelInvariantBlock
class ChannelInvariantCNN(nn.Module):
    def __init__(self, in_channels, num_classes=10, hidden_dim=256):
        super(ChannelInvariantCNN, self).__init__()
        self.block1 = ChannelInvariantBlock(in_channels)
        self.block2 = ChannelInvariantBlock(in_channels)
        self.hidden_layer = nn.Linear(in_channels, hidden_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.hidden_layer(x)
        x = self.fc(x)
        return x


# Define a CNN model using MLPMixerChannelInvariantBlock
class MLPMixerChannelInvariantCNN(nn.Module):
    def __init__(self, in_channels, num_classes=10, patch_size=4, hidden_dim=256):
        super(MLPMixerChannelInvariantCNN, self).__init__()
        self.block1 = MLPMixerChannelInvariantBlock(in_channels, patch_size, hidden_dim)
        self.block2 = MLPMixerChannelInvariantBlock(hidden_dim, patch_size, hidden_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Test code for each model
def test_models():
    dummy_input = torch.randn(2, 3, 32, 32)
    models = [
        DepthwiseChannelInvariantCNN(in_channels=3),
        PointwiseChannelInvariantCNN(in_channels=3),
        SelfAttentionChannelInvariantCNN(in_channels=3),
        DepthwiseSeparableCNN(in_channels=3),
        ChannelInvariantCNN(in_channels=3),
        MLPMixerChannelInvariantCNN(in_channels=3)
    ]
    
    for model in models:
        model_name = model.__class__.__name__
        try:
            print(f"Testing {model_name}...")
            output = model(dummy_input)
            print(f"{model_name} output shape: {output.shape}")
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{model_name} number of parameters: {num_params}\n")
        except Exception as e:
            print(f"Error in {model_name}: {e}\n")

if __name__ == "__main__":
    test_models()
