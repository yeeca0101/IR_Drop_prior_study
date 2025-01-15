import os
import sys
sys.path.append('/workspace/Next_step/blocks')
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import pandas as pd

from blocks.nets import * 



class ChannelInvariantNet(pl.LightningModule):
    def __init__(self, in_channels, num_classes, net):
        super(ChannelInvariantNet, self).__init__()
        self.net= net(in_channels = in_channels)     
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)
        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

# Prepare Data and Function to Randomly Shuffle RGB Channels
def prepare_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader

def random_channel_shuffle(x):
    channels = x.size(1)
    shuffled_indices = torch.randperm(channels)
    return x[:, shuffled_indices, :, :]

if __name__ == "__main__":
    n_classes= 10
    train_loader, val_loader, test_loader = prepare_data()
    nets = [DepthwiseChannelInvariantCNN,PointwiseChannelInvariantCNN,
            SelfAttentionChannelInvariantCNN,DepthwiseSeparableCNN,
            ChannelInvariantCNN,MLPMixerChannelInvariantCNN]
    results = []

    for net in nets:
        try:
            model = ChannelInvariantNet(in_channels=3, num_classes=n_classes,net=net)
            trainer = pl.Trainer(
                max_epochs=10,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1 if torch.cuda.is_available() else 'auto',
            )
            trainer.fit(model, train_loader, val_loader)

            model.eval()
            test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes).to(model.device)
            for batch in test_loader:
                x, y = batch
                x, y = x.to(model.device), y.to(model.device)
                x_shuffled = random_channel_shuffle(x)
                with torch.no_grad():
                    y_hat = model(x_shuffled)
                test_acc.update(y_hat.softmax(dim=-1), y)
            final_test_acc = test_acc.compute().item()
            print(f"net: {net.__name__}, Test Accuracy with Random Channel Shuffle: {final_test_acc}")
            results.append({"net": net.__name__, "Test Accuracy": final_test_acc})
        except Exception as e:
            print(f"Error with net {net.__name__}: {str(e)}")
            continue

    results_df = pd.DataFrame(results)
    print(results_df)
