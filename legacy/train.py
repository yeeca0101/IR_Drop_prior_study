import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict
from dataset import build_dataset
from tqdm import tqdm

from losses import CombinedMSEF1Loss,combined_loss
from model import AutoEncoder,AttUNet
from models.ce_fpn_model import IRdropModel
from models.inception_unet_ori import UNetWithAttention
from models import UNet

def train_on_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for batch in progress_bar:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(train_loader)

def validation(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        for batch in progress_bar:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(val_loader)

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args):
    writer = SummaryWriter(log_dir=args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_mse = float('inf')
    best_model_file_name = None
    
    progress_bar = tqdm(range(args.epochs), desc="Epochs")
    model.to(device)
    for epoch in progress_bar:
        train_loss = train_on_epoch(model, train_loader, criterion, optimizer, device)
        val_mse = validation(model, val_loader, criterion, device)
        scheduler.step()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_mse, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        progress_bar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Val MSE': f'{val_mse:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

        if val_mse < best_mse:
            if best_model_file_name:
                try:
                    os.remove(os.path.join(args.checkpoint_dir, best_model_file_name))
                except FileNotFoundError:
                    pass

            best_model_file_name = f'chkpt_{epoch}_{val_mse*1000:.2f}.pth'
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mse': val_mse,
            }
            torch.save(state, os.path.join(args.checkpoint_dir, best_model_file_name))
            best_mse = val_mse

    writer.close()

def main():
    args = EasyDict()
    args.log_dir = './logs/attn/mse'
    args.checkpoint_dir = './checkpoints/attn/mse'
    args.epochs = 500
    args.batch_size = 64
    args.lr = 1e-5
    args.gamma = 0.95  # Exponential decay rate
    device = torch.device('cuda:3')

    # model = IRdropModel(in_channel=3,device=device)
    model = AttUNet()
    trainset,valset,testset = build_dataset(img_size=32)

    train_loader = DataLoader(trainset,batch_size=args.batch_size,shuffle=True,drop_last=True)
    val_loader = DataLoader(valset,batch_size=args.batch_size,shuffle=False)

    # criterion = combined_loss # nn.MSELoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=0.9)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device=device, args=args)


if __name__ == '__main__':
    # device = torch.device('cuda:3')
    # model = IRdropModel(3,device=device)
    # model.to(device)
    # inp = torch.randn(1,3,256,256).to(device)
    # print(model(inp).shape)

    main()
