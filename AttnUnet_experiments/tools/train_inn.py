import sys
sys.path.append('../')
import argparse
import os
import numpy as np
import torch
import torch.backends as bc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import Callback

from ir_dataset import build_dataset_5m

# A100 설정 등 (필요시)
bc.cuda.matmul.allow_tf32 = True
bc.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='PyTorch Lightning INN Training for IR-Drop')
parser.add_argument('--img_size', default=256, type=int, help='res')
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
parser.add_argument('--epoch', default=100, type=int, help='max epoch')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--gpus', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--dataset', type=str, default='cus', help='')
parser.add_argument('--monitor', type=str, default='mae', help='')
parser.add_argument('--dbu_per_px', type=str, default='200nm', help='')
parser.add_argument('--in_channels', type=int, default=2, help='number of channels for INN (if single channel, replicate to 2)')
parser.add_argument('--hidden_channels', type=int, default=64, help='hidden channels in coupling layers')
parser.add_argument('--num_layers', type=int, default=4, help='number of coupling layers')
parser.add_argument('--log_dir', default='logs_inn', type=str, help='tensorboard log folder')
parser.add_argument('--post_fix', default='', type=str, help='')
parser.add_argument('--save_folder', default='checkpoints_inn', type=str, help='checkpoint save folder')
parser.add_argument('--repeat', default=1, type=int, help='number of repetitive training')
parser.add_argument('--mixed_precision', type=bool, default=False, help='use mixed precision training')

# ★ range_loss 가중치(α) 추가
parser.add_argument('--alpha_range', type=float, default=0.1, help='weight for the range_loss')

args = parser.parse_args()


# ============================================================
# Model: INN (RealNVP style)
# ============================================================
class AffineCoupling(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        """
        입력 텐서를 채널 기준으로 두 부분으로 나눈 후, 
        한쪽(x1)으로 scale(s)와 shift(t)를 예측하여 x2에 affine 변환을 수행.
        in_channels는 짝수여야 합니다.
        """
        super(AffineCoupling, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels // 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        h = self.net(x1)
        s, t = h.chunk(2, dim=1)
        s = torch.tanh(s)
        if not reverse:
            y2 = x2 * torch.exp(s) + t
        else:
            y2 = (x2 - t) * torch.exp(-s)
        return torch.cat([x1, y2], dim=1)

class Permutation(nn.Module):
    def __init__(self, num_channels):
        """
        채널 순서를 랜덤하게 섞는 layer. inverse 시 원래 순서로 복원.
        """
        super(Permutation, self).__init__()
        perm = torch.randperm(num_channels)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", torch.argsort(perm))
    
    def forward(self, x, reverse=False):
        if not reverse:
            return x[:, self.perm, ...]
        else:
            return x[:, self.inv_perm, ...]

class InvertibleNormalization(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=4):
        """
        RealNVP 스타일의 INN 모델.
        여러 coupling layer와 permutation layer를 쌓고, 마지막에 sigmoid를 적용하여
        출력이 [0,1] 범위가 되도록 강제합니다.
        """
        super(InvertibleNormalization, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(Permutation(in_channels))
            layers.append(AffineCoupling(in_channels, hidden_channels))
        self.layers = nn.ModuleList(layers)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        z = x
        for layer in self.layers:
            z = layer(z, reverse=False)
        # [0,1] 범위 강제
        z = self.sigmoid(z)
        return z
    
    def inverse(self, z):
        # 역변환을 위해 sigmoid의 역함수 (logit)를 적용합니다.
        eps = 1e-6
        z_clamped = torch.clamp(z, eps, 1 - eps)
        z_inv = torch.log(z_clamped / (1 - z_clamped))
        x = z_inv
        for layer in reversed(self.layers):
            x = layer(x, reverse=True)
        return x

# ============================================================
# Lightning Module for INN Training
# ============================================================
class INNTrainingModule(LightningModule):
    def __init__(self, lr, in_channels, hidden_channels, num_layers, alpha_range):
        super().__init__()
        self.lr = lr
        self.model = InvertibleNormalization(in_channels, hidden_channels, num_layers)
        self.criterion = nn.MSELoss()
        self.num_workers = 4
        self.alpha = alpha_range  # range_loss 가중치

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        _, ir_ori = batch
        ir_norm = self.min_max_norm(ir_ori)
        # ir_ori가 1채널이면 2채널로 복제 (repeat)
        if ir_ori.shape[1] == 1:
            ir_ori = ir_ori.repeat(1, 2, 1, 1)
        
        # forward
        z = self.model(ir_ori)
        # inverse -> 복원
        x_recon = self.model.inverse(z)

        # (1) 재구성 손실
        recon_loss = self.criterion(x_recon, ir_ori)
        
        # (2) sample per min–max 범위를 0~1에 가까이 맞추도록 하는 loss
        # 배치 차원(B) + (C,H,W)
        z = z[:,0,...].unsqueeze(1)
        z_min = z.amin(dim=[1,2,3])   # shape: (B,)
        z_max = z.amax(dim=[1,2,3])   # shape: (B,)
        range_loss = ((z_min - 0).abs() + (z_max - 1).abs()).mean()
        recon_norm_loss = F.mse_loss(ir_norm,z[:,0,...].unsqueeze(1))
        loss = recon_loss + self.alpha * range_loss + recon_norm_loss


        self.log("train_recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_range_loss", recon_norm_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def min_max_norm(self,x):
        return (x-x.min())/(x.max()-x.min())
    
    def validation_step(self, batch, batch_idx):
        _, ir_ori = batch
        ir_norm = self.min_max_norm(ir_ori)
        if ir_ori.shape[1] == 1:
            ir_ori = ir_ori.repeat(1, 2, 1, 1)
        
        # forward
        z = self.model(ir_ori)
        # inverse -> 복원
        x_recon = self.model.inverse(z)

        # 재구성 손실
        recon_loss = self.criterion(x_recon, ir_ori)

        # sample per min–max 범위를 0~1에 맞추는 loss
        z = z[:,0,...].unsqueeze(1)
        z_min = z.amin(dim=[1,2,3])
        z_max = z.amax(dim=[1,2,3])
        range_loss = ((z_min - 0).abs() + (z_max - 1).abs()).mean()
        recon_norm_loss = F.mse_loss(ir_norm,z[:,0,...].unsqueeze(1))
        loss = recon_loss + self.alpha * range_loss + recon_norm_loss

        # self.log("val_recon_loss", recon_loss, on_epoch=True, prog_bar=True)
        self.log("val_range_loss", recon_norm_loss, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-7)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset, self.val_dataset = build_dataset_5m(
                args.img_size, train=True,
                in_ch=3, 
                selected_folders=[f'{args.dbu_per_px}_numpy'],
                inn=True
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, 
                          num_workers=self.num_workers, persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False, 
                          num_workers=self.num_workers)


class CustomCheckpoint(Callback):
    def __init__(self, checkpoint_dir, metric_name='val_range_loss', mode='min', post_fix=''):
        super().__init__()
        checkpoint_dir = os.path.join(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir if post_fix == '' else f'{checkpoint_dir}/{post_fix}'
       
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_model_file_name = ""
        self.metric_name = metric_name
        self.mode = mode

    def on_validation_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            val_metric = trainer.callback_metrics.get(self.metric_name, None)
            if val_metric is not None:
                is_best = (self.mode == 'min' and val_metric <= self.best_metric) or \
                          (self.mode == 'max' and val_metric >= self.best_metric)
                if is_best:
                    self.best_metric = val_metric
                    state = {
                        'net': pl_module.model.state_dict(),
                        'epoch': trainer.current_epoch,
                        'mae': trainer.callback_metrics.get('val_mae', None),
                        'f1': trainer.callback_metrics.get('val_f1', None),
                    }
                    if self.best_model_file_name:
                        old_path = os.path.join(self.checkpoint_dir, self.best_model_file_name)
                        if os.path.exists(old_path):
                            os.remove(old_path)
                    self.best_model_file_name = f'INN_embd_{args.num_layers}layers_{args.hidden_channels}_{trainer.current_epoch}_{val_metric:.4f}.pth'
                    new_path = os.path.join(self.checkpoint_dir, self.best_model_file_name)
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    torch.save(state, new_path)
                    print(f'Saved new best model to: {new_path}')


def main():
    model = INNTrainingModule(
        lr=args.lr, 
        in_channels=args.in_channels, 
        hidden_channels=args.hidden_channels, 
        num_layers=args.num_layers,
        alpha_range=args.alpha_range  # 추가
    )
    
    logger = TensorBoardLogger(save_dir=args.log_dir, name="INN")
    os.makedirs(args.save_folder, exist_ok=True)

    checkpoint_callback = CustomCheckpoint(
        checkpoint_dir=args.save_folder,
        metric_name='val_range_loss', #if args.monitor == 'mae' else 'val_f1', 
        mode='min', # if args.monitor in 'val_range_loss' else 'max',
        post_fix=args.post_fix
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = Trainer(
        max_epochs=args.epoch,
        devices=args.gpus,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        strategy=DDPStrategy(find_unused_parameters=False) if args.gpus > 1 else 'auto',
        enable_checkpointing=False,
        precision=16 if args.mixed_precision else 32
    )
    
    trainer.fit(model)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
