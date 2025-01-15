import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from easydict import EasyDict
from dataset import build_dataset  # Make sure this is properly imported
from models import UNet  # Adjust the import based on your actual model module

class ImageReconstructionLightning(LightningModule):
    def __init__(self, model, lr, gamma, batch_size, img_size=32):
        super().__init__()
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.img_size = img_size
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('val_mse', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.9)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def prepare_data(self):
        # Implement if you have data preparation steps like downloading
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset, self.valset, _ = build_dataset(img_size=self.img_size)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, shuffle=False)

def main():
    args = EasyDict()
    args.log_dir = './logs/basic_unet/mse'
    args.checkpoint_dir = './checkpoints/basic_unet/mse'
    args.epochs = 500
    args.batch_size = 128
    args.lr = 1e-5
    args.gamma = 0.95  # Exponential decay rate
    args.device = 'cuda:3'
    args.img_size = 32

    # Set the visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(':')[-1]

    # Initialize your model
    model = UNet()  # Replace with your actual model initialization

    # Initialize the LightningModule
    image_recon_model = ImageReconstructionLightning(
        model=model,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        img_size=args.img_size
    )

    # Set up logger and callbacks
    logger = TensorBoardLogger(save_dir=args.log_dir, name='basic_unet_mse')
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='best_model',
        save_top_k=1,
        verbose=True,
        monitor='val_mse',
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Initialize Trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor]
    )

    # Start training
    trainer.fit(image_recon_model)

if __name__ == '__main__':
    main()
