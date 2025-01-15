import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Callback
from pytorch_lightning.strategies import DDPStrategy
from sklearn.model_selection import KFold
import pandas as pd

# Dataset loading
from unets import R2AttU_Net,AttU_Net,init_weights  
from models.papers_attn_unet import AttUNet
from models.papers_model import AttnUnetBase
from models.papers_model_v2 import AttnUnetV2 # add Dropblock, use sigmoid(out)
from metric import IRDropMetrics  
from ir_dataset import build_dataset_asap7_cross_val,get_val_casename
from loss import *

class IRDropPrediction(LightningModule):
    def __init__(self, lr, args,cross_val_id):
        super().__init__()
        self.lr = lr
        self.num_workers = 0
        self.cross_val_id = cross_val_id
        self.model = self.build_model(args.arch, args)
        if args.finetune:
            self.model = self.init_weights_chkpt(self.model, args.save_folder)
        else:
            init_weights(self.model)

        self.criterion = LossSelect(loss_type=args.loss,
                                    use_cache=True if args.loss == 'cache' else False,
                                    dice_q=args.dice_q,
                                    loss_with_logit=args.loss_with_logit
                                    )
        self.metrics = IRDropMetrics(loss_with_logit=args.loss_with_logit)
        self.save_hyperparameters(self.lr,self.cross_val_id,self.criterion)

    def build_model(self, arch, args):
        if arch == 'attn_12ch':
            model = AttU_Net(img_ch=12)
        elif arch == 'attn_base':
            model = AttnUnetBase()
        elif arch == 'attnv2':
            model = AttnUnetV2(dropout_name=args.dropout, dropout_p=0.05 if args.finetune else 0.1,in_ch=args.in_ch)
        else:
            raise NameError(f'arch type error : {arch}')
        return model

    def init_weights_chkpt(self, model, save_folder):
        checkpoint_files = [f for f in os.listdir(save_folder) if f.endswith('.pth')]
        checkpoint_files.sort(key=lambda x: int(x.split('_')[3].split('.')[0]), reverse=True)
        checkpoint_path = os.path.join(save_folder, checkpoint_files[0])
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['net'])
        return model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        metrics = self.metrics.compute_metrics(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_mae', metrics['mae'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_f1', metrics['f1'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        metrics = self.metrics.compute_metrics(outputs, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_f1', metrics['f1'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_mae', metrics['mae'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if args.optim == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=args.weight_decay)
        elif args.optim == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr,
                                  momentum=0.9, weight_decay=args.weight_decay)
        else:
            raise NameError(f'not support {args.optim}')
        if args.scheduler == 'cosineanealing':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=1e-7)
        elif args.scheduler == 'cosineanealingwarmup':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2, eta_min=0.000001)
        else:
            raise NameError(f'not support yet {args.scheduler}')
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_mae"}

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if args.dataset.lower() == 'asap7':
                self.train_dataset, self.val_dataset = build_dataset_asap7_cross_val(self.cross_val_id,get_case_name=False,img_size=args.img_size,in_ch=args.in_ch)
            else:
                raise NameError('check dataset name')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=self.num_workers)

class CustomCheckpoint(Callback):
    def __init__(self, checkpoint_dir, repeat_idx, metric_name='val_mae', mode='min', post_fix='',val_casename=''):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir if args.post_fix == '' else f'{checkpoint_dir}_{args.post_fix}'
        if args.finetune:
            self.checkpoint_dir = os.path.join(self.checkpoint_dir, 'finetune', args.loss, post_fix)
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_model_file_name = ""
        self.repeat_idx = repeat_idx
        self.metric_name = metric_name
        self.mode = mode
        self.val_casename = val_casename

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
                        os.remove(old_path)
                    
                    self.best_model_file_name = f'{self.val_casename}_{self.repeat_idx}_{trainer.current_epoch}_{val_metric:.4f}.pth'
                    new_path = os.path.join(self.checkpoint_dir, self.best_model_file_name)
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    torch.save(state, new_path)
                    print(f'Saved new best model to: {new_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Lightning IR-Drop Prediction')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate [pre-train:5e-4, fine-tuning:1e-5~5e-4]')
    parser.add_argument('--save_folder', default='checkpoint', type=str, help='checkpoint save folder')
    parser.add_argument('--log_dir', default='logs', type=str, help='tensorboard log folder')
    parser.add_argument('--repeat', default=1, type=int, help='number of repetitive training')
    parser.add_argument('--epoch', default=100, type=int, help='max epoch')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--gpus', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--dataset', type=str, default='asap7', help='select dataset')
    parser.add_argument('--finetune', type=bool, default=False, help='pre-train or fine-tuning')
    parser.add_argument('--optim', type=str, default='adam', help='optimizer')
    parser.add_argument('--dropout', type=str, default='dropblock', help='for attnV2')
    parser.add_argument('--arch', type=str, default='attn_12ch', help='sup model : [attn_12ch, attn_base_12ch, attnv2]')
    parser.add_argument('--loss', type=str, default='default', help='sup loss : [default, comb, default_edge,edge,ssim,dice]')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay rate')
    parser.add_argument('--dice_q', type=float, default=0.9, help='top %')
    parser.add_argument('--pdn_density_dropout', type=float, default=0.0, help='0.~1. 0: False, 1 : always')
    parser.add_argument('--loss_with_logit', type=bool, default=True, help='if False, apply sigmoid fn')
    parser.add_argument('--post_fix', type=str, default='', help='post fix of finetune checkpoint path')
    parser.add_argument('--scheduler', type=str, default='consineanealingwarmup', help='lr scheduler')
    parser.add_argument('--img_size', default=512, type=int, help='input  h,w')
    parser.add_argument('--in_ch', default=12, type=int, help='input channels')
    args = parser.parse_args()


    cross_val_ids = [0, 1, 2, 3]  # dataset of cross validation has 4 samples 
    for id in cross_val_ids:
        val_casename = get_val_casename(id)
        model = IRDropPrediction(lr=args.lr, args=args,cross_val_id=id)
        logdir = os.path.join(args.log_dir, f'{args.arch}/{args.dataset}/{args.loss}/cross_val_{id}_{val_casename}')
        logger = TensorBoardLogger(save_dir=logdir)
        checkpoint_callback = CustomCheckpoint(
            checkpoint_dir=args.save_folder,
            repeat_idx=id,
            metric_name='val_mae' if not args.finetune else 'val_f1',
            mode='min' if not args.finetune else 'max',
            post_fix=args.post_fix,
            val_casename=val_casename
        )
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        trainer = Trainer(
            max_epochs=args.epoch,
            devices=args.gpus,
            accelerator="gpu" if args.gpus > 0 else "cpu",
            logger=logger,
            callbacks=[checkpoint_callback, lr_monitor],
            strategy=DDPStrategy(find_unused_parameters=False) if args.gpus > 1 else 'auto',
            enable_checkpointing=False
        )

        trainer.fit(model)
        torch.cuda.empty_cache()
