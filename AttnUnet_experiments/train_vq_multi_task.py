'''
    author : ymin98@sogang.ac.kr
    CVQ-Attn-Unet v6_1 > version
    multi head output과 hr 모델 실험 코드
'''

import argparse
import os
import torch
import torch.backends as bc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Callback
from pytorch_lightning.strategies import DDPStrategy

from models import *
from models.parts.vqvae import init_weights

from metric import IRDropMetrics 
from ir_dataset import IRDropDataset,build_dataset_iccad,build_dataset,build_dataset_began_asap7,build_dataset_5m
from loss import *


# A100
bc.cuda.matmul.allow_tf32 = True
bc.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='PyTorch Lightning IR-Drop Prediction')
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate [pre-train:5e-4, fine-tuning:1e-5~5e-4]')
parser.add_argument('--vq_lr', default=5e-5, type=float, help='learning rate vq. if use ema, the lr be ignored')
parser.add_argument('--save_folder', default='checkpoint', type=str, help='checkpoint save folder')
parser.add_argument('--log_dir', default='logs', type=str, help='tensorboard log folder')
parser.add_argument('--repeat', default=1, type=int, help='number of repetitive training')
parser.add_argument('--epoch', default=100, type=int, help='max epoch')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--gpus', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--dataset', type=str, default='iccad', help='select BeGAN or iccad')
parser.add_argument('--finetune', type=bool, default=False, help='pre-train or fine-tuning')
parser.add_argument('--optim', type=str, default='adam', help='opitmizer')
parser.add_argument('--dropout', type=str, default='dropblock', help='for attnV2')
parser.add_argument('--arch', type=str, default='attn_12ch', help='sup model : [attn_12ch, attn_base_12ch, attnv2, vqvae]') # vqvae 추가
parser.add_argument('--loss', type=str, default='default', help='sup loss : [default, comb, default_edge,edge,ssim,dice]')
parser.add_argument('--loss_aux', type=str, default='default', help='sup loss : [default, comb, default_edge,edge,ssim,dice]')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='prev : 5e4')
parser.add_argument('--dice_q', type=float, default=0.9, help='top %')
parser.add_argument('--pdn_density_dropout', type=float, default=0.0, help='0.~1. 0: False, 1 : always')
parser.add_argument('--loss_with_logit', type=bool, default=True, help='if False, apply sigmoid fn')
parser.add_argument('--post_fix', type=str, default='', help='post fix of finetune checkpoint path')
parser.add_argument('--scheduler', type=str, default='consineanealingwarmup', help='lr scheduler')
parser.add_argument('--cross_val', type=bool, default=False, help='for asap7 dataset training')
parser.add_argument('--pdn_zeros', type=bool, default=False, help='eff_dist, pdn_density channels zeros')
parser.add_argument('--in_ch', type=int, default=12, help='1 : only use current map or auto_encoder mode')
parser.add_argument('--img_size', type=int, default=512, help='input img_size')
parser.add_argument('--mixed_precision', type=bool, default=False, help='mixed_precision')
parser.add_argument('--monitor', type=str, default='f1', help='monitor metric')
parser.add_argument('--use_raw', type=bool, default=False, help='not normalized ir drop map.')
parser.add_argument('--post_min_max', type=bool, default=False, help='if True, min_max_norm(model(x)) ') 
parser.add_argument('--use_ema', action='store_true', help='vq ema')
parser.add_argument('--dbu_per_px', type=str, default='1um', help='')
parser.add_argument('--checkpoint_path', type=str, default='', help='')
parser.add_argument('--num_embeddings', type=int, default=512, help='number of codebooks vector')
parser.add_argument('--auto_encoder', action='store_true', help='auto encoder mode ')
parser.add_argument('--metric_type', type=str, default='max', help='max or quantile')

args = parser.parse_args()



class IRDropPrediction(LightningModule):
    def __init__(self, lr,):
                #  epoch=args.epoch,arch=args.arch,finetune=args.finetune,metric_type=args.metric_type,in_ch=args.in_ch,pdn_zeros=args.pdn_zeros,loss=args.loss,num_ebedd=args.num_embeddings):
        super().__init__()
        self.lr = lr
        self.num_workers = 4 if args.dataset.lower() == 'began' else 0
        self.num_workers=4


        self.train_auto_encoder = True if args.auto_encoder else False

        self.model = build_model(args.arch,args.dropout,args.finetune,args.in_ch,args.use_ema,num_embeddings=args.num_embeddings)
        if args.finetune:
            self.model = init_weights_chkpt(self.model,args.save_folder)
        elif args.checkpoint_path:
            self.model = init_weights_chkpt(self.model,args.checkpoint_path)


        self.criterion = MultiTaskLoss(loss_types=(args.loss,args.loss_aux),
                                    dice_q=args.dice_q,
                                    )

        print(self.criterion.loss_type)
        self.metrics = IRDropMetrics(post_min_max=args.post_min_max)
        self.save_hyperparameters(args)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        dictionary_loss = outputs.get('dictionary_loss',0)

        recon_loss = self.criterion(outputs, targets)
        loss = recon_loss + dictionary_loss

        metrics = self.metrics.compute_metrics(outputs['x_recon'] , targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_mae', metrics['mae'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_f1', metrics['f1'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        dictionary_loss = outputs.get('dictionary_loss',0)

        recon_loss = self.criterion(outputs, targets)
        loss = recon_loss + dictionary_loss

        metrics = self.metrics.compute_metrics(outputs['x_recon'], targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_f1', metrics['f1'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_mae', metrics['mae'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        network_params = self.model.parameters()
        if args.optim == 'adam':
            optimizer_network = optim.Adam(network_params, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optim == 'sgd':
            optimizer_network = optim.SGD(network_params, lr=args.lr)
        elif args.optim == 'adamw':
            optimizer_network = optim.AdamW(network_params, lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise ValueError(f'Optimizer {args.optim} is not supported')

        # Configure network scheduler
        if args.scheduler == 'cosineanealing':
            scheduler_network = optim.lr_scheduler.CosineAnnealingLR(optimizer_network, T_max=args.epoch, eta_min=1e-7)
        elif args.scheduler == 'cosineanealingwarmup':
            scheduler_network = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_network, T_0=40, T_mult=2, eta_min=0.000001)
        else:
            raise ValueError(f'Scheduler {args.scheduler} is not supported')

        return {
            "optimizer": optimizer_network,
            "lr_scheduler": {
                "scheduler": scheduler_network,
                "monitor": "val_mae"
            }
        }

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if args.dataset.lower() == 'began':
                self.train_dataset, self.val_dataset = build_dataset(img_size=args.img_size,
                                                                     pdn_density_p=args.pdn_density_dropout,
                                                                     pdn_zeros=args.pdn_zeros,
                                                                     in_ch=args.in_ch,
                                                                     use_raw=args.use_raw
                                                                     )
            elif args.dataset.lower() == 'iccad':
                self.train_dataset, self.val_dataset = build_dataset_iccad(finetune=args.finetune,
                                                                            pdn_density_p=args.pdn_density_dropout,
                                                                            pdn_zeros=args.pdn_zeros,
                                                                            in_ch=args.in_ch,
                                                                            img_size=args.img_size,
                                                                            use_raw=args.use_raw

                                                                            )
            elif args.dataset.lower() == 'asap7':
                self.train_dataset, self.val_dataset = build_dataset_began_asap7(finetune=args.finetune,
                                                                                   train=True,
                                                                                    in_ch=args.in_ch,
                                                                                    img_size=args.img_size,
                                                                                    use_raw=args.use_raw
                                                                                   )
            elif args.dataset.lower() == 'cus':
                print('selected folders : ',args.dbu_per_px)
                self.train_dataset, self.val_dataset = build_dataset_5m(img_size=args.img_size,
                                                                        use_raw=args.use_raw,
                                                                        in_ch=args.in_ch,
                                                                        train=True,
                                                                        unit=args.dbu_per_px
                                                                                   )
            else:
                raise NameError('check dataset name')
        
        def test_dt(dt):
            inp = dt.__getitem__(0)[0]
            print('inp shape : ',inp.shape)
            assert args.in_ch == inp.size(0), f"{args.in_ch} is not matching {inp.size(0)}"
        test_dt(self.train_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=self.num_workers)

def init_weights_chkpt(model, save_folder):
    # List all files in the save folder
    checkpoint_files = [f for f in os.listdir(save_folder) if f.endswith('.pth')]

    # Assuming you want to load the latest checkpoint based on the file naming scheme
    # Sort files based on the version (assuming the format of the files is consistent)
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)

    # Construct full checkpoint path
    checkpoint_path = os.path.join(save_folder, checkpoint_files[0])
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint into the model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['net'])

    return model

def get_run_paths():
    # 기본 베이스 경로 (원하는 기본 디렉토리로 수정 가능)
    base_log_dir = args.log_dir
    base_ckpt_dir = args.save_folder

    # finetune 여부에 따라 하위 폴더 구성
    if args.finetune:
        subfolder = os.path.join(args.checkpoint_path, "finetune",args.loss)
    else:
        subfolder = os.path.join(str(args.in_ch), args.arch, args.dataset, args.loss)
    
    if args.post_fix:
        subfolder = os.path.join(subfolder, args.post_fix)

    # 최종 경로 생성
    log_dir = os.path.join(base_log_dir, subfolder)
    checkpoint_dir = os.path.join(base_ckpt_dir, subfolder)
    return log_dir, checkpoint_dir

class CustomCheckpoint(Callback):
    def __init__(self, checkpoint_dir,  metric_name='val_mae', mode='min'):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
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
                    self.best_model_file_name = f'{args.arch}_embd{args.num_embeddings}_{args.metric_type}_{trainer.current_epoch}_{val_metric:.4f}.pth'
                    new_path = os.path.join(self.checkpoint_dir, self.best_model_file_name)
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    torch.save(state, new_path)
                    print(f'Saved new best model to: {new_path}')

def main(i):
    model = IRDropPrediction(lr=args.lr)
    logdir, checkpoint_dir = get_run_paths()
    logger = TensorBoardLogger(save_dir=logdir, name=f'')
    monitor_metric = f'val_{args.monitor}'  
    print('monitor : ',monitor_metric)
    checkpoint_callback = CustomCheckpoint(
        checkpoint_dir=checkpoint_dir,
        metric_name=monitor_metric,
        mode='min' if monitor_metric == 'val_mae' else 'max'
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
        precision='16-mixed' if args.mixed_precision else '32-true'
    )

    trainer.fit(model)
    torch.cuda.empty_cache()

def main_cross_val(id):
    pass

if __name__ == '__main__':
    if args.cross_val and args.dataset=='asap7':
        cross_val_ids = [0,1,2,3] # dataset of cross validation has 4 samples 
        for id in cross_val_ids:
            main_cross_val(id)
    else:
        for i in range(1, args.repeat + 1):
            main(i)