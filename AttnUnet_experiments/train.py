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
from metric import IRDropMetrics  # 이전에 구현한 메트릭 클래스
from ir_dataset import IRDropDataset,build_dataset_iccad,build_dataset,build_dataset_began_asap7
from loss import *


# A100
bc.cuda.matmul.allow_tf32 = True
bc.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='PyTorch Lightning IR-Drop Prediction')
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate [pre-train:5e-4, fine-tuning:1e-5~5e-4]')
parser.add_argument('--save_folder', default='checkpoint', type=str, help='checkpoint save folder')
parser.add_argument('--log_dir', default='logs', type=str, help='tensorboard log folder')
# parser.add_argument('--data_folder', default='/path/to/ir_drop_data', type=str, help='IR-drop data root folder')
parser.add_argument('--repeat', default=1, type=int, help='number of repetitive training')
parser.add_argument('--epoch', default=100, type=int, help='max epoch')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--gpus', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--dataset', type=str, default='iccad', help='select BeGAN or iccad')
parser.add_argument('--finetune', type=bool, default=False, help='pre-train or fine-tuning')
parser.add_argument('--optim', type=str, default='adam', help='opitmizer')
parser.add_argument('--dropout', type=str, default='dropblock', help='for attnV2')
parser.add_argument('--arch', type=str, default='attn_12ch', help='sup model : [attn_12ch, attn_base_12ch, attnv2]')
parser.add_argument('--loss', type=str, default='default', help='sup loss : [default, comb, default_edge,edge,ssim,dice]')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='prev : 5e4')
parser.add_argument('--dice_q', type=float, default=0.9, help='top %')
parser.add_argument('--pdn_density_dropout', type=float, default=0.0, help='0.~1. 0: False, 1 : always')
parser.add_argument('--loss_with_logit', type=bool, default=True, help='if False, apply sigmoid fn')
parser.add_argument('--post_fix', type=str, default='', help='post fix of finetune checkpoint path')
parser.add_argument('--scheduler', type=str, default='consineanealingwarmup', help='lr scheduler')
parser.add_argument('--cross_val', type=bool, default=False, help='for asap7 dataset training')
parser.add_argument('--pdn_zeros', type=bool, default=False, help='eff_dist, pdn_density channels zeros')
parser.add_argument('--in_ch', type=int, default=12, help='1 : only use current map')
parser.add_argument('--img_size', type=int, default=512, help='input img_size')
parser.add_argument('--mixed_precision', type=bool, default=False, help='mixed_precision')
parser.add_argument('--monitor', type=bool, default='f1', help='monitor metric')


args = parser.parse_args()


# def build_model(arch):
#     if arch == 'attn_12ch':
#         model = AttU_Net(img_ch=12)
#     elif arch == 'attn_base':
#         model = AttnUnetBase()
#     elif arch == 'attnv2':
#         model = AttnUnetV2(dropout_name=args.dropout,dropout_p=0.05 if args.finetune else 0.1, in_ch=args.in_ch)
#     elif arch == 'attnv3':
#         model = AttnUnetV3(dropout_name=args.dropout,dropout_p=0.05 if args.finetune else 0.1, in_ch=args.in_ch)
#     elif arch == 'attnv4':
#         model = AttnUnetV4(dropout_name=args.dropout,dropout_p=0.05 if args.finetune else 0.1, in_ch=args.in_ch, num_head=2 if args.in_ch==2 else 4)    
#     else:
#         raise NameError(f'arch type error : {arch}')
#     return model


class IRDropPrediction(LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_workers = 4 if args.dataset.lower() == 'began' else 0
        self.num_workers=0

        self.model = build_model(args.arch)
        if args.finetune:
            self.model = init_weights_chkpt(self.model,args.save_folder)
        else:
            init_weights(self.model)

        self.criterion = LossSelect(loss_type=args.loss,
                                    use_cache=True if args.loss == 'cache' else False,
                                    dice_q=args.dice_q,
                                    loss_with_logit=args.loss_with_logit
                                    ) #nn.MSELoss() #combined_loss #CustomLoss() #nn.MSELoss()
        print(self.criterion.loss_fn)
        self.metrics = IRDropMetrics(loss_with_logit=args.loss_with_logit)

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
            optimizer = optim.Adam(self.parameters(), lr=self.lr,weight_decay=args.weight_decay)
        elif args.optim == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
        else:
            raise NameError(f'not support {args.optim}')
        if args.scheduler == 'cosineanealing':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epoch,eta_min=1e-7)
        elif args.scheduler == 'cosineanealingwarmup':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2, eta_min=0.000001)
        else:
            raise NameError(f'not support yet {args.scheduler}')
        
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epoch,eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_mae"}

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if args.dataset.lower() == 'began':
                self.train_dataset, self.val_dataset = build_dataset(img_size=args.img_size,
                                                                     pdn_density_p=args.pdn_density_dropout,
                                                                     pdn_zeros=args.pdn_zeros,
                                                                     in_ch=args.in_ch
                                                                     )
            elif args.dataset.lower() == 'iccad':
                self.train_dataset, self.val_dataset = build_dataset_iccad(finetune=args.finetune,
                                                                            pdn_density_p=args.pdn_density_dropout,
                                                                            pdn_zeros=args.pdn_zeros,
                                                                            in_ch=args.in_ch,
                                                                            img_size=args.img_size
                                                                            )
            elif args.dataset.lower() == 'asap7':
                self.train_dataset, self.val_dataset = build_dataset_began_asap7(finetune=args.finetune,
                                                                                   train=True,
                                                                                    in_ch=args.in_ch,
                                                                                    img_size=args.img_size
                                                                                   )
            else:
                raise NameError('check dataset name')
        
        def test_dt(dt):
            inp = dt.__getitem__(0)[0]
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
    checkpoint_files.sort(key=lambda x: int(x.split('_')[3].split('.')[0]), reverse=True)

    # Construct full checkpoint path
    checkpoint_path = os.path.join(save_folder, checkpoint_files[0])
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint into the model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['net'])

    return model

class CustomCheckpoint(Callback):
    def __init__(self, checkpoint_dir, repeat_idx, metric_name='val_mae', mode='min',post_fix=''):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir if args.post_fix == '' else f'{checkpoint_dir}/{args.post_fix}'
        if args.finetune:
            self.checkpoint_dir = os.path.join(self.checkpoint_dir,'finetune',args.loss,post_fix)
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_model_file_name = ""
        self.repeat_idx = repeat_idx
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
                        os.remove(old_path)
                    
                    self.best_model_file_name = f'IR_Drop_Model_{self.repeat_idx}_{trainer.current_epoch}_{val_metric:.4f}.pth'
                    new_path = os.path.join(self.checkpoint_dir, self.best_model_file_name)
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    torch.save(state, new_path)
                    print(f'Saved new best model to: {new_path}')

def make_logdir():
    
    if args.finetune:           
        pre_train_loss = args.save_folder.split('/')[-1]
        logdir = os.path.join(args.log_dir,f'{args.arch}/{args.dataset}/{pre_train_loss}')
        logdir = os.path.join(logdir,f'finetune/{args.loss}')
    else:
        logdir = os.path.join(args.log_dir,f'{args.arch}/{args.dataset}/{args.loss}')
        if args.arch == 'attnv2':   logdir = os.path.join(logdir,args.dropout)
    
    if not args.loss_with_logit:logdir = os.path.join(logdir,'sigmoid')
    logdir = f'{logdir}' if args.post_fix =='' else f'{logdir}/{args.post_fix}'

    return logdir

def main(i):
    model = IRDropPrediction(lr=args.lr)
    logdir = make_logdir()
    logger = TensorBoardLogger(save_dir=logdir, name=f'repeat_{i}')
    
    if args.monitor == 'mae':
        monitor_metric = 'val_mae'  
    else: monitor_metric = 'val_f1'
    print('monitor : ',monitor_metric)

    checkpoint_callback = CustomCheckpoint(
        checkpoint_dir=args.save_folder,
        repeat_idx=i,
        metric_name=monitor_metric, 
        mode='min' if monitor_metric in 'val_mae' else 'max' ,
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