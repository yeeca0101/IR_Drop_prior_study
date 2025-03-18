import argparse
import os
import torch
import torch.backends as bc
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy

from models import *
from models.parts.vqvae import init_weights

from metric import IRDropMetrics 
from ir_dataset import IRDropDataset, build_dataset_iccad, build_dataset, build_dataset_began_asap7, IRDropFineTuneDataset, build_dataset_5m
# A100 관련 설정
bc.cuda.matmul.allow_tf32 = True
bc.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='PyTorch Lightning IR-Drop Evaluation')
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
parser.add_argument('--vq_lr', default=5e-5, type=float, help='learning rate for vq')
parser.add_argument('--save_folder', default='checkpoint', type=str, help='checkpoint save folder')
parser.add_argument('--epoch', default=100, type=int, help='max epoch')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--gpus', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--dataset', type=str, default='iccad', help='select dataset: Began, iccad, asap7, cus')
parser.add_argument('--finetune', action='store_true', help='pre-train or fine-tuning')
parser.add_argument('--optim', type=str, default='adam', help='optimizer')
parser.add_argument('--dropout', type=str, default='dropblock', help='dropout type (for attnV2)')
parser.add_argument('--arch', type=str, default='attn_12ch', help='model architecture : [attn_12ch, attn_base_12ch, attnv2, vqvae]')
parser.add_argument('--loss', type=str, default='default', help='loss type : [default, comb, default_edge, edge, ssim, dice]')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--dice_q', type=float, default=0.9, help='dice q')
parser.add_argument('--pdn_density_dropout', type=float, default=0.0, help='pdn density dropout (0~1)')
parser.add_argument('--loss_with_logit', action='store_true', help='if False, apply sigmoid fn')
parser.add_argument('--post_fix', type=str, default='', help='post fix of finetune checkpoint path')
parser.add_argument('--scheduler', type=str, default='cosineanealingwarmup', help='lr scheduler')
parser.add_argument('--cross_val', action='store_true', help='for asap7 dataset training')
parser.add_argument('--pdn_zeros', action='store_true', help='pdn density channels zeros')
parser.add_argument('--in_ch', type=int, default=12, help='number of input channels')
parser.add_argument('--img_size', type=int, default=512, help='input image size')
parser.add_argument('--mixed_precision', action='store_true', help='mixed precision evaluation')
parser.add_argument('--monitor', type=str, default='f1', help='monitor metric')
parser.add_argument('--use_raw', action='store_true', help='use raw ir drop map (not normalized)')
parser.add_argument('--vqvae_size', type=str, default='default', help='vqvae model size : [default, small, large]')
parser.add_argument('--post_min_max', action='store_true', help='if True, apply min_max_norm to model output')
parser.add_argument('--use_ema', action='store_true', help='use vq ema')
parser.add_argument('--dbu_per_px', type=str, default='1um', help='210nm or 1um')
parser.add_argument('--checkpoint_path', type=str, default='', help='Path to checkpoint folder (required for evaluation)')
parser.add_argument('--num_embeddings', type=int, default=512, help='number of codebook vectors')
args = parser.parse_args()

print('pdn_zeros : ',args.pdn_zeros)
print('post_min_max : ',args.post_min_max)
print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
# 평가 전용 LightningModule (train.py와 동일)
class IRDropPrediction(LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.num_workers = 4
        self.use_ema = args.use_ema

        self.model = build_model(args.arch, args.dropout, args.finetune, args.in_ch, self.use_ema, num_embeddings=args.num_embeddings)
        if args.checkpoint_path:
            self.model = init_weights_chkpt(self.model, args.checkpoint_path)

        self.criterion = nn.MSELoss()

        print("Loss type:", self.criterion.__class__.__name__)
        self.metrics = IRDropMetrics(post_min_max=args.post_min_max)

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)['x_recon']
        loss = self.criterion(outputs, targets)

        # if args.use_raw:
        #     outputs = self.val_dataset.inverse(outputs)
        #     outputs = F.interpolate(outputs,targets.shape[-2:],mode='area')

        metrics = self.metrics.compute_metrics(outputs, targets)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_f1', metrics['f1'], prog_bar=True, sync_dist=True)
        self.log('val_mae', metrics['mae'], prog_bar=True, sync_dist=True)
        self.log('val_ssim', metrics['ssim'], prog_bar=True, sync_dist=True)
        return loss


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if args.dataset.lower() == 'began':
                self.train_dataset, self.val_dataset = build_dataset(img_size=args.img_size,
                                                                     pdn_density_p=args.pdn_density_dropout,
                                                                     pdn_zeros=args.pdn_zeros,
                                                                     in_ch=args.in_ch,
                                                                     use_raw=args.use_raw)
            elif args.dataset.lower() == 'iccad':
                
                self.val_dataset = IRDropDataset(root_path='/data/ICCAD_2023/fake-circuit-data_20230623',
                        selected_folders=['fake-circuit-data-npy'],
                        img_size=args.img_size,
                        post_fix_path='',
                        target_layers = ['m1', 'm4', 'm7', 'm8', 'm9', 'm14', 'm47', 'm78', 'm89'],
                        preload=False,
                        pdn_density_p=0.,
                        pdn_zeros=args.pdn_zeros,
                        in_ch=args.in_ch,
                        use_raw=args.use_raw,
                        train=False
                    )
            elif args.dataset.lower() in ['iccad_real', 'iccad_fine']:
                root_path='/data/ICCAD_2023/real-circuit-data_20230615'
                testcase_folders = os.listdir(root_path)
                self.val_dataset = IRDropFineTuneDataset(root_path=root_path,
                                    selected_folders=testcase_folders,
                                    img_size=args.img_size,
                                    target_layers = ['m1', 'm4', 'm7', 'm8', 'm9', 'm14', 'm47', 'm78', 'm89'],
                                    train=False,
                                    in_ch=args.in_ch,
                                    pdn_zeros=args.pdn_zeros,
                                    use_raw=args.use_raw
                                ) if args.dataset == 'iccad_real' else \
                                    build_dataset_iccad(
                                    finetune=True,
                                    img_size=args.img_size,
                                    pdn_density_p=args.pdn_density_dropout,
                                    pdn_zeros=args.pdn_zeros,
                                    in_ch=args.in_ch,
                                    use_raw=args.use_raw)[1]


            elif args.dataset in ['hidden', 'iccad_hidden']:
                root_path='/data/ICCAD_2023/hidden-real-circuit-data'
                testcase_folders = os.listdir(root_path)
                self.val_dataset = IRDropFineTuneDataset(root_path=root_path,
                                    selected_folders=testcase_folders,
                                    img_size=args.img_size,
                                    target_layers = ['m1', 'm4', 'm7', 'm8', 'm9', 'm14', 'm47', 'm78', 'm89'],
                                    train=False,
                                    in_ch=args.in_ch,
                                    pdn_zeros=args.pdn_zeros,
                                    use_raw=args.use_raw
                                )
                
            elif args.dataset.lower() == 'asap7':
                self.train_dataset, self.val_dataset = build_dataset_began_asap7(finetune=args.finetune,
                                                                                   train=True,
                                                                                   in_ch=args.in_ch,
                                                                                   img_size=args.img_size,
                                                                                   use_raw=args.use_raw)

        
        def test_dt(dt):
            inp = dt.__getitem__(0)[0]
            assert args.in_ch == inp.size(0), f"{args.in_ch} is not matching {inp.size(0)}"
        test_dt(self.val_dataset)

  
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=self.num_workers)

# checkpoint 폴더 내 최신 파일을 불러오는 함수 (train.py와 동일)
def init_weights_chkpt(model, checkpoint_folder):
    checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith('.pth')]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_folder}")
    # 파일 이름 마지막에 붙은 숫자를 기준으로 정렬 (최신 모델)
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
    checkpoint_path = os.path.join(checkpoint_folder, checkpoint_files[0])
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['net'])
    return model

def main():
    # checkpoint_path는 평가에 반드시 필요합니다.
    if not args.checkpoint_path:
        raise ValueError("check --checkpoint_path ")
    
    model = IRDropPrediction(lr=args.lr)
    # validation 데이터셋 구성 (Lightning이 자동으로 setup을 호출하지 않을 수 있으므로 수동 호출)
    model.setup('fit')
    
    trainer = Trainer(
        devices=args.gpus,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        logger=False,  # 터미널 출력만 사용
        enable_checkpointing=False,
        precision='16-mixed' if args.mixed_precision else '32-true',
        strategy=DDPStrategy(find_unused_parameters=False) if args.gpus > 1 else 'auto'
    )
    
    print("Starting evaluation...")
    results = trainer.validate(model, dataloaders=model.val_dataloader())
    print("Evaluation results: ", args.checkpoint_path.split('checkpoint/')[-1])
    for res in results:
        print(res)
    
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
