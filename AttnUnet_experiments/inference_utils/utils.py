import os
import sys
sys.path.append('/workspace')
sys.path.append('/workspace/AttnUnet_experiments')
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from unets import AttU_Net
from models.papers_attn_unet import AttUNet
from models.papers_model_v2 import AttnUnetV2
from ir_dataset import *
from models.papers_model import AttnUnetBase
from metric import IRDropMetrics
from models import *


def get_dataset(dt,split='train',get_case_name=True,pdn_zeros=False,in_ch=12,img_size=512,
                use_raw=False,types='1um',root_path=None,
                target_norm_type=None,input_norm_type=None):
    if dt == 'iccad_train':
        dataset=build_dataset_iccad(pdn_zeros=pdn_zeros,in_ch=in_ch,img_size=img_size,use_raw=use_raw)[0] if  split == 'train' else build_dataset_iccad(pdn_zeros=pdn_zeros,in_ch=in_ch,img_size=img_size,use_raw=use_raw)[1]
        print(f'iccad_pretrain {split}')
    elif dt == 'iccad_fine':                        
        dataset=build_dataset_iccad_finetune(pdn_zeros=pdn_zeros,in_ch=in_ch,img_size=img_size,use_raw=use_raw)[0] if  split == 'train' else build_dataset_iccad_finetune(return_case=True,pdn_zeros=pdn_zeros,in_ch=in_ch,img_size=img_size,use_raw=use_raw)[1]
        print(f'iccad_fine {split}')
    elif dt == 'iccad_hidden':                        
        dataset=build_dataset_iccad_hidden(pdn_zeros=pdn_zeros,in_ch=in_ch,img_size=img_size,use_raw=use_raw)
        print(f'iccad_hidden {split}')
    elif dt == 'asap7_train_val':                        
        dataset=build_dataset_began_asap7(in_ch=in_ch,img_size=img_size,use_raw=use_raw)[0] if  split == 'train' else build_dataset_began_asap7(in_ch=in_ch,img_size=img_size,use_raw=use_raw)[1]
        print(f'asap7_train_val : {split}')
    elif dt == 'cus':
        kwargs = {'root_path':root_path} if root_path else {}  
        train_dt,val_dt = build_dataset_5m(img_size=img_size,train=True,in_ch=in_ch,
                                           use_raw=use_raw,
                                           unit=types,target_norm_type=target_norm_type,input_norm_type=input_norm_type,**kwargs)
        if split =='train':
            return train_dt
        else: return val_dt
    elif dt == 'cus_1um':       
        if split == 'test':
            dataset=build_dataset_5m_test(in_ch=in_ch,img_size=img_size,use_raw=use_raw,selected_folders=['1um_numpy'])                 
        else : dataset=build_dataset_5m(in_ch=in_ch,img_size=img_size,use_raw=use_raw,selected_folders=['1um_numpy'])[0] if  split == 'train' else build_dataset_5m(in_ch=in_ch,img_size=img_size,use_raw=use_raw,selected_folders=['1um_numpy'])[1]
    elif dt == 'cus_super':       
        # TODO
        # if split == 'test':
        #     dataset=build_dataset_5m_test(in_ch=in_ch,img_size=img_size,use_raw=use_raw,selected_folders=['1um_numpy'])                 
        dataset=build_dataset_5m(in_ch=in_ch,img_size=img_size,use_raw=use_raw,train_auto_encoder=True)[0] if  split == 'train' else build_dataset_5m(in_ch=in_ch,img_size=img_size,use_raw=use_raw,train_auto_encoder=True)[1]
        print(f'cus_super_resolutoin : {split}')
    elif dt == 'asap7_fine':
        dataset = TestASAP7Dataset(root_path='/data/real-circuit-benchmarks/asap7/numpy_data',
                                   target_layers=['m2', 'm5', 'm6', 'm7', 'm8', 'm25', 'm56', 'm67', 'm78'],
                                   img_size=img_size, use_irreg=False, preload=False, train=False,return_case=get_case_name,
                                   in_ch=in_ch,debug=False,use_raw=use_raw)        
        print('asap7_fine')
    elif dt == 'began_fine':
        root_path = '/data/real-circuit-benchmarks/nangate45/numpy_data'
        selected_folders = os.listdir(root_path)
        dataset = TestASAP7Dataset(
            root_path=root_path,
            # selected_folders=selected_folders,
            target_layers=['m1', 'm4', 'm7', 'm8', 'm9', 'm14', 'm47', 'm78', 'm89'],
            img_size=img_size,in_ch=in_ch,
            # pdn_zeros=pdn_zeros,
            train=False,
            use_irreg=False,debug=False,use_raw=use_raw
            # post_fix_path=''
        )
        print(f'began fine {split}')
    else:
        dataset=build_dataset(pdn_zeros=pdn_zeros,in_ch=in_ch,img_size=img_size,use_raw=use_raw)[0] if  split == 'train' else build_dataset(pdn_zeros=pdn_zeros,in_ch=in_ch,img_size=img_size,use_raw=use_raw)[1]
        # dataset = IRDropDataset(root_path='/data/BeGAN-circuit-benchmarks',
        #                     selected_folders=['nangate45/set1_numpy','nangate45/set2_numpy'],
        #                     img_size=512,
        #                    target_layers = ['m1', 'm4', 'm7', 'm8', 'm9', 'm14', 'm47', 'm78', 'm89'],
        #               )
        print(f'began {split}')
        
    return dataset

def find_pth_files(folder_path, mode='max'):
    if os.path.isfile(folder_path) and folder_path.endswith('.pth'):
        return folder_path 

    pth_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pth'):
                pth_files.append(os.path.join(root, file))
        break
    
    if pth_files == []: raise FileNotFoundError('not found the checkpoint in ', folder_path)

    pth_files.sort()  

    if mode == 'min':
        return pth_files[0] if pth_files else None
    elif mode == 'max':
        return pth_files[-1] if pth_files else None
    return pth_files

def visualize_images(inp, target, cols=4):
    # 입력 텐서와 타겟 텐서를 결합
    if target.dim==2:
        target = target.unsqueeze(0)
    all_images = torch.cat([inp, target], dim=0)
    
    # 행 수 계산
    rows = (len(all_images) + cols - 1) // cols
    
    # 플롯 생성
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    
    for i, img in enumerate(all_images):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
        ax.imshow(img.cpu().numpy(), cmap='jet')
        ax.axis('off')
        if i < len(all_images) - 1:
            ax.set_title(f'Input {i+1}')
        else:
            ax.set_title('Target')
    
    # 남은 서브플롯 제거
    for i in range(len(all_images), rows * cols):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def load_score(chkpt_path):
    chkpt_path = find_pth_files(chkpt_path)
    d = torch.load(chkpt_path,map_location='cpu')
    mae,f1 = d['mae'], d['f1']
    print('mae : ',mae)
    print('f1 : ',f1)   


def min_max_norm(x):
    return (x-x.min())/(x.max()-x.min())


class DiceLoss:
    def __init__(self):
        pass

    def __call__(self, target, pred):
        smooth = 1.0
        intersection = (target * pred).sum()
        return 1 - (2. * intersection + smooth) / (target.sum() + pred.sum() + smooth)

def to_numpy(x):
    return x.detach().cpu().numpy()

def calculate_metrics(pred, target, mask_opt,top_region=0.9):
    # torch 기반으로 threshold 계산
    if mask_opt == 'max':
        threshold = torch.max(target) * top_region
        target_bin = (target >= threshold).int()
        pred_bin = (pred >= threshold).int()
    elif 'quantile' == mask_opt:
        threshold_t = torch.quantile(target, top_region)
        threshold_p = torch.quantile(pred, top_region)
        target_bin = (target >= threshold_t).int()
        pred_bin = (pred >= threshold_p).int()
    elif 'quantile_target' == mask_opt:
        threshold_t = torch.quantile(target, top_region)
        target_bin = (target >= threshold_t).int()
        pred_bin = (pred >= threshold_t).int()
    elif 'quantile_pred' == mask_opt:
        threshold_t = torch.quantile(pred, top_region)
        target_bin = (target >= threshold_t).int()
        pred_bin = (pred >= threshold_t).int()
    else:
        raise ValueError("Invalid mask_opt. Use 'max', 'quantile', or 'quantile_target'.")

    # # Dice coefficient (torch tensor로 유지)
    # dice_coeff = 1 - DiceLoss()(target_bin.unsqueeze(0).unsqueeze(0).float(), 
    #                             pred_bin.unsqueeze(0).unsqueeze(0).float())

    # f1_score는 numpy로 변환
    f1 = f1_score(target_bin.flatten().cpu().numpy(), pred_bin.flatten().cpu().numpy())

    # mae_10 계산 (mask된 부분만)
    mask = target_bin == 1
    diff = torch.abs(pred - target)
    print(f'top_region : {target[mask].shape[0]}')
    if mask.sum() > 0:  # 마스크된 부분이 있는 경우
        mae_10 = diff[mask].mean().item()
    else:  # 마스크가 없는 경우
        mae_10 = 0.0  # 또는 np.nan 으로 반환 가능

    return mae_10 ,f1


# Numpy에서 top 10% 마스크를 구하는 함수
def numpy_top_10_mask(arr,q=95):
    threshold = np.percentile(arr, q)
    print('np : ',threshold)
    mask = (arr >= threshold).astype(float)  # 상위 10% 마스크
    return mask

# Torch에서 top 10% 마스크를 구하는 함수
def torch_top_10_mask(tensor,q=0.95):
    threshold = torch.quantile(tensor, q)
    print('torch : ',threshold)
    mask = (tensor >= threshold).float()  # 상위 10% 마스크
    return mask


class IRDropMetrics(nn.Module):
    def __init__(self, top_percent=0.9, post_min_max=False, how='max'):
        super(IRDropMetrics, self).__init__()
        self.top_percent = top_percent
        self.eps = 1e-6
        self.q = top_percent
        self.post_min_max = post_min_max

        # SSIM 계산을 위한 모듈 (binary segmentation이므로 channel=1, data_range=1.0 사용)
        # self.ssim_loss = SSIMLoss(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
        self._forward = self.forward_max if how == 'max' else self.forward_quantile 

    @torch.no_grad()
    def forward(self, pred, target):
        return self._forward(pred, target)

    def forward_max(self, pred, target):
        # target_for_ssim = target if target.ndim == 4 else target.unsqueeze(1)
        # ssim_val = 1 - self.ssim_loss(pred, target_for_ssim)
        B, C, H, W = pred.shape
        assert C == 1, "This implementation is for binary segmentation"

        target = target.squeeze(1)
        pred = pred.squeeze(1)

        mae = torch.mean(torch.abs(pred - target))

                # F1 점수를 위한 threshold 계산
        max_target_per_sample = target.view(B, -1).max(dim=1, keepdim=True)[0]  # (B, 1)
        max_pred_per_sample = pred.view(B, -1).max(dim=1, keepdim=True)[0] *self.top_percent  # (B, 1)

        threshold_per_sample = max_target_per_sample * self.top_percent  # (B, 1)

        # (2) binary mask (hotspot 영역) 생성
        threshold_expanded = threshold_per_sample.view(B, 1, 1)  # (B, 1, 1)
        threshold_expanded2 = max_pred_per_sample.view(B, 1, 1)  # (B, 1, 1)
        target_hotspot = (target >= threshold_expanded).int()  # (B, H, W)
        pred_hotspot = (pred >= threshold_expanded).int()  # (B, H, W)

        # (3) TP, FP, FN 계산
        tp = (target_hotspot * pred_hotspot).sum(dim=(1, 2))  # (B,)
        fp = ((1 - target_hotspot) * pred_hotspot).sum(dim=(1, 2))  # (B,)
        fn = (target_hotspot * (1 - pred_hotspot)).sum(dim=(1, 2))  # (B,)

        # (4) Precision, Recall, F1 score
        precision = tp / (tp + fp + self.eps)  # (B,)
        recall = tp / (tp + fn + self.eps)  # (B,)
        f1 = 2 * precision * recall / (precision + recall + self.eps)  # (B,)

        f1_score = f1.mean()  

        return {"mae": mae.item(), "f1": f1_score.item()}


    def forward_quantile(self, pred, target):
        # target_for_ssim = target if target.ndim == 4 else target.unsqueeze(1)
        # ssim_val = 1 - self.ssim_loss(pred, target_for_ssim)

        B, C, H, W = pred.shape
        assert C == 1, "This implementation is for binary segmentation"
        pred = pred.view(B, -1)
        target = target.view(B, -1)
        mae = torch.mean(torch.abs(pred - target))

        # F1 점수를 위한 threshold 계산
        pred_threshold = torch.quantile(pred.float(), self.q, dim=1, keepdim=True)
        target_threshold = torch.quantile(target.float(), self.q, dim=1, keepdim=True)
        pred_bin = (pred > pred_threshold).float()
        target_bin = (target > target_threshold).float()

        # Dice 계수 (F1) 계산
        intersection = (pred_bin * target_bin).sum(dim=1)
        union = pred_bin.sum(dim=1) + target_bin.sum(dim=1)
        f1 = (2. * intersection + self.eps) / (union + self.eps)

        return {"mae": mae.item(), "f1": f1.mean().item()}
    
    @torch.no_grad()
    def compute_metrics(self, pred, target):
        if self.post_min_max: pred = min_max_norm(pred)
        return self.forward(pred, target)