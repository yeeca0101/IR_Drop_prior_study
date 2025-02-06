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


def get_dataset(dt,split='train',get_case_name=True,pdn_zeros=False,in_ch=12,img_size=512,use_raw=False):
    if dt == 'iccad_train':
        dataset=build_dataset_iccad(pdn_zeros=pdn_zeros,in_ch=in_ch,img_size=img_size,use_raw=use_raw)[0] if  split == 'train' else build_dataset_iccad(pdn_zeros=pdn_zeros,in_ch=in_ch,img_size=img_size,use_raw=use_raw)[1]
        print(f'iccad_pretrain {split}')
    elif dt == 'iccad_fine':                        
        dataset=build_dataset_iccad_finetune(pdn_zeros=pdn_zeros,in_ch=in_ch,img_size=img_size,use_raw=use_raw)[0] if  split == 'train' else build_dataset_iccad_finetune(return_case=True,pdn_zeros=pdn_zeros,in_ch=in_ch,img_size=img_size,use_raw=use_raw)[1]
        print(f'iccad_fine {split}')
    elif dt == 'asap7_train_val':                        
        dataset=build_dataset_began_asap7(in_ch=in_ch,img_size=img_size,use_raw=use_raw)[0] if  split == 'train' else build_dataset_began_asap7(in_ch=in_ch,img_size=img_size,use_raw=use_raw)[1]
        print(f'asap7_train_val : {split}')
    elif dt == 'cus_210nm':  
        if split == 'test':
            dataset=build_dataset_5m_test(in_ch=in_ch,img_size=img_size,use_raw=use_raw,selected_folders=['210nm_numpy'])                        
        else: dataset=build_dataset_5m(in_ch=in_ch,img_size=img_size,use_raw=use_raw,selected_folders=['210nm_numpy'])[0] if  split == 'train' else build_dataset_5m(in_ch=in_ch,img_size=img_size,use_raw=use_raw,selected_folders=['210nm_numpy'])[1]
        print(f'cus_210nm : {split}')
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
    all_images = torch.cat([inp, target.unsqueeze(0)], dim=0)
    
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

def calculate_metrics(pred, target, th):
    # Convert tensors to numpy
    pred_np = to_numpy(pred)
    target_np = to_numpy(target)
    
    # Threshold using percentile
    target_np = (target_np >= np.percentile(target_np, th)).astype(int)
    pred_np = (pred_np >= np.percentile(pred_np, th)).astype(int)
    
    # Dice coefficient
    dice_coeff = 1 - DiceLoss()(torch.tensor(target_np[np.newaxis, np.newaxis, ...]),
                                torch.tensor(pred_np[np.newaxis, np.newaxis, ...]))
    
    # F1 score
    f1 = f1_score(target_np.flatten(), pred_np.flatten())
    
    return dice_coeff, f1

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
