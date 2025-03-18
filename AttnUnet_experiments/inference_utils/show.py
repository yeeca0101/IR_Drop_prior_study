'''
    ymin98@sogang.ac.kr
'''
import sys
sys.path.append('../')
import os  
import re
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from .utils import find_pth_files, min_max_norm, calculate_metrics, to_numpy, DiceLoss
from models import *


def predict_and_visualize(dataset, model, checkpoint_path,
                          index, cols=4, colorbar=False,
                          norm_out=False, device='cuda:0',
                          casename=False, cmap='inferno',
                          use_raw=False,sr=False,with_out_inp=False,
                          plot_mask=False,top_region=0.9,mask_opt='max',plot_scatter=False,out_key = 'x_recon'):
    '''
        args:
            mask_opt : 'max' or 'quantile'
    '''
    model.to(device)
    checkpoint_path = find_pth_files(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path,map_location=device)['net'])
    model.eval()

    sample = dataset.__getitem__(index)
    inp, target = sample[0], sample[1].contiguous()
    print('target max : ',target.max())
    inp_batch = inp.unsqueeze(0).to(device)
    if casename:
        print(sample[2])

    with torch.no_grad():
        pred = model(inp_batch) if not sr else model(inp_batch, target.shape[-2:])
    if len(pred) >= 1: 
        pred = pred[out_key]
    
    pred = pred.detach().cpu() 
    print(pred.shape)

    if norm_out:
        pred = min_max_norm(pred)

    target_min = None
    target_max = None
    if use_raw :
        target_min = target.min()
        target_max = target.max()
        _, t_h, t_w = target.shape
        pred = (target_max-target_min)*pred + target_min
        # else: pred *=0.01
        print(t_h,t_w)
        pred = F.interpolate(pred.unsqueeze(0), size=(1,t_h, t_w), mode='area').squeeze(0)
    print('pred max : ',pred.max())

    mae_map = torch.abs(pred - target) 
    if use_raw:
        print(f'mae : {torch.mean(mae_map).item()*1000:.2f} mV')
    else:
        print('mae : ', round(torch.mean(mae_map).item(),5))

    pred = pred.squeeze(0)

    mae_10, f1 = calculate_metrics(pred, target, mask_opt=mask_opt)
    if use_raw : mae_10 *=1000
    print(f"F1 Score: {f1:.3f}")
    print(f'mae 10 : {mae_10:.3f}' + ' mV' if use_raw else '')
    if inp.dim() == 2:  # 단일 채널, 
        all_images = [inp]
    elif inp.dim() == 3:  # 다중 채널, 
        all_images = [inp[i] for i in range(inp.shape[0])]

    if with_out_inp:
        all_images = [target, pred, mae_map]
        titles = ['Target', 'Prediction', 'MAE Map']
    else:
        all_images += [target, pred, mae_map]
        titles = [f'Input {i + 1}' for i in range(len(all_images) - 3)] + ['Target', 'Prediction', 'MAE Map']

    if plot_mask:
        if mask_opt == 'max':
            threshold = target.max() * top_region
            target_mask = (target >= threshold).float()
            pred_mask = (pred >= threshold).float()
        elif mask_opt == 'quantile':
            pred_threshold = torch.quantile(pred.view(-1),top_region)
            target_threshold = torch.quantile(target.view(-1),top_region)
            pred_mask = (pred >= pred_threshold).float()
            target_mask = (target >= target_threshold).float()
        elif mask_opt == 'quantile_target':
            target_threshold = torch.quantile(target.view(-1),top_region)
            pred_mask = (pred >= target_threshold).float()
            target_mask = (target >= target_threshold).float()
        all_images += [target_mask, pred_mask]
        titles += ['Target Mask', 'Prediction Mask']

    if plot_scatter:
        all_images.append((target.flatten(), pred.flatten(), target.max() * 0.9))  # threshold와 함께 tuple 저장
        titles.append('Target vs. Pred (Scatter)')
        
    rows = (len(all_images) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])


    for i, (img, title) in enumerate(zip(all_images, titles)):
        ax = axes[i]
        if isinstance(img, tuple):  # Scatter인 경우
            target_flat, pred_flat, threshold = img[0].cpu().numpy(), img[1].cpu().numpy(), img[2].cpu().numpy()
            ax.scatter(target_flat, pred_flat, alpha=0.3, s=5, label='All Points')  # 모든 점
            min_val = min(target_flat.min(), pred_flat.min())
            max_val = max(target_flat.max(), pred_flat.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
            mask_high = target_flat >= threshold
            ax.scatter(target_flat[mask_high],pred_flat[mask_high], color='red', s=5, marker='s', label='Target ≥ 90% max')

            ax.set_xlabel('Target IR Drop')
            ax.set_ylabel('Prediction')
            ax.set_title('Scatter Plot: Target vs. Prediction')
            ax.legend()
        else:  # 일반 이미지인 경우
            img = img.squeeze().cpu().numpy()
            if 'Mask' in title:
                im = ax.imshow(img, vmin=img.min(), vmax=img.max())
            elif use_raw and ('Input' not in title):
                im = ax.imshow(img, cmap=cmap, vmin=target_min, vmax=target_max)
            else:
                im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title)
            if 'Input' not in title:
                fig.colorbar(im, ax=ax)

    # 남은 서브플롯 제거
    for i in range(len(all_images), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    return pred

######################################################################################


def plot_distributions(target, prediction, casename='', bins=50):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    distributions = [target.squeeze().cpu().numpy(), prediction.squeeze().cpu().numpy()]
    titles = ['Target Distribution', 'Normalized Pred Distribution']

    for ax, dist, title in zip(axes, distributions, titles):
        ax.hist(dist.flatten(), bins=bins, alpha=0.7, color='b')
        ax.set_title(f'{casename} - {title}')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def plot_distribution_pred(checkpoint_path, dataset,model,index,norm_out=False,device='cuda:0'):
    checkpoint_path = find_pth_files(checkpoint_path)
    model.eval()
    model.load_state_dict(torch.load(checkpoint_path)['net'])

    sample = dataset.__getitem__(index)
    inp, target = sample[0], sample[1]

    inp_batch = inp.unsqueeze(0).to(device)
    model.to(device)

    with torch.no_grad():
        pred = model(inp_batch)
    if len(pred) == 3:
        pred = pred['x_recon']
    pred = pred.detach().cpu()

    if norm_out:
        pred = min_max_norm(pred)

    # test(target,pred)
    plot_distributions(target,pred)


###########################################################################


def plot_f1_mask(dataset,model,checkpoint,index,device,threshold):
    model.to(device)
    model.eval()
    
    model.load_state_dict(torch.load(find_pth_files(checkpoint), map_location=device)['net'])
    sample = dataset.__getitem__(index)
    inp, target = sample[0].to(device), sample[1].to(device)

    # 입력 텐서 준비 (배치 차원 추가)
    inp_batch = inp.unsqueeze(0)

    # 예측 수행
    with torch.no_grad():
        pred = model(inp_batch)['x_recon']
    # 예측 결과 압축 (배치 차원 제거)
    pred = pred.squeeze(0).squeeze(0)
    
    th = 90
    print(pred.shape,target.shape) # 512,512 512,512
    _,f1 = calculate_metrics(pred,target)

    print('f1 : ',f1)
    plt.figure(figsize=(5,5))
    plt.subplot(121)
    plt.imshow(target)
    plt.subplot(122)
    plt.imshow(pred)
    plt.show()



def plot_mask_with_threshold(dataset,model,checkpoint,index,device,threshold):
    model.to(device)
    model.eval()

    # 모델의 파라미터 로드
    model.load_state_dict(torch.load(find_pth_files(checkpoint), map_location=device)['net'])
    sample = dataset.__getitem__(index)
    inp, target = sample[0].to(device), sample[1].to(device)
    inp_batch = inp.unsqueeze(0)

    with torch.no_grad():
        pred = model(inp_batch)['x_recon']
    pred = pred.squeeze(0).squeeze(0)
    

    print(pred.shape,target.shape) # 512,512 512,512
    pred = to_numpy(pred) #(pred >= 0.9).int()
    target = to_numpy(target) #(target >= 0.9).int()
    target = (target >= np.percentile(target, threshold)).astype(int)
    pred = (pred >= np.percentile(pred, threshold)).astype(int)
    print(1-DiceLoss()(torch.tensor(target[np.newaxis,np.newaxis,...]),torch.tensor(pred[np.newaxis,np.newaxis,...])))
    # print(calculate_f1(target,pred))
    plt.figure(figsize=(5,5))
    plt.subplot(121)
    plt.imshow(target)
    plt.subplot(122)
    plt.imshow(pred)
    plt.show()
    
############################################################################################


# Function to extract information from checkpoint path
def parse_checkpoint_path(path):
    pattern = r"(?P<channels>\d+ch)/(?P<version>attnv\d+(?:_\d+)?)/(?P<loss>.+)/default(?:/|$)"
    match = re.search(pattern, path)
    if match:
        return match.group('channels'), match.group('version'), match.group('loss').split('/')[-1]
    return None, None, None

# Function to determine num_embeddings

def get_num_embeddings(state_dict):
    for key, value in state_dict.items():
        if 'e_i_ts' in key.lower():
            return value.shape[1]
        elif 'embedding' in key.lower():
            return value.shape[0]
    return None

def plot_predictions_from_checkpoints(checkpoint_paths, datasets, sample_indices, plot_inputs=[], device='cuda:0', measure=['f1'],cmap='jet',
                                      fontsize=14,colorbar=False,vmin=0,vmax=1, move_raw=False, colorbar_target=False):
    # Setup plot grid dimensions
    rows = len(checkpoint_paths) + 1
    cols = len(sample_indices)
    if plot_inputs:
        rows += len(plot_inputs)

    first_checkpoint = find_pth_files(checkpoint_paths[0])
    first_channels, version_first, loss_first = parse_checkpoint_path(first_checkpoint)
    if first_channels not in datasets:
        raise ValueError(f"Dataset for channel configuration '{first_channels}' not provided.")
    base_dataset = datasets[first_channels]

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    for col_idx, idx in enumerate(sample_indices):
        sample = base_dataset.__getitem__(idx)
        inp, target = sample[0], sample[1]
        if move_raw: target = target * 0.25038

        # Plot Ground Truth (GT)
        if vmin == 'auto':
            im = axes[0, col_idx].imshow(target.squeeze().cpu().numpy(), cmap=cmap,vmin=target.squeeze().cpu().numpy().min(),vmax=target.squeeze().cpu().numpy().max())
        else:
            im = axes[0, col_idx].imshow(target.squeeze().cpu().numpy(), cmap=cmap,vmin=vmin,vmax=vmax)
        axes[0, col_idx].set_title(f"GT (Index {idx})",fontsize=fontsize+5)
        axes[0, col_idx].axis('off')
        if colorbar_target : fig.colorbar(im,ax=axes[0, col_idx])

        if plot_inputs:
            for i, channel_idx in enumerate(plot_inputs):
                im = axes[i + 1, col_idx].imshow(inp[channel_idx].cpu().numpy(), cmap=cmap,vmin=0,vmax=1)
                axes[i + 1, col_idx].set_title(f"Input Channel {channel_idx}")
                axes[i + 1, col_idx].axis('off')
                if colorbar:fig.colorbar(im,ax=axes[i + 1, col_idx])
        current_row = 1 + len(plot_inputs)

        for checkpoint_idx, checkpoint_path_any in enumerate(checkpoint_paths):
            checkpoint_path = find_pth_files(checkpoint_path_any)
            # Parse checkpoint information
            channels, version, loss = parse_checkpoint_path(checkpoint_path)

            cur_dataset = datasets[channels]
            sample = cur_dataset[idx]
            inp, target = sample[0], sample[1]
            inp_batch = inp.unsqueeze(0).to(device)
            
            # Load checkpoint
            state_dict = torch.load(checkpoint_path)['net']
            num_embeddings = get_num_embeddings(state_dict) 

            # Initialize model
            try:
                # channels가 "2ch", "3ch", ... 형태라고 가정하여 숫자 부분만 추출
                in_channels = int(channels.replace("ch", ""))
            except Exception as e:
                raise ValueError(f"Invalid channel format: {channels}. Expected format like '2ch', '3ch', etc.")

            try:
                if 'attnv5_1' in version:
                    model = AttnUnetV5_1(
                        in_ch=in_channels, 
                        out_ch=1, 
                        dropout_name='dropblock', 
                        dropout_p=0.3, 
                        num_embeddings=num_embeddings
                    )
                elif 'attnv5_2' in version:
                    model = AttnUnetV5_2(
                        in_ch=in_channels, 
                        out_ch=1, 
                        dropout_name='dropblock', 
                        dropout_p=0.3, 
                        num_embeddings=num_embeddings
                    )
                elif 'attnv5' in version:
                    model = AttnUnetV5(
                        in_ch=in_channels, 
                        out_ch=1, 
                        dropout_name='dropblock', 
                        dropout_p=0.3, 
                        num_embeddings=num_embeddings
                    )
                elif 'attnv6_1' in version:
                    if 'relu' in checkpoint_path:
                        import torch.nn as nn
                        kwargs = {'act': nn.ReLU()}
                    else:
                        kwargs = {}
                    model = AttnUnetV6_1(
                        in_ch=in_channels, 
                        out_ch=1, 
                        dropout_name='dropblock', 
                        dropout_p=0.3, 
                        num_embeddings=num_embeddings,
                        **kwargs
                    )
                elif 'attnv6_2' in version:
                    if 'relu' in checkpoint_path:
                        import torch.nn as nn
                        kwargs = {'act': nn.ReLU()}
                    else:
                        kwargs = {}
                    model = AttnUnetV6_2(
                        in_ch=in_channels,  # 기존에 in_ch=1로 되어있던 부분을 in_channels로 변경
                        out_ch=1, 
                        dropout_name='dropblock', 
                        dropout_p=0.3, 
                        num_embeddings=num_embeddings,
                        **kwargs
                    )
                elif 'attnv6' in version:
                    model = AttnUnetV6(
                        in_ch=in_channels, 
                        out_ch=1, 
                        dropout_name='dropblock', 
                        dropout_p=0.3, 
                        num_embeddings=num_embeddings
                    )
                else:
                    raise ValueError("알 수 없는 version")
            except Exception as e:
                raise ValueError(f'{checkpoint_path} not match {version} or embeddings : {num_embeddings}') from e

            try:
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()

                # Generate predictions
                with torch.no_grad():
                    pred = model(inp_batch)
                    if isinstance(pred, dict) and 'x_recon' in pred:
                        pred = pred['x_recon']
                    pred = pred.detach().cpu()
            except:
                raise ValueError(f'{checkpoint_path} not match {inp_batch.shape} or {channels}')
            
            # Plot predictions
            if move_raw : pred = pred * 0.025038
            if vmin == 'auto':
                im = axes[current_row, col_idx].imshow(pred.squeeze().numpy(), cmap=cmap,vmin=pred.squeeze().numpy().min(),vmax=pred.squeeze().numpy().max())
            else:
                im = axes[current_row, col_idx].imshow(pred.squeeze().numpy(), cmap=cmap,vmin=vmin,vmax=vmax)
            axes[current_row, col_idx].set_title(f"{version.split('attn')[-1]}, {channels}, {loss}",fontsize=fontsize)
            axes[current_row, col_idx].axis('off')
            if colorbar:fig.colorbar(im,ax=axes[current_row, col_idx])
            # Calculate and display measure
            text=''
            for m in measure:
                if m == 'f1':
                    _, f1 = calculate_metrics(pred, target, th=90)
                    text += f"F1: {f1:.4f}    "
                elif m == 'mae':
                    mae = torch.abs(pred - target).mean().item()
                    text += f"MAE: {mae:.4f}   "
                else:
                    text = f"{m}: N/A"

            axes[current_row, col_idx].text(0.5, -0.01, text, fontsize=fontsize, ha='center', va='top', transform=axes[current_row, col_idx].transAxes)

            current_row += 1

    plt.tight_layout()
    plt.show()

########################################################################################################
# 1um -> 210nm
import torch.nn.functional as F
from ir_dataset_5nm import IRDropInferenceAutoencoderDataset5nm
from losses import SSIMLoss
import albumentations as A

def set_lr_model(checkpoint_path, device='cuda:0'):
    """
    주어진 체크포인트 경로에서 LR 모델을 초기화하고 state_dict를 로드합니다.
    
    Args:
        checkpoint_path (str): 체크포인트 파일 경로.
        device (str or torch.device): 모델을 로드할 장치.
        
    Returns:
        torch.nn.Module: state_dict를 로드하고 eval 모드로 전환한 LR 모델.
    """
    # 1. 체크포인트 경로에서 채널, 버전, loss 정보를 파싱합니다.
    channels, version, loss = parse_checkpoint_path(checkpoint_path)
    if channels is None or version is None:
        raise ValueError(f"체크포인트 경로 파싱에 실패했습니다: {checkpoint_path}")
    
    # 2. 체크포인트 파일 로드 및 state_dict 추출
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['net']
    
    # 3. state_dict에서 임베딩 수 결정
    num_embeddings = get_num_embeddings(state_dict)
    
    # 4. 체크포인트 정보에 따라 입력 채널 수 설정 (2ch이면 2, 아니면 3)
    in_ch = 2 if channels == '2ch' else 3

    # 5. 버전에 따라 해당 모델 클래스를 초기화합니다.
    try:
        if 'attnv5_1' in version:
            model = AttnUnetV5_1(
                in_ch=in_ch, out_ch=1,
                dropout_name='dropblock', dropout_p=0.3,
                num_embeddings=num_embeddings
            )
        elif 'attnv5_2' in version:
            model = AttnUnetV5_2(
                in_ch=in_ch, out_ch=1,
                dropout_name='dropblock', dropout_p=0.3,
                num_embeddings=num_embeddings
            )
        elif 'attnv5' in version:
            model = AttnUnetV5(
                in_ch=in_ch, out_ch=1,
                dropout_name='dropblock', dropout_p=0.3,
                num_embeddings=num_embeddings
            )
        elif 'attnv6_1' in version:
            import torch.nn as nn
            kwargs = {'act': nn.ReLU()} if 'relu' in checkpoint_path else {}
            model = AttnUnetV6_1(
                in_ch=in_ch, out_ch=1,
                dropout_name='dropblock', dropout_p=0.3,
                num_embeddings=num_embeddings, **kwargs
            )
        elif 'attnv6_2' in version:
            import torch.nn as nn
            kwargs = {'act': nn.ReLU()} if 'relu' in checkpoint_path else {}
            # attnv6_2는 입력 채널이 1로 고정됨
            model = AttnUnetV6_2(
                in_ch=1, out_ch=1,
                dropout_name='dropblock', dropout_p=0.3,
                num_embeddings=num_embeddings, **kwargs
            )
        elif 'attnv6' in version:
            model = AttnUnetV6(
                in_ch=in_ch, out_ch=1,
                dropout_name='dropblock', dropout_p=0.3,
                num_embeddings=num_embeddings
            )
        else:
            raise ValueError(f"체크포인트의 버전 정보가 올바르지 않습니다: {version}")
    except Exception as e:
        raise ValueError(
            f"{checkpoint_path} 는 예상하는 버전({version}) 혹은 임베딩 개수({num_embeddings})와 맞지 않습니다."
        ) from e

    # 6. state_dict를 모델에 로드하고, device로 이동한 후 eval 모드로 전환합니다.
    try:
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    except Exception as e:
        raise ValueError(f"체크포인트 {checkpoint_path} 의 state_dict 로드에 실패했습니다.") from e

    return model


class InferencePipeline1umTo210nm:
    def __init__(self, lr_model_name=None, lr_model_checkpoint=None, hr_model_name=None,
                    hr_model_checkpoint=None, dataset=None, device='cuda:0', lr_in_channels=2,
                    metrics=None):
        """
        Args:
            lr_model_name (str): LR 모델 이름.
            lr_model_checkpoint (str): LR 모델 체크포인트 경로.
            hr_model_name (str): HR 모델 이름.
            hr_model_checkpoint (str): HR 모델 체크포인트 경로.
            dataset: 인퍼런스 데이터셋 (예: IRDropInferenceDataset5nm 인스턴스).
            device (str or torch.device): 사용할 장치.
            lr_in_channels (int): LR 모델 입력 채널 수 (2 또는 3).
            metrics (list of str): 계산할 metric 리스트, 예: ['ssim'], ['ssim', 'mae'], ['ssim', 'mae', 'f1']
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.ssim_loss_fn = SSIMLoss(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
        
        # 계산할 metric 설정 (기본은 'ssim')
        self.metrics = metrics if metrics is not None else ['ssim']
        if 'f1' in self.metrics:
            self.dice_loss_fn = DiceLoss()  # F1 계산 시 DiceLoss 사용

        # LR 모델 로드 (체크포인트 정보를 통해 모델이 생성됨)
        self.lr_model = set_lr_model(lr_model_checkpoint, device=self.device)
        
        # HR 모델 (예시로 SRModelV2 사용; run_pipe에서는 개별 체크포인트를 로드하므로 기본 HR 모델은 사용하지 않습니다)
        self.hr_model = SRModelV2(
            in_ch=1, out_ch=1,
            upscale_factor=4, num_features=64,
            num_rrdb=8, growth_rate=32
        )
        self.hr_model.load_state_dict(torch.load(hr_model_checkpoint)['net'])
        self.hr_model.to(self.device)
        self.hr_model.eval()
    
    def resize_as_1um(self,x):
        import cv2
        print(x.shape)
        val_transform = A.Resize(256, 256, interpolation=cv2.INTER_NEAREST_EXACT)
        x = val_transform(image=x)['image']
        print(x.shape)
        return torch.from_numpy(x).unsqueeze(0)
    
    def cal_ssim(self, pred, target):
        # pred와 target이 4D 텐서가 아니면 4D로 변환
        if pred.dim() != 4:
            pred = pred.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
        if target.dim() != 4:
            target = target.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

        # 채널 차원이 두 번째 차원이 되도록 조정 (필요한 경우)
        if pred.size(1) != 1:
            pred = pred.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        if target.size(1) != 1:
            target = target.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        # SSIM 손실 계산 후 1에서 빼서 score 계산 (1이면 완벽)
        ssim_loss = 1 - self.ssim_loss_fn(pred, target)
        return ssim_loss
    
    def cal_mae(self, pred, target):
        # MAE (Mean Absolute Error) 계산
        if pred.dim() != 4:
            pred = pred.unsqueeze(0)
        if target.dim() != 4:
            target = target.unsqueeze(0)
        mae_val = F.l1_loss(pred, target)
        return mae_val

    def cal_f1(self, pred, target):
        # F1 점수를 위한 threshold 계산
        self.q = 0.9
        self.smooth=1e-5
        B = pred.shape[0]
        pred = pred.contiguous().view(B, -1)
        target = target.contiguous().view(B, -1)
        pred_threshold = torch.quantile(pred.float(), self.q, dim=1, keepdim=True)
        target_threshold = torch.quantile(target.float(), self.q, dim=1, keepdim=True)
        pred_bin = (pred > pred_threshold).float()
        target_bin = (target > target_threshold).float()
        intersection = (pred_bin * target_bin).sum(dim=1)
        union = pred_bin.sum(dim=1) + target_bin.sum(dim=1)
        f1 = (2. * intersection + self.smooth) / (union + self.smooth)
        return f1.item()
    
    def min_max_norm(self, x):
        return (x - x.min()) / (x.max() - x.min())
    
    def run(self, idx, cols=3, cmap='jet', min_max_norm=False,cal_metric_as_1um=False,colorbar=False):
        """
        지정된 인덱스의 샘플에 대해 예측을 수행하고 시각화합니다.
        
        Args:
            idx (int): 데이터셋 내 샘플 인덱스.
            cols (int): 시각화 그리드의 열 수 (기본 3: 타깃, LR 예측, HR 예측).
            cmap (str): 사용할 컬러맵.
            min_max_norm (bool): 이미지에 min-max 정규화를 적용할지 여부.
            
        Returns:
            dict: {'lr_target': ..., 'lr_pred': ..., 'hr_target': ..., 'hr_pred': ...}
        """
        # 데이터셋은 dictionary를 반환한다고 가정합니다.
        sample = self.dataset[idx]
        lr_input = sample['lr_input']
        lr_target = sample['lr_target']
        lr_target_ori = sample['lr_target_ori'].unsqueeze(0)
        hr_target = sample['hr_target']
        lr_ori_shape = sample['lr_ori_shape']
        hr_ori_shape = sample['hr_ori_shape']

        with torch.no_grad():
            # 1. LR 예측
            lr_input_batch = lr_input.unsqueeze(0).to(self.device)  # (1, C, 256, 256)
            lr_pred = self.lr_model(lr_input_batch)['x_recon']         # (1, 1, 256, 256)
            lr_pred = lr_pred.squeeze(0).cpu()                         # (1, 256, 256)

            # 2. LR 예측을 원본 해상도(ori_shape)로 보간하여 HR 입력 생성
            hr_input = F.interpolate(
                lr_pred.unsqueeze(0), size=lr_ori_shape, mode='bilinear', align_corners=False
            )  # (1, 1, H, W)
            
            # 3. HR 예측 - 이전 LR 예측 사용
            hr_pred = self.hr_model(hr_input.to(self.device), hr_ori_shape)['x_recon']  # (1, 1, H, W)
            hr_pred = hr_pred.squeeze(0).cpu()                                          # (1, H, W)

            # 3-1. HR 예측 - 1um 타깃(lr_target_ori) 사용
            hr_pred_tar = self.hr_model(lr_target_ori.to(self.device), hr_ori_shape)['x_recon']  # (1, 1, H, W)
            hr_pred_tar = hr_pred_tar.squeeze(0).cpu()                                           # (1, H, W)

            # LR metric 계산
            metrics_lr = {}
            metrics_hr = {}
           
        # 4. 시각화
        # Tensor를 numpy array로 변환 (채널 차원 제거)
        lr_target_np = lr_target.squeeze(0).numpy()
        lr_pred_np = lr_pred.squeeze(0).numpy()
        hr_target_np = hr_target.squeeze(0).numpy()
        hr_pred_np = hr_pred.squeeze(0).numpy()
        hr_pred_tar_np = hr_pred_tar.squeeze(0).numpy()
                
        if cal_metric_as_1um:
            lr_pred,lr_target = self.resize_as_1um(lr_pred_np),self.resize_as_1um(lr_target_np)
            hr_pred,hr_target = self.resize_as_1um(hr_pred_np),self.resize_as_1um(hr_target_np)
            hr_pred_tar = self.resize_as_1um(hr_pred_tar_np)
            lr_target_np = lr_target.squeeze(0).numpy()
            lr_pred_np = lr_pred.squeeze(0).numpy()
            hr_target_np = hr_target.squeeze(0).numpy()
            hr_pred_np = hr_pred.squeeze(0).numpy()
            hr_pred_tar_np = hr_pred_tar.squeeze(0).numpy()

        # 2행 cols열 그리드 생성 (첫 행: LR, 두번째 행: HR)
        fig, axs = plt.subplots(2, cols, figsize=(5 * cols, 10))
        
        # LR 타깃 시각화
        title_lr_target = "1um Target"
        if 'ssim' in self.metrics:
            metrics_lr["lr_ssim_target"] = self.cal_ssim(lr_target, lr_target)
            title_lr_target += f"\nSSIM: {metrics_lr.get('lr_ssim_target', -1):.4f}"
        if 'mae' in self.metrics:
            metrics_lr["lr_mae_target"] = self.cal_mae(lr_target, lr_target)
            title_lr_target += f" | MAE: {metrics_lr.get('lr_mae_target', -1):.4f}"
        if 'f1' in self.metrics:
            metrics_lr["lr_f1_target"] = self.cal_f1(lr_target, lr_target)
            title_lr_target += f" | F1: {metrics_lr.get('lr_f1_target', -1):.4f}"  
        im = axs[0, 0].imshow(lr_target_np if not min_max_norm else self.min_max_norm(lr_target_np), cmap=cmap, vmin=0, vmax=1)
        axs[0, 0].set_title(title_lr_target, fontsize=16)
        axs[0, 0].axis('off')
        if colorbar:fig.colorbar(im, ax=axs[0, 0])

        # LR 예측 시각화
        title_lr_pred = "1um Prediction"
        if 'ssim' in self.metrics:
            metrics_lr["lr_ssim"] = self.cal_ssim(lr_pred, lr_target)
            title_lr_pred += f"\nSSIM: {metrics_lr.get('lr_ssim', -1):.4f}"
        if 'mae' in self.metrics:
            metrics_lr["lr_mae"] = self.cal_mae(lr_pred, lr_target)
            title_lr_pred += f" | MAE: {metrics_lr.get('lr_mae', -1):.4f}"
        if 'f1' in self.metrics:
            metrics_lr["lr_ssim"] = self.cal_f1(lr_pred, lr_target)
            title_lr_pred += f" | F1: {metrics_lr.get('lr_ssim', -1):.4f}"
        im = axs[0, 1].imshow(lr_pred_np if not min_max_norm else self.min_max_norm(lr_pred_np), cmap=cmap, vmin=0, vmax=1)
        axs[0, 1].set_title(title_lr_pred, fontsize=16)
        axs[0, 1].axis('off')
        if colorbar:fig.colorbar(im, ax=axs[0, 1])

        # HR 타깃 시각화
        title_hr_target = "210nm Target"
        if 'ssim' in self.metrics:
            metrics_hr["hr_ssim_target"] = self.cal_ssim(hr_target, hr_target)
            title_hr_target += f"\nSSIM: {metrics_hr.get('hr_ssim_target', -1):.4f}"
        if 'mae' in self.metrics:
            metrics_hr["hr_mae_target"] = self.cal_mae(hr_target, hr_target)
            title_hr_target += f" | MAE: {metrics_hr.get('hr_mae_target', -1):.4f}"
        if 'f1' in self.metrics:
            metrics_hr["hr_f1_target"] = self.cal_f1(hr_target, hr_target)
            title_hr_target += f" | F1: {metrics_hr.get('hr_f1_target', -1):.4f}"
        im=axs[1, 0].imshow(hr_target_np if not min_max_norm else self.min_max_norm(hr_target_np), cmap=cmap, vmin=0, vmax=1)
        axs[1, 0].set_title(title_hr_target, fontsize=16)
        axs[1, 0].axis('off')
        if colorbar:fig.colorbar(im, ax=axs[1, 0])
        
        # HR 예측 시각화
        title_hr_pred = "210nm Prediction"
        if 'ssim' in self.metrics:
            metrics_hr["hr_ssim"] = self.cal_ssim(hr_pred, hr_target)
            title_hr_pred += f"\nSSIM: {metrics_hr.get('hr_ssim', -1):.4f}"
        if 'mae' in self.metrics:
            metrics_hr["hr_mae"] = self.cal_mae(hr_pred, hr_target)
            title_hr_pred += f" | MAE: {metrics_hr.get('hr_mae', -1):.4f}"
        if 'f1' in self.metrics:
            metrics_hr["hr_f1"] = self.cal_f1(hr_pred, hr_target)
            title_hr_pred += f" | F1: {metrics_hr.get('hr_f1', -1):.4f}"
        im=axs[1, 1].imshow(hr_pred_np if not min_max_norm else self.min_max_norm(hr_pred_np), cmap=cmap, vmin=0, vmax=1)
        axs[1, 1].set_title(title_hr_pred, fontsize=16)
        axs[1, 1].axis('off')
        if colorbar:fig.colorbar(im, ax=axs[1, 1])
        
        # HR 예측 (1um 타깃 입력 사용) 시각화
        title_hr_pred_tar = "210nm Prediction (with 1um Target)"
        if 'ssim' in self.metrics:
            metrics_hr["hr_ssim_with_lr_tar"] = self.cal_ssim(hr_pred_tar, hr_target)
            title_hr_pred_tar += f"\nSSIM: {metrics_hr.get('hr_ssim_with_lr_tar', -1):.4f}"
        if 'mae' in self.metrics:
            metrics_hr["hr_mae_with_lr_tar"] = self.cal_mae(hr_pred_tar, hr_target)
            title_hr_pred_tar += f" | MAE: {metrics_hr.get('hr_mae_with_lr_tar', -1):.4f}"
        if 'f1' in self.metrics:
            metrics_hr["hr_f1_with_lr_tar"] = self.cal_f1(hr_pred_tar, hr_target)
            title_hr_pred_tar += f" | F1: {metrics_hr.get('hr_f1_with_lr_tar', -1):.4f}"
        im=axs[1, 2].imshow(hr_pred_tar_np if not min_max_norm else self.min_max_norm(hr_pred_tar_np), cmap=cmap, vmin=0, vmax=1)
        axs[1, 2].set_title(title_hr_pred_tar, fontsize=16)
        axs[1, 2].axis('off')
        if colorbar:fig.colorbar(im, ax=axs[1, 2])
        
        # 사용하지 않는 subplot 숨김 처리
        axs[0, 2].axis('off')
        plt.tight_layout()
        plt.show()

    def run_pipe(self, checkpoint_paths, sample_indices, cmap='jet', fontsize=14, colorbar=False,cal_metric_as_1um=False,inferpolation='bilinear'):
   
        
        # 플롯 그리드 설정: 행 = 체크포인트 수 + 1 (첫 행은 HR 타깃), 열 = 샘플 수
        rows = len(checkpoint_paths) + 1
        cols = len(sample_indices)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        
        # 각 샘플(열)에 대해 처리
        for col_idx, idx in enumerate(sample_indices):
            sample = self.dataset[idx]
            # 필요한 데이터 추출: LR 입력, HR 타깃, 원본 LR/HR 해상도
            lr_input = sample['lr_input']
            hr_target = sample['hr_target']
            lr_ori_shape = sample['lr_ori_shape']
            hr_ori_shape = sample['hr_ori_shape']
            
            lr_input_batch = lr_input.unsqueeze(0).to(self.device)
            with torch.no_grad():
                lr_pred = self.lr_model(lr_input_batch)['x_recon']
            
            hr_input = F.interpolate(lr_pred, size=lr_ori_shape, mode=inferpolation, align_corners=None)
            
            # 첫 번째 행에 HR 타깃 플롯 (210nm Target)
            hr_target_np = hr_target.squeeze().cpu().numpy()
            im=axes[0, col_idx].imshow(hr_target_np, cmap=cmap, vmin=0, vmax=1)
            axes[0, col_idx].set_title(f"210nm Target (Index {idx})", fontsize=fontsize)
            axes[0, col_idx].axis('off')
            if colorbar:fig.colorbar(im, ax=axes[0, col_idx])

            for row_idx, checkpoint_path in enumerate(checkpoint_paths, start=1):
                # SRModelV2 아키텍처를 기준으로 HR 모델 인스턴스 초기화
                temp_hr_model = SRModelV2(
                    in_ch=1, out_ch=1,
                    upscale_factor=4, num_features=64,
                    num_rrdb=8, growth_rate=32
                )
                # 체크포인트에서 state_dict 로드
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                state_dict = checkpoint['net']
                temp_hr_model.load_state_dict(state_dict)
                temp_hr_model.to(self.device)
                temp_hr_model.eval()
                
                with torch.no_grad():
                    hr_pred = temp_hr_model(hr_input.to(self.device), hr_ori_shape)['x_recon']
                hr_pred = hr_pred.cpu() # C,H,W
                hr_pred_np = hr_pred.squeeze().numpy() # H, W
                
                if cal_metric_as_1um:
                    hr_pred,hr_target = self.resize_as_1um(hr_pred_np),self.resize_as_1um(hr_target_np)
                    
                # metric 계산 (설정된 경우)
                metric_text = ""
                if 'ssim' in self.metrics:
                    ssim_val = self.cal_ssim(hr_pred, hr_target)
                    metric_text += f"SSIM: {ssim_val:.4f}  "
                if 'mae' in self.metrics:
                    mae_val = self.cal_mae(hr_pred, hr_target)
                    metric_text += f"MAE: {mae_val:.4f}  "
                if 'f1' in self.metrics:
                    f1_val = self.cal_f1(hr_pred, hr_target)
                    metric_text += f"F1: {f1_val:.4f}  "
                
                # 체크포인트 경로에서 정보를 파싱 (매칭되지 않으면 파일 이름 사용)
                channels, version, loss = parse_checkpoint_path(checkpoint_path)
                if version is not None:
                    ckpt_info = f"{version.split('attn')[-1]}, {channels}, {loss}"
                else:
                    ckpt_info = os.path.basename(checkpoint_path)
                
                title = f"{ckpt_info}\n{metric_text}"
                axes[row_idx, col_idx].imshow(hr_pred_np, cmap=cmap, vmin=0, vmax=1)
                axes[row_idx, col_idx].set_title(title, fontsize=fontsize)
                axes[row_idx, col_idx].axis('off')
                if colorbar:
                    im = axes[row_idx, col_idx].imshow(hr_pred_np, cmap=cmap, vmin=0, vmax=1)
                    fig.colorbar(im, ax=axes[row_idx, col_idx])
        
        plt.tight_layout()
        plt.show()


#####################################################################################
# 2025.02.12
# show inputs 1um ~ 100nm
import os
import numpy as np
import matplotlib.pyplot as plt

def parse_resistance_title(file_path):
    """
    resistance 파일 경로에서 타이틀 문자열을 생성합니다.
    예시)
      "47_m0_resistance.npy"                -> "m0"
      "47_m0_to_m1_via_resistance.npy"        -> "m0 to m1 via"
    """
    file_name = os.path.basename(file_path)            # 예: "47_m0_resistance.npy"
    name_without_ext = os.path.splitext(file_name)[0]    # 예: "47_m0_resistance"
    tokens = name_without_ext.split('_')
    # 만약 첫 토큰이 순번(숫자)라면 제거
    if tokens and tokens[0].isdigit():
        tokens = tokens[1:]
    # 'resistance' 토큰 제거
    tokens = [t for t in tokens if t.lower() != 'resistance']
    # (원한다면 metal layer의 경우 접두어 추가 가능)
    title = " ".join(tokens)
    return title

def visualize_raw_input(dataset, idx=0, cmap='viridis', vmin=None, vmax=None, cols=5, show_target=True,colorbar=False,fontsize=20):
    """
    dataset: IRDropDataset5nm와 같은 데이터셋 인스턴스.
    idx: 시각화할 샘플 인덱스.
    cmap: matplotlib의 colormap (기본값 'viridis')
    vmin, vmax: imshow에 전달할 최소/최대 값 (None이면 각 이미지의 min/max 사용)
    cols: 한 행(row)에 표시할 subplot의 개수
    show_target: True이면 target(ir_drop) 이미지도 subplot에 포함 (False이면 미포함)
    
    - dataset.data_files를 이용해 npy 파일들을 로드하며, 정규화 함수는 적용하지 않습니다.
    - in_ch에 따라 입력 데이터(input_data)를 구성하고, 각 채널의 타이틀은 다음과 같이 지정됩니다.
        * in_ch == 2:
            - 채널 0: "current"
            - 채널 1: "resistance total"
        * in_ch == 3:
            - 채널 0: "current"
            - 채널 1: "pad distance"
            - 채널 2: "resistance total"
        * in_ch == 25:
            - 채널 0: "current"
            - 채널 1: "pad distance"
            - 채널 2 이상: 해당 resistance 파일 경로에서 파싱한 타이틀
    - show_target가 True인 경우 target 이미지도 마지막 subplot에 표시합니다.
    """
    # dataset.data_files에서 파일 그룹 불러오기
    file_group = dataset.data_files[idx]
    
    # current와 ir_drop는 항상 로드
    current = np.load(file_group['current'])
    ir_drop = np.load(file_group['ir_drop'])
    
    # in_ch 값에 따라 input_data 구성
    if dataset.in_ch == 2:
        # resistance 파일들을 모두 로드 후 합산 → 단일 채널
        resistance_stack = [np.load(res) for res in file_group['resistances']]
        # resistance_total = np.stack(resistance_stack, axis=-1).sum(axis=-1)
        # 채널 순서: current, resistance_total
        input_data = np.stack([current] + resistance_stack, axis=-1)
        
    elif dataset.in_ch == 3:
        pad_distance = np.load(file_group['pad_distance'])
        resistance_stack = [np.load(res) for res in file_group['resistances']]
        resistance_total = np.stack(resistance_stack, axis=-1).sum(axis=-1)
        # 채널 순서: current, pad distance, resistance_total
        input_data = np.stack([current, pad_distance, resistance_total], axis=-1)
        
    elif dataset.in_ch == 25:
        pad_distance = np.load(file_group['pad_distance'])
        resistance_maps = [np.load(res) for res in file_group['resistances']]
        if len(resistance_maps) != 23:
            raise ValueError(f"Expected 23 resistance maps, but got {len(resistance_maps)}")
        # 채널 순서: current, pad distance, resistance_map1, ..., resistance_map23
        input_data = np.stack([current, pad_distance] + resistance_maps, axis=-1)
    else:
        raise ValueError(f"Not support {dataset.in_ch} channels.")
    
    # 입력 채널 수
    n_channels = input_data.shape[-1]
    # target 이미지 포함 여부에 따라 총 플롯 개수 결정
    total_plots = n_channels + 1 if show_target else n_channels
    
    # cols에 따라 행(row) 수 계산 (나머지 subplot은 숨김)
    rows = (total_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    # axes가 2D 배열일 경우 flatten
    axes = np.array(axes).flatten()
    
    # 순서대로 subplot에 이미지 그리기
    for i, ax in enumerate(axes):
        if i < total_plots:
            # 입력 채널인 경우
            if i < n_channels:
                image = input_data[..., i]
                # vmin, vmax가 None이면 각 이미지의 최솟값/최댓값 사용
                if None in [vmin, vmax]:
                    im = ax.imshow(image, cmap=cmap, vmin=image.min(), vmax=image.max())
                else:
                    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
                
                # 채널별 타이틀 지정
                if dataset.in_ch == 2:
                    if i == 0:
                        title = "current"
                    else:
                        res_index = i - 1
                        if res_index < len(file_group['resistances']):
                            res_file = file_group['resistances'][res_index]
                            title = parse_resistance_title(res_file)
                        else:
                            title = f"res_{res_index}"
                elif dataset.in_ch == 3:
                    if i == 0:
                        title = "current"
                    elif i == 1:
                        title = "pad distance"
                    elif i == 2:
                        title = "resistance total"
                elif dataset.in_ch == 25:
                    if i == 0:
                        title = "current"
                    elif i == 1:
                        title = "pad distance"
                    else:
                        res_index = i - 2
                        if res_index < len(file_group['resistances']):
                            res_file = file_group['resistances'][res_index]
                            title = parse_resistance_title(res_file)
                        else:
                            title = f"res_{res_index}"
                else:
                    title = f"channel {i}"
                ax.set_title(title,fontsize=fontsize)
                if colorbar : fig.colorbar(im, ax=ax)
            else:
                # 마지막 플롯에 target(ir_drop) 이미지 그리기
                if None in [vmin, vmax]:
                    im = ax.imshow(ir_drop, cmap=cmap, vmin=ir_drop.min(), vmax=ir_drop.max())
                else:
                    im = ax.imshow(ir_drop, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title('IR Drop (Target)',fontsize=fontsize)
                if colorbar : fig.colorbar(im, ax=ax)
        else:
            # 사용하지 않는 subplot은 숨김
            ax.set_visible(False)
            
    plt.tight_layout()
    plt.show()
# =============================================================================
# 사용 예시 (in_ch==3인 경우)
# (주의: get_dataset 함수와 IRDropDataset5nm 클래스가 미리 정의되어 있어야 합니다.)
# =============================================================================
# 예시: in_ch가 3일 때 (채널 구성: current, pad distance, resistance total)
# dataset = get_dataset('cus', split='val', use_raw=False, pdn_zeros=True, in_ch=3, img_size=256, types='1um')
# # 만약 get_dataset()가 DatasetWrapper 같은 객체를 리턴한다면 .dataset 속성으로 실제 Dataset에 접근할 수 있음.
# visualize_raw_input(dataset.dataset, idx=0, cmap='jet', vmin=None, vmax=1)

# =============================================================================
# 사용 예시 (in_ch==25인 경우)
# dataset = get_dataset('cus', split='val', use_raw=False, pdn_zeros=True, in_ch=25, img_size=256, types='100nm')
# visualize_raw_input(dataset.dataset, idx=0, cmap='jet', vmin=None, vmax=1)
