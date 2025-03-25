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
                          plot_mask=False,top_region=0.9,mask_opt='max',plot_scatter=False,out_key = 'x_recon',t_norm_type=None):
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
    inp_batch = inp.unsqueeze(0).to(device)
    if casename:
        print(sample[2])

    with torch.no_grad():
        pred = model(inp_batch) if not sr else model(inp_batch, target.shape[-2:])
    if len(pred) >= 1: 
        pred = pred[out_key]
    
    pred = pred.detach().cpu() 

    if norm_out:
        pred = min_max_norm(pred)

    target_min = target.min()
    target_max = target.max()
    if use_raw :
        _, t_h, t_w = target.shape
        if t_norm_type in ['g_max','global_max']:
            pred= pred * dataset.dataset.conf.ir_drop.max
        elif t_norm_type=='min_max':
            pred = (target_max-target_min)*pred + target_min
        elif t_norm_type == 'z_score':
            pred = (pred *dataset.dataset.conf.std) + dataset.dataset.conf.mean
        print(t_h,t_w)
        pred = F.interpolate(pred.unsqueeze(0), size=(1,t_h, t_w), mode='area').squeeze(0)

    t_max = target.max().item()
    p_max = pred.max().item()
    th = (target.max() * top_region).item()
    print('target max : ',round(t_max if not use_raw else t_max *1000,3),
          'mV pred max : ',round(p_max if not use_raw else p_max *1000,3), 
          'mV treshold : ',round(th if not use_raw else th *1000,3) ,' mV')

    mae_map = torch.abs(pred - target) 
    if use_raw:
        print(f'mae : {torch.mean(mae_map).item()*1000:.2f} mV')
    else:
        print('mae : ', round(torch.mean(mae_map).item(),5))

    pred = pred.squeeze(0)

    mae_10, f1 = calculate_metrics(pred, target, mask_opt=mask_opt,top_region=top_region)
    print(f"F1 Score: {f1:.3f}")
    print(f'mae 10 : {mae_10*1000:.3f}' + ' mV' if use_raw else '')

    target_div = target / torch.norm(target,p=2)
    print(f'norm : {torch.norm(target,p=2)} normed max : {target_div.max()} normed_min {target_div.min()}, target min : {target.min()}')
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
            target_threshold = target.max() * top_region
            target_mask = (target >= target_threshold).float()
            pred_mask = (pred >= target_threshold).float()
        elif mask_opt == 'quantile':
            pred_threshold = torch.quantile(pred.view(-1),top_region)
            target_threshold = torch.quantile(target.view(-1),top_region)
            pred_mask = (pred >= pred_threshold).float()
            target_mask = (target >= target_threshold).float()
        elif mask_opt == 'quantile_target':
            target_threshold = torch.quantile(target.view(-1),top_region)
            pred_mask = (pred >= target_threshold).float()
            target_mask = (target >= target_threshold).float()
        elif mask_opt == 'quantile_pred':
            target_threshold = torch.quantile(pred.view(-1),top_region)
            pred_mask = (pred >= target_threshold).float()
            target_mask = (target >= target_threshold).float()
        all_images += [target_mask, pred_mask]
        titles += ['Target Mask', 'Prediction Mask']

    if plot_scatter:
        all_images.append((target.flatten(), pred.flatten(), target_threshold if not use_raw else target_threshold*1000))  # threshold와 함께 tuple 저장
        titles.append('Target vs. Pred (Scatter)')
        
    rows = (len(all_images) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])


    for i, (img, title) in enumerate(zip(all_images, titles)):
        ax = axes[i]
        if isinstance(img, tuple):  # Scatter인 경우
            target_flat, pred_flat, threshold = img[0].cpu().numpy(), img[1].cpu().numpy(), img[2].cpu().numpy()
            if use_raw : target_flat, pred_flat = target_flat*1000, pred_flat*1000
            ax.scatter(target_flat, pred_flat, alpha=0.3, s=5, label='All Points')  # 모든 점
            min_val = min(target_flat.min(), pred_flat.min())
            max_val = max(target_flat.max(), pred_flat.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
            mask_high = target_flat >= threshold
            golden_ir , p_ir = target_flat[mask_high],pred_flat[mask_high]
            ax.scatter(golden_ir , p_ir, color='red', s=5, marker='s', label='Target ≥ 90% max')

            ax.set_xlabel('Target IR Drop')
            ax.set_ylabel('Prediction')
            ax.set_title('Scatter Plot: Target vs. Prediction'+ ('' if not use_raw else '(mV)'))
            ax.legend()
        else:  # 일반 이미지인 경우
            img = img.squeeze().cpu().numpy()
            if 'Mask' in title:
                im = ax.imshow(img, vmin=0, vmax=1)
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
    titles = ['Target Distribution', 'Pred Distribution']

    for ax, dist, title in zip(axes, distributions, titles):
        ax.hist(dist.flatten(), bins=bins, alpha=0.7, color='b')
        ax.set_title(f'{casename} - {title}')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def plot_distribution_pred(checkpoint_path, dataset,model,index,norm_out=False,device='cuda:0',use_raw=False,g_max=False):
    checkpoint_path = find_pth_files(checkpoint_path)
    model.eval()
    model.load_state_dict(torch.load(checkpoint_path)['net'])

    sample = dataset.__getitem__(index)
    inp, target = sample[0], sample[1]

    inp_batch = inp.unsqueeze(0).to(device)
    model.to(device)

    with torch.no_grad():
        pred = model(inp_batch)
    if len(pred) > 1:
        pred = pred['x_recon']
    pred = pred.detach().cpu()

    if norm_out:
        pred = min_max_norm(pred)
    if use_raw :
        if g_max : pred *=dataset.dataset.conf.max
        else:pred = (target.max()-target.min())*pred + target.min()
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
