'''
    ymin98@sogang.ac.kr
'''
import sys
sys.path.append('../')
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

from .utils import find_pth_files, min_max_norm, calculate_metrics, to_numpy, DiceLoss
from models import *


def predict_and_visualize(dataset, model, checkpoint_path,
                          index, cols=4, colorbar=False,
                          norm_out=False, device='cuda:0',
                          casename=False, cmap='inferno',use_raw=False):

    checkpoint_path = find_pth_files(checkpoint_path)
    model.eval()
    model.load_state_dict(torch.load(checkpoint_path)['net'])

    sample = dataset.__getitem__(index)
    inp, target = sample[0], sample[1]
    if casename:
        print(sample[2])

    inp_batch = inp.unsqueeze(0).to(device)
    model.to(device)

    with torch.no_grad():
        pred = model(inp_batch)
    if len(pred) >= 3: 
        pred = pred['x_recon']
    pred = pred.detach().cpu() 

    if norm_out:
        pred = min_max_norm(pred)
        if use_raw:
            target = min_max_norm(target)
            
    mae_map = torch.abs(pred - target)
    print('mae : ', torch.mean(mae_map))
    pred = pred.squeeze(0)

    dice_coeff, f1 = calculate_metrics(pred, target, th=90)
    print(f"Dice Coefficient: {dice_coeff:.3f}")
    print(f"F1 Score: {f1:.3f}")

    if inp.dim() == 2:  # 단일 채널, 
        all_images = [inp]
    elif inp.dim() == 3:  # 다중 채널, 
        all_images = [inp[i] for i in range(inp.shape[0])]

    all_images += [target, pred, mae_map]

    rows = (len(all_images) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])

    titles = [f'Input {i + 1}' for i in range(len(all_images) - 3)] + ['Target', 'Prediction', 'MAE Map']

    for i, (img, title) in enumerate(zip(all_images, titles)):
        img = img.squeeze().cpu().numpy()
        ax = axes[i]
        if use_raw:
            im = ax.imshow(img, cmap=cmap, vmin=img.min(), vmax=img.max()) 
        else:
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1) # if colorbar else ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
        if colorbar and (i > len(titles)-4):
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

def plot_predictions_from_checkpoints(checkpoint_paths, dataset, sample_indices, plot_inputs=[], device='cuda:0', measure=['f1'],cmap='jet',fontsize=14):
    # Setup plot grid dimensions
    rows = len(checkpoint_paths) + 1
    cols = len(sample_indices)
    if plot_inputs:
        rows += len(plot_inputs)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    for col_idx, idx in enumerate(sample_indices):
        sample = dataset.__getitem__(idx)
        inp, target = sample[0], sample[1]

        # Plot Ground Truth (GT)
        axes[0, col_idx].imshow(target.squeeze().cpu().numpy(), cmap=cmap,vmin=0,vmax=1)
        axes[0, col_idx].set_title(f"GT (Index {idx})",fontsize=fontsize+5)
        axes[0, col_idx].axis('off')

        if plot_inputs:
            for i, channel_idx in enumerate(plot_inputs):
                axes[i + 1, col_idx].imshow(inp[channel_idx].cpu().numpy(), cmap=cmap,vmin=0,vmax=1)
                axes[i + 1, col_idx].set_title(f"Input Channel {channel_idx}")
                axes[i + 1, col_idx].axis('off')
        current_row = 1 + len(plot_inputs)

        for checkpoint_idx, checkpoint_path_any in enumerate(checkpoint_paths):
            checkpoint_path = find_pth_files(checkpoint_path_any)
            # Parse checkpoint information
            channels, version, loss = parse_checkpoint_path(checkpoint_path)

            # Adjust input channels if checkpoint specifies 2ch
            inp_channels = inp if channels != '2ch' else inp[[0, 2]]
            inp_batch = inp_channels.unsqueeze(0).to(device)
            
            # Load checkpoint
            state_dict = torch.load(checkpoint_path)['net']
            num_embeddings = get_num_embeddings(state_dict) 

            # Initialize model
            try:
                if 'attnv5_1' in version:
                    model = AttnUnetV5_1(in_ch=2 if channels == '2ch' else 3, out_ch=1, dropout_name='dropblock', dropout_p=0.3, num_embeddings=num_embeddings)
                if 'attnv5_2' in version:
                    model = AttnUnetV5_2(in_ch=2 if channels == '2ch' else 3, out_ch=1, dropout_name='dropblock', dropout_p=0.3, num_embeddings=num_embeddings)
                elif 'attnv5' in version:
                    model = AttnUnetV5(in_ch=2 if channels == '2ch' else 3, out_ch=1, dropout_name='dropblock', dropout_p=0.3, num_embeddings=num_embeddings)
                elif 'attnv6_1' in version:
                    if 'swisht' not in checkpoint_path:
                        import torch.nn as nn
                        kwargs = {'act':nn.ReLU()}
                    else : kwargs = {}
                    model = AttnUnetV6_1(in_ch=2 if channels == '2ch' else 3, out_ch=1, dropout_name='dropblock', dropout_p=0.3, num_embeddings=num_embeddings,**kwargs)
                elif 'attnv6' in version:
                    model = AttnUnetV6(in_ch=2 if channels == '2ch' else 3, out_ch=1, dropout_name='dropblock', dropout_p=0.3, num_embeddings=num_embeddings)
            except:
                raise ValueError(f'{checkpoint_path} not match {version} or embeddings : {num_embeddings}')

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
            axes[current_row, col_idx].imshow(pred.squeeze().numpy(), cmap=cmap,vmin=0,vmax=1)
            axes[current_row, col_idx].set_title(f"{version.split('attn')[-1]}, {channels}, {loss}",fontsize=fontsize)
            axes[current_row, col_idx].axis('off')
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