'''
    ymin98@sogang.ac.kr
'''
import torch
import numpy as np
import matplotlib.pyplot as plt

from .utils import find_pth_files, min_max_norm, calculate_metrics, to_numpy


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
    if len(pred) == 3: 
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


def plot_f1_mask(dataset,model,checkpoint,index,device):
    model.to(device)
    model.eval()
    
    model.load_state_dict(torch.load(find_pth_files(checkpoint), map_location=device)['net'])
    sample = dataset.__getitem__(index)
    inp, target = sample[0].to(device), sample[1].to(device)

    # 입력 텐서 준비 (배치 차원 추가)
    inp_batch = inp.unsqueeze(0)

    # 예측 수행
    with torch.no_grad():
        pred = model(inp_batch)
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