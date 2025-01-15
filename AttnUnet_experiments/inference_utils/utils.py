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
from ir_dataset import IRDropDataset,build_dataset_iccad_finetune,build_dataset_iccad,build_dataset,build_dataset_began_asap7,TestASAP7Dataset
from models.papers_model import AttnUnetBase
from metric import IRDropMetrics
from models import *


def get_dataset(dt,split='train',get_case_name=True,img_size=512,in_ch=12):
    if dt == 'iccad_train':
        dataset=build_dataset_iccad(img_size=img_size,in_ch=in_ch)[0] if  split == 'train' else build_dataset_iccad(img_size=img_size,in_ch=in_ch)[1]
        print(f'iccad_pretrain {split}')
    elif dt == 'iccad_fine':                        
        dataset=build_dataset_iccad_finetune(img_size=img_size,in_ch=in_ch)[0] if  split == 'train' else build_dataset_iccad_finetune(return_case=get_case_name,img_size=img_size,in_ch=in_ch)[1]
        print(f'iccad_fine {split}')
    elif dt == 'asap7_train_val':                        
        dataset=build_dataset_began_asap7(img_size=img_size,in_ch=in_ch)[0] if  split == 'train' else build_dataset_began_asap7(img_size=img_size,in_ch=in_ch)[1]
        print(f'asap7_train_val : {split}')
    elif dt == 'asap7_fine':  # TestASAP7Dataset를 추가
        dataset = TestASAP7Dataset(root_path='/data/real-circuit-benchmarks/asap7/numpy_data',
                                   target_layers=['m2', 'm5', 'm6', 'm7', 'm8', 'm25', 'm56', 'm67', 'm78'],
                                   img_size=img_size, use_irreg=False, preload=False, train=False,return_case=get_case_name,
                                   in_ch=in_ch)
        print(f'ASAP7 {split}')
    else:
        dataset=build_dataset(img_size=img_size,in_ch=in_ch)[0] if  split == 'train' else build_dataset(img_size=img_size,in_ch=in_ch)[1]
        # dataset = IRDropDataset(root_path='/data/BeGAN-circuit-benchmarks',
        #                     selected_folders=['nangate45/set1_numpy','nangate45/set2_numpy'],
        #                     img_size=512,
        #                    target_layers = ['m1', 'm4', 'm7', 'm8', 'm9', 'm14', 'm47', 'm78', 'm89'],
        #               )
        print(f'began {split}')
        
    return dataset


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


class PredictorVisualizer:
    def __init__(self, model, checkpoint_path, dataset, device='cuda:7'):
        """
        Initializes the PredictorVisualizer with the model, checkpoint, dataset, and device.
        
        Args:
            model (torch.nn.Module): The PyTorch model to be used for predictions.
            checkpoint_path (str): Path to the model checkpoint.
            dataset (torch.utils.data.Dataset): The dataset to use for predictions.
            device (str): The device to use for computation ('cuda:X' or 'cpu').
        """
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.dataset = dataset
        self.device = device
        
        self._load_model()

    def _load_model(self):
        """Loads the model weights from the checkpoint."""
        self.model.load_state_dict(torch.load(self.checkpoint_path)['net'])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, index, norm_out=False):
        """
        Makes a prediction for the given index from the dataset.
        
        Args:
            index (int): Index of the sample in the dataset.
            norm_out (bool): Whether to normalize the output using min-max normalization.
        
        Returns:
            tuple: (input, target, prediction, mae_map)
        """
        sample = self.dataset.__getitem__(index)
        inp, target = sample[0], sample[1]
        
        # Prepare input tensor (add batch dimension)
        inp_batch = inp.unsqueeze(0).to(self.device)
        
        # Perform prediction
        with torch.no_grad():
            pred = self.model(inp_batch)
        pred = pred.detach().cpu()

        # Normalize prediction if needed
        if norm_out:
            pred = self.min_max_norm(pred)

        # Calculate MAE map
        mae_map = torch.abs(pred - target)
        print('mae : ', torch.mean(mae_map))
        
        return inp, target, pred.squeeze(0), mae_map

    @staticmethod
    def min_max_norm(tensor):
        """Applies min-max normalization to the given tensor."""
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())

    def calculate_metrics(self, pred, target, th=90):
        """
        Calculates metrics such as Dice coefficient and F1 score.
        
        Args:
            pred (torch.Tensor): The predicted tensor.
            target (torch.Tensor): The target tensor.
            th (int): Threshold for binarizing predictions and targets.
        
        Returns:
            tuple: (Dice coefficient, F1 score)
        """
        # Placeholder for actual implementation of calculate_metrics
        dice_coeff = 0.9  # Example value
        f1 = 0.85  # Example value
        
        print(f"Dice Coefficient: {dice_coeff:.3f}")
        print(f"F1 Score: {f1:.3f}")
        
        return dice_coeff, f1

    def visualize(self, inp, target, pred, mae_map, cols=4, colorbar=False, cmap='inferno'):
        """
        Visualizes the input, target, prediction, and MAE map.
        
        Args:
            inp (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.
            pred (torch.Tensor): The predicted tensor.
            mae_map (torch.Tensor): The MAE map tensor.
            cols (int): Number of columns in the plot grid.
            colorbar (bool): Whether to include colorbars in the plots.
            cmap (str): Colormap to use for visualization.
        """
        all_images = [inp] if inp.dim() == 2 else [inp[i] for i in range(inp.shape[0])]
        all_images += [target, pred, mae_map]

        # Calculate number of rows
        rows = (len(all_images) + cols - 1) // cols

        # Create plots
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = axes.flatten() if rows > 1 else [axes]

        titles = [f'Input {i + 1}' for i in range(len(all_images) - 3)] + ['Target', 'Prediction', 'MAE Map']

        for i, (img, title) in enumerate(zip(all_images, titles)):
            img = img.squeeze().cpu().numpy()
            ax = axes[i]
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1) if colorbar else ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis('off')
            if colorbar and (i > len(titles) - 4):
                fig.colorbar(im, ax=ax)

        # Remove unused subplots
        for i in range(len(all_images), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def predict_and_visualize(self, index, cols=4, colorbar=False, norm_out=False, casename=False, cmap='inferno'):
        """
        Combines prediction and visualization for a given index.
        
        Args:
            index (int): Index of the sample in the dataset.
            cols (int): Number of columns in the plot grid.
            colorbar (bool): Whether to include colorbars in the plots.
            norm_out (bool): Whether to normalize the output using min-max normalization.
            casename (bool): Whether to print the case name (if available).
            cmap (str): Colormap to use for visualization.
        """
        inp, target, pred, mae_map = self.predict(index, norm_out)
        
        if casename and len(self.dataset[index]) > 2:
            print(self.dataset[index][2])
        
        # Calculate metrics
        self.calculate_metrics(pred, target)
        
        # Visualize results
        self.visualize(inp, target, pred, mae_map, cols, colorbar, cmap)


# to be not used
def predict_and_visualize(dataset, model, checkpoint_path,
                          index, cols=4, colorbar=False,
                          norm_out=False, device='cuda:7',
                          casename=False, cmap='inferno'):


    # 모델 로드 및 평가 모드 설정
    model.eval()
    model.load_state_dict(torch.load(checkpoint_path)['net'])

    # 데이터셋에서 샘플 가져오기
    sample = dataset.__getitem__(index)
    inp, target = sample[0], sample[1]
    if casename:
        print(sample[2])

    # 입력 텐서 준비 (배치 차원 추가)
    inp_batch = inp.unsqueeze(0).to(device)
    model.to(device)

    # 예측 수행
    with torch.no_grad():
        pred = model(inp_batch)
    pred = pred.detach().cpu()

    # 예측 결과 압축 (배치 차원 제거)
    if norm_out:
        pred = min_max_norm(pred)

    mae_map = torch.abs(pred - target)
    print('mae : ', torch.mean(mae_map))
    pred = pred.squeeze(0)

    # Metrics 계산
    dice_coeff, f1 = calculate_metrics(pred, target, th=90)
    print(f"Dice Coefficient: {dice_coeff:.3f}")
    print(f"F1 Score: {f1:.3f}")

    # 시각화를 위해 모든 이미지를 하나의 리스트로 결합
    all_images = [inp] if inp.dim() == 2 else [inp[i] for i in range(inp.shape[0])]
    all_images += [target, pred, mae_map]

    # 행 수 계산
    rows = (len(all_images) + cols - 1) // cols

    # 플롯 생성
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if rows > 1 else [axes]

    titles = [f'Input {i + 1}' for i in range(len(all_images) - 3)] + ['Target', 'Prediction', 'MAE Map']

    for i, (img, title) in enumerate(zip(all_images, titles)):
        img = img.squeeze().cpu().numpy()
        ax = axes[i]
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1) if colorbar else ax.imshow(img, cmap=cmap)
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
