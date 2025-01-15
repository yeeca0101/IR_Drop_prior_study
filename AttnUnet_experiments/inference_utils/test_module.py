import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


def min_max_norm(x):
    return (x-x.min())/(x.max()-x.min())

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=None, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        assert C == 1, "This implementation is for binary segmentation"
        
        # Squeeze the channel dimension
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        
        # Create mask for ignored index
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float()
        else:
            mask = torch.ones_like(pred)
        
        # Multiply pred and target by mask
        pred = pred * mask
        target = target * mask
        
        # Flatten pred and target
        pred = pred.view(B, -1)
        target = target.view(B, -1)
        
        # Compute Dice coefficient
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1. - dice.mean()
    
class TestCaseModule:
    def __init__(self, model, checkpoint_path, dataset, batch_size=1, 
                 device='cuda:7',norm_out=True,loss_with_logit=True,testcase_name=True):
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.dataset = dataset
        self.device = device
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.results_df = pd.DataFrame(columns=['Testcase Name', 'MAE', 'F1 Score'])
        self.figures = []
        self.norm_out = norm_out
        self.loss_with_logit = loss_with_logit
        self.testcase_name = testcase_name
        self.colorbar = False

        # Load model
        self.model.load_state_dict(torch.load(self.checkpoint_path)['net'])
        self.model.to(self.device)
        self.model.eval()

    def run(self, visualize_input=False):
        results = []
        all_samples = []
        for i, sample in enumerate(self.dataloader):
            if self.testcase_name:
                inp, target, casename = sample[0].to(self.device), sample[1], sample[2][0]
            else:
                inp, target = sample[0].to(self.device), sample[1] 
                casename=f'dummy_{i}'
            with torch.no_grad():
                pred = self.model(inp)
                if not self.loss_with_logit:
                    pred = torch.sigmoid(pred)
                pred = pred.detach().cpu()
                pred_logit = pred.clone()
                # Normalization
                if self.norm_out:
                    pred = min_max_norm(pred)
                    normalized_mae_map = torch.abs(pred - target)
                    normalized_mae = torch.mean(normalized_mae_map).item()      
                else: 
                    normalized_mae = -1
            # Get metrics
            mae_map = torch.abs(pred_logit - target) if not self.norm_out else normalized_mae_map
            mae = torch.mean(mae_map).item()
            dice_coeff, f1 = self.calculate_metrics(pred_logit, target, th=90)

            # Store results in list
            results.append({'Testcase Name': casename, 'MAE': mae, 'F1 Score': f1,'Normalized MAE':normalized_mae})
            all_samples.append((inp, target, pred, mae_map, casename))

            # Plot distributions if norm_out is True
            if self.norm_out:
                self.plot_dist(target, pred_logit, pred, casename)

        # Convert results list to dataframe
        self.results_df = pd.DataFrame(results)

        # Save dataframe to CSV
        # self.results_df.to_csv('testcase_results.csv', index=False)

        # Visualize all samples
        self.visualize_all_samples(all_samples, visualize_input)

    def visualize_all_samples(self, all_samples, visualize_input, transpose=True):
        num_samples = len(all_samples)
        cols = 3 if not visualize_input else 3 + all_samples[0][0].shape[1]
        
        if transpose:
            fig, axes = plt.subplots(cols, num_samples, figsize=(num_samples * 4, cols * 4))
        else:
            fig, axes = plt.subplots(num_samples, cols, figsize=(cols * 4, num_samples * 4))

        for i, (inp, target, pred, mae_map, casename) in enumerate(all_samples):
            if transpose:
                col_axes = axes[:, i] if num_samples > 1 else axes
            else:
                row_axes = axes[i] if num_samples > 1 else axes

            # Visualize input if required
            if visualize_input:
                for j in range(inp.shape[1]):
                    ax = col_axes[j] if transpose else row_axes[j]
                    ax.imshow(inp[0, j].cpu().numpy(), cmap='jet')
                    ax.set_title(f'{casename} - Input {j+1}')
                    ax.axis('off')

            # Visualize GT, Pred, MAE map
            start_idx = 0 if not visualize_input else inp.shape[1]
            images = [target.squeeze().cpu().numpy(), 
                    pred.squeeze().cpu().numpy(), 
                    mae_map.squeeze().cpu().numpy()]
            titles = ['GT', 'Pred', 'MAE map']

            for j, (img, title) in enumerate(zip(images, titles)):
                ax = col_axes[start_idx + j] if transpose else row_axes[start_idx + j]
                kwargs = {'vmin':0, 'vmax':1} if j !=0 else {}
                im = ax.imshow(img, cmap='jet', **kwargs)  # Store the imshow result
                ax.set_title(f'{i} - {casename} - {title}')
                ax.axis('off')
                if self.colorbar:
                    fig.colorbar(im, ax=ax)

        plt.tight_layout()
        self.figures.append(fig)

    def show(self):
        for fig in self.figures:
            plt.show(fig)

    def plot_dist(self, target, pred_logit, pred, casename):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        distributions = [target.squeeze().cpu().numpy(), 
                         pred_logit.squeeze().cpu().numpy(), 
                         pred.squeeze().cpu().numpy()]
        titles = ['Target Distribution', 'Pred Logit Distribution', 'Normalized Pred Distribution']

        for ax, dist, title in zip(axes, distributions, titles):
            ax.hist(dist.flatten(), bins=50, alpha=0.7, color='b')
            ax.set_title(f'{casename} - {title}')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency')

        plt.tight_layout()
        self.figures.append(fig)
        
    @staticmethod
    def calculate_metrics(pred, target, th):
        pred_np = pred.squeeze().cpu().numpy()
        target_np = target.squeeze().cpu().numpy()

        # Threshold using percentile
        target_np = (target_np >= np.percentile(target_np, th)).astype(int)
        pred_np = (pred_np >= np.percentile(pred_np, th)).astype(int)

        # Dice coefficient
        dice_coeff = 1 - DiceLoss(smooth=1.)(torch.tensor(pred_np[np.newaxis, np.newaxis, ...]),
                                            torch.tensor(target_np[np.newaxis, np.newaxis, ...]),
                                            )

        # F1 score
        f1 = f1_score(target_np.flatten(), pred_np.flatten())

        return dice_coeff.item(), f1
    

def get_results(model,dataset,checkpoint_path,norm_out=False,show_input=False,loss_with_logit=True, device='cuda:7',testcase_name=True):
    load_score(chkpt_path=checkpoint_path)
    tester = TestCaseModule(model,checkpoint_path, dataset,norm_out=norm_out,loss_with_logit=loss_with_logit, device=device,testcase_name=testcase_name)
    tester.run(visualize_input=show_input)
    tester.show()
    result_df = tester.results_df
    result_df.iloc[:,1:] = result_df.iloc[:,1:].apply(lambda x : x.round(3))
    display(result_df)
    f1,mae,n_mae = tester.results_df['F1 Score'].mean(),tester.results_df['MAE'].mean(),tester.results_df['Normalized MAE'].mean()
    print(f'f1 : {f1:.3f}, mae : {mae:.3f}, normalized mae : {n_mae:.3f}')
