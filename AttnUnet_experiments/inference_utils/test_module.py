import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm


from utils import min_max_norm, DiceLoss, IRDropMetrics


class TestCaseModule:
    def __init__(self, model, checkpoint_path, dataset, batch_size=1, 
                 device='cuda:0',norm_out=True,loss_with_logit=True,testcase_name=True,th=0.9,mask_opt='max'):
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
        self.metric = IRDropMetrics(top_percent=th,how=mask_opt)


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
                
                if len(pred) > 1 :
                    pred = pred['x_recon']
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
            dice_coeff, f1 = self.calculate_metrics(pred, target)

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
        
    def calculate_metrics(self, pred, target):
        return 0, self.metric(pred,target)['f1']

    # def calculate_metrics(pred, target, th):
    #     pred_np = pred.squeeze().cpu().numpy()
    #     target_np = target.squeeze().cpu().numpy()

    #     # Threshold using percentile
    #     target_np = (target_np >= np.percentile(target_np, th)).astype(int)
    #     pred_np = (pred_np >= np.percentile(pred_np, th)).astype(int)

    #     # Dice coefficient
    #     dice_coeff = 1 - DiceLoss()(torch.tensor(pred_np[np.newaxis, np.newaxis, ...]),
    #                                         torch.tensor(target_np[np.newaxis, np.newaxis, ...]),
    #                                         )

    #     # F1 score
    #     f1 = f1_score(target_np.flatten(), pred_np.flatten())

    #     return dice_coeff.item(), f1
    

class F1ScoreEvaluator(nn.Module):
    def __init__(self, threshold=90):
        """
        F1 score를 계산하기 위한 클래스.
        Args:
        - threshold (int): 상위 퍼센티지를 결정하는 기준. 기본값은 상위 10% 기준 (90 퍼센타일).
        """
        super(F1ScoreEvaluator, self).__init__()
        self.threshold = threshold
        self.reset()

    def reset(self):
        """내부적으로 값을 초기화하는 함수."""
        self.true_labels = []
        self.pred_labels = []

    def update(self, preds, targets):
        """
        F1 스코어 업데이트를 위해 예측 값과 실제 값 추가.
        
        Args:
        - preds (torch.Tensor): 모델의 예측 값 (IR drops).
        - targets (torch.Tensor): 실제 라벨 (0 또는 1).
        """
        # numpy로 변환
        preds_np = preds.cpu().detach().numpy()
        targets_np = targets.cpu().detach().numpy()

        # 상위 10% 기준 계산
        threshold_value = np.percentile(preds_np, self.threshold)

        # 예측값을 상위 10% 기준으로 레이블링 (1: positive, 0: negative)
        pred_labels = (preds_np >= threshold_value).astype(int)

        # 실제 타겟 레이블도 비슷하게 레이블링 (상위 10% 양성)
        target_labels = (targets_np >= np.percentile(targets_np, self.threshold)).astype(int)

        # 리스트에 예측 값과 실제 값 추가
        self.true_labels.extend(target_labels.flatten())
        self.pred_labels.extend(pred_labels.flatten())

    def compute(self):
        """현재까지의 F1 score를 계산."""
        return f1_score(self.true_labels, self.pred_labels)

def predict_and_evaluate_f1(dataset, model, checkpoint_path, index=None, threshold=90, test_mode=False, batch_size=32, device='cpu'):
    # 모델을 지정된 device로 이동
    model.to(device)
    model.eval()

    # 모델의 파라미터 로드
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['net'])

    # F1ScoreEvaluator 객체 생성
    f1_evaluator = F1ScoreEvaluator(threshold=threshold)

    # test_mode가 True일 경우 DataLoader를 사용
    if test_mode:
        # DataLoader 생성
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # 배치마다 평가 수행
        with torch.no_grad():
            for batch in tqdm(loader):
                inputs, targets = batch[0].to(device), batch[1].to(device)  # 데이터를 지정된 device로 이동
                preds = model(inputs)  # 모델 예측
                preds = preds.squeeze(0)  # 배치 차원 제거
                f1_evaluator.update(preds, targets)  # F1 score 업데이트
    else:
        # 단일 샘플 평가
        if index is None:
            raise ValueError("If test_mode is False, an index must be provided.")
        
        # 데이터셋에서 샘플 가져오기
        sample = dataset.__getitem__(index)
        inp, target = sample[0].to(device), sample[1].to(device)

        # 입력 텐서 준비 (배치 차원 추가)
        inp_batch = inp.unsqueeze(0)

        # 예측 수행
        with torch.no_grad():
            pred = model(inp_batch)

        # 예측 결과 압축 (배치 차원 제거)
        pred = pred.squeeze(0)
        
        # F1 score 업데이트
        f1_evaluator.update(pred, target)

    # F1 score 계산
    f1 = f1_evaluator.compute()
    
    print(f"F1 Score: {f1}")
    
    return f1