#!/usr/bin/env python
import os
import argparse
import torch
import torch.nn.functional as F
import albumentations as A
import cv2


# =============================================================================
# 필요한 함수 및 클래스 (실제 코드에서는 별도 모듈에서 임포트)
# =============================================================================

# --- 체크포인트 경로 파싱 예시 함수 ---
def parse_checkpoint_path(checkpoint_path):
    """
    체크포인트 파일명에서 채널, 버전, loss 정보를 파싱합니다.
    예시 파일명: "model_2ch_attnv5_1_loss.pth"
    """
    base = os.path.basename(checkpoint_path)
    parts = base.split('_')
    channels = None
    version = None
    loss = None
    for part in parts:
        if part in ['2ch', '3ch']:
            channels = part
        elif part.startswith('attnv'):
            version = part
        elif part.startswith('loss'):
            loss = part
    return channels, version, loss

# --- 예시: state_dict에서 임베딩 개수를 결정하는 함수 ---
def get_num_embeddings(state_dict):
    # 실제 state_dict 내용을 확인하여 임베딩 수를 결정하세요.
    # 여기서는 예시로 512를 리턴합니다.
    return 512

# --- 더미 SSIMLoss (실제 구현 코드로 교체) ---
class SSIMLoss(torch.nn.Module):
    def __init__(self, win_size, win_sigma, data_range, size_average, channel):
        super().__init__()
        # 실제 파라미터를 사용하여 초기화
    def forward(self, img1, img2):
        # 실제 SSIM 계산 대신, 예시로 고정값을 리턴합니다.
        return torch.tensor(0.5)

# --- 더미 DiceLoss (F1 계산 시 사용) ---
class DiceLoss:
    def __init__(self):
        pass
    def __call__(self, input, target):
        return torch.tensor(0.0)

# --- 예시: 모델 클래스들 (실제 구현 코드로 교체) ---
class AttnUnetV5_1(torch.nn.Module):
    def __init__(self, in_ch, out_ch, dropout_name, dropout_p, num_embeddings):
        super().__init__()
        self.net = torch.nn.Identity()
    def forward(self, x):
        return {'x_recon': x}

class AttnUnetV5_2(torch.nn.Module):
    def __init__(self, in_ch, out_ch, dropout_name, dropout_p, num_embeddings):
        super().__init__()
        self.net = torch.nn.Identity()
    def forward(self, x):
        return {'x_recon': x}

class AttnUnetV5(torch.nn.Module):
    def __init__(self, in_ch, out_ch, dropout_name, dropout_p, num_embeddings):
        super().__init__()
        self.net = torch.nn.Identity()
    def forward(self, x):
        return {'x_recon': x}

class AttnUnetV6_1(torch.nn.Module):
    def __init__(self, in_ch, out_ch, dropout_name, dropout_p, num_embeddings, **kwargs):
        super().__init__()
        self.net = torch.nn.Identity()
    def forward(self, x):
        return {'x_recon': x}

class AttnUnetV6_2(torch.nn.Module):
    def __init__(self, in_ch, out_ch, dropout_name, dropout_p, num_embeddings, **kwargs):
        super().__init__()
        self.net = torch.nn.Identity()
    def forward(self, x):
        return {'x_recon': x}

class AttnUnetV6(torch.nn.Module):
    def __init__(self, in_ch, out_ch, dropout_name, dropout_p, num_embeddings):
        super().__init__()
        self.net = torch.nn.Identity()
    def forward(self, x):
        return {'x_recon': x}

# --- 예시: HR 모델 (SRModelV2) ---
class SRModelV2(torch.nn.Module):
    def __init__(self, in_ch, out_ch, upscale_factor, num_features, num_rrdb, growth_rate):
        super().__init__()
        self.net = torch.nn.Identity()
    def forward(self, x, target_shape):
        # 간단히 입력을 target_shape 크기로 보간하는 예시 구현
        x_recon = F.interpolate(x, size=target_shape, mode='bilinear', align_corners=False)
        return {'x_recon': x_recon}

# --- 예시: 데이터셋 클래스 ---
class IRDropInferenceAutoencoderDataset5nm(torch.utils.data.Dataset):
    def __init__(self, data_path=None):
        """
        data_path를 이용하여 데이터셋을 초기화합니다.
        실제 구현에 맞게 수정하세요.
        """
        # 예시로 임의의 5개 샘플 생성
        self.data = [self._create_dummy_sample(i) for i in range(5)]
    
    def _create_dummy_sample(self, idx):
        # (C, H, W) 텐서 생성 예시 (여기서는 1채널 256x256 이미지를 임의 생성)
        lr_input = torch.rand(2, 256, 256)      # LR 입력 (채널 수는 2 또는 3)
        lr_target = torch.rand(1, 256, 256)       # 1um 타깃
        lr_target_ori = torch.rand(1, 256, 256)   # 원본 해상도 1um 타깃
        hr_target = torch.rand(1, 512, 512)         # 210nm 타깃 (해상도 예시)
        # 원본 해상도 (tuple) 예시
        lr_ori_shape = (256, 256)
        hr_ori_shape = (512, 512)
        return {
            'lr_input': lr_input,
            'lr_target': lr_target,
            'lr_target_ori': lr_target_ori,
            'hr_target': hr_target,
            'lr_ori_shape': lr_ori_shape,
            'hr_ori_shape': hr_ori_shape,
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# =============================================================================
# LR 모델 로드 함수
# =============================================================================
def set_lr_model(checkpoint_path, device='cuda:0'):
    """
    주어진 체크포인트 경로에서 LR 모델을 초기화하고 state_dict를 로드합니다.
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
            model = AttnUnetV5_1(in_ch=in_ch, out_ch=1, dropout_name='dropblock', dropout_p=0.3, num_embeddings=num_embeddings)
        elif 'attnv5_2' in version:
            model = AttnUnetV5_2(in_ch=in_ch, out_ch=1, dropout_name='dropblock', dropout_p=0.3, num_embeddings=num_embeddings)
        elif 'attnv5' in version:
            model = AttnUnetV5(in_ch=in_ch, out_ch=1, dropout_name='dropblock', dropout_p=0.3, num_embeddings=num_embeddings)
        elif 'attnv6_1' in version:
            import torch.nn as nn
            kwargs = {'act': nn.ReLU()} if 'relu' in checkpoint_path else {}
            model = AttnUnetV6_1(in_ch=in_ch, out_ch=1, dropout_name='dropblock', dropout_p=0.3,
                                 num_embeddings=num_embeddings, **kwargs)
        elif 'attnv6_2' in version:
            import torch.nn as nn
            kwargs = {'act': nn.ReLU()} if 'relu' in checkpoint_path else {}
            # attnv6_2는 입력 채널이 1로 고정됨
            model = AttnUnetV6_2(in_ch=1, out_ch=1, dropout_name='dropblock', dropout_p=0.3,
                                 num_embeddings=num_embeddings, **kwargs)
        elif 'attnv6' in version:
            model = AttnUnetV6(in_ch=in_ch, out_ch=1, dropout_name='dropblock', dropout_p=0.3,
                               num_embeddings=num_embeddings)
        else:
            raise ValueError(f"체크포인트의 버전 정보가 올바르지 않습니다: {version}")
    except Exception as e:
        raise ValueError(f"{checkpoint_path} 는 예상하는 버전({version}) 혹은 임베딩 개수({num_embeddings})와 맞지 않습니다.") from e

    # 6. state_dict를 모델에 로드, device 이동 후 eval 모드 전환
    try:
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    except Exception as e:
        raise ValueError(f"체크포인트 {checkpoint_path} 의 state_dict 로드에 실패했습니다.") from e

    return model

# =============================================================================
# 평가 파이프라인 클래스 (시각화 없이 metric 스코어만 출력)
# =============================================================================
class InferencePipeline1umTo210nm:
    def __init__(self, lr_model_checkpoint, hr_model_checkpoint, dataset, device='cuda:0', metrics=None):
        """
        Args:
            lr_model_checkpoint (str): LR 모델 체크포인트 경로.
            hr_model_checkpoint (str): HR 모델 체크포인트 경로.
            dataset: 인퍼런스 데이터셋 (예: IRDropInferenceAutoencoderDataset5nm 인스턴스).
            device (str or torch.device): 사용할 장치.
            metrics (list of str): 계산할 metric 리스트, 예: ['ssim'], ['ssim','mae','f1']
        """
        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dataset = dataset
        self.metrics = metrics if metrics is not None else ['ssim']
        if 'f1' in self.metrics:
            self.dice_loss_fn = DiceLoss()
        self.ssim_loss_fn = SSIMLoss(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
        
        # LR 모델 로드
        self.lr_model = set_lr_model(lr_model_checkpoint, device=self.device)
        
        # HR 모델 로드 (SRModelV2 사용 예시)
        self.hr_model = SRModelV2(in_ch=1, out_ch=1, upscale_factor=4, num_features=64, num_rrdb=8, growth_rate=32)
        checkpoint_hr = torch.load(hr_model_checkpoint, map_location=self.device)
        self.hr_model.load_state_dict(checkpoint_hr['net'])
        self.hr_model.to(self.device)
        self.hr_model.eval()
    
    def resize_as_1um(self, x):
        """
        numpy array인 x를 1um 해상도 (예: 256x256)로 리사이즈한 후 tensor로 변환.
        """
        # 입력 x는 numpy array (H, W)로 가정
        transformed = A.Resize(256, 256, interpolation=cv2.INTER_NEAREST_EXACT)
        x_resized = transformed(image=x)['image']
        return torch.from_numpy(x_resized).unsqueeze(0)  # (1, H, W)
    
    def cal_ssim(self, pred, target):
        # 입력 pred와 target이 4D 텐서가 아니면 4D로 변환 (B, C, H, W)
        if pred.dim() != 4:
            pred = pred.unsqueeze(0)
        if target.dim() != 4:
            target = target.unsqueeze(0)
        # 채널이 두번째 차원이 아닐 경우 변환 (필요시)
        if pred.size(1) != 1:
            pred = pred.permute(0, 3, 1, 2)
        if target.size(1) != 1:
            target = target.permute(0, 3, 1, 2)
        # SSIM 스코어: 1 - (ssim_loss)를 계산 (1이면 완벽)
        ssim_loss = 1 - self.ssim_loss_fn(pred, target)
        return ssim_loss

    def cal_mae(self, pred, target):
        if pred.dim() != 4:
            pred = pred.unsqueeze(0)
        if target.dim() != 4:
            target = target.unsqueeze(0)
        mae_val = F.l1_loss(pred, target)
        return mae_val

    def cal_f1(self, pred, target):
        # F1 점수를 위한 threshold 계산
        q = 0.9
        smooth = 1e-5
        B = pred.shape[0]
        pred_flat = pred.contiguous().view(B, -1)
        target_flat = target.contiguous().view(B, -1)
        pred_threshold = torch.quantile(pred_flat.float(), q, dim=1, keepdim=True)
        target_threshold = torch.quantile(target_flat.float(), q, dim=1, keepdim=True)
        pred_bin = (pred_flat > pred_threshold).float()
        target_bin = (target_flat > target_threshold).float()
        intersection = (pred_bin * target_bin).sum(dim=1)
        union = pred_bin.sum(dim=1) + target_bin.sum(dim=1)
        f1 = (2. * intersection + smooth) / (union + smooth)
        return f1.item()

    def evaluate_sample(self, idx, cal_metric_as_1um=False):
        """
        주어진 인덱스의 샘플에 대해 예측을 수행하고 metric 스코어를 계산하여 dict로 반환합니다.
        Args:
            idx (int): 데이터셋 내 샘플 인덱스.
            cal_metric_as_1um (bool): True이면 1um 해상도로 리사이즈한 후 metric 계산.
        Returns:
            dict: {'lr': {metric: value, ...}, 'hr': {metric: value, ...}}
        """
        sample = self.dataset[idx]
        lr_input = sample['lr_input']
        lr_target = sample['lr_target']
        lr_target_ori = sample['lr_target_ori'].unsqueeze(0)
        hr_target = sample['hr_target']
        lr_ori_shape = sample['lr_ori_shape']
        hr_ori_shape = sample['hr_ori_shape']

        with torch.no_grad():
            # LR 예측
            lr_input_batch = lr_input.unsqueeze(0).to(self.device)  # (1, C, 256, 256)
            lr_pred = self.lr_model(lr_input_batch)['x_recon']         # (1, 1, 256, 256)
            lr_pred = lr_pred.squeeze(0).cpu()                         # (1, 256, 256)

            # LR 예측을 원본 해상도(lr_ori_shape)로 보간하여 HR 입력 생성
            hr_input = F.interpolate(lr_pred.unsqueeze(0), size=lr_ori_shape, mode='bilinear', align_corners=False)
            
            # HR 예측 - LR 예측 사용
            hr_pred = self.hr_model(hr_input.to(self.device), hr_ori_shape)['x_recon']  # (1, 1, H, W)
            hr_pred = hr_pred.squeeze(0).cpu()                                          # (1, H, W)

            # HR 예측 - 1um 타깃(lr_target_ori) 사용
            hr_pred_tar = self.hr_model(lr_target_ori.to(self.device), hr_ori_shape)['x_recon']  # (1, 1, H, W)
            hr_pred_tar = hr_pred_tar.squeeze(0).cpu()                                           # (1, H, W)

        # 만약 cal_metric_as_1um 옵션이 켜져 있다면, numpy 변환 후 리사이즈 적용
        if cal_metric_as_1um:
            # tensor -> numpy
            lr_pred_np = lr_pred.squeeze(0).numpy()
            lr_target_np = lr_target.squeeze(0).numpy()
            hr_pred_np = hr_pred.squeeze(0).numpy()
            hr_target_np = hr_target.squeeze(0).numpy()
            hr_pred_tar_np = hr_pred_tar.squeeze(0).numpy()
            # 리사이즈
            lr_pred = self.resize_as_1um(lr_pred_np)
            lr_target = self.resize_as_1um(lr_target_np)
            hr_pred = self.resize_as_1um(hr_pred_np)
            hr_target = self.resize_as_1um(hr_target_np)
            hr_pred_tar = self.resize_as_1um(hr_pred_tar_np)
        
        # metric 계산
        metrics_lr = {}
        metrics_hr = {}
        if 'ssim' in self.metrics:
            metrics_lr["lr_ssim_target"] = self.cal_ssim(lr_target, lr_target).item()
            metrics_lr["lr_ssim"] = self.cal_ssim(lr_pred, lr_target).item()
            metrics_hr["hr_ssim_target"] = self.cal_ssim(hr_target, hr_target).item()
            metrics_hr["hr_ssim"] = self.cal_ssim(hr_pred, hr_target).item()
            metrics_hr["hr_ssim_with_lr_tar"] = self.cal_ssim(hr_pred_tar, hr_target).item()
        if 'mae' in self.metrics:
            metrics_lr["lr_mae_target"] = self.cal_mae(lr_target, lr_target).item()
            metrics_lr["lr_mae"] = self.cal_mae(lr_pred, lr_target).item()
            metrics_hr["hr_mae_target"] = self.cal_mae(hr_target, hr_target).item()
            metrics_hr["hr_mae"] = self.cal_mae(hr_pred, hr_target).item()
            metrics_hr["hr_mae_with_lr_tar"] = self.cal_mae(hr_pred_tar, hr_target).item()
        if 'f1' in self.metrics:
            metrics_lr["lr_f1_target"] = self.cal_f1(lr_target, lr_target)
            metrics_lr["lr_f1"] = self.cal_f1(lr_pred, lr_target)
            metrics_hr["hr_f1_target"] = self.cal_f1(hr_target, hr_target)
            metrics_hr["hr_f1"] = self.cal_f1(hr_pred, hr_target)
            metrics_hr["hr_f1_with_lr_tar"] = self.cal_f1(hr_pred_tar, hr_target)
        
        return {"lr": metrics_lr, "hr": metrics_hr}

# =============================================================================
# 메인 함수: 인자 파싱 후 평가 실행 (스코어만 출력)
# =============================================================================
def main(args):
    device = args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 데이터셋 인스턴스 생성 (필요 시 data_path 등 추가 인자 전달)
    dataset = IRDropInferenceAutoencoderDataset5nm(data_path=args.data_path) if args.data_path else IRDropInferenceAutoencoderDataset5nm()
    
    # 평가 파이프라인 인스턴스 생성
    pipeline = InferencePipeline1umTo210nm(
        lr_model_checkpoint=args.lr_checkpoint,
        hr_model_checkpoint=args.hr_checkpoint,
        dataset=dataset,
        device=device,
        metrics=args.metrics.split(',') if args.metrics else ['ssim']
    )
    
    # 평가할 샘플 인덱스 (쉼표로 구분된 문자열 -> 리스트)
    sample_indices = [int(idx.strip()) for idx in args.sample_indices.split(',')]
    
    # 각 샘플에 대해 평가 수행 및 스코어 출력
    for idx in sample_indices:
        print(f"\n--- Sample index: {idx} ---")
        scores = pipeline.evaluate_sample(idx, cal_metric_as_1um=args.cal_metric_as_1um)
        print("LR Metrics:")
        for key, value in scores['lr'].items():
            print(f"  {key}: {value:.4f}")
        print("HR Metrics:")
        for key, value in scores['hr'].items():
            print(f"  {key}: {value:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate LR and HR models on IR dataset (Score only)")
    parser.add_argument('--lr_checkpoint', type=str, required=True,
                        help='Path to the LR model checkpoint')
    parser.add_argument('--hr_checkpoint', type=str, required=True,
                        help='Path to the HR model checkpoint')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Optional path to dataset data')
    parser.add_argument('--sample_indices', type=str, default='0',
                        help='Comma separated sample indices to evaluate, e.g. "0,1,2"')
    parser.add_argument('--metrics', type=str, default='ssim',
                        help='Comma separated list of metrics (e.g., "ssim,mae,f1")')
    parser.add_argument('--cal_metric_as_1um', action='store_true',
                        help='Rescale predictions/targets to 1um resolution before computing metrics')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run inference on (e.g., "cuda:0" or "cpu")')
    
    args = parser.parse_args()
    main(args)
