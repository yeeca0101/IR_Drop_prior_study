# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
import warnings
from typing import List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input: Tensor, win: Tensor) -> Tensor:
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float,
    win: Tensor,
    size_average: bool = True,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03)
) -> Tuple[Tensor, Tensor]:
    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        data_range (float or int): value range of input images. (usually 1.0 or 255)
        win (torch.Tensor): 1-D gauss kernel
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
) -> Tensor:
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    #if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    weights: Optional[List[float]] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03)
) -> Tensor:
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    #if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
        2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights_tensor = X.new_tensor(weights)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights_tensor.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # type: ignore  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights_tensor.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class SSIMLoss(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
        nonnegative_ssim: bool = False,
    ) -> None:
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIMLoss, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        Y = Y.unsqueeze(1)
        return 1.-ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )


class MS_SSIMLoss(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        weights: Optional[List[float]] = None,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    ) -> None:
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MS_SSIMLoss, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        Y = Y.unsqueeze(1)
        return 1- ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        )

class IW_SSIMLoss(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        importance_weights: Optional[Tensor] = None,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
        nonnegative_ssim: bool = False,
    ) -> None:
        r""" class for IW-SSIM
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            importance_weights (torch.Tensor, optional): pixel-wise importance weights
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(IW_SSIMLoss, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.importance_weights = importance_weights
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        Y = Y.unsqueeze(1)
        ssim_map = ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=False,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )
        if self.importance_weights is not None:
            importance_weights = self.importance_weights.to(X.device, dtype=X.dtype)
            ssim_map = ssim_map * importance_weights

        if self.size_average:
            return 1. - ssim_map.mean()
        else:
            return 1. - ssim_map.mean(1)



class FSIMLoss(nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, gradient_operator='scharr'):
        super(FSIMLoss, self).__init__()
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.gradient_operator = gradient_operator.lower()
        self.gaussian_kernel = self._create_gaussian_kernel(win_size, win_sigma)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        self.C3 = 0.03 ** 2

    def _create_gaussian_kernel(self, win_size, win_sigma):
        """
        Create a 1D Gaussian kernel.
        """
        coords = torch.arange(win_size, dtype=torch.float32) - win_size // 2
        g_kernel = torch.exp(-coords**2 / (2 * win_sigma**2))
        g_kernel /= g_kernel.sum()
        return g_kernel.view(1, 1, -1)

    def gaussian_filter(self, x, kernel):
        """
        Apply Gaussian filter to the input tensor.
        """
        c = x.shape[1]
        kernel = kernel.to(x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
        x = F.conv2d(x, kernel, padding=(self.win_size//2, 0), groups=c)
        x = F.conv2d(x, kernel.transpose(2, 3), padding=(0, self.win_size//2), groups=c)
        return x

    def gradient_magnitude(self, x):
        """
        Calculate the gradient magnitude using the specified gradient operator.
        """
        if self.gradient_operator == 'sobel':
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.shape[1])
            grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.shape[1])
        elif self.gradient_operator == 'scharr':
            scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            grad_x = F.conv2d(x, scharr_x, padding=1, groups=x.shape[1])
            grad_y = F.conv2d(x, scharr_y, padding=1, groups=x.shape[1])
        else:
            raise ValueError(f"Unsupported gradient operator: {self.gradient_operator}. Use 'sobel' or 'scharr'.")
        
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    def phase_congruency(self, x):
        fft = torch.fft.fft2(x)
        amplitude = torch.abs(fft)
        phase = torch.angle(fft)

        num_scales = 4
        num_orientations = 4
        _, _, height, width = x.shape
        y, x = torch.meshgrid(torch.linspace(-0.5, 0.5, height, device=fft.device),
                            torch.linspace(-0.5, 0.5, width, device=fft.device), indexing='ij')
        radius = torch.sqrt(x ** 2 + y ** 2)
        radius = torch.fft.fftshift(radius)
        radius[radius == 0] = 1  # Avoid division by zero

        pc_sum = torch.zeros_like(amplitude)
        epsilon = 1e-4

        for scale in range(num_scales):
            for orientation in range(num_orientations):
                theta = torch.tensor(orientation * torch.pi / num_orientations,device=fft.device)
                x_theta = x * torch.cos(theta) + y * torch.sin(theta)
                y_theta = -x * torch.sin(theta) + y * torch.cos(theta)
                log_gabor = torch.exp((-(torch.log(radius / (0.1 * (2 ** scale)))) ** 2) / (2 * (0.55 ** 2)))
                log_gabor *= torch.exp(-((x_theta ** 2 + y_theta ** 2) / (2 * (0.4 ** 2))))
                log_gabor = log_gabor.unsqueeze(0).unsqueeze(0)

                filtered = fft * log_gabor
                response = torch.fft.ifft2(filtered).real
                pc_sum += torch.relu(response - torch.mean(response)) / (torch.std(response) + epsilon)

        return pc_sum

    def forward(self, X, Y):
        """
        Calculate FSIM between X and Y.
        """
        if not X.shape == Y.shape:
            # raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")
            Y = Y.unsqueeze(1)

        # Calculate Phase Congruency (PC) using Fourier transform
        pc_X = self.phase_congruency(X)
        pc_Y = self.phase_congruency(Y)

        # Calculate mean and variance using Gaussian filtering
        mu1 = self.gaussian_filter(X, self.gaussian_kernel)
        mu2 = self.gaussian_filter(Y, self.gaussian_kernel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self.gaussian_filter(X * X, self.gaussian_kernel) - mu1_sq
        sigma2_sq = self.gaussian_filter(Y * Y, self.gaussian_kernel) - mu2_sq
        sigma12 = self.gaussian_filter(X * Y, self.gaussian_kernel) - mu1_mu2

        # Gradient Magnitude (GM)
        gm_X = self.gradient_magnitude(X)
        gm_Y = self.gradient_magnitude(Y)

        # Similarity measures for PC, GM, and statistical variance
        S_pc = (2 * pc_X * pc_Y + self.C1) / (pc_X ** 2 + pc_Y ** 2 + self.C1)
        S_gm = (2 * gm_X * gm_Y + self.C2) / (gm_X ** 2 + gm_Y ** 2 + self.C2)
        S_sigma = (2 * sigma12 + self.C3) / (sigma1_sq + sigma2_sq + self.C3)
        
        PC_weight = (pc_X + pc_Y) / 2
        # Final FSIM calculation
        T = S_pc * S_gm * S_sigma * PC_weight
        fsim_score = torch.sum(T) / (torch.sum(PC_weight) + 1e-6)

        return 1-fsim_score
    


    

# # 예제 코드
if __name__ == '__main__':

    ssim_loss_fn =SSIMLoss(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
    ms_ssim_loss_fn=MS_SSIMLoss(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
    loss_fn = IW_SSIMLoss(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
    fsim_loss_fn = FSIMLoss()

    # Suppose pred and target are torch tensors with shape (batch_size, channels, height, width)
    pred = torch.randn(32, 1, 512, 512).to('cuda:3')
    target = torch.randn(32, 512, 512).to('cuda:3')

    def loss_test(pred,target,loss_fn):
        fn = torch.sigmoid
        fn = F.relu
        pred = fn(pred)
        target = fn(target)

        loss = loss_fn(pred, target)
        print(loss)

    loss_test(pred,target,ms_ssim_loss_fn)
    loss_test(pred,target,ssim_loss_fn)
    loss_test(pred,target,loss_fn)
    loss_test(pred,target,fsim_loss_fn)
