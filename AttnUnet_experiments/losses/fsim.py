import torch
import torch.nn as nn
import torch.nn.functional as F


# class FSIMLoss(nn.Module):
#     def __init__(self, win_size=11, win_sigma=1.5, gradient_operator='scharr', data_range=1.0):
#         super(FSIMLoss, self).__init__()
#         self.win_size = win_size
#         self.win_sigma = win_sigma
#         self.gradient_operator = gradient_operator.lower()
#         self.data_range = data_range
#         self.gaussian_kernel = self._create_gaussian_kernel(win_size, win_sigma)

#         if data_range == 1.0:
#             self.C1 = 0.01
#             self.C2 = 0.03
#             self.C3 = 0.03
#         else:
#             self.C1 = (0.01 * self.data_range) ** 2
#             self.C2 = (0.03 * self.data_range) ** 2
#             self.C3 = (0.03 * self.data_range) ** 2

#     def _create_gaussian_kernel(self, win_size, win_sigma):
#         """
#         Create a 2D Gaussian kernel.
#         """
#         coords = torch.arange(win_size, dtype=torch.float32) - win_size // 2
#         g_kernel_1d = torch.exp(-coords**2 / (2 * win_sigma**2))
#         g_kernel_1d /= g_kernel_1d.sum()
#         g_kernel_2d = torch.outer(g_kernel_1d, g_kernel_1d)
#         return g_kernel_2d.view(1, 1, win_size, win_size)

#     def gaussian_filter(self, x, kernel):
#         """
#         Apply Gaussian filter to the input tensor.
#         """
#         c = x.shape[1]
#         kernel = kernel.to(x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
#         x = F.conv2d(x, kernel, padding=self.win_size // 2, groups=c)
#         return x

#     def gradient_magnitude(self, x):
#         """
#         Calculate the gradient magnitude using the specified gradient operator.
#         """
#         if self.gradient_operator == 'scharr':
#             scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
#             scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
#             grad_x = F.conv2d(x, scharr_x, padding=1, groups=x.shape[1])
#             grad_y = F.conv2d(x, scharr_y, padding=1, groups=x.shape[1])
#         else:
#             raise ValueError(f"Unsupported gradient operator: {self.gradient_operator}. Use 'scharr'.")
        
#         return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

#     def phase_congruency(self, x):
#         num_scales = 4
#         num_orientations = 4
#         _, _, height, width = x.shape
        
#         fft_x = torch.fft.fft2(x)
#         pc_sum = torch.zeros_like(x)
        
#         for scale in range(num_scales):
#             for orientation in range(num_orientations):
#                 # 로그 가보 필터 생성
#                 log_gabor = self.create_log_gabor_filter(height, width, scale, orientation).to(x.device)
                
#                 # 주파수 도메인에서 필터링
#                 filtered = fft_x * log_gabor
                
#                 # 역 FFT로 공간 도메인으로 변환
#                 response = torch.fft.ifft2(filtered).real
                
#                 # 위상 일치성 누적
#                 amplitude = torch.abs(response)
#                 pc_sum += response / (amplitude + 1e-8)
        
#         return pc_sum

#     def create_log_gabor_filter(self,height, width, scale, orientation, min_wavelength=3, mult=2.1, sigma_f=0.55, sigma_theta=0.5):
#         """
#         Create a log-Gabor filter in the frequency domain.
        
#         Args:
#         height, width: Dimensions of the filter
#         scale: Scale index (determines the filter's center frequency)
#         orientation: Orientation index (determines the filter's angle)
#         min_wavelength: Wavelength of the smallest scale filter
#         mult: Scaling factor between successive filters
#         sigma_f: Radial bandwidth
#         sigma_theta: Angular bandwidth
        
#         Returns:
#         2D torch tensor representing the log-Gabor filter in the frequency domain
#         """
#         # Create meshgrid
#         y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
#         y = y - height // 2
#         x = x - width // 2
        
#         # Convert to polar coordinates
#         radius = torch.sqrt(x**2 + y**2)
#         theta = torch.atan2(y, x)
        
#         # Avoid division by zero
#         radius[radius == 0] = 1
        
#         # Calculate the wavelength for this scale
#         wavelength = min_wavelength * (mult ** scale)
        
#         # Calculate the center frequency
#         f0 = 1.0 / wavelength
        
#         # Create the radial component
#         log_rad = torch.log2(radius / f0)
#         radial = torch.exp(-(log_rad**2) / (2 * sigma_f**2))
        
#         # Create the angular component
#         angle = orientation * math.pi / 8  # Assuming 8 orientations
#         d_theta = torch.remainder(theta - angle, math.pi)
#         angular = torch.exp(-(d_theta**2) / (2 * sigma_theta**2))
        
#         # Combine radial and angular components
#         gabor = radial * angular
        
#         # Set DC to 0
#         gabor[height//2, width//2] = 0
        
#         return gabor


#     def forward(self, X, Y):
#         """
#         Calculate FSIM between X and Y.
#         """
#         if not X.shape == Y.shape:
#             Y=Y.unsqueeze(1)
#             # raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

#         # Calculate Phase Congruency (PC) using Fourier transform
#         pc_X = self.phase_congruency(X)
#         pc_Y = self.phase_congruency(Y)

#         # Normalize PC values to [0, 1] range to improve stability
#         pc_X = (pc_X - pc_X.min()) / (pc_X.max() - pc_X.min() + 1e-6)
#         pc_Y = (pc_Y - pc_Y.min()) / (pc_Y.max() - pc_Y.min() + 1e-6)

#         # Calculate mean and variance using Gaussian filtering
#         mu1 = self.gaussian_filter(X, self.gaussian_kernel)
#         mu2 = self.gaussian_filter(Y, self.gaussian_kernel)

#         mu1_sq = mu1 ** 2
#         mu2_sq = mu2 ** 2
#         mu1_mu2 = mu1 * mu2

#         sigma1_sq = self.gaussian_filter(X * X, self.gaussian_kernel) - mu1_sq
#         sigma2_sq = self.gaussian_filter(Y * Y, self.gaussian_kernel) - mu2_sq
#         sigma12 = self.gaussian_filter(X * Y, self.gaussian_kernel) - mu1_mu2

#         # Gradient Magnitude (GM)
#         gm_X = self.gradient_magnitude(X)
#         gm_Y = self.gradient_magnitude(Y)

#         # Normalize GM values to [0, 1] range
#         gm_X = (gm_X - gm_X.min()) / (gm_X.max() - gm_X.min() + 1e-6)
#         gm_Y = (gm_Y - gm_Y.min()) / (gm_Y.max() - gm_Y.min() + 1e-6)

#         # Similarity measures for PC, GM, and statistical variance
#         S_pc = (2 * pc_X * pc_Y + self.C1) / (pc_X ** 2 + pc_Y ** 2 + self.C1 + 1e-6)
#         S_gm = (2 * gm_X * gm_Y + self.C2) / (gm_X ** 2 + gm_Y ** 2 + self.C2 + 1e-6)
#         S_sigma = (2 * sigma12 + self.C3) / (sigma1_sq + sigma2_sq + self.C3 + 1e-6)

#         # Final FSIM calculation with PC weighting
#         PC_weight = (pc_X + pc_Y) / 2
#         T = S_pc * S_gm * S_sigma * PC_weight
#         fsim_score = torch.sum(T) / (torch.sum(PC_weight) + 1e-6)

#         # Clamp the result to [0, 1] to ensure valid similarity range
#         return 1 - fsim_score


#     def coefficient(self, X, Y):
#         """
#         Calculate FSIM-based loss between X and Y.
#         """
#         return 1 - self.forward(X, Y)

class FSIMLoss(nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, gradient_operator='scharr'):
        super(FSIMLoss, self).__init__()
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.gradient_operator = gradient_operator.lower()
        self.gaussian_kernel = self._create_gaussian_kernel(win_size, win_sigma)

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
        """
        Calculate the Phase Congruency (PC) of the input tensor.
        """
        # Fourier Transform of the input image
        fft = torch.fft.fft2(x)
        amplitude = torch.abs(fft)
        phase = torch.angle(fft)

        # Constructing log-Gabor filters
        num_scales = 4
        num_orientations = 4
        _, _, height, width = x.shape
        y, x = torch.meshgrid(torch.linspace(-0.5, 0.5, height, device=fft.device),
                              torch.linspace(-0.5, 0.5, width, device=fft.device))
        radius = torch.sqrt(x ** 2 + y ** 2)
        radius = torch.fft.fftshift(radius)
        radius[0, 0] = 1  # Avoid division by zero

        pc_sum = torch.zeros_like(amplitude)
        epsilon = 1e-4

        for scale in range(num_scales):
            for orientation in range(num_orientations):
                theta = orientation * torch.pi / num_orientations
                theta = torch.tensor(theta,device=fft.device)
                x_theta = x * torch.cos(theta) + y * torch.sin(theta)
                y_theta = -x * torch.sin(theta) + y * torch.cos(theta)
                log_gabor = torch.exp((-(torch.log(radius / (0.1 * (2 ** scale)))) ** 2) / (2 * (0.55 ** 2)))
                log_gabor *= torch.exp(-((x_theta ** 2 + y_theta ** 2) / (2 * (0.4 ** 2))))
                log_gabor = log_gabor.unsqueeze(0).unsqueeze(0)

                response = torch.fft.ifft2(fft * log_gabor).real
                pc_sum += torch.relu(response - torch.mean(response)) / (torch.std(response) + epsilon)

        return pc_sum

    def forward(self, X, Y):
        """
        Calculate FSIM between X and Y.
        """
        if not X.shape == Y.shape:
            raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

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
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        C3 = 0.03 ** 2

        S_pc = (2 * pc_X * pc_Y + C1) / (pc_X ** 2 + pc_Y ** 2 + C1)
        S_gm = (2 * gm_X * gm_Y + C2) / (gm_X ** 2 + gm_Y ** 2 + C2)
        S_sigma = (2 * sigma12 + C3) / (sigma1_sq + sigma2_sq + C3)
        
        PC_weight = (pc_X + pc_Y) / 2
        # Final FSIM calculation
        T = S_pc * S_gm * S_sigma * PC_weight
        fsim_score = torch.sum(T) / (torch.sum(PC_weight) + 1e-6)

        return 1-fsim_score

# Example usage
if __name__ == "__main__":
    # Define dummy images
    device = torch.device('cuda:3')
    org_img = torch.rand((1, 1, 256, 256), requires_grad=True).to(device)  # Batch size of 1, 1 channel, 256x256 image
    pred_img = torch.rand((1, 1, 256, 256), requires_grad=True).to(device)

    # Instantiate FSIM loss and calculate loss
    criterion = FSIMLoss()
    loss = criterion(org_img, org_img)
    loss.backward()  # Ensure backward pass works
    print("FSIM Loss:", loss.item())
