import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import *

class CustomLoss(nn.Module):
    def __init__(self, lambda_param=2):
        super(CustomLoss, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, predictions, targets):
        # 예측 값과 실제 값의 차이를 계산
        diff = predictions - targets

        # 조건에 따라 loss 계산
        loss = torch.where(predictions >= targets, 
                           torch.abs(diff),              # 예측 값이 실제 값보다 크거나 같을 때
                           self.lambda_param * torch.abs(diff))  # 예측 값이 실제 값보다 작을 때
        
        # 전체 loss를 평균하여 반환
        return loss.mean()

def gradient_loss(predicted, target):
    # Check the input shapes (B,C,H,W)
    target = target.unsqueeze(1)
    # Compute gradient in the x direction
    predicted_dx = torch.abs(predicted[:, :, 1:, :] - predicted[:, :, :-1, :])
    target_dx = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    
    # Compute gradient in the y direction
    predicted_dy = torch.abs(predicted[:, :, :, 1:] - predicted[:, :, :, :-1])
    target_dy = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    
    # Pad the gradients to match the original size
    predicted_dx = torch.nn.functional.pad(predicted_dx, (0, 0, 1, 0))  # Pad height (x-direction)
    target_dx = torch.nn.functional.pad(target_dx, (0, 0, 1, 0))
    
    predicted_dy = torch.nn.functional.pad(predicted_dy, (1, 0, 0, 0))  # Pad width (y-direction)
    target_dy = torch.nn.functional.pad(target_dy, (1, 0, 0, 0))
    
    # Compute the gradient loss (MSE of the gradients)
    return torch.mean((predicted_dx - target_dx)**2 + (predicted_dy - target_dy)**2)

def weighted_mse_loss(predicted, target, threshold=0.9):
    # Create weight map where larger IR drop values get higher weight
    mask = (target > torch.quantile(target, threshold)).float()
    return torch.mean(mask * (predicted - target) ** 2)

def sobel_operator(x):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32,device=x.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32,device=x.device).unsqueeze(0).unsqueeze(0)
    grad_x = F.conv2d(x, sobel_x)
    grad_y = F.conv2d(x, sobel_y)
    return grad_x, grad_y

def edge_loss(predicted, target):
    target = target.unsqueeze(1)
    predicted_edges_x, predicted_edges_y = sobel_operator(predicted)
    target_edges_x, target_edges_y = sobel_operator(target)
    return torch.mean((predicted_edges_x - target_edges_x) ** 2 + (predicted_edges_y - target_edges_y) ** 2)

# You can also add losses that consider how the current map and pdn_density affect IR drop
def combined_loss(predicted, target, alpha=0.5, beta=0.3, gamma=0.2):
    hotspot_loss = weighted_mse_loss(predicted, target)
    mse = F.mse_loss(predicted.squeeze(1), target)
    grad_loss = gradient_loss(predicted, target)
    
    return mse + alpha * grad_loss + beta * hotspot_loss + gamma * edge_loss(predicted, target)

def default_loss(predictions, targets):
        # 예측 값과 실제 값의 차이를 계산
        diff = predictions - targets
        # 조건에 따라 loss 계산
        loss = torch.where(predictions >= targets, 
                           torch.abs(diff),              # 예측 값이 실제 값보다 크거나 같을 때
                           2 * torch.abs(diff))  # 예측 값이 실제 값보다 작을 때
        # 전체 loss를 평균하여 반환
        return loss.mean()

def mae_loss(predicted, target):
    loss = torch.abs(predicted-target)
    return loss.mean()

def default_and_edge(predicted, target,):
    mae_l = default_loss(predictions=predicted,targets=target)
    edge = edge_loss(predicted, target)

    return mae_l + edge

class LossSelect(nn.Module):
    def __init__(self, loss_type='default',lambda_fn_dict={},
                 use_cache=False,dice_q=0.9,loss_with_logit=True,post_min_max=False) -> None:
        '''
        lambda_fn_dict : {
                    'loss_1':[weight_1,loss_type_1],
                    'loss_2':[weight_2,loss_type_2],
                    ...
        }
        '''
        super().__init__()
        print(f'loss type : {loss_type}')
        self.loss_with_logit =loss_with_logit
        self.post_min_max = post_min_max

        if use_cache:
            lambda_fn_dict = {
                '1000':[0.5,'ssim'], # min_max (target), (pred)
                'min_max':[0.2,'mae'],     # target*1,000
                'dice' : [0.3,'dice']
                # 'linear'              # x=x
            }
            print('use cache dict : ',lambda_fn_dict)
        self.loss_type = loss_type
        self.lambda_fn_dict = lambda_fn_dict
        self.loss_fn = None
        self.dice_q=dice_q

        if loss_type:
            self.loss_fn = self.get_fn()
        
        if lambda_fn_dict:
            self.loss_fn = self.pipe_loss

        assert self.loss_fn is not None, 'check LossSelect opt'

        self.apply_fn = {
                'min_max':min_max_norm,
                '1000': lambda x : x*1000 ,
                '1000_min_max': lambda x : min_max_norm(x*1000) ,
        }

    def get_fn(self,):
        loss_fn=None
        if self.loss_type=='default':
            loss_fn = CustomLoss(lambda_param=2)
        elif self.loss_type=='mse':
            loss_fn = F.mse_loss
        elif self.loss_type=='mae':
            loss_fn = mae_loss
        elif self.loss_type=='default_edge':
            loss_fn = default_and_edge
        elif self.loss_type=='comb':
            loss_fn = combined_loss
        elif self.loss_type=='edge':
            loss_fn = edge_loss
        elif self.loss_type == 'gradient':
            loss_fn = gradient_loss
        elif self.loss_type=='ssim':
            loss_fn = SSIMLoss(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
        elif self.loss_type=='ms_ssim':
            loss_fn = MS_SSIMLoss(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
        elif self.loss_type=='fsim':
            loss_fn = FSIMLoss()
        elif self.loss_type == 'dice':
            loss_fn = IRDiceLoss(self.dice_q)
        elif self.loss_type in ['kl', 'js', 'wasserstein', 'correlation', 'histogram_matching','kl_restoration']:
            loss_fn = PixelDistributionLoss(loss_type=self.loss_type)
        elif self.loss_type == 'ec_ssim': # Error-Centric SSIM Loss (EC-SSIM Loss)
            loss_fn = ECSSIMLoss()
        elif self.loss_type == 'ssim_mae_ec_ssim_dice':
            self.lambda_fn_dict={
                'loss_1':[0.4,'ssim'],
                'loss_2':[0.1,'ec_ssim'],
                'loss_3':[0.4,'mae'],
                'loss_3':[0.1,'dice'],
            }
            loss_fn = self.combined_loss
        elif self.loss_type == 'ms_ssim_mae_ec_ssim_dice':
            self.lambda_fn_dict={
                'loss_1':[0.4,'ms_ssim'],
                'loss_2':[0.1,'ec_ssim'],
                'loss_3':[0.4,'mae'],
                'loss_3':[0.1,'dice'],
            }
            loss_fn = self.combined_loss
        elif self.loss_type == 'ssim_ec_ssim':
            self.lambda_fn_dict={
                'loss_2':[0.5,'ssim'],
                'loss_3':[0.5,'ec_ssim'],
            }
            loss_fn = self.combined_loss
        elif self.loss_type == 'ec_ssim_mae':
            self.lambda_fn_dict={
                'loss_2':[0.5,'ec_ssim'],
                'loss_3':[0.5,'mae'],
            }
            loss_fn = self.combined_loss
        elif self.loss_type == 'ms_ssim_ec_ssim':
            self.lambda_fn_dict={
                'loss_2':[0.5,'ms_ssim'],
                'loss_3':[0.5,'ec_ssim'],
            }
            loss_fn = self.combined_loss
        elif self.loss_type == 'ssim_dice':
            self.lambda_fn_dict={
                'loss_2':[1,'ssim'],
                'loss_3':[1,'dice'],
            }
            loss_fn = self.combined_loss
        elif self.loss_type == 'ms_ssim_fsim':
            self.lambda_fn_dict={
                'loss_2':[1,'ms_ssim'],
                'loss_3':[0.05,'fsim'],
            }
            loss_fn = self.combined_loss
        elif self.loss_type == 'ssim_dice_mse':
            self.lambda_fn_dict={
                'loss_2':[1,'ssim'],
                'loss_3':[0.1,'dice'],
                'loss_1':[1,'mse']
            }
            loss_fn = self.combined_loss
        elif self.loss_type == 'ssim_mae':
            self.lambda_fn_dict={
                'loss_2':[1,'ssim'],
                'loss_3':[1,'mae'],
            }
            loss_fn = self.combined_loss 
        elif self.loss_type == 'ssim_default':
            self.lambda_fn_dict={
                'loss_2':[1,'ssim'],
                'loss_3':[0.2,'default'],
            }
            loss_fn = self.combined_loss  
        elif self.loss_type == 'ssim_mse':
            self.lambda_fn_dict={
                'loss_2':[0.5,'ssim'],
                'loss_3':[0.5,'mse'],
            }
            loss_fn = self.combined_loss
        elif self.loss_type == 'ms_ssim_mse':
            self.lambda_fn_dict={
                'loss_2':[1,'ms_ssim'],
                'loss_3':[1,'mse'],
            }
            loss_fn = self.combined_loss    
        elif self.loss_type == 'ms_ssim_mae':
            self.lambda_fn_dict={
                'loss_2':[1,'ms_ssim'],
                'loss_3':[1,'mae'],
            }
            loss_fn = self.combined_loss 
        elif self.loss_type == 'ms_ssim_dice':
            self.lambda_fn_dict={
                'loss_2':[1,'ms_ssim'],
                'loss_3':[1,'dice'],
            }
            loss_fn = self.combined_loss  
        elif self.loss_type == 'ssim_dice_mae':
            self.lambda_fn_dict={
                'loss_2':[0.8,'ssim'],
                'loss_3':[0.1,'dice'],
                'loss_1':[0.1,'mae']
            }
            loss_fn = self.combined_loss 
        elif self.loss_type == 'ssim_dice_mae_fsim':
            self.lambda_fn_dict={
                'loss_1':[1.,'ssim'],
                'loss_2':[0.05,'dice'],
                'loss_3':[0.5,'mae'],
                'loss_4':[0.05,'fsim'],
            }
            loss_fn = self.combined_loss   
        elif self.loss_type == 'ms_ssim_dice_mae':
            self.lambda_fn_dict={
                'loss_1':[1,'ms_ssim'],
                'loss_2':[0.1,'dice'],
                'loss_3':[1,'mae'],
            }
            loss_fn = self.combined_loss
        elif self.loss_type == 'ssim_mae_ec_ssim':
            self.lambda_fn_dict={
                'loss_1':[1,'ssim'],
                'loss_2':[0.1,'ec_ssim'],
                'loss_3':[1,'mae'],
            }
            loss_fn = self.combined_loss
        elif self.loss_type == 'ms_ssim_mae_ec_ssim':
            self.lambda_fn_dict={
                'loss_1':[1,'ms_ssim'],
                'loss_2':[0.1,'ec_ssim'],
                'loss_3':[1,'mae'],
            }
            loss_fn = self.combined_loss
        elif self.loss_type == 'ms_ssim_dice_mae_fsim':
            self.lambda_fn_dict={
                'loss_1':[1.,'ms_ssim'],
                'loss_2':[0.05,'dice'],
                'loss_3':[0.5,'mae'],
                'loss_4':[0.05,'fsim'],
            }
            loss_fn = self.combined_loss  
        ##### distribution #########
        elif self.loss_type == 'ms_ssim_kl':
            self.lambda_fn_dict={
                'loss_1':[1.,'ms_ssim'],
                'loss_2':[1.,'kl'],
            }
            loss_fn = self.combined_loss  
        elif self.loss_type == 'ms_ssim_wasserstein':
            self.lambda_fn_dict={
                'loss_1':[1.,'ms_ssim'],
                'loss_2':[0.1,'wasserstein'],
            }
            loss_fn = self.combined_loss  
        elif self.loss_type == 'ssim_kl_restoration':
            self.lambda_fn_dict={
                'loss_1':[1.,'ssim'],
                'loss_2':[1.,'kl_restoration'],
            }
            loss_fn = self.combined_loss  
        
        else:
            return None
        return loss_fn
    
    def combined_loss(self, pred, target):
        total_loss = 0.
        for apply_name, (weight, loss_type) in self.lambda_fn_dict.items():
            # pred,target = self.make_logits(pred,target,apply_name)
            self.loss_type = loss_type
            loss_fn = self.get_fn()
            loss = loss_fn(pred, target)
            total_loss += weight * loss
        return total_loss
    
    def pipe_loss(self,pred,target):
        # target = min_max_norm(target * 100)
        # pred = min_max_norm(pred * 100)
        target, target_min, target_max = min_max_norm(target)
        pred_scaled, pred_min, pred_max = min_max_norm(pred)

        total_loss = 0.
        for apply_name, (weight, loss_type) in self.lambda_fn_dict.items():
            self.loss_type = loss_type
            loss_fn = self.get_fn()
            loss = loss_fn(pred, target)
            total_loss += weight * loss
        
        total_loss += ( 
                       F.mse_loss(pred_min,target_min)+
                       F.mse_loss(pred_max,target_max))

        return total_loss


    def forward(self,pred,target,):
        if self.post_min_max: pred= min_max_norm(pred)
        return self.loss_fn(pred,target)

    def make_logits(self,pred,target,apply_name):
        if apply_name in 'min_max':
            return self.apply_fn['min_max'](pred), self.apply_fn['min_max'](target)
        elif apply_name in '1000':
            return pred, self.apply_fn['1000'](target)
        else:
            return pred, target


############ Scaling ############################
# def min_max_norm(x):
#     return (x-x.min()+1e-5)/(x.max()-x.min()+1e-5)
def min_max_norm(x, min_val=None, max_val=None):
    if min_val is None:
        min_val = x.amin(dim=(1, 2), keepdim=True)
    if max_val is None:
        max_val = x.amax(dim=(1, 2), keepdim=True)
    return (x - min_val) / (max_val - min_val + 1e-8), min_val, max_val

def inverse_min_max_norm(x, min_val, max_val):
    return x * (max_val - min_val) + min_val

def log_scale(x, epsilon=1.0):
    return torch.log(x + epsilon)

def inverse_log_scale(x, epsilon=1.0):
    return torch.exp(x) - epsilon

# # 모델의 예측 값과 실제 값 (예시로 가상의 데이터를 사용)
# predictions = torch.tensor([[0.9, 0.8], [0.4, 0.6]], requires_grad=True)
# targets = torch.tensor([[1.0, 0.7], [0.5, 0.5]])
################################################################################
def gen_test_data():
    predictions = torch.randn((4,1,32,32),requires_grad=True)
    targets  =torch.randint(0, 2, (4, 32, 32)).float()
    return predictions, targets

# Custom Loss 함수 정의 및 적용
def loss_test(loss_type,predictions,targets):
    criterion = LossSelect(loss_type)
    loss = criterion(predictions, targets)
    # 역전파 수행
    loss.backward()
    print(f'Loss: {loss.item()}')

def total_loss_test():
    predictions, targets = gen_test_data()
    for _loss in ['default','default_edge','comb','edge','gradient','ssim','dice']:
        try:
            loss_test(_loss,predictions,targets)
        except:
            loss_test(_loss,predictions,targets.unsqueeze(1))

def comb_loss_test(loss_with_logit=True):
    predictions, targets = gen_test_data()
    criterion = LossSelect('loss_type',use_cache=True,loss_with_logit=loss_with_logit)
    loss = criterion(predictions, targets)
    # 역전파 수행
    # loss.backward()
    print(f'Loss: {loss.item()}')

###############################################################################
if __name__ == '__main__':
    comb_loss_test()
    comb_loss_test(loss_with_logit=False)
