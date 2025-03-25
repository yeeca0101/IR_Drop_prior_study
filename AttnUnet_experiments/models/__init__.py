import torch.nn as nn
from .papers_attn_unet import AttUNet 
from .papers_model import AttnUnetBase
from .papers_model_v2 import AttnUnetV2
from .papers_model_v3 import AttnUnetV3
from .papers_model_v4 import AttnUnetV4
from .papers_model_v5 import AttnUnetV5, AttnUnetV5_1,AttnUnetV5_2
from .papers_model_v6 import AttnUnetV6, AttnUnetV6_1, AttnUnetV6_2, SwishT
from .v7_cvq_resnxt_unet import AttnUnetV7, SwishT_C,SwishT_B, SwishT


from .sr_models import *

# developing
from .model_factory.resmlp_unet import ResMLP_UNet
from .model_factory.efficientformer.efficientformer import EfficientFormer
from .model_factory.efficientformer.config import  (efficientformerv2_s0_irdrop_config,
                                                    efficientformerv2_s1_irdrop_config,
                                                    efficientformerv2_s2_irdrop_config,
                                                    )

from .model_factory.cfirstnet.net import Net as CFIRST
from .model_factory.resnextunet import build_resnext_unet
from .parts.vqvae import create_model



def build_model(arch,dropout_type,is_fintune,in_ch,use_ema,num_embeddings=512):
    if arch == 'attn_12ch':
        pass
    elif arch == 'attn_base':
        model = AttnUnetBase()
    elif arch == 'attnv2':
        model = AttnUnetV2(dropout_name=dropout_type,dropout_p=0.05 if is_fintune else 0.1, in_ch=in_ch)
    elif arch == 'attnv3':
        model = AttnUnetV3(dropout_name=dropout_type,dropout_p=0.05 if is_fintune else 0.1, in_ch=in_ch)
    elif arch == 'attnv4':
        model = AttnUnetV4(dropout_name=dropout_type,dropout_p=0.05 if is_fintune else 0.1, in_ch=in_ch, num_head=2 if in_ch==2 else 4)    
    elif arch == 'attnv5':
        model = AttnUnetV5(dropout_name=dropout_type,dropout_p=0.05 if is_fintune else 0.1, in_ch=in_ch,use_ema=use_ema,num_embeddings=num_embeddings)        
    elif arch == 'attnv5_1':
        model = AttnUnetV5_1(dropout_name=dropout_type,dropout_p=0.05 if is_fintune else 0.1, in_ch=in_ch,use_ema=use_ema,num_embeddings=num_embeddings)        
    elif arch == 'attnv5_2':
        model = AttnUnetV5_2(dropout_name=dropout_type,dropout_p=0.05 if is_fintune else 0.1, in_ch=in_ch,use_ema=use_ema,num_embeddings=num_embeddings)        
    elif arch == 'attnv6':
        model = AttnUnetV6(dropout_name=dropout_type,dropout_p=0.05 if is_fintune else 0.1, in_ch=in_ch,use_ema=use_ema,num_embeddings=num_embeddings)    
    elif arch == 'attnv6_1':
        model = AttnUnetV6_1(dropout_name=dropout_type,dropout_p=0.05 if is_fintune else 0.1, in_ch=in_ch,use_ema=use_ema,num_embeddings=num_embeddings)    
    elif arch == 'attnv6_2':
        model = AttnUnetV6_2(dropout_name=dropout_type,dropout_p=0.05 if is_fintune else 0.1, in_ch=in_ch,use_ema=use_ema,num_embeddings=num_embeddings)
    elif arch == 'sr_v1':
        model = SRModelV1(dropout_name=dropout_type,dropout_p=0.05 if is_fintune else 0.1, in_ch=in_ch)    
    elif arch == 'sr_v2':
        model = SRModelV2(in_ch=1, out_ch=1, upscale_factor=4, num_features=64, num_rrdb=8, growth_rate=32)
    elif arch == 'efficientformers0':
        model = EfficientFormer(resolution=256, in_ch=in_ch,num_classes=1, **efficientformerv2_s0_irdrop_config)
    elif arch == 'efficientformers1':
        model = EfficientFormer(resolution=256, in_ch=in_ch,num_classes=1, **efficientformerv2_s1_irdrop_config)
    elif arch == 'efficientformers2':
        model = EfficientFormer(resolution=256, in_ch=in_ch,num_classes=1, **efficientformerv2_s2_irdrop_config)
    elif arch == 'attnv7':
        model = AttnUnetV7(dropout_name=dropout_type,
                           dropout_p=0.05 if is_fintune else 0.1,
                           in_ch=in_ch,
                           num_embeddings=num_embeddings)
    elif arch in ['resnextunet','resnextunet_small','resnextunet_base','resnextunet_large']:
        if arch == 'resnextunet':
            size = 'small'
        else:
            size = arch.split('_')[-1]
        model = build_resnext_unet(size, in_ch=in_ch)
    elif arch == 'cfirst':
        model = CFIRST(
                model_backbone="convnextv2_tiny.fcmae",
                model_pretrained='True',
                in_channels=in_ch,
                stochastic_depth=0.5,
                dropout=0.0,
                decoder_channels=128,
                out_channels=1,
    )

    elif arch =='resmlp_unet':
        model = ResMLP_UNet(in_ch=in_ch,base_dim=32)
    elif arch == 'vqvae': # legacy
        # model = create_model(args.vqvae_size,in_ch=in_ch,use_ema = use_ema) # vqvae 모델 생성 추가
        pass
    else:
        raise NameError(f'arch type error : {arch}')
    
    if 'attn' in arch and is_fintune:
        try:
            for param in model.vq.parameters():
                param.requires_grad = False
            print('model.vq is off-line mode')    
        except:
            print('model.vq off-line mode error')
            pass
    return model
