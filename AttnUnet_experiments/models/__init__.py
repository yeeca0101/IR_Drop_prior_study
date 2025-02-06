import torch.nn as nn
from .papers_attn_unet import AttUNet 
from .papers_model import AttnUnetBase
from .papers_model_v2 import AttnUnetV2
from .papers_model_v3 import AttnUnetV3
from .papers_model_v4 import AttnUnetV4
from .papers_model_v5 import AttnUnetV5, AttnUnetV5_1,AttnUnetV5_2
from .papers_model_v6 import AttnUnetV6, AttnUnetV6_1, AttnUnetV6_2, SwishT
from .sr_models import *
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
    elif arch == 'vqvae': # legacy
        # model = create_model(args.vqvae_size,in_ch=in_ch,use_ema = use_ema) # vqvae 모델 생성 추가
        pass
    else:
        raise NameError(f'arch type error : {arch}')
    return model
