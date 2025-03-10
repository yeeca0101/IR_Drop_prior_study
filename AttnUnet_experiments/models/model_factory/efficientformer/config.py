import math

# 모델 너비 및 깊이 설정
EfficientFormer_width = {
    'L': [40, 80, 192, 384],   # L 모델
    'S2': [32, 64, 144, 288],   # S2 모델
    'S1': [32, 48, 120, 224],   # S1 모델
    'S0': [32, 48, 96, 176],    # S0 모델
}

EfficientFormer_depth = {
    'L': [5, 5, 15, 10],
    'S2': [4, 4, 12, 8],
    'S1': [3, 3, 9, 6],
    'S0': [2, 2, 6, 4],
}

# 각 stage의 확장 비율
expansion_ratios_L = {
    '0': [4, 4, 4, 4, 4],
    '1': [4, 4, 4, 4, 4],
    '2': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 4, 3, 3, 3, 3, 4, 4, 4],
}

expansion_ratios_S2 = {
    '0': [4, 4, 4, 4],
    '1': [4, 4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 3, 3, 3, 3, 4, 4],
}

expansion_ratios_S1 = {
    '0': [4, 4, 4],
    '1': [4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 4, 4, 4],
    '3': [4, 4, 3, 3, 4, 4],
}

expansion_ratios_S0 = {
    '0': [4, 4],
    '1': [4, 4],
    '2': [4, 3, 3, 3, 4, 4],
    '3': [4, 3, 3, 4],
}

###############################################################################
# Classification Configurations (fork_feat=False)
###############################################################################
efficientformerv2_s0_config = {
    "layers": EfficientFormer_depth['S0'],
    "embed_dims": EfficientFormer_width['S0'],
    "downsamples": [True, True, True, True],
    "fork_feat": False,
    "drop_path_rate": 0.0,
    "vit_num": 2,
    "e_ratios": expansion_ratios_S0,
}

efficientformerv2_s1_config = {
    "layers": EfficientFormer_depth['S1'],
    "embed_dims": EfficientFormer_width['S1'],
    "downsamples": [True, True, True, True],
    "fork_feat": False,
    "drop_path_rate": 0.0,
    "vit_num": 2,
    "e_ratios": expansion_ratios_S1,
}

efficientformerv2_s2_config = {
    "layers": EfficientFormer_depth['S2'],
    "embed_dims": EfficientFormer_width['S2'],
    "downsamples": [True, True, True, True],
    "fork_feat": False,
    "drop_path_rate": 0.02,
    "vit_num": 4,
    "e_ratios": expansion_ratios_S2,
}

efficientformerv2_l_config = {
    "layers": EfficientFormer_depth['L'],
    "embed_dims": EfficientFormer_width['L'],
    "downsamples": [True, True, True, True],
    "fork_feat": False,
    "drop_path_rate": 0.1,
    "vit_num": 6,
    "e_ratios": expansion_ratios_L,
}

###############################################################################
# Segmentation Configurations (fork_feat=True, FPN Head 사용)
###############################################################################
efficientformerv2_s0_seg_config = {
    "layers": EfficientFormer_depth['S0'],
    "embed_dims": EfficientFormer_width['S0'],
    "downsamples": [True, True, True, True],
    "fork_feat": True,
    "drop_path_rate": 0.0,
    "vit_num": 2,
    "e_ratios": expansion_ratios_S0,
    "head_cfg": {
         "type": "FPNHead",
         "out_channels": 256,
         "num_classes": 21,
         "dropout": 0.1,
    },
}

efficientformerv2_s1_seg_config = {
    "layers": EfficientFormer_depth['S1'],
    "embed_dims": EfficientFormer_width['S1'],
    "downsamples": [True, True, True, True],
    "fork_feat": True,
    "drop_path_rate": 0.0,
    "vit_num": 2,
    "e_ratios": expansion_ratios_S1,
    "head_cfg": {
         "type": "FPNHead",
         "out_channels": 256,
         "num_classes": 21,
         "dropout": 0.1,
    },
}

efficientformerv2_s2_seg_config = {
    "layers": EfficientFormer_depth['S2'],
    "embed_dims": EfficientFormer_width['S2'],
    "downsamples": [True, True, True, True],
    "fork_feat": True,
    "drop_path_rate": 0.02,
    "vit_num": 4,
    "e_ratios": expansion_ratios_S2,
    "head_cfg": {
         "type": "FPNHead",
         "out_channels": 256,
         "num_classes": 21,
         "dropout": 0.1,
    },
}

efficientformerv2_l_seg_config = {
    "layers": EfficientFormer_depth['L'],
    "embed_dims": EfficientFormer_width['L'],
    "downsamples": [True, True, True, True],
    "fork_feat": True,
    "drop_path_rate": 0.1,
    "vit_num": 6,
    "e_ratios": expansion_ratios_L,
    "head_cfg": {
         "type": "FPNHead",
         "out_channels": 256,
         "num_classes": 21,
         "dropout": 0.1,
    },
}

###############################################################################
# IR Drop Prediction Configurations (fork_feat=True, IRDropHead 사용)
###############################################################################
efficientformerv2_s0_irdrop_config = {
    # "in_ch":25, # build model에서 접근
    "layers": EfficientFormer_depth['S0'],
    "embed_dims": EfficientFormer_width['S0'],
    "downsamples": [True, True, True, True],
    "fork_feat": True,  # multi-scale feature 출력
    "drop_path_rate": 0.0,
    "vit_num": 2,
    "e_ratios": expansion_ratios_S0,
    "head_cfg": {
         "type": "IRDropHead",
         "mid_channels": 256,
         "out_channels": 1,   # IR drop prediction의 경우 회귀 출력 (예: 1)
    },
}

# 필요에 따라 S1, S2, L 버전의 IR drop prediction config도 추가 가능
efficientformerv2_s1_irdrop_config = {
    # "in_ch":25, # build model에서 접근
    "layers": EfficientFormer_depth['S1'],
    "embed_dims": EfficientFormer_width['S1'],
    "downsamples": [True, True, True, True],
    "fork_feat": True,  # multi-scale feature 출력
    "drop_path_rate": 0.0,
    "vit_num": 2,
    "e_ratios": expansion_ratios_S1,
    "head_cfg": {
         "type": "IRDropHead",
         "mid_channels": 256,
         "out_channels": 1,   # IR drop prediction의 경우 회귀 출력 (예: 1)
    },
}

efficientformerv2_s2_irdrop_config = {
    # "in_ch":25, # build model에서 접근
    "layers": EfficientFormer_depth['S2'],
    "embed_dims": EfficientFormer_width['S2'],
    "downsamples": [True, True, True, True],
    "fork_feat": True,  # multi-scale feature 출력
    "drop_path_rate": 0.0,
    "vit_num": 4,
    "e_ratios": expansion_ratios_S2,
    "head_cfg": {
         "type": "IRDropHead",
         "mid_channels": 256,
         "out_channels": 1,   # IR drop prediction의 경우 회귀 출력 (예: 1)
    },
}
