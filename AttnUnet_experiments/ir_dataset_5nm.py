import os
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Subset, Dataset
from sklearn.model_selection import train_test_split
import torch
import glob
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import re

from config import get_config  # config.json을 통해 통계정보를 불러옴

class IRDropDataset5nm(Dataset):
    def __init__(self, root_path, selected_folders,
                 img_size=256, post_fix_path='', train=True,
                 in_ch=2, use_raw=False, dbu_per_px='',
                 target_norm_type=None, input_norm_type=None):
        self.root_path = root_path
        self.selected_folders = selected_folders
        self.target_size = img_size
        self.post_fix = post_fix_path
        self.cached_data = None
        self.train = train
        self.in_ch = in_ch
        self.use_raw = use_raw
        self.dbu_per_px = dbu_per_px
        
        # full config 객체 불러오기 (ir_drop, current, pad, resistances 등 모두 포함)
        self.conf = get_config(dbu_per_px)
        
        # norm type: 타겟은 ir_drop, 입력은 current, pad, resistance 각각 적용 가능하도록 처리
        self.target_norm_type = target_norm_type if target_norm_type else 'sample_min_max'
        self.input_norm_type = input_norm_type if input_norm_type else 'sample_min_max'
        
        print(f'input norm : {self.input_norm_type}, target norm : {self.target_norm_type}')
        # 타겟 정규화: ir_drop 채널에 적용
        self.target_norm_fn = self.get_norm_fn(self.target_norm_type, 'ir_drop')
        # 입력 정규화: 2,3채널의 경우 resistance는 total 통계, 25채널의 경우 각 layer별 적용 (이때는 _parse_resistance_layer를 사용)
        self.input_norm_fns = {
            'current': self.get_norm_fn(self.input_norm_type, 'current'),
            'pad': self.get_norm_fn(self.input_norm_type, 'pad'),
            # 2,3채널 모드에서는 total을 사용하고, 25채널 모드에서는 _load_data_from_disk_25ch에서 개별적으로 처리함
            'resistance': self.get_norm_fn(self.input_norm_type, 'resistance')
        }
        
        self.transform = A.ReplayCompose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=(90, 90), p=0.3),
            A.Rotate(limit=(180, 180), p=0.3),
            A.Rotate(limit=(270, 270), p=0.3),
            A.NoOp(p=0.3),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'}, is_check_shapes=False)

        self.val_transform = A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            ToTensorV2()
        ], is_check_shapes=False)

        self.data_files = []
        self._find_files()

        if self.in_ch == 25:
            self.load_data_fn = self._load_data_from_disk_25ch
        elif self.in_ch == 2:
            self.load_data_fn = self._load_data_from_disk_2ch
        elif self.in_ch == 3:
            self.load_data_fn = self._load_data_from_disk_3ch
        else:
            raise ValueError(f"Not support {self.in_ch} channels.")

    def get_norm_fn(self, norm_type, channel, layer_key=None):
        """
        norm_type: 'sample_max', 'global_max', 'z_score', 'sample_min_max', 'exp', 'log_exp', 'none'
        channel: 'ir_drop', 'current', 'pad', 'resistance'
        layer_key: resistance의 경우 25채널 모드에서 개별 layer 키 (예: "m8", "via56" 등)
        """
        if norm_type in ['sample_max', 'norm','max']:
            return lambda x: x / x.max() if x.max() > 0 else x
        elif norm_type in ['global_max', 'g_max']:
            if channel == 'ir_drop':
                max_val = self.conf.ir_drop.max if self.conf.ir_drop is not None else None
            elif channel == 'current':
                max_val = self.conf.current.max if self.conf.current is not None else None
            elif channel == 'pad':
                max_val = self.conf.pad.max if self.conf.pad is not None else None
            elif channel == 'resistance':
                # 25채널 모드에서 개별 layer 정규화
                if self.in_ch == 25 and layer_key is not None:
                    layer_cfg = self.conf.resistances.layers.get(layer_key)
                    max_val = layer_cfg.max if layer_cfg is not None else None
                else:
                    max_val = self.conf.resistances.total.max if (self.conf.resistances and self.conf.resistances.total is not None) else None
            else:
                max_val = None
            return lambda x: x / max_val if max_val is not None and max_val != 0 else x
        elif norm_type in ['z_score']:
            if channel == 'ir_drop':
                mean_val = self.conf.ir_drop.mean if self.conf.ir_drop is not None else None
                std_val = self.conf.ir_drop.std if self.conf.ir_drop is not None else None
            elif channel == 'current':
                mean_val = self.conf.current.mean if self.conf.current is not None else None
                std_val = self.conf.current.std if self.conf.current is not None else None
            elif channel == 'pad':
                mean_val = self.conf.pad.mean if self.conf.pad is not None else None
                std_val = self.conf.pad.std if self.conf.pad is not None else None
            elif channel == 'resistance':
                if self.in_ch == 25 and layer_key is not None:
                    layer_cfg = self.conf.resistances.layers.get(layer_key)
                    mean_val = layer_cfg.mean if layer_cfg is not None else None
                    std_val = layer_cfg.std if layer_cfg is not None else None
                else:
                    mean_val = self.conf.resistances.total.mean if (self.conf.resistances and self.conf.resistances.total is not None) else None
                    std_val = self.conf.resistances.total.std if (self.conf.resistances and self.conf.resistances.total is not None) else None
            else:
                mean_val, std_val = None, None
            return lambda x: (x - mean_val) / std_val if std_val is not None and std_val != 0 else x
        elif norm_type in ['sample_min_max', 'min_max']:
            return lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
        elif norm_type in ['exp']:
            return lambda x: np.exp((x - x.min()) / (x.max() - x.min() + 1e-8) - 1)
        elif norm_type in ['log_exp']:
            return lambda x: np.exp(((np.log(x + 1e-5) - np.log(x + 1e-5).min()) /
                                     (np.log(x + 1e-5).max() - np.log(x + 1e-5).min() + 1e-8)) - 1)
        elif norm_type in ['none']:
            return lambda x: x
        else:
            print(f"Warning: Unknown normalization type '{norm_type}'. Using default 'sample_max'.")
            return lambda x: x / x.max() if x.max() > 0 else x

    def _parse_resistance_layer(self, file_path):
        """
        resistance 파일명에서 layer key를 추출 (예: "m8", "via56" 등)
        """
        base = os.path.basename(file_path)
        pattern_m = re.compile(r'.*_(m\d+)_resistance\.npy')
        pattern_via = re.compile(r'.*_m(\d+)_to_m(\d+)_via_resistance\.npy')
        m_match = pattern_m.match(base)
        if m_match:
            return m_match.group(1)
        else:
            via_match = pattern_via.match(base)
            if via_match:
                return f"via{via_match.group(1)}{via_match.group(2)}"
        return None

    def set_target_norm_fn(self, norm_type):
        self.target_norm_type = norm_type
        self.target_norm_fn = self.get_norm_fn(norm_type, 'ir_drop')

    def set_input_norm_fn(self, norm_type):
        self.input_norm_type = norm_type
        # 2,3채널 모드에서는 resistance는 total을 사용
        self.input_norm_fns = {
            'current': self.get_norm_fn(norm_type, 'current'),
            'pad': self.get_norm_fn(norm_type, 'pad'),
            'resistance': self.get_norm_fn(norm_type, 'resistance')
        }

    def _find_files(self):
        self.data_files = []
        for folder in self.selected_folders:
            folder_path = os.path.join(self.root_path, folder, self.post_fix)
            file_groups = {
                'current': glob.glob(os.path.join(folder_path, '*_current*.npy')),
                'ir_drop': glob.glob(os.path.join(folder_path, '*_ir_drop*.npy')),
                'resistance': glob.glob(os.path.join(folder_path, 'layer_data', '*resistance*.npy'))
            }
            if self.in_ch in [3, 25]:
                file_groups['pad_distance'] = glob.glob(os.path.join(folder_path, '*pad*.npy'))
            for key in file_groups:
                file_groups[key].sort()
            resistance_files_dict = {}
            for file_path in file_groups['resistance']:
                index = os.path.basename(file_path).split('_')[0]
                resistance_files_dict.setdefault(index, []).append(file_path)
            if len(file_groups['current']) != len(file_groups['ir_drop']):
                raise ValueError(f"Mismatch in the number of current and ir_drop files in folder {folder}!")
            self.__load_data_files(file_groups, resistance_files_dict)

    def __load_data_files(self, file_groups, resistance_files_dict):
        pad_distance_files = file_groups.get('pad_distance', [])
        if self.in_ch == 2:
            zipped_files = zip(file_groups['current'], file_groups['ir_drop'])
        elif self.in_ch in [3, 25]:
            if len(file_groups['current']) != len(pad_distance_files):
                raise ValueError("Mismatch in the number of current and pad_distance files!")
            zipped_files = zip(file_groups['current'], file_groups['ir_drop'], pad_distance_files)
        else:
            raise ValueError(f"Not support {self.in_ch} channels.")
        for files in zipped_files:
            current = files[0]
            ir_drop = files[1]
            pad_distance = files[2] if self.in_ch in [3, 25] else None
            index = os.path.basename(current).split('_')[0]
            resistances = resistance_files_dict.get(index, [])
            self.data_files.append({
                'current': current,
                'ir_drop': ir_drop,
                'resistances': resistances,
                'pad_distance': pad_distance
            })
        
    def _load_data_from_disk_2ch(self, idx):
        file_group = self.data_files[idx]
        current = self.input_norm_fns['current'](np.load(file_group['current']))
        ir_drop = np.load(file_group['ir_drop'])
        if ir_drop.ndim == 2:
            ir_drop = np.expand_dims(ir_drop, axis=-1)
        resistance_stack = []
        for res_file in file_group['resistances']:
            resistance_stack.append(np.load(res_file))
        try:
            resistance_stack = np.stack(resistance_stack, axis=-1).sum(-1)
        except Exception as e:
            print("Resistance shape error:", [np.load(r).shape for r in file_group['resistances']])
            raise ValueError('load_data_from_disk error') from e
        # 2,3채널 모드에서는 resistance는 total 통계를 사용
        resistance_total = self.input_norm_fns['resistance'](resistance_stack)
        input_data = np.stack([current, resistance_total], axis=-1)
        return input_data, ir_drop

    def _load_data_from_disk_3ch(self, idx):
        input_data_2ch, ir_drop = self._load_data_from_disk_2ch(idx)
        file_group = self.data_files[idx]
        pad_distance = self.input_norm_fns['pad'](np.load(file_group['pad_distance']))
        current = input_data_2ch[..., 0]
        resistance_total = input_data_2ch[..., 1]
        input_data_3ch = self._combine_3ch_data(current, pad_distance, resistance_total)
        return input_data_3ch, ir_drop

    def _load_data_from_disk_25ch(self, idx):
        file_group = self.data_files[idx]
        current = self.input_norm_fns['current'](np.load(file_group['current']))
        pad_distance = self.input_norm_fns['pad'](np.load(file_group['pad_distance']))
        resistance_maps = []
        # 25채널 모드에서는 각 resistance 파일마다 개별 layer 정규화 적용
        for res_file in file_group['resistances']:
            layer_key = self._parse_resistance_layer(res_file)
            norm_fn = self.get_norm_fn(self.input_norm_type, 'resistance', layer_key)
            resistance_map = norm_fn(np.load(res_file))
            resistance_maps.append(resistance_map)
        if len(resistance_maps) != 23:
            raise ValueError(f"Expected 23 resistance maps, but got {len(resistance_maps)} for index {os.path.basename(file_group['current'])}")
        shapes = [current.shape, pad_distance.shape] + [rm.shape for rm in resistance_maps]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError(f"Shape mismatch among channels for index {os.path.basename(file_group['current'])}: {shapes}")
        input_data = np.stack([current, pad_distance] + resistance_maps, axis=-1)
        ir_drop = np.load(file_group['ir_drop'])
        if ir_drop.ndim == 2:
            ir_drop = np.expand_dims(ir_drop, axis=-1)
        return input_data, ir_drop

    def __getitem__(self, idx):
        input_data, ir_drop = self.load_data_fn(idx)
        ir_drop = self.target_norm_fn(ir_drop) if not self.use_raw else ir_drop
        if self.train:
            transformed = self.transform(image=input_data, mask=ir_drop)
        else:
            transformed = self.val_transform(image=input_data, mask=ir_drop)
        input_tensor = transformed['image']
        target_tensor = transformed['mask'] if not self.use_raw else torch.as_tensor(ir_drop)
        target_tensor = target_tensor.permute(2, 0, 1)
        return input_tensor.float(), target_tensor.float()

    def __len__(self):
        return len(self.data_files)

    def _combine_3ch_data(self, current, pad_distance, resistance_total):
        return np.stack([current, pad_distance, resistance_total], axis=-1)

    def _norm(self, x):
        return x / x.max() if x.max() > 0 else x
      
    def _min_max_norm(self, x):
        return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x

    def _exp_norm(self, x):
        normalized = (x - x.min()) / (x.max() - x.min() + 1e-8) if x.max() > x.min() else x
        return np.exp(normalized - 1)

    def _log_exp_norm(self, x):
        log_data = np.log(x + 1e-5)
        log_min, log_max = log_data.min(), log_data.max()
        normalized = (log_data - log_min) / (log_max - log_min + 1e-8)
        return np.exp(normalized - 1)

    def inverse(self, x, norm_type):
        if norm_type in ['global_max', 'g_max']:
            return x * (self.conf.ir_drop.max if norm_type=='global_max' else 1)
        elif norm_type in ['z_score']:
            return x * self.conf.ir_drop.std + self.conf.ir_drop.mean if self.conf.ir_drop and self.conf.ir_drop.std is not None else x
        elif norm_type in ['min_max', 'sample_min_max']:
            return x  # 개별 sample min/max 역변환은 어려움
        else:
            print('Inverse normalization not implemented for norm type:', norm_type)
            return x

    def getitem_ir_ori(self, idx):
        input_data, ir_drop = self.load_data_fn(idx)
        if self.train:
            transformed = self.transform(image=input_data, mask=ir_drop)
        else:
            transformed = self.val_transform(image=input_data, mask=ir_drop)
        input_tensor = transformed['image']
        target_tensor = transformed['mask']
        target_tensor = target_tensor.permute(2, 0, 1)
        return input_tensor.float(), target_tensor.float()

if __name__ =='__main__':
    def test_new_data_error_2(per='1um'):
        root_path = "/data"
        selected_folders = [
            f'pdn_4th_4types/{per}_numpy',
            f'pdn_3rd_4types/{per}_numpy', 
            f'pdn_data_6th/{per}_numpy'
        ]
        post_fix = ""
        dataset = IRDropDataset5nm(
            root_path=root_path,
            selected_folders=selected_folders,
            post_fix_path=post_fix,
            train=False,
            use_raw=True,
            in_ch=25,
            dbu_per_px=per,
            target_norm_type='global_max',   # 초기 타겟 정규화 (ir_drop)
            input_norm_type='sample_min_max'   # 초기 입력 정규화 (current, pad, resistance)
        )
        error_count = 0
        error_indices = []
        for i in range(len(dataset)):
            print(f'{i} processing')
            try:
                a = dataset.__getitem__(i)
                print(a[0].shape)
            except Exception as e:
                error_count += 1
                error_indices.append(i)
                print(f"Error at index {i}: {e}")
                continue
        print('error_count : ', error_count)
        print('error_indices : ', error_indices)

    # 테스트: 다양한 공정 노드에 대해 dataset이 정상 동작하는지 확인
    test_new_data_error_2('1um')
    test_new_data_error_2('500nm')
    test_new_data_error_2('200nm')
    test_new_data_error_2('100nm')
