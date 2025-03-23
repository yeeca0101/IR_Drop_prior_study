import os
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import glob
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import get_config

class IRDropDataset5nm(Dataset):
    def __init__(self, root_path, selected_folders,
                 img_size=256, post_fix_path='', train=True,
                 in_ch=2, use_raw=False,dbu_per_px='',target_norm_type=None, input_norm_type=None):
        self.root_path = root_path
        self.selected_folders = selected_folders
        self.target_size = img_size
        self.post_fix = post_fix_path
        self.cached_data = None
        self.train = train
        self.in_ch = in_ch
        self.use_raw = use_raw
        self.dbu_per_px = dbu_per_px
        self.conf = get_config(dbu_per_px).ir_drop
        print('ir_conf : ',self.conf)
        self.target_norm_type = target_norm_type
        self.input_norm_type = input_norm_type

        self.target_norm_fn = self.set_norm_fn(self.target_norm_type)
        self.input_norm_fn = self.set_norm_fn(self.input_norm_type)

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
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA) ,
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

    def _find_files(self):
        self.data_files = []

        for folder in self.selected_folders:
            folder_path = os.path.join(self.root_path, folder, self.post_fix)
            
            # 파일 그룹 초기화
            file_groups = {
                'current': glob.glob(os.path.join(folder_path, '*_current*.npy')),
                'ir_drop': glob.glob(os.path.join(folder_path, '*_ir_drop*.npy')),
                'resistance': glob.glob(os.path.join(folder_path, 'layer_data', '*resistance*.npy'))
            }
            if self.in_ch in [3, 25]:
                file_groups['pad_distance'] = glob.glob(os.path.join(folder_path, '*pad*.npy'))

            # 정렬
            for key in file_groups:
                file_groups[key].sort()

            # resistance 파일은 인덱스별로 그룹화
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
        current = self._norm(np.load(file_group['current']))
        ir_drop = np.load(file_group['ir_drop'])
        if ir_drop.ndim == 2:
            ir_drop = np.expand_dims(ir_drop, axis=-1)

        resistance_stack = []
        debug_list = []
        for res_file in file_group['resistances']:
            resistance_stack.append(np.load(res_file))
            debug_list.append(resistance_stack[-1].shape)

        try:
            resistance_stack = np.stack(resistance_stack, axis=-1).sum(-1)
        except:
            print(debug_list)
            raise ValueError('load_data_from_disk error')

        resistance_total = self._norm(resistance_stack)
        input_data = np.stack([current, resistance_total], axis=-1)
        return input_data, ir_drop

    def _load_data_from_disk_3ch(self, idx):
        input_data_2ch, ir_drop = self._load_data_from_disk_2ch(idx)
        file_group = self.data_files[idx]
        pad_distance = self._norm(np.load(file_group['pad_distance']))
        
        current = input_data_2ch[..., 0]      # current 채널
        resistance_total = input_data_2ch[..., 1]  # resistance 합산 채널

        input_data_3ch = self._combine_3ch_data(current, pad_distance, resistance_total)
        return input_data_3ch, ir_drop

    def _load_data_from_disk_25ch(self, idx):
        file_group = self.data_files[idx]
        current = self._norm(np.load(file_group['current']))
        pad_distance = self._norm(np.load(file_group['pad_distance']))

        resistance_maps = []
        for res_file in file_group['resistances']:
            resistance_map = np.load(res_file)
            resistance_maps.append(self._norm(resistance_map))
        
        if len(resistance_maps) != 23:
            raise ValueError(f"Expected 23 resistance maps, but got {len(resistance_maps)} for index {os.path.basename(file_group['current'])}")

        # 모든 채널의 shape가 동일한지 확인
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
        ir_drop = self._global_max_norm(ir_drop) if not self.use_raw else ir_drop
        
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

    def _target_norm(self,x):
        return self.target_norm_fn(x)
    
    def _norm(self, x):
        return x / x.max() if x.max() > 0 else x
      
    def _global_max_norm(self,x):
        return x / self.conf.max

    def _z_score(self,x):
        return (x-self.conf.mean)/self.conf.std

    def _min_max_norm(self, x):
        return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x

    def _combine_3ch_data(self, current, pad_distance, resistance_total):
        return np.stack([current, pad_distance, resistance_total], axis=-1)

    def set_target_norm_fn(self,fn_name):
        self.target_norm_type = fn_name
        self.set_norm_fn(self.target_norm_type)
    
    def getitem_ir_ori(self, idx):
        input_data, ir_drop = self.load_data_fn(idx)
        # ir_drop not normalized
        if self.train:
            transformed = self.transform(image=input_data, mask=ir_drop)
        else:
            transformed = self.val_transform(image=input_data, mask=ir_drop)
        input_tensor = transformed['image']
        target_tensor = transformed['mask'] 
        target_tensor = target_tensor.permute(2, 0, 1)
        return input_tensor.float(), target_tensor.float()
    

    def set_norm_fn(self, norm_type):
        norm_functions = {
            'norm': self._norm,
            'global_max': self._global_max_norm,
            'z_score': self._z_score,
            'min_max': self._min_max_norm,
            'exp': self._exp_norm,
            'log_exp': self._log_exp_norm,
            'none': lambda x: x  # 정규화 없음
        }
        
        if norm_type in norm_functions:
            return norm_functions[norm_type]
        else:
            print(f"Warning: Unknown normalization type '{norm_type}'. Using default 'norm'.")
            return self._norm
    
    def inverse(self,x,fn_name):
        if fn_name in ['g_max', 'global_max']:
            return x * self.conf.max
        elif fn_name in ['min_max',]:
            return x
        else:
            print('not yet inverse')


    def _exp_norm(self, x):
        # 안전한 exp 정규화 (오버플로우 방지)
        normalized = (x - x.min()) / (x.max() - x.min() + 1e-8) if x.max() > x.min() else x
        return np.exp(normalized - 1)  # exp(-1) ~ exp(0) 범위로 매핑

    def _log_exp_norm(self, x):
        # 로그 변환 후 exp 정규화
        log_data = np.log(x + 1e-5)  # 작은 값 추가하여 로그 안전성 확보
        log_min, log_max = log_data.min(), log_data.max()
        normalized = (log_data - log_min) / (log_max - log_min + 1e-8)
        return np.exp(normalized - 1)

if __name__ =='__main__':
    def test_new_data_error_2(per= '1um'):
        root_path = "/data"
        selected_folders = [f'pdn_4th_4types/{per}_numpy',
                            f'pdn_3rd_4types/{per}_numpy', 
                            f'pdn_data_6th/{per}_numpy']
        post_fix = ""

        dataset = IRDropDataset5nm(root_path=root_path,
                                    selected_folders=selected_folders,
                                    post_fix_path=post_fix,
                                    train=False,
                                    use_raw=True,
                                    in_ch=25,
                                    dbu_per_px=per)
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

    # 원하는 테스트 함수를 호출하여 실행합니다.
    test_new_data_error_2('1um')
    test_new_data_error_2('500nm')
    test_new_data_error_2('200nm')
    test_new_data_error_2('100nm')
