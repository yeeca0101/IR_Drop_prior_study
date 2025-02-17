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


'''
    v2 dataset :
    IR Drop
        "mean": 0.001601,
        "std": 0.002327,
        "min": 0.0,
        "max": 0.025038
'''
IR_DROP_MEAN = 0.001601 
IR_DROP_STD = 0.002327

class IRDropDataset5nm(Dataset):
    def __init__(self, root_path, selected_folders,
                 img_size=256, post_fix_path='', train=True,
                 in_ch=2, use_raw=False):
        """
        기본 데이터셋:
          - in_ch==2: current + (전체 resistance를 합산한 값)
          - in_ch==3: current + pad_distance + (전체 resistance 합산)
          - in_ch==25: current (1) + pad_distance (1) + resistance map 23개 (총 25채널)
        """
        self.root_path = root_path
        self.selected_folders = selected_folders
        self.target_size = img_size
        self.post_fix = post_fix_path
        self.cached_data = None
        self.train = train
        self.in_ch = in_ch
        self.use_raw = use_raw

        self.transform = A.ReplayCompose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.Rotate(limit=(90, 90), p=1),
                A.Rotate(limit=(180, 180), p=1),
                A.Rotate(limit=(270, 270), p=1),
                A.NoOp(p=1)
            ], p=1),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'}, is_check_shapes=False)

        self.val_transform = A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
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
        """
        각 폴더에서 current, ir_drop, resistance (및 pad_distance)를 탐색합니다.
        pad_distance는 in_ch가 3 또는 25인 경우에 로드합니다.
        """
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
        
        # print(f'__load_data_files : {len(self.data_files)} found.')

    def _load_data_from_disk_2ch(self, idx):
        file_group = self.data_files[idx]
        current = self._norm(np.load(file_group['current']))
        ir_drop = np.load(file_group['ir_drop'])
        ir_drop = self._zscore_norm(ir_drop) if not self.use_raw else ir_drop
        if ir_drop.ndim == 2:
            ir_drop = np.expand_dims(ir_drop, axis=-1)

        resistance_stack = []
        debug_list = []
        for res_file in file_group['resistances']:
            resistance_stack.append(np.load(res_file))
            debug_list.append(resistance_stack[-1].shape)

        try:
            # resistance들을 합산하여 단일 채널로 만듦
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
        """
        in_ch==25인 경우:
          - current (1채널)
          - pad_distance (1채널)
          - resistance map 23개 (각각 1채널)
          → 총 25채널 데이터를 만듭니다.
        """
        file_group = self.data_files[idx]
        # current와 pad_distance 로드 및 정규화
        current = self._norm(np.load(file_group['current']))
        pad_distance = self._norm(np.load(file_group['pad_distance']))

        # resistance map들을 개별 채널로 로드
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

        # 채널 순서: current, pad_distance, 그리고 23 resistance map → 총 25채널
        input_data = np.stack([current, pad_distance] + resistance_maps, axis=-1)
        
        ir_drop = np.load(file_group['ir_drop'])
        ir_drop = self._zscore_norm(ir_drop) if not self.use_raw else ir_drop
        if ir_drop.ndim == 2:
            ir_drop = np.expand_dims(ir_drop, axis=-1)
        return input_data, ir_drop

    def __getitem__(self, idx):
        input_data, ir_drop = self.load_data_fn(idx)
        if self.train:
            transformed = self.transform(image=input_data, mask=ir_drop)
        else:
            transformed = self.val_transform(image=input_data, mask=ir_drop)
        input_tensor = transformed['image']
        target_tensor = transformed['mask']
        target_tensor = target_tensor.permute(2, 0, 1)
        return input_tensor.float(), target_tensor.float()

    def getitem_ori(self, idx):
        return self.load_data_fn(idx)
    
    def __len__(self):
        return len(self.data_files)

    def _norm(self, x):
        return x / x.max() if x.max() > 0 else x

    def _min_max_norm(self, x):
        return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x

    def _zscore_norm(self,x): # for ir drop
        return (x - IR_DROP_MEAN) / IR_DROP_STD

    def _combine_3ch_data(self, current, pad_distance, resistance_total):
        return np.stack([current, pad_distance, resistance_total], axis=-1)


class IRDropInferenceAutoencoderDataset5nm(IRDropDataset5nm):
    """
    [Autoencoder Inference 전용 데이터셋]
    
    - LR 데이터는 1um_numpy 폴더의 데이터를 사용하여 256×256으로 resize한 후 LR 모델의 입력과 타깃으로 사용합니다.
    - HR 타깃은 210nm_numpy 폴더의 ir_drop 파일을 원본 해상도로 로드합니다.
    
    각 샘플은 아래 정보를 dictionary로 반환합니다.
      - 'lr_input'     : 1um 데이터 → Tensor, (C,256,256)
      - 'lr_target'    : 1um ir_drop (256×256) → Tensor, (1,256,256)
      - 'lr_target_ori': 변환 전 원본 1um ir_drop (원본 해상도)
      - 'lr_ori_shape' : 원본 해상도 (H, W)
      - 'hr_target'    : 210nm ir_drop (원본 해상도) → Tensor, (1, H, W)
      - 'hr_ori_shape' : 210nm ir_drop의 원본 해상도 (H, W)
    """
    
    def __init__(self, *args, **kwargs):
        # 인퍼런스이므로 train=False로 강제
        # kwargs['train'] = False  
        super().__init__(*args, **kwargs)
        self.val_transform = A.Compose([
            A.Resize(self.target_size, self.target_size, interpolation=cv2.INTER_NEAREST),
            ToTensorV2()
        ], is_check_shapes=False)
        
        # LR과 HR 파일을 매칭하는 autoencoder 전용 파일 탐색 함수
        self._find_files_autoencoder()
        self.load_data_fn = self._load_data_inference

    def _find_files_autoencoder(self):
        """
        1um_numpy (LR 데이터)와 210nm_numpy (HR 타깃) 폴더에서 인덱스 기준으로 파일을 매칭합니다.
        LR 관련 파일은 기존 키('current', 'ir_drop', 'resistances', 'pad_distance')로 저장하고,
        HR 타깃은 'hr_target' 키에 저장합니다.
        """
        self.data_files = []
        
        input_folder = os.path.join(self.root_path, '1um_numpy', self.post_fix)
        target_folder = os.path.join(self.root_path, '210nm_numpy', self.post_fix)
        
        file_groups_lr = {}
        file_groups_lr['current'] = glob.glob(os.path.join(input_folder, '*_current*.npy'))
        file_groups_lr['ir_drop'] = glob.glob(os.path.join(input_folder, '*_ir_drop*.npy'))
        file_groups_lr['resistances'] = glob.glob(os.path.join(input_folder, 'layer_data', '*resistance*.npy'))
        if self.in_ch == 3:
            file_groups_lr['pad_distance'] = glob.glob(os.path.join(input_folder, '*pad*.npy'))
        for key in file_groups_lr:
            file_groups_lr[key].sort()
        
        lr_dict = {}
        for file in file_groups_lr.get('current', []):
            index = os.path.basename(file).split('_')[0]
            lr_dict.setdefault(index, {})['current'] = file
        for file in file_groups_lr.get('ir_drop', []):
            index = os.path.basename(file).split('_')[0]
            lr_dict.setdefault(index, {})['ir_drop'] = file
        resistance_dict = {}
        for file in file_groups_lr.get('resistances', []):
            index = os.path.basename(file).split('_')[0]
            resistance_dict.setdefault(index, []).append(file)
        for index, files in resistance_dict.items():
            lr_dict.setdefault(index, {})['resistances'] = files
        if self.in_ch == 3:
            for file in file_groups_lr.get('pad_distance', []):
                index = os.path.basename(file).split('_')[0]
                lr_dict.setdefault(index, {})['pad_distance'] = file
        
        hr_files = glob.glob(os.path.join(target_folder, '*_ir_drop*.npy'))
        hr_files.sort()
        hr_dict = {}
        for file in hr_files:
            index = os.path.basename(file).split('_')[0]
            hr_dict[index] = file
        
        common_indices = sorted(list(set(lr_dict.keys()) & set(hr_dict.keys())))
        if not common_indices:
            raise ValueError("1um과 210nm 폴더에서 매칭되는 인덱스가 없습니다!")
        
        for idx in common_indices:
            entry = lr_dict[idx]
            entry['hr_target'] = hr_dict[idx]
            self.data_files.append(entry)
    
    def _load_data_inference(self, idx):
        if self.in_ch == 2:
            return self._load_data_from_disk_2ch(idx)
        elif self.in_ch == 3:
            return self._load_data_from_disk_3ch(idx)
        else:
            raise ValueError(f"{self.in_ch} channels는 지원하지 않습니다.")
    
    def __getitem__(self, idx):
        # 1. LR 데이터 로드 (원본 해상도)
        lr_input_data, lr_ir_drop = self.load_data_fn(idx)
        lr_ori_shape = lr_ir_drop.shape[:2]
        lr_target_ori = torch.from_numpy(lr_ir_drop).permute(2, 0, 1).float()

        # 2. LR transform 적용 (256×256 resize + ToTensor)
        transformed = self.val_transform(image=lr_input_data, mask=lr_ir_drop)
        lr_input = transformed['image'].float()
        lr_target = transformed['mask']
        if lr_target.ndim == 3:
            lr_target = lr_target.permute(2, 0, 1).float()

        # 3. HR 타깃 로드 (원본 해상도)
        hr_file = self.data_files[idx]['hr_target']
        hr_target = self._min_max_norm(np.load(hr_file))
        hr_ori_shape = hr_target.shape[:2]
        if hr_target.ndim == 2:
            hr_target = np.expand_dims(hr_target, axis=-1)
        hr_target_tensor = torch.from_numpy(hr_target).permute(2, 0, 1).float()

        return {
            'lr_input': lr_input,
            'lr_target': lr_target,
            'lr_target_ori': lr_target_ori,
            'lr_ori_shape': lr_ori_shape,
            'hr_target': hr_target_tensor,
            'hr_ori_shape': hr_ori_shape
        }


if __name__ =='__main__':
    def test_dataset(i):
        root_path = "/data/gen_pdn/pdn_data_3rd"
        selected_folders = ['1um_numpy']
        post_fix = ""

        dataset = IRDropDataset5nm(root_path=root_path,
                                    selected_folders=selected_folders,
                                    post_fix_path=post_fix,
                                    train=False,
                                    in_ch=3)
        a = dataset.__getitem__(i)
        print(a[0].shape)
        print(len(dataset))

    def test_new_data_error():
        root_path = "/data/gen_pdn/pdn_data_3rd"
        selected_folders = ['210nm_numpy']
        post_fix = ""

        dataset = IRDropDataset5nm(root_path=root_path,
                                    selected_folders=selected_folders,
                                    post_fix_path=post_fix,
                                    train=False,
                                    in_ch=3)
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

    def test_autoencoder_mode():
        root_path = "/data/gen_pdn/pdn_data_3rd"
        post_fix = ""
        # Autoencoder 모드는 IRDropInferenceAutoencoderDataset5nm을 사용합니다.
        dataset = IRDropInferenceAutoencoderDataset5nm(root_path=root_path,
                                                        selected_folders=[],  # 내부에서 고정됨
                                                        post_fix_path=post_fix,
                                                        in_ch=1)
        print("Total samples:", len(dataset))
        input_tensor, target_tensor = dataset.__getitem__(0)
        print("Input shape (1um ir drop):", input_tensor.shape)
        print("Target shape (210nm ir drop):", target_tensor.shape)

    def test_infer_dataset_hr():
        root_path = "/data/gen_pdn"
        post_fix = ""
        dataset = IRDropInferenceAutoencoderDataset5nm(root_path=root_path,
                                                        selected_folders=['1um_numpy','210nm_numpy'],
                                                        post_fix_path=post_fix,
                                                        in_ch=2)
        print("Total samples:", len(dataset))
        data = dataset.__getitem__(0)
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(f'{key} : ', value.shape)
            else:
                print(f'{key} : ', value)

    def test_new_data_error_2():
        root_path = "/data/pdn_3rd_4types"
        selected_folders = ['100nm_numpy']
        post_fix = ""

        dataset = IRDropDataset5nm(root_path=root_path,
                                    selected_folders=selected_folders,
                                    post_fix_path=post_fix,
                                    train=False,
                                    in_ch=25)
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
    test_new_data_error_2()
