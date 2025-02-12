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

class IRDropDataset5nm(Dataset):
    def __init__(self, root_path, selected_folders,
                 img_size=256, post_fix_path='', train=True,
                 in_ch=2, use_raw=False, train_auto_encoder=False):
        self.root_path = root_path
        self.selected_folders = selected_folders
        self.target_size = img_size
        self.post_fix = post_fix_path
        self.cached_data = None
        self.train = train
        self.in_ch = in_ch
        self.use_raw = use_raw
        self.train_auto_encoder = train_auto_encoder

        # train_auto_encoder 모드인 경우 1um 폴더의 ir drop map을 입력, 210nm 폴더의 ir drop map을 타깃으로 사용
        if self.train_auto_encoder:
            self.in_ch = 1
            # autoencoder 모드에서는 두 폴더(1um, 210nm)를 고정으로 사용합니다.
            self.selected_folders = ['1um_numpy', '210nm_numpy']

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
        ],additional_targets={'mask': 'mask'},is_check_shapes=False)

        self.val_transform = A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
            ToTensorV2()
        ],is_check_shapes=False)

        self.data_files = []

        # train_auto_encoder 모드에 따라 파일 탐색 방식을 분기
        self._find_files()

        # 데이터 로딩 함수 선택 (autoencoder 모드이면 _load_data_autoencoder 사용)
        if self.train_auto_encoder:
            self.load_data_fn = self._load_data_autoencoder
        elif self.in_ch == 2:
            self.load_data_fn = self._load_data_from_disk_2ch
        else:
            self.load_data_fn = self._load_data_from_disk_3ch

    def _find_files(self):
        if self.train_auto_encoder:
            self._find_files_autoencoder()
            return

        # 기존 로직: 각 폴더에서 current, ir_drop, resistance (및 pad_distance)를 탐색
        self.data_files = []

        for folder in self.selected_folders:
            folder_path = os.path.join(self.root_path, folder, self.post_fix)
            
            # 파일 그룹 초기화
            file_groups = {
                'current': glob.glob(os.path.join(folder_path, '*_current*.npy')),
                'ir_drop': glob.glob(os.path.join(folder_path, '*_ir_drop*.npy')),
                'resistance': glob.glob(os.path.join(folder_path, 'layer_data', '*resistance*.npy'))
            }

            if self.in_ch == 3:
                file_groups['pad_distance'] = glob.glob(os.path.join(folder_path, '*pad*.npy'))

            for key in file_groups:
                file_groups[key].sort()

            resistance_files_dict = {}
            for file_path in file_groups['resistance']:
                index = os.path.basename(file_path).split('_')[0]
                resistance_files_dict.setdefault(index, []).append(file_path)

            if len(file_groups['current']) != len(file_groups['ir_drop']):
                raise ValueError(f"Mismatch in the number of current and ir_drop files in folder {folder}!")

            # 데이터 로드
            self.__load_data_files(file_groups, resistance_files_dict)

    def _find_files_autoencoder(self):
        """
        train_auto_encoder 모드일 때, 1um 폴더의 ir_drop 파일과 210nm 폴더의 ir_drop 파일을 동일 인덱스로 매칭합니다.
        """
        self.data_files = []
        input_folder = os.path.join(self.root_path, '1um_numpy', self.post_fix)
        target_folder = os.path.join(self.root_path, '210nm_numpy', self.post_fix)
        
        # 각 폴더에서 ir_drop 파일만 탐색
        input_ir_files = glob.glob(os.path.join(input_folder, '*_ir_drop*.npy'))
        target_ir_files = glob.glob(os.path.join(target_folder, '*_ir_drop*.npy'))
        input_ir_files.sort()
        target_ir_files.sort()
        
        # 파일명 앞부분(인덱스) 기준으로 매핑
        input_dict = {}
        for file in input_ir_files:
            index = os.path.basename(file).split('_')[0]
            input_dict[index] = file

        target_dict = {}
        for file in target_ir_files:
            index = os.path.basename(file).split('_')[0]
            target_dict[index] = file

        common_indices = sorted(list(set(input_dict.keys()) & set(target_dict.keys())))
        if not common_indices:
            raise ValueError("No matching indices found between 1um and 210nm folders for autoencoder training!")
        for idx in common_indices:
            self.data_files.append({
                'input_ir': input_dict[idx],
                'target_ir': target_dict[idx]
            })

    def __load_data_files(self, file_groups, resistance_files_dict):
        pad_distance_files = file_groups.get('pad_distance', [])

        if self.in_ch == 2:
            zipped_files = zip(file_groups['current'], file_groups['ir_drop'])
        elif self.in_ch == 3:
            if len(file_groups['current']) != len(pad_distance_files):
                raise ValueError("Mismatch in the number of current and pad_distance files!")
            zipped_files = zip(file_groups['current'], file_groups['ir_drop'], pad_distance_files)
        else:
            raise ValueError(f"Not support {self.in_ch} channels.")

        for files in zipped_files:
            current, ir_drop = files[:2]
            pad_distance = files[2] if self.in_ch == 3 else None

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
        ir_drop = self._min_max_norm(ir_drop) if not self.use_raw else ir_drop
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
        
        current = input_data_2ch[..., 0]  # 2채널에서 current 추출
        resistance_total = input_data_2ch[..., 1]  # 2채널에서 resistance_total 추출

        input_data_3ch = self._combine_3ch_data(current, pad_distance, resistance_total)
        return input_data_3ch, ir_drop

    def _load_data_autoencoder(self, idx):
        """
        train_auto_encoder 모드일 때, 1um ir_drop 파일은 입력, 210nm ir_drop 파일은 타깃으로 불러옵니다.
        """
        file_group = self.data_files[idx]
        input_ir = np.load(file_group['input_ir'])
        target_ir = np.load(file_group['target_ir'])
        if not self.use_raw:
            input_ir = self._min_max_norm(input_ir)
            target_ir = self._min_max_norm(target_ir)

        if input_ir.ndim == 2:
            input_ir = np.expand_dims(input_ir, axis=-1)
        if target_ir.ndim == 2:
            target_ir = np.expand_dims(target_ir, axis=-1)
        return input_ir, target_ir

    def __getitem__(self, idx):
        input_data, ir_drop = self.load_data_fn(idx)
        if self.train_auto_encoder:
            input_tensor, target_tensor = torch.from_numpy(input_data).permute(2,0,1).contiguous(),torch.from_numpy(ir_drop).permute(2,0,1).contiguous() 
            return input_tensor.float(), target_tensor.float()
        else:
            if self.train:
                transformed = self.transform(image=input_data, mask=ir_drop)
            else:
                transformed = self.val_transform(image=input_data, mask=ir_drop)
            input_tensor = transformed['image']
            target_tensor = transformed['mask']
            target_tensor = target_tensor.permute(2,0,1)
            return input_tensor.float(), target_tensor.float()

    def getitem_ori(self,idx):
        return self.load_data_fn(idx)
    
    def __len__(self):
        return len(self.data_files)

    def _norm(self, x):
        return x / x.max() if x.max() > 0 else x

    def _min_max_norm(self, x):
        return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x

    def _combine_3ch_data(self, current, pad_distance, resistance_total):
        return np.stack([current, pad_distance, resistance_total], axis=-1)




class IRDropInferenceAutoencoderDataset5nm(IRDropDataset5nm):
    """
    [Autoencoder Inference 전용 데이터셋]
    
    - LR 데이터는 1um_numpy 폴더의 데이터를 사용합니다.
      → LR 데이터는 기존에 사용하던 모든 파일 (current, ir_drop, resistance, (pad_distance))를 로드하고,
         이후 256×256 해상도로 resize하여 LR 모델의 입력과 타깃으로 사용합니다.
    
    - HR 타깃은 210nm_numpy 폴더의 ir_drop 파일을 원본 해상도로 로드합니다.
    
    각 샘플은 아래 정보를 dictionary로 반환합니다.
      - 'lr_input'   : 1um 데이터 (current, resistance 등 통합된 입력) → Tensor, (C,256,256)
      - 'lr_target'  : 1um ir_drop (256×256) → Tensor, (1,256,256)
      - 'hr_target'  : 210nm ir_drop, 원본 해상도 → Tensor, (1, H, W)
      - 'ori_shape'  : 1um 데이터의 원본 해상도 (H, W); 후에 HR 예측 시 보간에 사용
    """
    
    def __init__(self, *args, **kwargs):
        """
        인퍼런스용으로 train=False로 강제합니다.
        """
        kwargs['train'] = False  # 인퍼런스이므로 augmentation 적용하지 않음
        # 기존 train_auto_encoder 플래그는 사용하지 않고, 별도 인퍼런스 모드로 사용
        super().__init__(*args, **kwargs)
        # 기존의 val_transform (256×256 resize + ToTensor) 사용
        self.val_transform = A.Compose([
            A.Resize(self.target_size, self.target_size, interpolation=cv2.INTER_NEAREST),
            ToTensorV2()
        ], is_check_shapes=False)
        
        # _find_files_autoencoder 함수를 재정의합니다.
        self._find_files_autoencoder()
        # 인퍼런스에서는 LR 데이터 로딩 함수로 _load_data_inference()를 사용합니다.
        self.load_data_fn = self._load_data_inference

    def _find_files_autoencoder(self):
        """
        1um_numpy (LR 데이터)와 210nm_numpy (HR 타깃) 폴더에서
        기존에 사용하던 모든 파일들을 로드한 후 인덱스 기준으로 매칭합니다.
        
        LR 관련 파일은 아래 키로 저장됩니다.
           - 'current'
           - 'ir_drop'
           - 'resistances'  (여러 resistance 파일의 리스트)
           - 'pad_distance' (in_ch==3 인 경우)
        
        HR 타깃은 'hr_target' 키로 210nm 폴더의 ir_drop 파일 경로를 저장합니다.
        """
        self.data_files = []
        
        # 폴더 경로 설정
        input_folder = os.path.join(self.root_path, '1um_numpy', self.post_fix)
        target_folder = os.path.join(self.root_path, '210nm_numpy', self.post_fix)
        
        # === LR 데이터 (1um) 파일 그룹 ===
        file_groups_lr = {}
        file_groups_lr['current'] = glob.glob(os.path.join(input_folder, '*_current*.npy'))
        file_groups_lr['ir_drop'] = glob.glob(os.path.join(input_folder, '*_ir_drop*.npy'))
        file_groups_lr['resistances'] = glob.glob(os.path.join(input_folder, 'layer_data', '*resistance*.npy'))
        if self.in_ch == 3:
            file_groups_lr['pad_distance'] = glob.glob(os.path.join(input_folder, '*pad*.npy'))
        # 정렬
        for key in file_groups_lr:
            file_groups_lr[key].sort()
        
        # LR 데이터: 인덱스별 딕셔너리 생성
        lr_dict = {}
        # current 파일
        for file in file_groups_lr.get('current', []):
            index = os.path.basename(file).split('_')[0]
            lr_dict.setdefault(index, {})['current'] = file
        # ir_drop 파일 (LR 타깃로 사용됨)
        for file in file_groups_lr.get('ir_drop', []):
            index = os.path.basename(file).split('_')[0]
            lr_dict.setdefault(index, {})['ir_drop'] = file
        # resistance 파일들
        resistance_dict = {}
        for file in file_groups_lr.get('resistances', []):
            index = os.path.basename(file).split('_')[0]
            resistance_dict.setdefault(index, []).append(file)
        for index, files in resistance_dict.items():
            lr_dict.setdefault(index, {})['resistances'] = files
        # pad_distance (in_ch==3 인 경우)
        if self.in_ch == 3:
            for file in file_groups_lr.get('pad_distance', []):
                index = os.path.basename(file).split('_')[0]
                lr_dict.setdefault(index, {})['pad_distance'] = file
        
        # === HR 타깃 (210nm) 파일 그룹 ===
        # 여기서는 ir_drop 파일만 로드 (원본 해상도 유지)
        hr_files = glob.glob(os.path.join(target_folder, '*_ir_drop*.npy'))
        hr_files.sort()
        hr_dict = {}
        for file in hr_files:
            index = os.path.basename(file).split('_')[0]
            hr_dict[index] = file
        
        # LR와 HR에서 공통 인덱스 추출
        common_indices = sorted(list(set(lr_dict.keys()) & set(hr_dict.keys())))
        if not common_indices:
            raise ValueError("1um과 210nm 폴더에서 매칭되는 인덱스가 없습니다!")
        
        # 최종 data_files 리스트 구성 (각 항목은 LR 관련 파일들과 HR 타깃 파일 경로를 포함)
        for idx in common_indices:
            entry = lr_dict[idx]
            entry['hr_target'] = hr_dict[idx]
            self.data_files.append(entry)
    
    def _load_data_inference(self, idx):
        """
        LR 데이터(1um)의 입력과 타깃을 로드합니다.
        (기존 _load_data_from_disk_2ch 또는 _load_data_from_disk_3ch 로직을 재사용)
        """
        if self.in_ch == 2:
            return self._load_data_from_disk_2ch(idx)
        elif self.in_ch == 3:
            return self._load_data_from_disk_3ch(idx)
        else:
            raise ValueError(f"{self.in_ch} channels는 지원하지 않습니다.")
    
    def __getitem__(self, idx):
        """
        하나의 샘플을 로드하여 아래 정보를 dictionary로 반환:
        - 'lr_input'     : 1um 데이터 (current, resistance 등) → transform 적용 후 (256×256, C채널)
        - 'lr_target'    : 1um ir_drop (256×256, 1채널)
        - 'lr_target_ori': Transform을 적용하지 않은 1um ir_drop (원본 해상도)
        - 'lr_ori_shape' : Transform을 적용하지 않은 1um ir_drop의 원본 해상도 (H, W)
        - 'hr_target'    : 210nm ir_drop, 원본 해상도 (1, H, W)
        - 'hr_ori_shape' : 210nm ir_drop의 원본 해상도 (H, W)
        """
        # 1. LR 데이터 (원본 해상도)
        lr_input_data, lr_ir_drop = self.load_data_fn(idx) # min_max 적용된 ir drop
        lr_ori_shape = lr_ir_drop.shape[:2]  # 원본 해상도 저장

        # Transform 적용하지 않은 lr_target_ori 추가
        lr_target_ori = torch.from_numpy(lr_ir_drop).permute(2, 0, 1).float()  # (1, H, W)

        # 2. LR transform 적용 (256×256 resize + ToTensor)
        transformed = self.val_transform(image=lr_input_data, mask=lr_ir_drop)
        lr_input = transformed['image'].float()   # (C, 256, 256)
        lr_target = transformed['mask']
        if lr_target.ndim == 3:
            # ToTensorV2가 반환하는 mask는 (H, W, 1) → (1, H, W)로 변경
            lr_target = lr_target.permute(2, 0, 1).float()

        # 3. HR 타깃은 210nm 폴더의 파일을 np.load (원본 해상도, 보정 등 transform 적용하지 않음)
        hr_file = self.data_files[idx]['hr_target']
        hr_target = self._min_max_norm(np.load(hr_file))
        hr_ori_shape = hr_target.shape[:2]  # 원본 해상도 저장
        if hr_target.ndim == 2:
            hr_target = np.expand_dims(hr_target, axis=-1)
        hr_target_tensor = torch.from_numpy(hr_target).permute(2, 0, 1).float()

        return {
            'lr_input': lr_input,         # LR 모델 입력 (256×256)
            'lr_target': lr_target,       # LR 모델 타깃 (256×256)
            'lr_target_ori': lr_target_ori,  # Transform을 적용하지 않은 원본 (1, H, W)
            'lr_ori_shape': lr_ori_shape,  # Transform을 적용하지 않은 LR 타깃의 원본 해상도 (H, W)
            'hr_target': hr_target_tensor,  # HR 모델 타깃 (원본 해상도)
            'hr_ori_shape': hr_ori_shape   # HR 타깃의 원본 해상도 (H, W)
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
        print(dataset.__len__())

    def test_new_data_error():
        root_path = "/data/gen_pdn/pdn_data_3rd"
        selected_folders = ['210nm_numpy']
        post_fix = ""

        dataset = IRDropDataset5nm(root_path=root_path,
                                    selected_folders=selected_folders,
                                    post_fix_path=post_fix,
                                    train=False,
                                    in_ch=3)
        # 3rd error : 18, 26 (118,126)
        error_count = 0
        error_indices = []
        for i in range(dataset.__len__()):
            print(f'{i} processing')
            try:
                a = dataset.__getitem__(i)
                print(a[0].shape)
                print(dataset.__len__())
            except:
                error_count +=1
                error_indices.append(i)
                continue
        print('error_count : ',error_count)
        print('error_indices : ',error_indices)


    # test_new_data_error()

    def test_autoencoder_mode():
        root_path = "/data/gen_pdn/pdn_data_3rd"
        # train_auto_encoder 모드에서는 1um과 210nm 폴더의 파일이 매칭됩니다.
        post_fix = ""
        dataset = IRDropDataset5nm(root_path=root_path,
                                    selected_folders=[],  # 내부에서 고정됨
                                    post_fix_path=post_fix,
                                    train=True,
                                    in_ch=1,
                                    train_auto_encoder=True)
        print("Total samples:", len(dataset))
        # 첫 번째 샘플 확인
        input_tensor, target_tensor = dataset.__getitem__(0)
        # input_tensor, target_tensor = dataset.getitem_ori(0)
        print("Input shape (1um ir drop):", input_tensor.shape)
        print("Target shape (210nm ir drop):", target_tensor.shape)

    def test_infer_dataset_hr():
        root_path = "/data/gen_pdn"
        # train_auto_encoder 모드에서는 1um과 210nm 폴더의 파일이 매칭됩니다.
        post_fix = ""
        dataset = IRDropInferenceAutoencoderDataset5nm(root_path=root_path,
                                    selected_folders=['1um_numpy','210nm_numpy'],  # 내부에서 고정됨
                                    post_fix_path=post_fix,
                                    in_ch=2,
                                    )
        print("Total samples:", len(dataset))
        # 첫 번째 샘플 확인
        data = dataset.__getitem__(0)
        for key,value in data.items():
            if isinstance(value, torch.Tensor):
                print(f'{key} : ',value.shape) 
            else:
                print(f'{key} : ',value) 

    test_infer_dataset_hr()