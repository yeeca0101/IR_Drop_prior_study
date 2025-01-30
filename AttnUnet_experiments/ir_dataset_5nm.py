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
                 in_ch=2, use_raw=False):
        self.root_path = root_path
        self.selected_folders = selected_folders
        self.target_size = img_size
        self.post_fix = post_fix_path
        self.cached_data = None
        self.train = train
        self.in_ch = in_ch
        self.use_raw = use_raw

        self.transform = A.Compose([
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
        ])

        self.val_transform = A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
            ToTensorV2()
        ])

        self.data_files = []
        self._find_files()

        self.load_data_fn = self._load_data_from_disk_2ch if self.in_ch == 2 else self._load_data_from_disk_3ch

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

    def __getitem__(self, idx):
        input_data, ir_drop = self.load_data_fn(idx)
        transformed = self.transform(image=input_data, mask=ir_drop) if self.train else self.val_transform(image=input_data, mask=ir_drop)
        input_tensor = transformed['image']
        target_tensor = transformed['mask']
        if self.in_ch == 1:
            input_tensor = input_tensor[0].unsqueeze(0)
        return input_tensor.float(), target_tensor.float()

    def __len__(self):
        return len(self.data_files)

    def _norm(self, x):
        return x / x.max() if x.max() > 0 else x

    def _min_max_norm(self, x):
        return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x

    def _combine_3ch_data(self, current, pad_distance, resistance_total):
        return np.stack([current, pad_distance, resistance_total], axis=-1)



    
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


    test_new_data_error()