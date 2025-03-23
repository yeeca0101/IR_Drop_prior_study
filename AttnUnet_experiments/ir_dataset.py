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
from sklearn.model_selection import KFold

from ir_dataset_5nm import IRDropDataset5nm
class IRDropDataset(Dataset):
    def __init__(self, root_path, selected_folders, target_layers, 
                 csv_path=None, img_size=512, post_fix_path='data', preload=False,train=True,pdn_density_p=0.0,
                 asap7=False,pdn_zeros=False,in_ch=12,use_raw=False,use_irreg=False,):
        """
        Args:
            root_path (str): The root directory where datasets are stored.
            selected_folders (list of str): A list of folder names to include (e.g., ['asap7', 'sky130hd']).
            target_layers (list of str): A list of layer names (e.g., ['m1', 'm4', 'm7']).
            csv_path (str): Path to the CSV file to load/save file paths. If provided, load paths from this file.
            preload (bool): If True, preload all data into memory.
            in_ch : default 12. 2 ch : current map, resistance total map 
        """
        self.root_path = root_path
        self.selected_folders = selected_folders
        self.target_layers = target_layers
        self.csv_path = csv_path
        self.img_size = img_size
        self.post_fix = post_fix_path
        self.preload = preload  # 메모리 캐싱 여부
        self.cached_data = None  # 캐시된 데이터 저장용
        self.train = train
        self.pdn_density_p = pdn_density_p
        self.is_asap7 = asap7
        self.pdn_zeros = pdn_zeros
        self.in_ch=in_ch
        self.use_raw = use_raw # return raw data for ir drop
        self.use_irreg = use_irreg 
        if self.is_asap7 : self.use_irreg =True

        self.transform = A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
                A.OneOf([
                A.Compose([
                    A.HorizontalFlip(p=0.2),
                    A.VerticalFlip(p=0.2),
                    A.Rotate(limit=(90, 90), p=0.2),
                    A.Rotate(limit=(180, 180), p=0.2),
                    A.Rotate(limit=(270, 270), p=0.2)]),
                A.NoOp(p=1),
                ],p=1),
                ToTensorV2()
        ],is_check_shapes=False)

        self.val_transform = A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            ToTensorV2()
        ],is_check_shapes=False)

        if csv_path and os.path.exists(csv_path):
            # Load file paths from the CSV file
            self.data_files = self._load_from_csv(csv_path)
        else:
            # Generate the file paths from the folders and save them to CSV
            self.data_files = self._find_files()
            if csv_path:
                self._save_to_csv(self.data_files, csv_path)

        if preload:
            # 데이터를 미리 메모리에 로드
            self.cached_data = self._preload_data()

    def _find_files(self):
        data_files = []

        # Initialize these lists outside the loop to accumulate paths from all folders
        current_files = []
        eff_dist_files = []
        pdn_density_files = []
        ir_drop_files = []
        resistance_files_dict = {layer: [] for layer in self.target_layers}

        # Iterate through selected folders and accumulate the file paths
        for folder in self.selected_folders:
            folder_path = os.path.join(self.root_path, folder,self.post_fix)
            # Extend to accumulate files from all selected folders
            current_files.extend(glob.glob(os.path.join(folder_path, '*_current*.npy')))
            if self.is_asap7:
                ir_drop_files.extend(glob.glob(os.path.join(folder_path, '*_voltage_map_regular.npy')))    
            else:
                eff_dist_files.extend(glob.glob(os.path.join(folder_path, '*_eff_dist*.npy')))
                pdn_density_files.extend(glob.glob(os.path.join(folder_path, '*_pdn_density*.npy')))  # Extend instead of assign
                ir_drop_files.extend(glob.glob(os.path.join(folder_path, '*_ir_drop*.npy')))      # Extend instead of assign

            # Collect resistance matrix files per layer, and extend for each folder
            
            for layer in self.target_layers:
                resistance_files_dict[layer].extend(glob.glob(os.path.join(folder_path,'layer_data', f'*{layer}_resistance*.npy')))

        # Sort the accumulated file lists outside the loop
        current_files.sort()
        eff_dist_files.sort()
        pdn_density_files.sort()
        ir_drop_files.sort()
        for layer in self.target_layers:
            resistance_files_dict[layer].sort()  # Sort resistance files for each layer

        # Ensure that the lengths of all file lists match
        if (not (len(current_files) == len(eff_dist_files) == len(pdn_density_files) == len(ir_drop_files)) and (not self.is_asap7)):
            raise ValueError("Mismatch in the number of current, eff_dist, pdn_density, and ir_drop files!")

        # Create data groups by combining the corresponding files
        if self.is_asap7:
            for current, ir_drop in zip(current_files, ir_drop_files):
                # Add the matching resistance files for each layer
                resistances = {layer: resistance_files_dict[layer] for layer in self.target_layers}
                data_files.append({
                    'current': current,
                    'ir_drop': ir_drop,
                    'resistances': resistances
                })
        else:    
            for current, eff_dist, pdn_density, ir_drop in zip(current_files, eff_dist_files, pdn_density_files, ir_drop_files):
                # Add the matching resistance files for each layer
                resistances = {layer: resistance_files_dict[layer] for layer in self.target_layers}
                data_files.append({
                    'current': current,
                    'eff_dist': eff_dist,
                    'pdn_density': pdn_density,
                    'ir_drop': ir_drop,
                    'resistances': resistances
                })

        return data_files

    def _save_to_csv(self, data_files, csv_path):
        csv_folder = os.path.join(*(csv_path.split('/')[:-1]))
        os.makedirs(csv_folder, exist_ok=True)

        data_list = []
        for files in data_files:
            res_str = {layer: ",".join(files['resistances'][layer]) for layer in self.target_layers}
            if self.is_asap7:
                data_list.append([files['current'], files['ir_drop']] + [res_str[layer] for layer in self.target_layers])
            else:
                data_list.append([files['current'], files['eff_dist'], files['pdn_density'], files['ir_drop']] + [res_str[layer] for layer in self.target_layers])
        if self.is_asap7:
            columns = ['current', 'ir_drop'] + [f'resistance_{layer}' for layer in self.target_layers]
        else:
            columns = ['current', 'eff_dist', 'pdn_density', 'ir_drop'] + [f'resistance_{layer}' for layer in self.target_layers]
        df = pd.DataFrame(data_list, columns=columns)
        df.to_csv(csv_path, index=False)

    def _load_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        data_files = []
        for _, row in df.iterrows():
            res_dict = {layer: row[f'resistance_{layer}'].split(',') for layer in self.target_layers}
            if self.is_asap7:
                data_files.append({
                    'current': row['current'],
                    'ir_drop': row['ir_drop'],
                    'resistances': res_dict
                })
            else:
                data_files.append({
                    'current': row['current'],
                    'eff_dist': row['eff_dist'], 
                    'pdn_density': row['pdn_density'],
                    'ir_drop': row['ir_drop'],
                    'resistances': res_dict
                })
        return data_files

    def _preload_data(self):
        """ Preload data into memory for faster access. """
        cached_data = []
        for idx in range(len(self.data_files)):
            input_data, target_data = self._load_data_from_disk(idx)
            cached_data.append((input_data, target_data))
        return cached_data

    def _load_data_from_disk(self, idx):
        """ Helper function to load data from disk for a given index. """
        file_group = self.data_files[idx]
        current = self._norm(np.load(file_group['current']))
        if self.is_asap7 or self.pdn_zeros:
            eff_dist = np.ones_like(current)
        else:    
            eff_dist = self._norm(np.load(file_group['eff_dist']))
        # ir_drop = self._min_max_norm(np.load(file_group['ir_drop']))
        ir_drop = np.load(file_group['ir_drop'])
        if self.use_irreg:
            target_shape = current.shape[:2]  # current의 높이와 너비만 사용
            ir_drop = cv2.resize(ir_drop, (target_shape[1], target_shape[0]))

        ir_drop = self._min_max_norm(ir_drop) if not self.use_raw else ir_drop
        
        if ((np.random.random() < self.pdn_density_p) and self.train) or self.is_asap7 or self.pdn_zeros:
            pdn_density = np.ones_like(current)
        else:
            pdn_density = self._norm(np.load(file_group['pdn_density']))

        resistance_stack = []
        for layer in self.target_layers:
            res_file = file_group['resistances'][layer][idx]  # idx에 맞는 resistance 데이터 가져옴
            resistance_stack.append(self._norm(np.load(res_file)))
        
        resistance_stack = np.stack(resistance_stack, axis=-1)
        input_data = np.stack([current, eff_dist, pdn_density], axis=-1)

        input_data = np.concatenate([input_data, resistance_stack], axis=-1)
        
        return input_data, ir_drop

    def _load_data_from_disk_2ch(self, idx):
        """ Helper function to load data from disk for a given index. """
        file_group = self.data_files[idx]
        current = self._norm(np.load(file_group['current']))
        
        ir_drop = np.load(file_group['ir_drop'])
        if self.is_asap7:
            target_shape = current.shape[:2]  # current의 높이와 너비만 사용
            ir_drop = cv2.resize(ir_drop, (target_shape[1], target_shape[0]))

        ir_drop = self._min_max_norm(ir_drop) if not self.use_raw else ir_drop

        resistance_stack = []
        for layer in self.target_layers:
            res_file = file_group['resistances'][layer][idx]  # idx에 맞는 resistance 데이터 가져옴
            resistance_stack.append(np.load(res_file))
        resistance_stack = np.stack(resistance_stack, axis=-1).sum(-1)
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
    
    def __len__(self):
        return len(self.data_files)

    def _norm(self, x):
        return x / x.max()

    def _min_max_norm(self,x):
        return x / x.max()
        # return (x-x.min())/(x.max()-x.min())
    
    def __getitem__(self, idx):
        input_data, ir_drop = self._load_data_from_disk(idx) if self.in_ch != 2 else self._load_data_from_disk_2ch(idx)
        
        # 입력 데이터 변환
        transformed = self.transform(image=input_data, mask=ir_drop) if self.train else self.val_transform(image=input_data, mask=ir_drop)
        input_tensor = transformed['image']
        target_tensor = transformed['mask'] 
            
        return input_tensor.float(), target_tensor.unsqueeze(0).float()

    def _getitem_ori(self, idx):
        file_group = self.data_files[idx]
        
        current = np.load(file_group['current'])
        eff_dist = np.load(file_group['eff_dist'])
        pdn_density = np.load(file_group['pdn_density'])
        ir_drop = np.load(file_group['ir_drop'])
        
        # 여러 레이어에 대한 resistance 데이터 처리
        resistance_stack = []
        for layer in self.target_layers:
            resistances_for_layer = [np.load(res_file) for res_file in file_group['resistances'][layer]][idx]
            resistance_stack.extend(resistances_for_layer)

        # 입력 데이터 변환
        input_data = np.stack([current, eff_dist, pdn_density], axis=-1)
        input_data = np.stack(input_data,resistance_stack,axis=-1)

        return input_data, ir_drop
    
    def _combine_3ch_data(self, current, pad_distance, resistance_total):
        return np.stack([current, pad_distance, resistance_total], axis=-1)


class IRDropFineTuneDataset(Dataset):
    def __init__(self, root_path, selected_folders, target_layers, csv_path=None, img_size=512, 
                 preload=False,train=True,pdn_density_p=0.0,return_case=False,
                 use_global_min_max_scale=False,in_ch=12,pdn_zeros=False,use_raw=False):
        """
        Args:
            root_path (str): The root directory where datasets are stored.
            selected_folders (list of str): A list of folder names to include (e.g., ['testcase1', 'testcase2']).
            target_layers (list of str): A list of layer names (e.g., ['m1', 'm4', 'm7']).
            csv_path (str): Path to the CSV file to load/save file paths. If provided, load paths from this file.
            preload (bool): If True, preload all data into memory.
        """
        self.root_path = root_path
        self.selected_folders = selected_folders
        self.target_layers = target_layers
        self.csv_path = csv_path
        self.img_size = img_size
        self.preload = preload  # Cache data in memory
        self.cached_data = None  # Cached data storage
        self.pdn_density_p = pdn_density_p
        self.train = train
        self.return_case = return_case
        self.use_global_min_max_scale = use_global_min_max_scale
        self.in_ch=in_ch
        self.pdn_zeros = pdn_zeros
        self.use_raw = use_raw

        self.resize = A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA)
        self.train_transform = A.Compose([
                A.OneOf([
                A.Compose([
                    A.HorizontalFlip(p=0.2),
                    A.VerticalFlip(p=0.2),
                    A.Rotate(limit=(90, 90), p=0.2),
                    A.Rotate(limit=(180, 180), p=0.2),
                    A.Rotate(limit=(270, 270), p=0.2)]),
                A.NoOp(p=1),
                ],p=1),
                ToTensorV2()
        ],is_check_shapes=False)

        self.val_transform = A.Compose([
            ToTensorV2()
        ],is_check_shapes=False)


        if csv_path and os.path.exists(csv_path):
            # Load file paths from the CSV file
            self.data_files = self._load_from_csv(csv_path)
        else:
            # Generate the file paths from the folders and save them to CSV
            self.data_files = self._find_files()
            if csv_path:
                self._save_to_csv(self.data_files, csv_path)

    def _find_files(self):
        data_files = []

        # Initialize these lists outside the loop to accumulate paths from all folders
        current_files = []
        eff_dist_files = []
        pdn_density_files = []
        ir_drop_files = []
        resistance_files_dict = {layer: [] for layer in self.target_layers}

        # Iterate through selected folders and accumulate the file paths
        for folder in self.selected_folders:
            folder_path = os.path.join(self.root_path, folder)
            
            # Extend to accumulate files from all selected folders
            current_files.extend(glob.glob(os.path.join(folder_path, 'current_map.npy')))
            eff_dist_files.extend(glob.glob(os.path.join(folder_path, 'eff_dist_map.npy')))
            pdn_density_files.extend(glob.glob(os.path.join(folder_path, 'pdn_density.npy')))  # Extend instead of assign
            ir_drop_files.extend(glob.glob(os.path.join(folder_path, 'ir_drop_map.npy')))      # Extend instead of assign

            # Collect resistance matrix files per layer, and extend for each folder
            
            for layer in self.target_layers:
                resistance_files_dict[layer].extend(glob.glob(os.path.join(folder_path,'layer_data', f'*{layer}_resistance*.npy')))

        # Sort the accumulated file lists outside the loop
        current_files.sort()
        eff_dist_files.sort()
        pdn_density_files.sort()
        ir_drop_files.sort()
        for layer in self.target_layers:
            resistance_files_dict[layer].sort()  # Sort resistance files for each layer

        # Ensure that the lengths of all file lists match
        if not (len(current_files) == len(eff_dist_files) == len(pdn_density_files) == len(ir_drop_files)):
            raise ValueError("Mismatch in the number of current, eff_dist, pdn_density, and ir_drop files!")

        # Create data groups by combining the corresponding files
        for current, eff_dist, pdn_density, ir_drop in zip(current_files, eff_dist_files, pdn_density_files, ir_drop_files):
            # Add the matching resistance files for each layer
            resistances = {layer: resistance_files_dict[layer] for layer in self.target_layers}
            data_files.append({
                'current': current,
                'eff_dist': eff_dist,
                'pdn_density': pdn_density,
                'ir_drop': ir_drop,
                'resistances': resistances
            })
        return data_files

    def _save_to_csv(self, data_files, csv_path):
        csv_folder = os.path.join(*(csv_path.split('/')[:-1]))
        os.makedirs(csv_folder, exist_ok=True)

        data_list = []
        for files in data_files:
            res_str = {layer: ",".join(files['resistances'][layer]) for layer in self.target_layers}
            data_list.append([files['current'], files['eff_dist'], files['pdn_density'], files['ir_drop']] + [res_str[layer] for layer in self.target_layers])
        
        columns = ['current', 'eff_dist', 'pdn_density', 'ir_drop'] + [f'resistance_{layer}' for layer in self.target_layers]
        df = pd.DataFrame(data_list, columns=columns)
        df.to_csv(csv_path, index=False)

    def _load_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        data_files = []
        for _, row in df.iterrows():
            res_dict = {layer: row[f'resistance_{layer}'].split(',') for layer in self.target_layers}
            data_files.append({
                'current': row['current'],
                'eff_dist': row['eff_dist'],
                'pdn_density': row['pdn_density'],
                'ir_drop': row['ir_drop'],
                'resistances': res_dict
            })
        return data_files

    def _preload_data(self):
        """ Preload data into memory for faster access. """
        cached_data = []
        for idx in range(len(self.data_files)):
            input_data, target_data = self._load_data_from_disk(idx) if self.in_ch != 2 else self._load_data_from_disk_2ch(idx)
            cached_data.append((input_data, target_data))
        return cached_data

    def _load_data_from_disk(self, idx):
        """ Helper function to load data from disk for a given index. """
        file_group = self.data_files[idx]
        
        current = np.load(file_group['current'])
        eff_dist = np.load(file_group['eff_dist'])
        ir_drop = np.load(file_group['ir_drop'])
                                        
        if np.random.random() < self.pdn_density_p and self.train or self.pdn_zeros:
            pdn_density = np.zeros_like(current)
        else:
            pdn_density = np.load(file_group['pdn_density'])

        resistance_stack = []
        for layer in self.target_layers:
            res_file = file_group['resistances'][layer][idx]
            resistance_stack.append(np.load(res_file))
        
        resistance_stack = np.stack(resistance_stack, axis=-1)
        input_data = np.stack([current, eff_dist, pdn_density], axis=-1)
        input_data = np.concatenate([input_data, resistance_stack], axis=-1)
        
        return input_data, ir_drop

    def _load_data_from_disk_2ch(self, idx):
        """ Helper function to load data from disk for a given index. """
        file_group = self.data_files[idx]
        current = np.load(file_group['current'])
        ir_drop = np.load(file_group['ir_drop'])

        resistance_stack = []
        for layer in self.target_layers:
            res_file = file_group['resistances'][layer][idx]  # idx에 맞는 resistance 데이터 가져옴
            resistance_stack.append(np.load(res_file))
        resistance_stack = np.stack(resistance_stack, axis=-1).sum(-1)
        resistance_total = resistance_stack
            
        input_data = np.stack([current, resistance_total], axis=-1)
        return input_data, ir_drop
    
    def __len__(self):
        return len(self.data_files)

    def _norm(self, x):
        return x / x.max()
    
    def _min_max_norm(self,x):
        return (x-x.min())/(x.max()-x.min())

    def channel_wise_min_max(self,x):
        channel_max = np.max(x, axis=(0, 1), keepdims=True)
        channel_min = np.min(x, axis=(0, 1), keepdims=True)
        channel_max = np.where(channel_max == 0, 1, channel_max)
        x = (x-channel_min) / (channel_max-channel_min)

        return x
    
    def channel_wise_max(self,x):
        channel_max = np.max(x, axis=(0, 1), keepdims=True)
        channel_max = np.where(channel_max == 0, 1, channel_max)
        x = x/ channel_max

        return x
       # input_data : 3 or 12ch [curr(1), effective distance(1), empty or pdn density(1) resistances(9) or total resisntance(1)]
        # how to apply norm before rotate transforms
    
    def sample_per_z_norm(self,x):
        return (x - x.mean())/x.std()

    def std_norm(self,x):
        return x/0.00038944 # cache384
    
    def z_norm(self,x):
        return (x-0.00112233)/0.00038944
    
    def __getitem__(self, idx):
        input_data, ir_drop = self._load_data_from_disk(idx) if self.in_ch != 2 else self._load_data_from_disk_2ch(idx)
        input_data= self.resize.apply(input_data,interpolation=cv2.INTER_AREA)
        input_data = self.channel_wise_min_max(input_data)

        if not self.use_raw:
            ir_drop = self.resize.apply_to_mask(ir_drop,interpolation=cv2.INTER_AREA)
            # ir_drop *= 100 
            ir_drop = self._norm(ir_drop)

        if self.train:
            transformed = self.train_transform(image=input_data, mask=ir_drop)
        else:
            transformed = self.val_transform(image=input_data, mask=ir_drop)
        input_tensor = transformed['image']
        target_tensor = transformed['mask'].unsqueeze(0) if not self.use_raw else torch.as_tensor(ir_drop).unsqueeze(0)
        

        if self.return_case:
            testcase_name = [_ for _ in self.data_files[idx]['current'].split('/') if 'testcase' in _][0]
            return input_tensor.float(), target_tensor.float(), testcase_name
        return input_tensor.float(), target_tensor.float()


class TestASAP7Dataset(Dataset):
    def __init__(self, root_path, target_layers, img_size=512, use_irreg=False, 
                 preload=False, train=True,return_case=False,in_ch=12,use_raw=False,debug=True):
        """
        Args:
            root_path (str): The root directory where datasets are stored.
            target_layers (list of str): A list of layer names (e.g., ['m1', 'm4', 'm7']).
            img_size (int): The target image size for resizing.
            use_irreg (bool): If True, use files from 'irreg' folder; otherwise, use 'reg' folder.
            preload (bool): If True, preload all data into memory.
            train (bool): If True, apply data augmentations for training; otherwise, use validation transform.
        """
        self.root_path = root_path
        self.target_layers = target_layers
        self.img_size = img_size
        self.use_irreg = use_irreg
        self.preload = preload
        self.train = train
        self.debug = debug
        self.return_case=return_case
        self.in_ch = in_ch
        self.use_raw = use_raw

        self.transform = A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
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
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            ToTensorV2()
        ])

        self.transform = self.transform if self.train else self.val_transform
        # Load files from all subdirectories, excluding "dynamic_node"
        self.data_files = self._find_files(exclude_folder="dynamic_node")

        if preload:
            self.cached_data = self._preload_data()

    def _find_files(self, exclude_folder="dynamic_node"):
        data_files = []
        grid_type = 'irreg' if self.use_irreg else 'reg'

        # Iterate over folders in root_path, excluding the specified folder
        for folder in os.listdir(self.root_path):
            if folder == exclude_folder:
                continue  # Skip the excluded folder
            
            folder_path = os.path.join(self.root_path, folder)

            # File paths for current map and voltage maps
            current_files = glob.glob(os.path.join(folder_path, f'*_current*.npy'))
            voltage_files = glob.glob(os.path.join(folder_path, f'*_voltage_map_{grid_type}*.npy'))
            # for BeGAN-45-real
            if len(voltage_files)==0:voltage_files=glob.glob(os.path.join(folder_path, f'*ir_drop_{grid_type}*.npy'))
            if len(current_files) != len(voltage_files): current_files = glob.glob(os.path.join(folder_path, f'*_current_{grid_type}*.npy'))
            # Debug output to check file counts
            if self.debug:
                print(f"Processing folder: {folder}")
                print(f"Found {len(current_files)} current map files.")
                print(f"Found {len(voltage_files)} voltage map files for grid type '{grid_type}'.")

            # Resistance matrix files for each layer - only load once per layer
            layer_data_path = os.path.join(folder_path, 'layer_data', grid_type)
            if not os.path.isdir(layer_data_path):layer_data_path = os.path.join(folder_path, 'layer_data')
            resistances = {}
            
            for layer in self.target_layers:
                resistance_file = os.path.join(layer_data_path, f'{layer}_resistance_matrix.npy')
                if not os.path.isfile(resistance_file):
                    raise FileNotFoundError(f"Missing or multiple resistance files for layer {layer} in folder {folder}. Expected 1 file.")
                resistances[layer] = resistance_file  # Store the single file path for each layer

            current_files.sort()
            voltage_files.sort()

            # Ensure the number of current and voltage files match
            if len(current_files) != len(voltage_files):
                raise ValueError(f"Mismatch in the number of current and voltage map files in folder {folder}!")

            # Add each data group with the preloaded resistance files for each layer
            for i in range(len(current_files)):
                data_files.append({
                    'current': current_files[i],
                    'voltage': voltage_files[i],
                    'resistances': resistances
                })

        return data_files

    def _preload_data(self):
        cached_data = []
        for idx in range(len(self.data_files)):
            input_data, target_data = self._load_data_from_disk(idx)
            cached_data.append((input_data, target_data))
        return cached_data

    def _load_data_from_disk_2ch(self, idx):
        """ Helper function to load data from disk for a given index. """
        file_group = self.data_files[idx]
        current = self._norm(np.load(file_group['current']))
        
        try:
            ir_drop = np.load(file_group['voltage'])
            if not self.use_raw : ir_drop = self._min_max_norm(ir_drop)# voltage is the target
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing voltage map file at index {idx}")
        if not self.use_irreg and current.shape != ir_drop.shape:
            ir_drop = cv2.resize(ir_drop, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_AREA)

        resistance_stack = []
        for layer in self.target_layers:
            res_file = file_group['resistances'][layer]  # idx에 맞는 resistance 데이터 가져옴
            resistance_stack.append(np.load(res_file))
        resistance_stack = np.stack(resistance_stack, axis=-1).sum(-1)
        resistance_total = self._norm(resistance_stack)
            
        input_data = np.stack([current, resistance_total], axis=-1)
        return input_data, ir_drop
    
    def _load_data_from_disk(self, idx):
        file_group = self.data_files[idx]
        # Load and check mandatory files
        try:
            current = self._norm(np.load(file_group['current']))
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing current map file at index {idx}")

        try:
            ir_drop = self._min_max_norm(np.load(file_group['voltage']))  # voltage is the target
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing voltage map file at index {idx}")

        # Resize voltage if use_irreg is False and shapes don't match
        if not self.use_irreg and current.shape != ir_drop.shape:
            ir_drop = cv2.resize(ir_drop, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_AREA)

        resistance_stack = []
        for layer in self.target_layers:
            try:
                res_file = file_group['resistances'][layer]
                resistance_map = np.load(res_file)
                # Check if all values in the resistance map are zero
                if np.all(resistance_map == 0):
                    # If all values are zero, use zeros_like instead of normalization
                    normalized_resistance = np.ones_like(resistance_map)
                else:
                    # Otherwise, normalize
                    normalized_resistance = resistance_map / resistance_map.max()
                resistance_stack.append(normalized_resistance)
            except (IndexError, FileNotFoundError):
                raise FileNotFoundError(f"Missing resistance file {res_file} for layer {layer} at index {idx}")

        # Handle optional files: effective_dist and pdn_density_map
        effective_dist = np.ones_like(current)  # Set to ones if missing
        pdn_density_map = np.ones_like(current)  # Set to ones if missing

        # Ensure all arrays have the same 3D shape
        current = np.expand_dims(current, axis=-1)
        effective_dist = np.expand_dims(effective_dist, axis=-1)
        pdn_density_map = np.expand_dims(pdn_density_map, axis=-1)
        resistance_stack = np.stack(resistance_stack, axis=-1)

        # Concatenate input arrays along the third dimension
        input_data = np.concatenate([current, effective_dist, pdn_density_map, resistance_stack], axis=-1)

        return input_data, ir_drop  # voltage is used as the target

    def _norm(self, x):
        return x / x.max()

    def _min_max_norm(self,x):
        return (x-x.min())/(x.max()-x.min())

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        if self.preload:
            input_data, target_data = self.cached_data[idx]
        else:
            input_data, target_data = self._load_data_from_disk(idx) if self.in_ch != 2 else self._load_data_from_disk_2ch(idx)
        
        transformed = self.transform(image=input_data, mask=target_data)
        input_tensor = transformed['image']
        target_tensor = transformed['mask'].unsqueeze(0)
        if self.in_ch==1:
            input_tensor = input_tensor[0].unsqueeze(0)

        if self.return_case:
            testcase_name = [_ for _ in self.data_files[idx]['current'].split('/') if 'current' in _][0]
            testcase_name = testcase_name.split('_current_map.npy')[0]
            return input_tensor.float(), target_tensor.float(), testcase_name
        return input_tensor.float(), target_tensor.float()

def build_dataset_asap7_cross_val(root_path, target_layers, img_size=512, use_irreg=False, preload=True,in_ch=12,use_raw=False):
    # 전체 데이터셋 생성
    full_dataset = TestASAP7Dataset(root_path=root_path,
                                    target_layers=target_layers,
                                    img_size=img_size,
                                    use_irreg=use_irreg,
                                    preload=preload,
                                    train=False,
                                    in_ch=in_ch,
                                    use_raw=use_raw
                                    )
    
    # 데이터셋의 전체 길이
    dataset_size = len(full_dataset)
    
    # 4-fold cross-validation을 위한 인덱스 생성
    indices = list(range(dataset_size))
    
    # 각 fold에 대한 데이터셋을 저장할 리스트
    cross_val_datasets = []
    
    for i in range(dataset_size):
        # i번째 샘플을 검증 세트로 사용
        val_idx = [i]
        train_idx = [idx for idx in indices if idx != i]
        
        # 훈련 및 검증 세트 생성
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        
        cross_val_datasets.append((train_dataset, val_dataset))


    return cross_val_datasets

def split_dataset(dataset, train_ratio=0.8, valid_ratio=0.195, test_ratio=0.005, random_state=42,train=True):
    assert np.isclose(train_ratio + valid_ratio + test_ratio, 1.0), " 1.0"
    from copy import deepcopy
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    train_valid_size = int((train_ratio + valid_ratio) * dataset_size)
    train_valid_indices, test_indices = train_test_split(indices, train_size=train_valid_size, random_state=random_state)
    
    train_size = int(train_ratio * dataset_size)
    train_indices, valid_indices = train_test_split(train_valid_indices, train_size=train_size, random_state=random_state)
    
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(deepcopy(dataset), valid_indices)
    test_dataset = Subset(deepcopy(dataset), test_indices)

    valid_dataset.dataset.__setattr__('train',False)
    test_dataset.dataset.__setattr__('train',False)
    train_dataset.dataset.__setattr__('train',True)

    if train:
        return train_dataset, valid_dataset
    else:
        return train_dataset, valid_dataset, test_dataset

def build_dataset(root_path='/data/BeGAN-circuit-benchmarks',img_size=512,train=True,
                  pdn_density_p=0.0,pdn_zeros=False,in_ch=12,use_raw=False):
    dataset = IRDropDataset(root_path=root_path,
                            selected_folders=['nangate45/set1_numpy','nangate45/set2_numpy'],
                            # csv_path='./csv/total_numpy_12ch.csv',
                            img_size=img_size,
                           target_layers = ['m1', 'm4', 'm7', 'm8', 'm9', 'm14', 'm47', 'm78', 'm89'],
                           pdn_density_p=pdn_density_p,
                           pdn_zeros=pdn_zeros,
                           in_ch=in_ch,
                            use_raw=use_raw
                      )
    print(dataset.__len__())
    return split_dataset(dataset,train=train)

############ 5nm ###########################################################
def build_dataset_5m(img_size=256,train=True,
                 in_ch=2,use_raw=False, unit='1um',train_auto_encoder=False,
                 root_path = '/data',inn=False):
    # root_path = "/data"
    # 5th는 기존 ir max 기준으로 수정버전
    selected_folders = [f'pdn_4th_4types/{unit}_numpy',
                        f'pdn_3rd_4types/{unit}_numpy', 
                        f'pdn_data_6th/{unit}_numpy']
    post_fix = ""
    dataset = IRDropDataset5nm(root_path=root_path,
                               img_size=img_size,
                               train=train,
                                selected_folders=selected_folders,
                                post_fix_path=post_fix,
                                in_ch=in_ch,
                                use_raw=use_raw,
                                dbu_per_px=unit,
                                )


    print(dataset.__len__())
    if train:
        return split_train_val(dataset,)
    else:
        return dataset


def build_dataset_5m_auto(img_size=256,use_raw=False,train=True):
    root_path = "/data/gen_pdn"
    selected_folders = ['1um_numpy','210nm_numpy']
    post_fix = ""
    dataset = IRDropDataset5nm(root_path=root_path,
                               img_size=img_size,
                                selected_folders=selected_folders,  # 내부에서 고정됨
                                post_fix_path=post_fix,
                                train=train,
                                in_ch=1,
                                train_auto_encoder=True,
                                use_raw=use_raw)
    
    print(dataset.__len__())
    return dataset


def build_dataset_5m_test(img_size=256,
                 in_ch=2,use_raw=False,selected_folders = ['210nm_numpy',]):

    root_path = "/data/gen_pdn/pdn_data_3rd"
    post_fix = ""
    dataset = IRDropDataset5nm(root_path=root_path,
                               img_size=img_size,
                               train=False,
                                selected_folders=selected_folders,
                                post_fix_path=post_fix,
                                in_ch=in_ch,
                                use_raw=use_raw)
    print(dataset.__len__())
    return dataset
    
################################################################################
def split_train_val(dataset, train_ratio=0.8, random_state=42):
    from copy import deepcopy
    # 전체 데이터셋 크기
    dataset_size = len(dataset)
    
    # 인덱스 생성
    indices = list(range(dataset_size))
    
    # train과 validation을 8:2로 나눔
    train_size = int(train_ratio * dataset_size)
    train_indices, valid_indices = train_test_split(indices, train_size=train_size, random_state=random_state)
    
    # 데이터셋 서브셋 생성
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(deepcopy(dataset), valid_indices)
    valid_dataset.dataset.__setattr__('train',False)

    return train_dataset, valid_dataset

def build_dataset_iccad(finetune=False,pdn_density_p=0.0,pdn_zeros=False,in_ch=12,img_size=512,use_raw=False):
    if finetune:
        return build_dataset_iccad_finetune(pdn_density_p=pdn_density_p,pdn_zeros=pdn_zeros,img_size=img_size,in_ch=in_ch,use_raw=use_raw)
    
    dataset = IRDropDataset(root_path='/data/ICCAD_2023/fake-circuit-data_20230623',
                        selected_folders=['fake-circuit-data-npy'],
                        img_size=img_size,
                        post_fix_path='',
                        target_layers = ['m1', 'm4', 'm7', 'm8', 'm9', 'm14', 'm47', 'm78', 'm89'],
                        preload=False,
                        pdn_density_p=pdn_density_p,
                        pdn_zeros=pdn_zeros,
                        in_ch=in_ch,
                        use_raw=use_raw
                    )
    return split_train_val(dataset)

def build_dataset_iccad_finetune(pdn_density_p=0.0,return_case=False,in_ch=12,img_size=512,pdn_zeros=False,preload=False,use_raw=False):
    root_path='/data/ICCAD_2023/real-circuit-data_20230615'
    testcase_folders = os.listdir(root_path)
    dataset = IRDropFineTuneDataset(root_path=root_path,
                        selected_folders=testcase_folders,
                        img_size=img_size,
                        target_layers = ['m1', 'm4', 'm7', 'm8', 'm9', 'm14', 'm47', 'm78', 'm89'],
                        preload=preload,
                        train=True,
                        pdn_density_p=pdn_density_p,
                        in_ch=in_ch,
                        pdn_zeros=pdn_zeros,
                        use_raw=use_raw
                    )


    return split_train_val(dataset,train_ratio=0.8)


def build_dataset_iccad_hidden(pdn_density_p=0.0,return_case=False,in_ch=12,img_size=512,pdn_zeros=False,preload=False,use_raw=False):
    root_path='/data/ICCAD_2023/hidden-real-circuit-data'
    testcase_folders = os.listdir(root_path)
    test_dataset = IRDropFineTuneDataset(root_path=root_path,
                        selected_folders=testcase_folders,
                        img_size=img_size,
                        target_layers = ['m1', 'm4', 'm7', 'm8', 'm9', 'm14', 'm47', 'm78', 'm89'],
                        preload=preload,
                        train=False,
                        pdn_density_p=pdn_density_p,
                        return_case=return_case,
                        in_ch=in_ch,
                        pdn_zeros=pdn_zeros,
                        use_raw=use_raw
                    )
    return test_dataset

def build_dataset_began_asap7(finetune=False,train=True,in_ch=12,img_size=512,use_raw=False):
    if finetune:
        pass
    else:
        root_path = '/data/BeGAN-circuit-benchmarks/asap7/numpy_data'  # Replace with actual path
        target_layers = ['m2', 'm5', 'm6', 'm7', 'm8', 'm25', 'm56', 'm67', 'm78']
        dataset = IRDropDataset(
                    root_path=root_path,
                    selected_folders=['BeGAN'],
                    post_fix_path='',
                    target_layers=target_layers,
                    img_size=img_size,
                    preload=False,
                    train=train,  
                    asap7=True,
                    in_ch=in_ch,
                    use_raw=use_raw
                )
        return split_train_val(dataset)

###### ASAP7 real Cross Val ###########################################
def build_dataset_asap7_cross_val(cross_val_id, n_splits=4,get_case_name=True,in_ch=12,img_size=512,use_raw=False):
    from copy import deepcopy

    dataset = TestASAP7Dataset(root_path='/data/real-circuit-benchmarks/asap7/numpy_data',
                    target_layers=['m2', 'm5', 'm6', 'm7', 'm8', 'm25', 'm56', 'm67', 'm78'],
                    img_size=img_size, use_irreg=False, preload=False, train=True,return_case=get_case_name,
                    in_ch=in_ch)
    
    kf = KFold(n_splits=n_splits)
    indices = list(range(len(dataset)))
    train_indices, val_indices = list(kf.split(indices))[cross_val_id]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(deepcopy(dataset), val_indices)
    val_dataset.dataset.__setattr__('train', False)
    return train_dataset, val_dataset

def get_val_casename(cross_val_id, n_splits=4):
            _, val_dataset = build_dataset_asap7_cross_val(cross_val_id, n_splits, get_case_name=True)
            val_case_names = [val_dataset.__getitem__(i)[-1] for i in range(len(val_dataset))]
            return '_'.join(val_case_names)

##########################################################################

if __name__ =='__main__':
    from torch.utils.data import DataLoader

    def test1():
        dt = IRDropDataset(root_path='/data/BeGAN-circuit-benchmarks',
                                selected_folders=['nangate45/set1_numpy','nangate45/set2_numpy'],
                                img_size=512,
                            target_layers = ['m1', 'm4', 'm7', 'm8', 'm9', 'm14', 'm47', 'm78', 'm89'],
                            in_ch=2
                        )
        # print(dt.__len__())
        print(dt.__getitem__(1)[0].shape,dt.__getitem__(1)[1].shape)
        # print(dt.__getitem__(64)[0].shape,dt.__getitem__(64)[1].shape)

    ##########################################
    # td,vd = build_dataset_iccad()
    # td.__getitem__(1)
    # vd.__getitem__(1)
    # print(td.__len__(),vd.__len__())
    ###########################################

    # dt,_ = build_dataset_iccad()
    # train_loader = DataLoader(dt,batch_size=32,shuffle=True,drop_last=True,num_workers=0,pin_memory=True)
    # for batch in train_loader:
    #     inp, target = batch
    #     print(inp.shape,target.shape)

    # dataset = IRDropFineTuneDataset(root_path='/data/ICCAD_2023/hidden-real-circuit-data',
    #                     selected_folders=['testcase10','testcase13','testcase14'],
    #                     img_size=512,
    #                     target_layers = ['m1', 'm4', 'm7', 'm8', 'm9', 'm14', 'm47', 'm78', 'm89'],
    #                     preload=True
    #                 )
    # print(dataset.data_files)

    # td,vd = build_dataset_iccad_finetune()

    # print(td.__len__(),vd.__len__())

    #########################################
    # return testcase test
    def test3():
        root_path='/data/ICCAD_2023/hidden-real-circuit-data'
        testcase_folders = os.listdir(root_path)
        val_dataset = IRDropFineTuneDataset(root_path=root_path,
                            selected_folders=testcase_folders,
                            img_size=512,
                            target_layers = ['m1', 'm4', 'm7', 'm8', 'm9', 'm14', 'm47', 'm78', 'm89'],
                            preload=False,
                            train=False,
                            pdn_density_p=0,
                            return_case=True,
                            in_ch=2
                        )
        i,o,name = val_dataset.__getitem__(9)
        print(i.shape, o.shape)
        print(name)

    # test
    def test4():
        def test_asap7_dataset(root_path, target_layers, img_size=512, use_irreg=False, preload=False, batch_size=2):
            # Initialize the dataset
            dataset = TestASAP7Dataset(
                root_path=root_path,
                target_layers=target_layers,
                img_size=img_size,
                use_irreg=use_irreg,
                preload=preload,
                train=False  # Enable training mode to apply augmentations
            )
            print('len : ',dataset.__len__())
            
            # Initialize DataLoader for batch processing
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            
            # Fetch a single batch
            for batch_idx, (input_data, target_data) in enumerate(dataloader):
                print(f"Batch {batch_idx + 1}")
                print(f"Input Data Shape: {input_data.shape}")
                print(f"Target Data Shape: {target_data.shape}")
                import matplotlib.pyplot as plt
                # plt.imsave(f'target_{batch_idx}.png',target_data[0])
                # Display only the first batch
            print(dataset.data_files)
        root_path = '/data/real-circuit-benchmarks/asap7/numpy_data'  # Replace with actual path
        target_layers = ['m2', 'm5', 'm6', 'm7', 'm8', 'm25', 'm56', 'm67', 'm78']
        # Run the test function
        test_asap7_dataset(root_path, target_layers, img_size=512, use_irreg=False, preload=False)
    # test4()

    def test5(): # fintune
        def test_asap7_dataset2(root_path, target_layers, img_size=512, use_irreg=False, preload=False, batch_size=2):
            # Initialize the dataset
            dataset = IRDropDataset(
                root_path=root_path,
                selected_folders=['BeGAN'],
                post_fix_path='',
                target_layers=target_layers,
                img_size=img_size,
                preload=preload,
                train=True,  # Enable training mode to apply augmentations
                asap7=True
            )
            print('len : ',dataset.__len__())
            dataset.__getitem__(1)
            # Initialize DataLoader for batch processing
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            
            # Fetch a single batch
            for batch_idx, (input_data, target_data) in enumerate(dataloader):
                print(f"Batch {batch_idx + 1}")
                print(f"Input Data Shape: {input_data.shape}")
                print(f"Target Data Shape: {target_data.shape}")
                break

        root_path = '/data/BeGAN-circuit-benchmarks/asap7/numpy_data'  # Replace with actual path
        target_layers = ['m2', 'm5', 'm6', 'm7', 'm8', 'm25', 'm56', 'm67', 'm78']
        # Run the test function
        test_asap7_dataset2(root_path, target_layers, img_size=512, use_irreg=False, preload=False)
    # test5()

    def test6_can_train_asap7():
        td,vd = build_dataset_began_asap7(fintune=False,train=True)
        td.__getitem__(1)
        vd.__getitem__(1)
        print(td.__len__(),vd.__len__())

    # test6_can_train_asap7()
    def test7_cross_val_asap7():
        cross_val_ids = [0, 1, 2, 3]  # 4-fold cross validation
        n_splits = len(cross_val_ids)

        # Iterate through each cross-validation fold
        for cross_val_id in cross_val_ids:
            train_dt,val_dt = build_dataset_asap7_cross_val(cross_val_id,n_splits) 
            print('-'*50)
            print(f'train_{cross_val_id}')
            for i in range(len(train_dt)):
                print(train_dt.__getitem__(i)[-1])
            print('-'*50)
            print(f'val_{cross_val_id}')
            for i in range(len(val_dt)):
                print(val_dt.__getitem__(i)[-1])

    def test7_1_cross_val_get_case():
        # Example usage
        cross_val_id = 0
        val_case_name = get_val_casename(cross_val_id)
        logdir = os.path.join('/log', f'arch/dataset/loss/cross_val_{cross_val_id}_{val_case_name}')
        print(logdir)        
    # test7_1_cross_val_get_case()

    # test1()
    # test3()

    def test_8_5nm():
        td, vd = build_dataset_5m()


    root_path = "/data/pdn_3rd_4types" # fine 
    selected_folders = ['200nm_numpy']
    post_fix = ""
    dataset = IRDropDataset5nm(root_path=root_path,
                               img_size=256,
                               train=True,
                                selected_folders=selected_folders,
                                post_fix_path=post_fix,
                                in_ch=3,
                                use_raw=False,
                                train_auto_encoder=False)
    
    root_path = '/data/pdn_3rd_4types'
    selected_folders = ['200nm_numpy']
    post_fix = ""

    dataset = IRDropDataset5nm(root_path=root_path,
                                selected_folders=selected_folders,
                                post_fix_path=post_fix,
                                train=True,
                                in_ch=3)
    print(dataset.__len__())

    