import os
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

inp_means = torch.tensor([6.2446e-08, 1.4142e+01, 1.4631e+00])
inp_stds = torch.tensor([4.9318e-08, 7.2629e+00, 1.1194e+00])

target_mean = torch.tensor([0.0011250363299623132])
target_std = torch.tensor([4.9318e-08])
# target_std = torch.tensor([4.9318e-08, 7.2629e+00, 1.1194e+00])


# class IRDropDataset(Dataset):
#     def __init__(self, root_path, selected_folders, target_layers, 
#                  csv_path=None, img_size=512, post_fix_path='data', preload=False,train=True,pdn_density_p=0.0,
#                  asap7=False,pdn_zeros=False,in_ch=12,return_case=None):
#         """
#         Args:
#             root_path (str): The root directory where datasets are stored.
#             selected_folders (list of str): A list of folder names to include (e.g., ['asap7', 'sky130hd']).
#             target_layers (list of str): A list of layer names (e.g., ['m1', 'm4', 'm7']).
#             csv_path (str): Path to the CSV file to load/save file paths. If provided, load paths from this file.
#             preload (bool): If True, preload all data into memory.
#         """
#         self.root_path = root_path
#         self.selected_folders = selected_folders
#         self.target_layers = target_layers
#         self.csv_path = csv_path
#         self.target_size = img_size
#         self.post_fix = post_fix_path
#         self.preload = preload  # 메모리 캐싱 여부
#         self.cached_data = None  # 캐시된 데이터 저장용
#         self.train = train
#         self.pdn_density_p = pdn_density_p
#         self.is_asap7 = asap7
#         self.pdn_zeros = pdn_zeros
#         self.in_ch=in_ch
#         self.return_case = return_case
        
#         self.transform = A.Compose([
#             A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
#             A.OneOf([
#                 A.HorizontalFlip(p=1),
#                 A.VerticalFlip(p=1),
#                 A.Rotate(limit=(90, 90), p=1),
#                 A.Rotate(limit=(180, 180), p=1),
#                 A.Rotate(limit=(270, 270), p=1),
#                 A.NoOp(p=1)  # 아무 변환도 적용하지 않을 확률
#             ], p=1),   # OneOf 내의 변환 중 하나를 선택
#             ToTensorV2()
#         ]) if self.train else A.Compose([
#             A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
#             ToTensorV2()
#         ])

#         if csv_path and os.path.exists(csv_path):
#             # Load file paths from the CSV file
#             self.data_files = self._load_from_csv(csv_path)
#         else:
#             # Generate the file paths from the folders and save them to CSV
#             self.data_files = self._find_files()
#             if csv_path:
#                 self._save_to_csv(self.data_files, csv_path)

#         if preload:
#             # 데이터를 미리 메모리에 로드
#             self.cached_data = self._preload_data()

#     def _find_files(self):
#         data_files = []

#         # Initialize these lists outside the loop to accumulate paths from all folders
#         current_files = []
#         eff_dist_files = []
#         pdn_density_files = []
#         ir_drop_files = []
#         resistance_files_dict = {layer: [] for layer in self.target_layers}

#         # Iterate through selected folders and accumulate the file paths
#         for folder in self.selected_folders:
#             folder_path = os.path.join(self.root_path, folder,self.post_fix)
#             # Extend to accumulate files from all selected folders
#             current_files.extend(glob.glob(os.path.join(folder_path, '*_current*.npy')))
#             if self.is_asap7:
#                 ir_drop_files.extend(glob.glob(os.path.join(folder_path, '*_voltage_map_regular.npy')))    
#             else:
#                 eff_dist_files.extend(glob.glob(os.path.join(folder_path, '*_eff_dist.npy')))
#                 pdn_density_files.extend(glob.glob(os.path.join(folder_path, '*_pdn_density.npy')))  # Extend instead of assign
#                 ir_drop_files.extend(glob.glob(os.path.join(folder_path, '*_ir_drop.npy')))      # Extend instead of assign

#             # Collect resistance matrix files per layer, and extend for each folder
            
#             for layer in self.target_layers:
#                 resistance_files_dict[layer].extend(glob.glob(os.path.join(folder_path,'layer_data', f'*{layer}_resistance*.npy')))

#         # Sort the accumulated file lists outside the loop
#         current_files.sort()
#         eff_dist_files.sort()
#         pdn_density_files.sort()
#         ir_drop_files.sort()
#         for layer in self.target_layers:
#             resistance_files_dict[layer].sort()  # Sort resistance files for each layer

#         # Ensure that the lengths of all file lists match
#         if (not (len(current_files) == len(eff_dist_files) == len(pdn_density_files) == len(ir_drop_files)) and (not self.is_asap7)):
#             raise ValueError("Mismatch in the number of current, eff_dist, pdn_density, and ir_drop files!")

#         # Create data groups by combining the corresponding files
#         if self.is_asap7:
#             for current, ir_drop in zip(current_files, ir_drop_files):
#                 # Add the matching resistance files for each layer
#                 resistances = {layer: resistance_files_dict[layer] for layer in self.target_layers}
#                 data_files.append({
#                     'current': current,
#                     'ir_drop': ir_drop,
#                     'resistances': resistances
#                 })
#         else:    
#             for current, eff_dist, pdn_density, ir_drop in zip(current_files, eff_dist_files, pdn_density_files, ir_drop_files):
#                 # Add the matching resistance files for each layer
#                 resistances = {layer: resistance_files_dict[layer] for layer in self.target_layers}
#                 data_files.append({
#                     'current': current,
#                     'eff_dist': eff_dist,
#                     'pdn_density': pdn_density,
#                     'ir_drop': ir_drop,
#                     'resistances': resistances
#                 })

#         return data_files

#     def _save_to_csv(self, data_files, csv_path):
#         csv_folder = os.path.join(*(csv_path.split('/')[:-1]))
#         os.makedirs(csv_folder, exist_ok=True)

#         data_list = []
#         for files in data_files:
#             res_str = {layer: ",".join(files['resistances'][layer]) for layer in self.target_layers}
#             if self.is_asap7:
#                 data_list.append([files['current'], files['ir_drop']] + [res_str[layer] for layer in self.target_layers])
#             else:
#                 data_list.append([files['current'], files['eff_dist'], files['pdn_density'], files['ir_drop']] + [res_str[layer] for layer in self.target_layers])
#         if self.is_asap7:
#             columns = ['current', 'ir_drop'] + [f'resistance_{layer}' for layer in self.target_layers]
#         else:
#             columns = ['current', 'eff_dist', 'pdn_density', 'ir_drop'] + [f'resistance_{layer}' for layer in self.target_layers]
#         df = pd.DataFrame(data_list, columns=columns)
#         df.to_csv(csv_path, index=False)

#     def _load_from_csv(self, csv_path):
#         df = pd.read_csv(csv_path)
#         data_files = []
#         for _, row in df.iterrows():
#             res_dict = {layer: row[f'resistance_{layer}'].split(',') for layer in self.target_layers}
#             if self.is_asap7:
#                 data_files.append({
#                     'current': row['current'],
#                     'ir_drop': row['ir_drop'],
#                     'resistances': res_dict
#                 })
#             else:
#                 data_files.append({
#                     'current': row['current'],
#                     'eff_dist': row['eff_dist'], 
#                     'pdn_density': row['pdn_density'],
#                     'ir_drop': row['ir_drop'],
#                     'resistances': res_dict
#                 })
#         return data_files

#     def _preload_data(self):
#         """ Preload data into memory for faster access. """
#         cached_data = []
#         for idx in range(len(self.data_files)):
#             input_data, target_data = self._load_data_from_disk(idx)
#             cached_data.append((input_data, target_data))
#         return cached_data

#     def _load_data_from_disk(self, idx):
#         """ Helper function to load data from disk for a given index. """
#         file_group = self.data_files[idx]
#         current = np.load(file_group['current'])
#         if self.is_asap7 or self.pdn_zeros:
#             eff_dist = np.ones_like(current)
#         else:    
#             eff_dist = np.load(file_group['eff_dist'])
#         # ir_drop = self._min_max_norm(np.load(file_group['ir_drop']))
#         ir_drop = np.load(file_group['ir_drop'])
#         target_shape = current.shape[:2]  # current의 높이와 너비만 사용
#         ir_drop = cv2.resize(ir_drop, (target_shape[1], target_shape[0]))
        
#         if ((np.random.random() < self.pdn_density_p) and self.train) or self.is_asap7 or self.pdn_zeros:
#             pdn_density = np.ones_like(current)
#         else:
#             pdn_density = np.load(file_group['pdn_density'])

#         resistance_stack = []
#         for layer in self.target_layers:
#             res_file = file_group['resistances'][layer][idx]  # idx에 맞는 resistance 데이터 가져옴
#             resistance_stack.append(np.load(res_file))
        
#         resistance_stack = np.stack(resistance_stack, axis=-1)
#         input_data = np.stack([current, eff_dist, pdn_density], axis=-1)
#         input_data = np.concatenate([input_data, resistance_stack], axis=-1)
        
#         return input_data, ir_drop

#     def _load_data_from_disk_2ch(self, idx):
#         """ Helper function to load data from disk for a given index. """
#         file_group = self.data_files[idx]
#         current = self._norm(np.load(file_group['current']))
        
#         ir_drop = np.load(file_group['ir_drop'])
#         ir_drop = self._min_max_norm(ir_drop)

#         resistance_stack = []
#         for layer in self.target_layers:
#             res_file = file_group['resistances'][layer][idx]  # idx에 맞는 resistance 데이터 가져옴
#             resistance_stack.append(np.load(res_file))
#         resistance_stack = np.stack(resistance_stack, axis=-1).sum(-1)
#         resistance_total = self._norm(resistance_stack)
            
#         input_data = np.stack([current, resistance_total], axis=-1)

#         return input_data, ir_drop
    
#     def __len__(self):
#         return len(self.data_files)
    
#     def parse_testcase_name(self,idx):
#         testcase_name = [_ for _ in self.data_files[idx]['current'].split('/') if 'current' in _][0]
#         testcase_name = testcase_name.split('_current_map.npy')[0]
#         return testcase_name
    
#     def __getitem__(self, idx):
#         input_data, ir_drop = self._load_data_from_disk(idx)

#         transformed = self.transform(image=input_data, mask=ir_drop)
#         input_tensor = transformed['image']
#         target_tensor = transformed['mask']
#         if self.in_ch==1:
#             input_tensor = input_tensor[0].unsqueeze(0)

#         if self.return_case:
#             return input_tensor.float(), target_tensor.float(), self.parse_testcase_name(idx)
#         return input_tensor.float(), target_tensor.float()

#     def get_ori(self,idx):
#         input_data, ir_drop = self._load_data_from_disk(idx)
#         if self.return_case:
#             return input_data, ir_drop, self.parse_testcase_name()
#         return input_data, ir_drop
    
    
class NormalizeTransform:
    def __init__(self, input_means, input_stds, target_mean, target_std):
        self.input_means = input_means
        self.input_stds = input_stds
        self.target_mean = target_mean
        self.target_std = target_std

    def __call__(self, input_data, target):
        normalized_input = (input_data - self.input_means[:, None, None]) / self.input_stds[:, None, None]
        normalized_target = (target - self.target_mean) / self.target_std
        return normalized_input, normalized_target


class CustomDataset(Dataset):
    def __init__(self, root_path, selected_folders, csv_path=None,img_size=256):
        """
        Args:
            root_path (str): The root directory where datasets are stored.
            selected_folders (list of str): A list of folder names to include (e.g., ['asap7', 'sky130hd']).
            csv_path (str): Path to the CSV file to load/save file paths. If provided, load paths from this file.
        """
        self.root_path = root_path
        self.selected_folders = selected_folders
        self.csv_path = csv_path
        self.target_size = img_size
        self.norm_transform = NormalizeTransform(inp_means, inp_stds, target_mean, target_std)  # Assuming defined elsewhere
        
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=inp_means.tolist(), std=inp_stds.tolist()),
            ToTensorV2()
        ])
        
        # self.transform = A.Compose([
        #     A.Resize(img_size, img_size,interpolation=cv2.INTER_CUBIC),
        #     A.HorizontalFlip(p=0.5),
        #     A.VerticalFlip(p=0.5),
        #     A.Rotate((90,90),p=0.5),
        #     A.Rotate((180,180),p=0.5),
        #     A.Rotate((270,270),p=0.5),
        #     ToTensorV2()
        # ])

        self.target_transform = A.Compose([
            A.Resize(img_size, img_size),
            ToTensorV2()
        ])

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
        
        # Iterate over each selected folder
        for folder in self.selected_folders:
            folder_path = os.path.join(self.root_path, folder,'data')
            # Find the relevant files
            # current_files = glob.glob(os.path.join(folder_path, '*_current.csv.gz'))
            # eff_dist_files = glob.glob(os.path.join(folder_path, '*_eff_dist.csv.gz'))
            # pdn_density_files = glob.glob(os.path.join(folder_path, '*_pdn_density.csv.gz'))
            # ir_drop_files = glob.glob(os.path.join(folder_path, '*_ir_drop.csv.gz'))
            
            current_files = glob.glob(os.path.join(folder_path, '*_current.npy'))
            eff_dist_files = glob.glob(os.path.join(folder_path, '*_eff_dist.npy'))
            pdn_density_files = glob.glob(os.path.join(folder_path, '*_pdn_density.npy'))
            ir_drop_files = glob.glob(os.path.join(folder_path, '*_ir_drop.npy'))
            
            # Sort the files to ensure they are aligned
            current_files.sort()
            eff_dist_files.sort()
            pdn_density_files.sort()
            ir_drop_files.sort()

            # Append each set of files as a dictionary to the data_files list
            for current, eff_dist, pdn_density, ir_drop in zip(current_files, eff_dist_files, pdn_density_files, ir_drop_files):
                data_files.append({
                    'current': current,
                    'eff_dist': eff_dist,
                    'pdn_density': pdn_density,
                    'ir_drop': ir_drop
                })
        return data_files

    def _save_to_csv(self, data_files, csv_path):
        """
        Save the list of file paths to a CSV file.
        """
        csv_folder = os.path.join(*(csv_path.split('/')[:-1]))
        print(csv_folder)
        os.makedirs(csv_folder,exist_ok=True)

        data_list = []
        for files in data_files:
            data_list.append([files['current'], files['eff_dist'], files['pdn_density'], files['ir_drop']])
        
        df = pd.DataFrame(data_list, columns=['current', 'eff_dist', 'pdn_density', 'ir_drop'])
        df.to_csv(csv_path, index=False)

    def _load_from_csv(self, csv_path):
        """
        Load file paths from a CSV file.
        """
        df = pd.read_csv(csv_path)
        data_files = []
        for _, row in df.iterrows():
            data_files.append({
                'current': row['current'],
                'eff_dist': row['eff_dist'],
                'pdn_density': row['pdn_density'],
                'ir_drop': row['ir_drop']
            })
        print(f"File paths loaded from {csv_path}")
        return data_files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_group = self.data_files[idx]
        
        # CSV 파일 로드
        # current = pd.read_csv(file_group['current'], compression='gzip').to_numpy()
        # eff_dist = pd.read_csv(file_group['eff_dist'], compression='gzip').to_numpy()
        # pdn_density = pd.read_csv(file_group['pdn_density'], compression='gzip').to_numpy()
        # ir_drop = pd.read_csv(file_group['ir_drop'], compression='gzip').to_numpy()
        
        current = np.load(file_group['current'])
        eff_dist = np.load(file_group['eff_dist'])
        pdn_density = np.load(file_group['pdn_density'])
        ir_drop = np.load(file_group['ir_drop'])
        
        # 입력 데이터 변환
        input_data = np.stack([current, eff_dist, pdn_density], axis=-1)
        transformed = self.transform(image=input_data)
        input_tensor = transformed['image']
        
        # 타겟 데이터 변환
        target_transformed = self.target_transform(image=ir_drop)
        target_tensor = target_transformed['image']
        target_tensor= target_tensor/target_tensor.max()

        return input_tensor, target_tensor.float()



def split_dataset(dataset, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, random_state=42):
    assert np.isclose(train_ratio + valid_ratio + test_ratio, 1.0), " 1.0"
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    train_valid_size = int((train_ratio + valid_ratio) * dataset_size)
    train_valid_indices, test_indices = train_test_split(indices, train_size=train_valid_size, random_state=random_state)
    
    train_size = int(train_ratio * dataset_size)
    train_indices, valid_indices = train_test_split(train_valid_indices, train_size=train_size, random_state=random_state)
    
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, valid_dataset, test_dataset

def build_dataset(root_path='/data/BeGAN-circuit-benchmarks',img_size=256):
    dataset = CustomDataset(root_path=root_path,
                            selected_folders=['nangate45/set1_numpy','nangate45/set2_numpy'],
                            csv_path='./csv/total_numpy.csv',
                            img_size=img_size
                      )
    return split_dataset(dataset)


if __name__ =='__main__':
    from torch.utils.data import DataLoader

    dt = CustomDataset(root_path='/data/BeGAN-circuit-benchmarks',
                            selected_folders=['nangate45/set1_numpy','nangate45/set2_numpy'],
                            csv_path='./csv/total_numpy.csv',
                            img_size=32
                      )
    print(dt.data_files)
    print(dt.__getitem__(1)[0].shape,dt.__getitem__(1)[1].shape)

    train_loader = DataLoader(dt,batch_size=32,shuffle=True,drop_last=True)
    for batch in train_loader:
        inp, target = batch
        print(inp.shape,target.shape)