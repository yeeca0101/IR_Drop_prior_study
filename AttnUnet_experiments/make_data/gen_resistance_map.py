'''
author : Youngmin Seo
         yeeca0401@gmail.com
Extract the Resistance maps from .sp file(netlist) 
         
'''

import os
import glob
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

# Define via layer maps as constants
VIA_LAYERS_NANGATE45 = {
    ('m1', 'm4'): 'm14',
    ('m4', 'm7'): 'm47',
    ('m7', 'm8'): 'm78',
    ('m8', 'm9'): 'm89'
}

VIA_LAYERS_ASAP7 = {
    ('m2', 'm5'): 'm25',
    ('m5', 'm6'): 'm56',
    ('m6', 'm7'): 'm67',
    ('m7', 'm8'): 'm78'
}

# you can consider the via_layer_map argument

class ResistanceMapExtractor:
    def __init__(self, file_path, target_layers, csv_files, output_dir='layer_data', 
                 show_only=False,fig_out_dir='./results',via_layer_map=VIA_LAYERS_ASAP7, prefix=""):
        self.file_path = file_path
        self.target_layers = target_layers
        self.csv_files = csv_files
        self.output_dir = output_dir
        self.fig_out_dir = fig_out_dir
        self.show_only = show_only
        self.via_layer_map = via_layer_map
        self.prefix = prefix
        os.makedirs(output_dir, exist_ok=True)
        self.dataset_statistics = {layer: [] for layer in target_layers}

    def parse_spice_file(self):
        via_layer_map = self.via_layer_map
        all_layers = self.target_layers + list(via_layer_map.values())
        layer_data = {layer: [] for layer in all_layers}
        
        open_func = gzip.open if self.file_path.endswith('.gz') else open
        mode = 'rt' if self.file_path.endswith('.gz') else 'r'

        with open_func(self.file_path, mode) as file:
            for line in file:
                match = re.match(r"R\d+\s+n1_(\w+)_(\d+)_(\d+)\s+n1_(\w+)_(\d+)_(\d+)\s+([\d.]+)", line, re.IGNORECASE)

                if match:
                    layer1 = match.group(1).lower()
                    layer2 = match.group(4).lower()
                    x1, y1 = int(match.group(2)), int(match.group(3))
                    x2, y2 = int(match.group(5)), int(match.group(6))
                    resistance = float(match.group(7))

                    if layer1 in self.target_layers and layer2 in self.target_layers and layer1 == layer2:
                        layer_data[layer1].append(((x1, y1), (x2, y2), resistance))
                    elif layer1 in self.target_layers and layer2 in self.target_layers:
                        via_layer = via_layer_map.get((layer1, layer2)) or via_layer_map.get((layer2, layer1))
                        if via_layer:
                            layer_data[via_layer].append(((x1, y1), (x2, y2), resistance))

        return layer_data

    def get_csv_matrix_size(self, file_path):
        open_func = gzip.open if file_path.endswith('.gz') else open
        
        with open_func(file_path, 'rt') as f:
            matrix = pd.read_csv(f, header=None)
        
        return matrix.shape

    def convert_to_matrix_indices(self, layer_resistances, matrix_size=(100, 100)):
        matrix = np.zeros((matrix_size[0], matrix_size[1]))
        
        for (x1, y1), (x2, y2), resistance in layer_resistances:
            x1, y1 = x1 // 2000, y1 // 2000
            x2, y2 = x2 // 2000, y2 // 2000
            
            if 0 <= x1 < matrix_size[1] and 0 <= y1 < matrix_size[0]:
                matrix[x1, y1] += resistance
            if 0 <= x2 < matrix_size[1] and 0 <= y2 < matrix_size[0]:
                matrix[x2, y2] += resistance
        
        return matrix

    def collect_statistics(self, matrix, layer_name):
        """Only collect statistics without visualization"""
        self.dataset_statistics[layer_name].append({
            'mean': matrix.mean(),
            'max': matrix.max(),
            'min': matrix.min(),
            'values': matrix.flatten()
        })

    def process_spice_file(self):
        layer_data = self.parse_spice_file()
        resolution = self.get_csv_matrix_size(self.csv_files['current_map'])
        os.makedirs(self.output_dir, exist_ok=True)
        
        for layer in self.target_layers:
            resistance_matrix = self.convert_to_matrix_indices(layer_data[layer], matrix_size=resolution)
            
            if not self.show_only:
                output_file_path = os.path.join(self.output_dir, f'{self.prefix}_{layer}_resistance_matrix.npy')
                np.save(output_file_path, resistance_matrix)
            
            # Only collect statistics without individual visualization
            self.collect_statistics(resistance_matrix, layer)

        print(f"Process completed. Results saved in {self.output_dir}")

    def visualize_dataset_statistics(self):
        """Visualize dataset-level statistics using histogram and statistical lines"""
        for layer, stats_list in self.dataset_statistics.items():
            if not stats_list:
                continue
            
            # Collect all values and statistics
            all_values = np.concatenate([stats['values'] for stats in stats_list])
            mean_value = np.mean(all_values)
            max_value = np.max(all_values)
            min_value = np.min(all_values)
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Plot histogram with KDE
            sns.histplot(all_values, bins=50, kde=True)
            
            # Add statistical lines
            plt.axvline(mean_value, color='r', linestyle='--', 
                       label=f'Mean: {mean_value:.4f}')
            plt.axvline(max_value, color='g', linestyle='-', 
                       label=f'Max: {max_value:.4f}')
            plt.axvline(min_value, color='b', linestyle='-', 
                       label=f'Min: {min_value:.4f}')
            
            # Add titles and labels
            plt.title(f"Layer {layer} Dataset Statistics", fontsize=14)
            plt.xlabel("Resistance (ohms)", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            
            # Customize legend and grid
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Optional: Add additional statistics text box
            stats_text = f'Total Samples: {len(all_values):,}\n'
            stats_text += f'Std Dev: {np.std(all_values):.4f}\n'
            stats_text += f'Median: {np.median(all_values):.4f}'
            plt.text(0.95, 0.95, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            if self.show_only:
                plt.show()
            else:
                os.makedirs(self.fig_out_dir,exist_ok=True)
                output_image_path = os.path.join(self.fig_out_dir, f'{self.prefix}_{layer}_dataset_statistics.png')
                plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
                plt.close()

# 전체 폴더 내 모든 SPICE 파일 처리 함수
def process_all_sp_files_train(root_dir, output_dir, show_only=False,target_layers=['m1', 'm4', 'm7', 'm8', 'm9', 'm14', 'm47', 'm78', 'm89']):
    sp_files = [f for f in os.listdir(root_dir) if f.endswith('.sp') or f.endswith('.sp.gz')]
    
    for sp_file in sp_files:
        # Extract prefix from the SPICE file name
        prefix = '_'.join(sp_file.split('_')[:2])
        
        # Find matching CSV files for the prefix
        csv_files = {
            'current_map': glob.glob(os.path.join(root_dir, f"{prefix}*current*.csv.gz"))[0],
        }

        # Check if all required CSV files exist
        if not all(os.path.exists(csv_file) for csv_file in csv_files.values()):
            print(f"Warning: Not all CSV files found for prefix {sp_file}. Skipping...")
            continue

        sp_file_path = os.path.join(root_dir, sp_file)
        print(f"Processing {sp_file} with CSV files {csv_files}...")
        
        # Set output directory path specific to each SPICE file
        file_output_dir = os.path.join(output_dir,'layer_data')
        
        # Process the SPICE file with the matching CSV files and resolution
        extractor = ResistanceMapExtractor(sp_file_path, target_layers, csv_files, output_dir=file_output_dir, show_only=show_only, prefix=prefix)
        extractor.process_spice_file()

    # Visualize dataset level statistics after processing all SPICE files
    extractor.visualize_dataset_statistics()

# 전체 폴더 내 모든 SPICE 파일 처리 함수 (ASAP7)
def process_all_sp_files_asap7_testcase(root_dir, output_dir, show_only=False):
    # List all SPICE files in the directory
    sp_files = [f for f in os.listdir(root_dir) if f.endswith('.sp') or f.endswith('.sp.gz')]
    
    # Target layers to process
    target_layers = ['m2', 'm5', 'm6', 'm7', 'm8', 'm25', 'm56', 'm67', 'm78']
    
    # Process each SPICE file separately
    for sp_file in sp_files:
        # Extract prefix and grid type (irreg or reg) from the SPICE file name
        parts = sp_file.split('_')
        prefix = parts[0]
        grid_type = 'irreg' if 'irreg' in parts else 'reg'

        if prefix in 'dynamic_node':
            prefix = 'dynamic_node'
        
        # Define patterns to locate each relevant CSV file for the current prefix
        csv_files = {
            'current_map': glob.glob(os.path.join(root_dir, f"{prefix}_current_map.csv.gz")),
            'voltage_map_irregular': glob.glob(os.path.join(root_dir, f"{prefix}_voltage_map_irregular.csv.gz")),
            'voltage_map_regular': glob.glob(os.path.join(root_dir, f"{prefix}_voltage_map_regular.csv.gz")),
            'irreg_grid': glob.glob(os.path.join(root_dir, f"{prefix}_irreg_grid.sp.gz")),
            'reg_grid': glob.glob(os.path.join(root_dir, f"{prefix}_reg_grid.sp.gz"))
        }

        # Flatten the lists from glob to select the first matched file or None if no file is found
        csv_files = {key: files[0] if files else None for key, files in csv_files.items()}
        
        # Print missing files information
        missing_files = [key for key, file in csv_files.items() if file is None]
        if missing_files:
            print(f"Warning: Missing files for prefix '{prefix}': {', '.join(missing_files)}. Proceeding with available files.")
        
        # Set the SPICE file path
        sp_file_path = os.path.join(root_dir, sp_file)
        
        # Create specific output directory based on prefix and grid type
        file_output_dir = os.path.join(output_dir, prefix, 'layer_data', grid_type)
        os.makedirs(file_output_dir, exist_ok=True)
        
        print(f"Processing {sp_file} with available CSV files {csv_files}...")

        # Process the SPICE file with the matching CSV files (only available files)
        extractor = ResistanceMapExtractor(sp_file_path, target_layers, csv_files, output_dir=file_output_dir, show_only=show_only)
        extractor.process_spice_file()

    # Visualize dataset level statistics after processing all SPICE files
    extractor.visualize_dataset_statistics()
