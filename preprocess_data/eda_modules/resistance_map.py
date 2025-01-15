import os
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

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

VIA_LAYERS_ALL = {
    ('m0', 'm1'): 'm01',
    ('m1', 'm2'): 'm12',
    ('m1', 'm3'): 'm13',
    ('m1', 'm4'): 'm14',
    ('m1', 'm5'): 'm15',
    ('m1', 'm6'): 'm16',
    ('m1', 'm7'): 'm17',
    ('m1', 'm8'): 'm18',
    ('m1', 'm9'): 'm19',
    ('m2', 'm3'): 'm23',
    ('m2', 'm4'): 'm24',
    ('m2', 'm5'): 'm25',
    ('m2', 'm6'): 'm26',
    ('m2', 'm7'): 'm27',
    ('m2', 'm8'): 'm28',
    ('m2', 'm9'): 'm29',
    ('m3', 'm4'): 'm34',
    ('m3', 'm5'): 'm35',
    ('m3', 'm6'): 'm36',
    ('m3', 'm7'): 'm37',
    ('m3', 'm8'): 'm38',
    ('m3', 'm9'): 'm39',
    ('m4', 'm5'): 'm45',
    ('m4', 'm6'): 'm46',
    ('m4', 'm7'): 'm47',
    ('m4', 'm8'): 'm48',
    ('m4', 'm9'): 'm49',
    ('m5', 'm6'): 'm56',
    ('m5', 'm7'): 'm57',
    ('m5', 'm8'): 'm58',
    ('m5', 'm9'): 'm59',
    ('m6', 'm7'): 'm67',
    ('m6', 'm8'): 'm68',
    ('m6', 'm9'): 'm69',
    ('m7', 'm8'): 'm78',
    ('m7', 'm9'): 'm79',
    ('m8', 'm9'): 'm89'
}

class EDA:
    def __init__(self, current_map_path, sp_file_path, output_dir,
                  show_only=False, grid_mode=False, grid_col=3, show_current=False, voltage_map_path='', show_all_layers=False, via_layer_map=VIA_LAYERS_ALL,
                  match_str:str=None):
        self.current_map_path = current_map_path
        self.sp_file_path = sp_file_path
        self.output_dir = output_dir
        self.show_only = show_only
        self.grid_mode = grid_mode
        self.grid_col = grid_col
        self.show_current = show_current
        self.voltage_map_path = voltage_map_path
        self.show_all_layers = show_all_layers
        self.via_layer_map = via_layer_map
        self.target_layers = None
        self.resolution = None
        os.makedirs(output_dir, exist_ok=True)
        self.match_str = match_str

    def parse_spice_file(self):
        # Determine target layers from SPICE file if not provided
        all_layers = set()
        layer_data = {}

        open_func = gzip.open if self.sp_file_path.endswith('.gz') else open
        mode = 'rt' if self.sp_file_path.endswith('.gz') else 'r'

        match_str = r"R\d+\s+n1_(\w+)_(\d+)_(\d+)\s+n1_(\w+)_(\d+)_(\d+)\s+([\d.]+)" if self.match_str is None else self.match_str

        with open_func(self.sp_file_path, mode) as file:
            for line in file:
                line = line.replace('met','m')
                
                match = re.match(match_str, line, re.IGNORECASE)
                if match:
                    layer1 = match.group(1).lower()
                    layer2 = match.group(4).lower()
                    all_layers.update([layer1, layer2])

        if not self.target_layers:
            if self.show_all_layers:
                self.target_layers = [f'm{i}' for i in range(1, 10)] + list(self.via_layer_map.values())
            else:
                self.target_layers = list(all_layers)
                self.target_layers += [self.via_layer_map.get((l1, l2)) for l1, l2 in self.via_layer_map.keys() if l1 in self.target_layers and l2 in self.target_layers]

        for layer in self.target_layers:
            layer_data[layer] = []

        # Re-read and store relevant layer data
        with open_func(self.sp_file_path, mode) as file:
            for line in file:
                line = line.replace('met','m')
                match = re.match(match_str, line, re.IGNORECASE)
                if match:
                    layer1 = match.group(1).lower()
                    layer2 = match.group(4).lower()
                    x1, y1 = int(match.group(2)), int(match.group(3))
                    x2, y2 = int(match.group(5)), int(match.group(6))
                    resistance = float(match.group(7))

                    if layer1 in self.target_layers and layer2 in self.target_layers and layer1 == layer2:
                        layer_data[layer1].append(((x1, y1), (x2, y2), resistance))
                    elif layer1 in self.target_layers and layer2 in self.target_layers:
                        via_layer = self.via_layer_map.get((layer1, layer2)) or self.via_layer_map.get((layer2, layer1))
                        if via_layer:
                            layer_data[via_layer].append(((x1, y1), (x2, y2), resistance))

        return layer_data

    def get_csv_matrix_size(self):
        open_func = gzip.open if self.current_map_path.endswith('.gz') else open

        with open_func(self.current_map_path, 'rt') as f:
            matrix = pd.read_csv(f, header=None)
        
        self.resolution = matrix.shape
        return matrix.shape

    def convert_to_matrix_indices(self, layer_resistances):
        matrix = np.zeros((self.resolution[0], self.resolution[1]))

        for (x1, y1), (x2, y2), resistance in layer_resistances:
            x1, y1 = x1 // 2000, y1 // 2000
            x2, y2 = x2 // 2000, y2 // 2000
            
            if 0 <= x1 < self.resolution[1] and 0 <= y1 < self.resolution[0]:
                matrix[x1, y1] += resistance
            if 0 <= x2 < self.resolution[1] and 0 <= y2 < self.resolution[0]:
                matrix[x2, y2] += resistance
        
        return matrix

    def save_matrix_as_image(self, matrices, layers):
        if self.grid_mode:
            if self.show_current:
                layers.append('current_map')
                current_matrix = pd.read_csv(self.current_map_path, header=None).to_numpy()
                matrices.append(current_matrix)

            if self.voltage_map_path:
                layers.append('voltage_map')
                voltage_matrix = pd.read_csv(self.voltage_map_path, header=None).to_numpy()
                matrices.append(voltage_matrix)

            num_layers = len(layers)
            num_cols = self.grid_col
            num_rows = (num_layers + num_cols - 1) // num_cols

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 6))
            axes = axes.flatten() if num_layers > 1 else [axes]

            for i, (matrix, layer) in enumerate(zip(matrices, layers)):
                ax = axes[i]
                cmap = 'viridis' if layer in ['current_map', 'voltage_map'] else 'hot'
                im = ax.imshow(matrix, cmap=cmap, interpolation='nearest')
                fig.colorbar(im, ax=ax, label='Current (A)' if layer == 'current_map' else ('Voltage (V)' if layer == 'voltage_map' else 'Resistance (ohms)'))
                ax.set_title(f'Map for {layer.upper()}')

            # Hide any unused subplots
            for j in range(len(layers), len(axes)):
                fig.delaxes(axes[j])

            if self.show_only:
                plt.show()
            else:
                output_image_path = os.path.join(self.output_dir, 'resistance_maps_grid.png')
                plt.savefig(output_image_path)
                plt.close()
        else:
            for matrix, layer in zip(matrices, layers):
                plt.figure(figsize=(6, 6))
                cmap = 'viridis' if layer in ['current_map', 'voltage_map'] else 'hot'
                plt.imshow(matrix, cmap=cmap, interpolation='nearest')
                plt.colorbar(label='Current (A)' if layer == 'current_map' else ('Voltage (V)' if layer == 'voltage_map' else 'Resistance (ohms)'))
                plt.title(f'Map for {layer.upper()}')

                if self.show_only:
                    plt.show()
                else:
                    output_file_path = os.path.join(self.output_dir, f'{layer}_map.png')
                    plt.savefig(output_file_path)
                    plt.close()

    def process_spice_file(self):
        layer_data = self.parse_spice_file()
        self.get_csv_matrix_size()

        matrices = []
        for layer in self.target_layers:
            resistance_matrix = self.convert_to_matrix_indices(layer_data[layer])
            matrices.append(resistance_matrix)
            if not self.show_only:
                output_file_path = os.path.join(self.output_dir, f'{layer}_resistance_matrix.npy')
                np.save(output_file_path, resistance_matrix)

        # Save as image or display
        self.save_matrix_as_image(matrices, self.target_layers)

        print(f"Process completed. Resistance matrices {'displayed' if self.show_only else 'saved in ' + self.output_dir}.")


if __name__ == '__main__':
    def test_1():
        # Example usage:
        root = '/data/real-circuit-benchmarks/asap7/data/'
        current_map_path = 'ibex_current_map.csv.gz'
        sp_file_path = 'ibex_reg_grid.sp.gz'
        # root = '/data/real-circuit-benchmarks/nangate45/data/'
        # current_map_path = 'aes_current_reg.csv.gz'
        # sp_file_path = 'aes_reg.sp.gz'
        eda = EDA(current_map_path=os.path.join(root, current_map_path),
                sp_file_path=os.path.join(root, sp_file_path),
                output_dir='output_dir', show_only=True, grid_mode=True, grid_col=3, show_current=True, 
                show_all_layers=True, via_layer_map=VIA_LAYERS_ASAP7)
        eda.process_spice_file()
