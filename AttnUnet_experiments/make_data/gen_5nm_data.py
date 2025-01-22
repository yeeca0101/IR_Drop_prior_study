import os
import numpy as np
import pandas as pd
import re

class DatasetConverter:
    def __init__(self, base_path, resolution_folder, dbu_per_pixel):
        self.base_path = base_path
        self.resolution_folder = resolution_folder
        self.dbu_per_pixel = dbu_per_pixel
        self.output_dir = os.path.join(base_path, f"{resolution_folder}_numpy")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "layer_data"), exist_ok=True)

    def get_csv_matrix_size(self, index):
        file_name = f"{index}_power.csv"
        csv_path = os.path.join(self.base_path, self.resolution_folder, file_name)
        
        with open(csv_path, 'r') as f:
            matrix = pd.read_csv(f, header=None)
        
        return matrix.shape

    def convert_csv_to_npy(self):
        input_folder = os.path.join(self.base_path, self.resolution_folder)
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.csv'):
                csv_path = os.path.join(input_folder, file_name)
                data = pd.read_csv(csv_path, header=None).to_numpy()

                # Rename based on rules
                base_name = file_name[:-4]  # Remove '.csv'
                if "irdrop" in base_name:
                    base_name = base_name.replace("irdrop", "ir_drop")
                elif "power" in base_name:
                    base_name = base_name.replace("power", "current")

                npy_path = os.path.join(self.output_dir, f"{base_name}.npy")
                np.save(npy_path, data)

    def parse_sp_file(self, sp_path):
        match_str = r"R\d+\s+m(\d+)_(\d+)_(\d+)\s+m(\d+)_(\d+)_(\d+)\s+([\d.]+)"
        layer_data = {}

        with open(sp_path, 'r') as file:
            for line in file:
                match = re.match(match_str, line)
                if match:
                    layer1 = match.group(1)
                    x1, y1 = int(match.group(2)), int(match.group(3))
                    layer2 = match.group(4)
                    x2, y2 = int(match.group(5)), int(match.group(6))
                    resistance = float(match.group(7))

                    if layer1 != layer2:
                        layer_name = f"m{layer1}_to_m{layer2}_via"
                    else:
                        layer_name = f"m{layer1}"

                    if layer_name not in layer_data:
                        layer_data[layer_name] = []
                    layer_data[layer_name].append(((x1, y1), (x2, y2), resistance))

        return layer_data

    def convert_layer_data_to_npy(self):
        sp_folder = os.path.join(self.base_path, "layer_data")
        for sp_file in os.listdir(sp_folder):
            if sp_file.endswith('.sp'):
                index = sp_file.split("_")[0]
                self.resolution = self.get_csv_matrix_size(index)
                sp_path = os.path.join(sp_folder, sp_file)
                layer_data = self.parse_sp_file(sp_path)

                # Convert each layer's data to a matrix
                for layer, connections in layer_data.items():
                    matrix = self.convert_to_matrix(connections)
                    npy_path = os.path.join(self.output_dir, "layer_data", f"{index}_{layer}_resistance.npy")
                    np.save(npy_path, matrix)

    def convert_to_matrix(self, connections):
        matrix = np.zeros(self.resolution)

        for (x1, y1), (x2, y2), resistance in connections:
            x1_idx, y1_idx = x1 // self.dbu_per_pixel, y1 // self.dbu_per_pixel
            x2_idx, y2_idx = x2 // self.dbu_per_pixel, y2 // self.dbu_per_pixel

            if 0 <= x1_idx < self.resolution[1] and 0 <= y1_idx < self.resolution[0]:
                matrix[y1_idx, x1_idx] += resistance
            if 0 <= x2_idx < self.resolution[1] and 0 <= y2_idx < self.resolution[0]:
                matrix[y2_idx, x2_idx] += resistance

        return matrix

if __name__ == '__main__':
    # Main Execution
    base_path = "/data/gen_pdn/pdn_data_3rd"
    resolution_folders = ["1um",'210nm']  # or "210nm"

    for resolution_folder in resolution_folders:
        dbu_per_pixel = 2000 if resolution_folder == "1um" else 420

        converter = DatasetConverter(base_path, resolution_folder, dbu_per_pixel)
        converter.convert_csv_to_npy()
        converter.convert_layer_data_to_npy()

        print(f"{resolution_folder} Data conversion completed.")
