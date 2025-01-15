import os
import glob
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm

'''
csv.gz data to .npy
    - current 
    - effective distance
    - ir drop (voltage)
'''

# asap7
def extract_and_save_as_numpy_asap7(folder_path, output_folder_path):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Define patterns to locate each type of CSV file based on the new structure
    file_patterns = {
        'current_map': '*_current_map.csv.gz',
        'voltage_map_irregular': '*_voltage_map_irregular.csv.gz',
        'voltage_map_regular': '*_voltage_map_regular.csv.gz',
        'irreg_grid': '*_irreg_grid.sp.gz',
        'reg_grid': '*_reg_grid.sp.gz'
    }

    for key, pattern in file_patterns.items():
        # Find all files that match the current pattern
        files = glob.glob(os.path.join(folder_path, pattern))
        
        for file in tqdm(files, desc=f"Processing {key} files"):
            # Extract filename without extension for saving as numpy
            base_filename = os.path.basename(file).replace('.csv.gz', '').replace('.sp.gz', '')

            # Get the prefix to create a subfolder for it (assumes prefix is the first part before '_')
            prefix = base_filename.split('_')[0]

            # Read the compressed CSV or SPICE file
            if file.endswith('.csv.gz'):
                with gzip.open(file, 'rt') as f:
                    df = pd.read_csv(f, header=None)
                data = df.to_numpy()
            elif file.endswith('.sp.gz'):
                # Optionally, you could handle .sp.gz files differently if necessary
                with gzip.open(file, 'rt') as f:
                    data = f.read()

            # Create subfolder for each prefix if necessary
            output_subfolder = os.path.join(output_folder_path, prefix)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)
            
            # Save the data as a .npy or .txt file based on the file type
            output_file_path = os.path.join(output_subfolder, base_filename + ('.npy' if file.endswith('.csv.gz') else '.txt'))
            if file.endswith('.csv.gz'):
                np.save(output_file_path, data)
            else:
                with open(output_file_path, 'w') as f:
                    f.write(data)

            print(f"Saved: {output_file_path}")


# nangate45
def extract_and_save_as_numpy_began_real(folder_path, output_folder_path):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Define patterns to locate each type of CSV file
    file_patterns = {
        'current_irreg': '*_current_irreg.csv.gz',
        'current_reg': '*_current_reg.csv.gz',
        'eff_dist_irreg': '*_eff_dist_irreg.csv.gz',
        'eff_dist_reg': '*_eff_dist_reg.csv.gz',
        'ir_drop_irreg': '*_ir_drop_irreg.csv.gz',
        'ir_drop_reg': '*_ir_drop_reg.csv.gz',
        'pdn_density_irreg': '*_pdn_density_irreg.csv.gz',
        'pdn_density_reg': '*_pdn_density_reg.csv.gz'
    }

    for key, pattern in file_patterns.items():
        # Find all files that match the current pattern
        files = glob.glob(os.path.join(folder_path, pattern))
        
        for file in tqdm(files):
            # Extract filename without extension for saving as numpy
            base_filename = os.path.basename(file).replace('.csv.gz', '')

            # Get the prefix to create a subfolder for it
            prefix = base_filename.split('_')[0]  # Assumes prefix is the first part before '_'

            # Read the compressed CSV file
            with gzip.open(file, 'rt') as f:
                df = pd.read_csv(f, header=None)

            # Convert the DataFrame to a NumPy array
            data = df.to_numpy()
            
            # Create subfolder for each prefix if necessary
            output_subfolder = os.path.join(output_folder_path, prefix)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)
            
            # Save the NumPy array as .npy file in the prefix-specific folder
            np.save(os.path.join(output_subfolder, base_filename + '.npy'), data)
            print(f"Saved: {base_filename}.npy in {output_subfolder}")


def extract_and_save_as_numpy(folder_path, output_folder_path):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Define patterns to locate each type of CSV file
    file_patterns = {
        'current': '*_current.csv.gz',
        'eff_dist': '*_eff_dist.csv.gz',
        'pdn_density': '*_pdn_density.csv.gz',
        'ir_drop': '*_ir_drop.csv.gz'
    }

    for key, pattern in file_patterns.items():
        # Find all files that match the current pattern
        files = glob.glob(os.path.join(folder_path, pattern))
        
        for file in tqdm(files):
            # Extract filename without extension for saving as numpy
            base_filename = os.path.basename(file).replace('.csv.gz', '')

            # Read the compressed CSV file
            with gzip.open(file, 'rt') as f:
                df = pd.read_csv(f,header=None)

            # Convert the DataFrame to a NumPy array
            data = df.to_numpy()

            # Save the NumPy array as .npy file
            np.save(os.path.join(output_folder_path, base_filename + '.npy'), data)
            print(f"Saved: {base_filename}.npy")

def csv_to_numpy(folder_path, output_folder_path,testcase=False):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Define patterns to locate each type of CSV file
    file_patterns = {
        'current': '*_current.csv',
        'eff_dist': '*_eff_dist.csv',
        'pdn_density': '*_pdn_density.csv',
        'ir_drop': '*_ir_drop.csv'
    }
    if testcase:
        file_patterns = {
            'current': '*current_map.csv',      
            'eff_dist': '*eff_dist_map.csv',    
            'pdn_density': '*pdn_density.csv', 
            'ir_drop': '*ir_drop_map.csv'     
        }


    for key, pattern in file_patterns.items():
        # Find all files that match the current pattern
        files = glob.glob(os.path.join(folder_path, pattern))
        
        for file in tqdm(files):
            # Extract filename without extension for saving as numpy
            base_filename = os.path.basename(file).replace('.csv', '')

            # Read the compressed CSV file
            df = pd.read_csv(file,header=None)

            # Convert the DataFrame to a NumPy array
            data = df.to_numpy()

            # Save the NumPy array as .npy file
            np.save(os.path.join(output_folder_path, base_filename + '.npy'), data)
            print(f"Saved: {base_filename}.npy")

if __name__ == '__main__':

    # csv.gz to numpy (BeGAN)
    # ---------------------------------------------------------------
    # Process both set1 and set2
    # Paths for set1 and set2
    # set1_input_path = '/data/BeGAN-circuit-benchmarks/nangate45/set1/data/'
    # set1_output_path = '/data/BeGAN-circuit-benchmarks/nangate45/set1_numpy/data/'

    # set2_input_path = '/data/BeGAN-circuit-benchmarks/nangate45/set2/data/'
    # set2_output_path = '/data/BeGAN-circuit-benchmarks/nangate45/set2_numpy/data/'
    # extract_and_save_as_numpy(set1_input_path, set1_output_path)
    # extract_and_save_as_numpy(set2_input_path, set2_output_path)


    # ---------------------------------------------------------------
    
    # BeGAN real data (nangete45) 2024.10.31
    # input_path = '/data/real-circuit-benchmarks/nangate45/data'
    # output_path = '/data/real-circuit-benchmarks/nangate45/numpy_data'
    # extract_and_save_as_numpy_began_real(input_path, output_path)

    # BeGAN real data (asap7)
    # input_path = '/data/real-circuit-benchmarks/asap7/data'
    # output_path = '/data/real-circuit-benchmarks/asap7/numpy_data'
    # extract_and_save_as_numpy_asap7(input_path, output_path) 

    # BeGAN fake data (asap7)
    fake_asap7_input_path = '/data/BeGAN-circuit-benchmarks/asap7/data/'
    fake_asap7_output_path = '/data/BeGAN-circuit-benchmarks/asap7/numpy_data/'
    extract_and_save_as_numpy_asap7(fake_asap7_input_path, fake_asap7_output_path)

    # ---------------------------------------------------------------
    # csv to numpy (chellenge data)
    # ---------------------------------------------------------------
    # fake_data_path = '/data/ICCAD_2023/fake-circuit-data_20230623/fake-circuit-data'
    # csv_to_numpy(fake_data_path,'/data/ICCAD_2023/fake-circuit-data_20230623/fake-circuit-data-npy')

    # real_data_path = '/data/ICCAD_2023/real-circuit-data_20230615'
    # for testcase_folder in os.listdir(real_data_path):
    #     tc_path = os.path.join(real_data_path,testcase_folder)
    #     print(testcase_folder)
    #     csv_to_numpy(tc_path,tc_path,testcase=True)

    # hidden case
    # hidden_data_path = '/data/ICCAD_2023/hidden-real-circuit-data'
    # for testcase_folder in os.listdir(hidden_data_path):
    #     tc_path = os.path.join(hidden_data_path,testcase_folder)
    #     print(testcase_folder)
    #     csv_to_numpy(tc_path,tc_path,testcase=True)    
    # ---------------------------------------------------------------



