import os
'''
파일 개수 체크
'''
def count_files_in_folders(root_folder):
    folder_file_count = {}
    
    # Walk through all subdirectories and files
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Count number of files in current directory
        num_files = len(filenames)
        folder_file_count[dirpath] = num_files
        
        print(f"Folder: {dirpath} contains {num_files} file(s)")
    
    return folder_file_count

# Example usage
root_folder = '/data/BeGAN-circuit-benchmarks/sky130hd/data'
file_counts = count_files_in_folders(root_folder)
