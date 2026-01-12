import h5py
import numpy as np

file_path = '/home/hongzefu/dataset_generate/record_dataset_BinFill.h5'

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: {obj.shape} ({obj.dtype})")
    else:
        print(f"{name}/")

try:
    with h5py.File(file_path, 'r') as f:
        print(f"Inspecting {file_path}...")
        f.visititems(print_structure)
        
        # Also check attributes of the root or some groups if needed
        print("\n--- Root Attributes ---")
        for k, v in f.attrs.items():
            print(f"{k}: {v}")
            
except Exception as e:
    print(f"Error reading file: {e}")
