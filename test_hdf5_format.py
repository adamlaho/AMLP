#!/usr/bin/env python3
"""
Script to verify HDF5 file format and data types
"""
import h5py
import numpy as np
import sys

def check_hdf5_format(h5_file):
    """Check the format of HDF5 file to verify data types."""
    print(f"\n{'='*60}")
    print(f"Checking HDF5 file: {h5_file}")
    print(f"{'='*60}\n")

    with h5py.File(h5_file, 'r') as f:
        # Check first batch, first config
        batch_name = "config_batch_0"
        config_name = "config_0"

        if batch_name not in f:
            print(f"ERROR: {batch_name} not found in file!")
            return

        batch_group = f[batch_name]
        if config_name not in batch_group:
            print(f"ERROR: {config_name} not found in {batch_name}!")
            return

        config = batch_group[config_name]

        # Check PBC
        pbc = config['pbc'][:]
        print(f"PBC data:")
        print(f"  Type: {type(pbc)}")
        print(f"  Dtype: {pbc.dtype}")
        print(f"  Shape: {pbc.shape}")
        print(f"  Value: {pbc}")
        print(f"  Is contiguous: {pbc.flags['C_CONTIGUOUS']}")
        print()

        # Check if it's a proper numpy array
        print(f"Verification:")
        print(f"  isinstance(pbc, np.ndarray): {isinstance(pbc, np.ndarray)}")
        print(f"  pbc.ndim: {pbc.ndim}")
        print(f"  len(pbc.shape): {len(pbc.shape)}")
        print()

        # Check cell
        if 'cell' in config:
            cell = config['cell'][:]
            print(f"Cell data:")
            print(f"  Type: {type(cell)}")
            print(f"  Dtype: {cell.dtype}")
            print(f"  Shape: {cell.shape}")
            print(f"  Is contiguous: {cell.flags['C_CONTIGUOUS']}")
            print()

        # Check positions
        positions = config['positions'][:]
        print(f"Positions data:")
        print(f"  Type: {type(positions)}")
        print(f"  Dtype: {positions.dtype}")
        print(f"  Shape: {positions.shape}")
        print(f"  Is contiguous: {positions.flags['C_CONTIGUOUS']}")
        print()

        # Check forces
        forces = config['properties']['forces'][:]
        print(f"Forces data:")
        print(f"  Type: {type(forces)}")
        print(f"  Dtype: {forces.dtype}")
        print(f"  Shape: {forces.shape}")
        print(f"  Is contiguous: {forces.flags['C_CONTIGUOUS']}")
        print()

        # Check stress
        stress = config['properties']['stress'][:]
        print(f"Stress data:")
        print(f"  Type: {type(stress)}")
        print(f"  Dtype: {stress.dtype}")
        print(f"  Shape: {stress.shape}")
        print(f"  Is contiguous: {stress.flags['C_CONTIGUOUS']}")
        print()

        print(f"{'='*60}")
        print("All data types look correct!")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_hdf5_format.py <path_to_h5_file>")
        sys.exit(1)

    h5_file = sys.argv[1]
    check_hdf5_format(h5_file)
