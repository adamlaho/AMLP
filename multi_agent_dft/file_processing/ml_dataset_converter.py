"""
Module: ml_dataset_converter.py
Purpose: Convert processed DFT data (in JSON format) into datasets formatted for 
         machine learning potentials (like MACE).

This module provides functionality to:
1. Convert JSON data from DFT calculations to HDF5 format for MACE ML potentials
2. Support data preparation for ML potential training and validation
"""

import os
import json
import h5py
import numpy as np
from math import sin, cos, sqrt, radians
from pathlib import Path

# Map element symbols to atomic numbers. Extend as needed.
ELEMENT_MAP = {
    # Period 1
    "H": 1,   "He": 2,
    
    # Period 2
    "Li": 3,  "Be": 4,  "B": 5,   "C": 6,   "N": 7,   "O": 8,   "F": 9,   "Ne": 10,
    
    # Period 3
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,  "S": 16,  "Cl": 17, "Ar": 18,
    
    # Period 4
    "K": 19,  "Ca": 20, "Sc": 21, "Ti": 22, "V": 23,  "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36,
    
    # Period 5
    "Rb": 37, "Sr": 38, "Y": 39,  "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44,
    "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52,
    "I": 53,  "Xe": 54,
    
    # Period 6
    "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62,
    "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,  "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
    
    # Period 7
    "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,  "Np": 93, "Pu": 94,
    "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101, "No": 102,
    "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110,
    "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
}

def cellpar_to_cell(cellpar):
    """
    Convert unit cell parameters [a, b, c, alpha, beta, gamma] into a 3x3 lattice matrix.
    
    Parameters
    ----------
    cellpar : list or array
        Cell parameters [a, b, c, alpha, beta, gamma] where a, b, c are lengths in Å
        and alpha, beta, gamma are angles in degrees.
        
    Returns
    -------
    cell : ndarray
        3x3 cell matrix in Å
    """
    a, b, c, alpha, beta, gamma = cellpar
    alpha, beta, gamma = radians(alpha), radians(beta), radians(gamma)
    v = sqrt(1 - cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 + 2*cos(alpha)*cos(beta)*cos(gamma))
    cell = np.zeros((3, 3), dtype=np.float32)
    cell[0, 0] = a
    cell[1, 0] = b * cos(gamma)
    cell[1, 1] = b * sin(gamma)
    cell[2, 0] = c * cos(beta)
    cell[2, 1] = c * (cos(alpha) - cos(beta)*cos(gamma)) / sin(gamma)
    cell[2, 2] = c * v / sin(gamma)
    return cell

def create_mace_h5_dataset(json_file, output_dir=None, dataset_name=None, train_ratio=0.85, batch_size=4, 
                          max_force_threshold=300.0, conversion_factor=1.0, pbc_handling="auto"):
    """
    Convert JSON data to MACE-compatible HDF5 format, splitting into training and validation sets.
    
    Parameters
    ----------
    json_file : str
        Path to the JSON file containing DFT configuration data
    output_dir : str, optional
        Directory to save the output HDF5 files. If None, uses the directory of the JSON file.
    dataset_name : str, optional
        Base name for the output datasets. If None, derives from the JSON filename.
    train_ratio : float, optional
        Ratio of data to use for training (default: 0.85)
    batch_size : int, optional
        Number of configurations per batch in the HDF5 file (default: 4)
    max_force_threshold : float, optional
        Maximum allowed force magnitude in eV/Å (default: 300.0)
    conversion_factor : float, optional
        Factor to convert energies and forces if they're not already in eV and eV/Å (default: 1.0)
    pbc_handling : str, optional
        How to handle periodic boundary conditions:
        - "auto": Use PBC based on presence of cell parameters (default)
        - "always": Always use PBC [True, True, True] for all configurations with cells
        - "never": Always use non-PBC [False, False, False]
        - "x", "y", "z": Use PBC only in specified directions, e.g. "xy" → [True, True, False]
        
    Returns
    -------
    tuple
        Paths to the created training and validation HDF5 files
    """
    json_path = Path(json_file)
    
    # Determine output directory and dataset name
    if output_dir is None:
        output_dir = json_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset_name is None:
        dataset_name = json_path.stem
    
    # Load JSON data
    try:
        with open(json_path, 'r') as f:
            all_data = json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading JSON file: {e}")
    
    if not isinstance(all_data, list):
        raise ValueError("Expected the JSON to contain a list of configurations.")
    
    # Split data into training and validation sets
    split_index = int(train_ratio * len(all_data))
    train_data = all_data[:split_index]
    valid_data = all_data[split_index:]
    
    # Create output HDF5 files
    train_h5 = output_dir / f"{dataset_name}_train.h5"
    valid_h5 = output_dir / f"{dataset_name}_valid.h5"
    
    # Process and write data
    _create_h5_from_data(train_data, train_h5, batch_size, max_force_threshold, conversion_factor, pbc_handling)
    _create_h5_from_data(valid_data, valid_h5, batch_size, max_force_threshold, conversion_factor, pbc_handling)
    
    print(f"\n[INFO] Created training dataset with {len(train_data)} configurations: {train_h5}")
    print(f"[INFO] Created validation dataset with {len(valid_data)} configurations: {valid_h5}")
    
    return str(train_h5), str(valid_h5)

def _get_pbc_array(pbc_handling, config):
    """
    Determine the PBC array based on the handling option and configuration.
    
    Parameters
    ----------
    pbc_handling : str
        How to handle periodic boundary conditions
    config : dict
        Configuration dictionary
        
    Returns
    -------
    pbc : ndarray
        Array of boolean values for PBC in each direction [x, y, z]
    """
    # Check if we have cell information
    has_cell = "cell_lengths" in config and "cell_angles" in config
    
    if pbc_handling == "auto":
        # Use configuration's PBC if available, otherwise default based on cell presence
        if "pbc" in config:
            return np.array(config["pbc"], dtype=bool)
        else:
            return np.array([has_cell, has_cell, has_cell], dtype=bool)
    elif pbc_handling == "always":
        # Always use PBC if cell is present
        return np.array([has_cell, has_cell, has_cell], dtype=bool)
    elif pbc_handling == "never":
        # Never use PBC
        return np.array([False, False, False], dtype=bool)
    else:
        # Custom PBC directions (x, y, z or combinations)
        pbc = [False, False, False]
        if "x" in pbc_handling:
            pbc[0] = has_cell
        if "y" in pbc_handling:
            pbc[1] = has_cell
        if "z" in pbc_handling:
            pbc[2] = has_cell
        return np.array(pbc, dtype=bool)

def _create_h5_from_data(data, h5_file, batch_size=4, max_force_threshold=300.0, 
                        conversion_factor=1.0, pbc_handling="auto"):
    """
    Writes configurations to an HDF5 file in a format compatible with MACE.
    
    Parameters
    ----------
    data : list
        List of configuration dictionaries (from JSON)
    h5_file : str or Path
        Output HDF5 filename
    batch_size : int, optional
        Number of configurations per batch (default: 4)
    max_force_threshold : float, optional
        Maximum allowed force magnitude in eV/Å (default: 300.0)
    conversion_factor : float, optional
        Factor to convert energies and forces if not already in eV (default: 1.0)
    pbc_handling : str, optional
        How to handle periodic boundary conditions
    """
    valid_configs = []
    skipped_configs = 0
    force_exceeded = 0
    missing_data = 0
    unknown_elements = 0

    for config_idx, config in enumerate(data):
        atom_types = config.get("atom_types", [])
        n_atoms = len(atom_types)
        if n_atoms == 0:
            missing_data += 1
            continue

        # Forces: expected in eV/Å (may need conversion)
        forces_list = config.get("forces", [])
        if len(forces_list) != n_atoms:
            missing_data += 1
            continue
        forces = np.array([[f["x"], f["y"], f["z"]] for f in forces_list],
                          dtype=np.float32) * conversion_factor

        # Skip configuration if any atom's force exceeds the threshold
        force_magnitudes = np.linalg.norm(forces, axis=1)
        if np.max(force_magnitudes) >= max_force_threshold:
            force_exceeded += 1
            continue

        # Positions: expected in Å
        coords_list = config.get("coordinates", [])
        if len(coords_list) != n_atoms:
            missing_data += 1
            continue
        positions = np.array([[c["x"], c["y"], c["z"]] for c in coords_list],
                             dtype=np.float32)

        # Energy: expected in eV
        if "energy" not in config:
            missing_data += 1
            continue
        energy_eV = np.array(float(config["energy"]) * conversion_factor, dtype=np.float32)

        # Convert atom types to atomic numbers
        atomic_numbers = []
        for symbol in atom_types:
            if symbol not in ELEMENT_MAP:
                unknown_elements += 1
                atomic_numbers = None
                break
            atomic_numbers.append(ELEMENT_MAP[symbol])
        if atomic_numbers is None:
            continue
        atomic_numbers = np.array(atomic_numbers, dtype=np.int64)

        # Determine cell and PBC status based on pbc_handling
        if "cell_lengths" in config and "cell_angles" in config:
            cellpar = [float(x) for x in config["cell_lengths"] + config["cell_angles"]]
            cell_matrix = np.array(cellpar_to_cell(cellpar), dtype=np.float32)
            pbc = _get_pbc_array(pbc_handling, config)
        else:
            cell_matrix = None
            pbc = np.array([False, False, False], dtype=bool)

        valid_configs.append({
            "atomic_numbers": atomic_numbers,
            "positions": positions,
            "energy": energy_eV,
            "forces": forces,
            "pbc": pbc,
            "cell": cell_matrix
        })

    skipped_configs = force_exceeded + missing_data + unknown_elements
    print(f"[INFO] Valid configurations: {len(valid_configs)}/{len(data)} ({len(valid_configs)/len(data)*100:.1f}%)")
    if skipped_configs > 0:
        print(f"[INFO] Skipped configurations: {skipped_configs}")
        print(f"       - Force threshold exceeded: {force_exceeded}")
        print(f"       - Missing or inconsistent data: {missing_data}")
        print(f"       - Unknown elements: {unknown_elements}")

    # Print PBC information
    if len(valid_configs) > 0:
        pbc_counts = {"[True, True, True]": 0, "[False, False, False]": 0, "other": 0}
        for config in valid_configs:
            pbc_str = str(config["pbc"].tolist())
            if pbc_str == "[True, True, True]":
                pbc_counts["[True, True, True]"] += 1
            elif pbc_str == "[False, False, False]":
                pbc_counts["[False, False, False]"] += 1
            else:
                pbc_counts["other"] += 1
        
        print(f"[INFO] PBC configuration counts:")
        print(f"       - Full 3D periodicity [True, True, True]: {pbc_counts['[True, True, True]']}")
        print(f"       - No periodicity [False, False, False]: {pbc_counts['[False, False, False]']}")
        if pbc_counts["other"] > 0:
            print(f"       - Partial periodicity (other): {pbc_counts['other']}")

    # Pad the last batch if necessary
    if len(valid_configs) % batch_size != 0 and len(valid_configs) > 0:
        configs_needed = batch_size - (len(valid_configs) % batch_size)
        for _ in range(configs_needed):
            valid_configs.append(valid_configs[-1].copy())
        print(f"[INFO] Added {configs_needed} padding configurations to complete the last batch")

    with h5py.File(h5_file, 'w') as hf:
        hf.attrs["layout"] = "Ace"
        hf.attrs["layout_version"] = "1.0"
        hf.attrs["name"] = os.path.basename(h5_file)
        hf.attrs["drop_last"] = "False"

        num_batches = (len(valid_configs) + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            batch_name = f"config_batch_{batch_idx}"
            batch_group = hf.create_group(batch_name)

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(valid_configs))
            batch_configs = valid_configs[start_idx:end_idx]

            for conf_idx, conf_data in enumerate(batch_configs):
                config_name = f"config_{conf_idx}"
                config_group = batch_group.create_group(config_name)

                # Create the properties group that MACE expects
                properties_group = config_group.create_group("properties")
                
                # Store physical properties in the properties group
                properties_group.create_dataset("energy", data=conf_data["energy"], dtype=np.float32)
                properties_group.create_dataset("forces", data=conf_data["forces"], dtype=np.float32)
                # Store stress and virials in properties if available, otherwise create empty datasets
                properties_group.create_dataset("stress", data=np.zeros((3, 3), dtype=np.float32))
                properties_group.create_dataset("virials", data=np.zeros((3, 3), dtype=np.float32))
                
                # Create the property_weights group that MACE expects
                property_weights_group = config_group.create_group("property_weights")
                property_weights_group.create_dataset("energy", data=np.array(1.0, dtype=np.float64))
                property_weights_group.create_dataset("forces", data=np.array(1.0, dtype=np.float64))
                property_weights_group.create_dataset("stress", data=np.array(1.0, dtype=np.float64))
                property_weights_group.create_dataset("virials", data=np.array(1.0, dtype=np.float64))
                
                # Store system information in the config group
                config_group.create_dataset("atomic_numbers", data=conf_data["atomic_numbers"], dtype=np.int64)
                config_group.create_dataset("positions", data=conf_data["positions"], dtype=np.float32)
                config_group.create_dataset("pbc", data=conf_data["pbc"], dtype=bool)

                if conf_data["cell"] is not None:
                    config_group.create_dataset("cell", data=conf_data["cell"], dtype=np.float32)
                else:
                    # Create an empty cell dataset for non-periodic systems
                    empty_cell = np.zeros((3, 3), dtype=np.float32)
                    config_group.create_dataset("cell", data=empty_cell, dtype=np.float32)

                # Global weight for the configuration
                config_group.create_dataset("weight", data=np.array(1.0, dtype=np.float64))
                
                # Other optional fields
                for field in ["dipole", "charges", "config_type"]:
                    config_group.create_dataset(field, data="None")

        print(f"[INFO] Written {len(valid_configs)} configurations to '{h5_file}'")
        print(f"[INFO] Created {num_batches} complete batches")
