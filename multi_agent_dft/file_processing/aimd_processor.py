#!/usr/bin/env python3
"""
Module: aimd_processor.py
Purpose: Process AIMD JSON files to extract the final geometry from each trajectory
         and generate corresponding CP2K MD input files and a new XYZ file.
         
This module provides functions for:
1. Loading and validating AIMD JSON files containing trajectory data
2. Extracting final configurations from AIMD trajectories
3. Converting atomic coordinates and cell parameters to ASE Atoms objects
4. Generating CP2K input files optimized for AIMD simulations at different temperatures
5. Writing XYZ files for visualization and further processing

The functions in this module are designed to be called from the main MultiAgentSystem class,
but can also be used independently when imported.
"""

import os
import glob
import json
import yaml
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional, Any

import numpy as np

# Try to import ASE, which is required for this module
try:
    from ase import Atoms
    from ase.geometry import cellpar_to_cell, cell_to_cellpar
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


def check_requirements() -> bool:
    """Check if required packages are available."""
    if not ASE_AVAILABLE:
        print("ERROR: ASE (Atomic Simulation Environment) is required for AIMD processing.")
        print("Please install it with: pip install ase")
        return False
    return True


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration parameters from a YAML file.
    
    Expected keys (example):
      json_dir: "/path/to/json_files"
      output_dir: "/path/to/aimd_outputs"
      melting_point: 300          # in Kelvin (optional)
      temperatures: [175, 200, 225, 250, 300, 350]
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the config file has invalid YAML syntax
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate that a configuration dictionary has all required keys.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_keys = ['json_dir', 'output_dir', 'temperatures']
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        return False, f"Missing required configuration keys: {', '.join(missing_keys)}"
    
    # Validate directory existence
    json_dir = Path(config['json_dir'])
    if not json_dir.exists() or not json_dir.is_dir():
        return False, f"JSON directory does not exist or is not a directory: {json_dir}"
    
    # Validate temperatures
    if not config['temperatures'] or not all(isinstance(t, (int, float)) for t in config['temperatures']):
        return False, "Temperatures list is empty or contains non-numeric values"
    
    return True, ""


def extract_final_config(json_file: Path) -> Optional[Dict[str, Any]]:
    """
    Extract the final configuration from a JSON file containing AIMD trajectory data.
    
    Args:
        json_file: Path to JSON file
        
    Returns:
        Dictionary containing the final configuration or None if an error occurs
    """
    try:
        with json_file.open('r') as f:
            data = json.load(f)
        
        if not data or not isinstance(data, list):
            print(f"Warning: Invalid or empty JSON data in {json_file}")
            return None
        
        final_config = data[-1]
        
        # Verify that the configuration has required fields
        if not all(key in final_config for key in ["coordinates", "atom_types", "cell_lengths", "cell_angles"]):
            missing = [k for k in ["coordinates", "atom_types", "cell_lengths", "cell_angles"] if k not in final_config]
            print(f"Warning: Missing required fields in configuration: {missing}")
            return None
        
        return final_config
    except Exception as e:
        print(f"Error reading JSON file {json_file}: {str(e)}")
        return None


def create_atoms_from_config(config: Dict[str, Any]) -> Optional[Atoms]:
    """
    Create an ASE Atoms object from a configuration dictionary.
    
    Args:
        config: Configuration dictionary with coordinates, atom_types, cell_lengths, cell_angles
        
    Returns:
        ASE Atoms object or None if an error occurs
    """
    try:
        # Extract coordinates
        coords = config.get("coordinates", [])
        if not coords:
            return None
            
        positions = np.array([[c["x"], c["y"], c["z"]] for c in coords], dtype=np.float32)
        
        # Extract atom types
        symbols = config.get("atom_types", [])
        if len(symbols) != len(positions):
            print(f"Warning: Number of atom types ({len(symbols)}) doesn't match coordinates ({len(positions)})")
            return None
        
        # Extract cell parameters
        cell_lengths = config.get("cell_lengths", [])
        cell_angles = config.get("cell_angles", [])
        if not cell_lengths or not cell_angles:
            print("Warning: Missing cell parameters")
            return None
            
        # Convert cell parameters to cell vectors
        cellpars = cell_lengths + cell_angles
        cell_vectors = cellpar_to_cell(cellpars)
        
        # Create and return ASE Atoms object
        return Atoms(symbols=symbols, positions=positions, cell=cell_vectors, pbc=True)
    except Exception as e:
        print(f"Error creating Atoms object: {str(e)}")
        return None


def write_xyz_file(atoms: Atoms, output_path: Path, comment: str = "XYZ from AIMD processing") -> bool:
    """
    Write an XYZ file from an ASE Atoms object.
    
    Args:
        atoms: ASE Atoms object
        output_path: Path to output XYZ file
        comment: Comment line for XYZ file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        
        with output_path.open('w') as f:
            f.write(f"{len(atoms)}\n")
            f.write(f"{comment}\n")
            for sym, pos in zip(symbols, positions):
                f.write(f"{sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
        
        return True
    except Exception as e:
        print(f"Error writing XYZ file {output_path}: {str(e)}")
        return False


def write_cp2k_aimd_input(atoms: Atoms, output_path: Path, temperature: float) -> bool:
    """
    Write a CP2K AIMD input file for a specific temperature.
    
    Args:
        atoms: ASE Atoms object
        output_path: Path to output CP2K input file
        temperature: MD simulation temperature in Kelvin
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Extract cell parameters
        cell_vectors = atoms.cell.array
        cellpars = cell_to_cellpar(cell_vectors)  # [a, b, c, alpha, beta, gamma]
        ABC = cellpars[:3]
        alpha_beta_gamma = cellpars[3:]
        
        # Extract atomic symbols and positions
        atom_symbols = atoms.get_chemical_symbols()
        atom_positions = atoms.get_positions()
        
        # Standard mapping for basis sets and potentials
        kind_parameters = {
            "H":  {"BASIS_SET": "ORB TZV2P-MOLOPT-PBE-GTH-q1", "POTENTIAL": "GTH-PBE-q1"},
            "C":  {"BASIS_SET": "ORB TZV2P-MOLOPT-PBE-GTH-q4", "POTENTIAL": "GTH-PBE-q4"},
            "N":  {"BASIS_SET": "ORB TZV2P-MOLOPT-PBE-GTH-q5", "POTENTIAL": "GTH-PBE-q5"},
            "O":  {"BASIS_SET": "ORB TZV2P-MOLOPT-PBE-GTH-q6", "POTENTIAL": "GTH-PBE-q6"},
            "F":  {"BASIS_SET": "ORB aug-TZV2P-GTH-q7",         "POTENTIAL": "GTH-PBE-q7"},
            "Si": {"BASIS_SET": "ORB aug-TZV2P-GTH-q4",         "POTENTIAL": "GTH-PBE-q4"},
            "Cl": {"BASIS_SET": "ORB aug-TZV2P-GTH-q7",         "POTENTIAL": "GTH-PBE-q7"},
            "Br": {"BASIS_SET": "ORB TZV2P-MOLOPT-PBE-GTH-q7",  "POTENTIAL": "GTH-PBE-q7"},
            "I":  {"BASIS_SET": "ORB TZV2P-MOLOPT-PBE-GTH-q7",  "POTENTIAL": "GTH-PBE-q7"}
        }
        
        # Create CP2K input file content
        lines = []
        
        # Global section
        lines.append("&GLOBAL")
        lines.append(f"  PROJECT AIMD_T{temperature}")
        lines.append("  RUN_TYPE MD")
        lines.append("  PRINT_LEVEL LOW")
        lines.append("&END GLOBAL\n")
        
        # Force evaluation section
        lines.append("&FORCE_EVAL")
        lines.append("  METHOD Quickstep")
        lines.append("  &DFT")
        lines.append("    BASIS_SET_FILE_NAME BASIS_MOLOPT")
        lines.append("    POTENTIAL_FILE_NAME GTH_POTENTIALS")
        
        # MGRID section
        lines.append("    &MGRID")
        lines.append("      CUTOFF 500")
        lines.append("      REL_CUTOFF 50")
        lines.append("      NGRIDS 4")
        lines.append("    &END MGRID")
        
        # SCF section
        lines.append("    &SCF")
        lines.append("      SCF_GUESS ATOMIC")
        lines.append("      MAX_SCF 50")
        lines.append("      EPS_SCF 5.0E-7")
        lines.append("      &OT")
        lines.append("        MINIMIZER DIIS")
        lines.append("        PRECONDITIONER FULL_SINGLE_INVERSE")
        lines.append("      &END OT")
        lines.append("      &OUTER_SCF")
        lines.append("        MAX_SCF 20")
        lines.append("        EPS_SCF 5.0E-6")
        lines.append("      &END OUTER_SCF")
        lines.append("      &PRINT")
        lines.append("        &RESTART")
        lines.append("          ADD_LAST NUMERIC")
        lines.append("          &EACH")
        lines.append("            QS_SCF 0")
        lines.append("          &END EACH")
        lines.append("        &END RESTART")
        lines.append("      &END PRINT")
        lines.append("    &END SCF")
        
        # XC section with vdW correction
        lines.append("    &XC")
        lines.append("      &XC_FUNCTIONAL")
        lines.append("        &PBE")
        lines.append("          PARAMETRIZATION REVPBE")
        lines.append("        &END PBE")
        lines.append("      &END XC_FUNCTIONAL")
        lines.append("      &XC_GRID")
        lines.append("        XC_DERIV NN50_SMOOTH")
        lines.append("      &END XC_GRID")
        lines.append("      &VDW_POTENTIAL")
        lines.append("        POTENTIAL_TYPE PAIR_POTENTIAL")
        lines.append("        &PAIR_POTENTIAL")
        lines.append("          TYPE DFTD3")
        lines.append("          R_CUTOFF 12.0")
        lines.append("          LONG_RANGE_CORRECTION")
        lines.append("          REFERENCE_FUNCTIONAL revPBE")
        lines.append("          PARAMETER_FILE_NAME dftd3.dat")
        lines.append("        &END PAIR_POTENTIAL")
        lines.append("      &END VDW_POTENTIAL")
        lines.append("    &END XC")
        lines.append("  &END DFT")
        
        # Print forces
        lines.append("  &PRINT")
        lines.append("    &FORCES ON")
        lines.append("    &END FORCES")
        lines.append("  &END PRINT")
        
        # Subsystem section (cell and coordinates)
        lines.append("  &SUBSYS")
        lines.append("    &CELL")
        lines.append(f"      ABC [angstrom] {ABC[0]:.6f} {ABC[1]:.6f} {ABC[2]:.6f}")
        lines.append(f"      ALPHA_BETA_GAMMA [deg] {alpha_beta_gamma[0]:.6f} {alpha_beta_gamma[1]:.6f} {alpha_beta_gamma[2]:.6f}")
        lines.append("      PERIODIC XYZ")
        lines.append("    &END CELL")
        
        # Add KIND sections for each element type
        for atom in set(atom_symbols):
            if atom in kind_parameters:
                params = kind_parameters[atom]
                lines.append(f"    &KIND {atom}")
                lines.append(f"      BASIS_SET {params['BASIS_SET']}")
                lines.append(f"      POTENTIAL {params['POTENTIAL']}")
                lines.append("    &END KIND")
            else:
                lines.append(f"    &KIND {atom}")
                lines.append("      BASIS_SET DZVP-MOLOPT-SR-GTH")
                lines.append("      POTENTIAL GTH-PBE-q0")
                lines.append("    &END KIND")
        
        # Add atomic coordinates
        lines.append("    &COORD")
        for sym, pos in zip(atom_symbols, atom_positions):
            lines.append(f"      {sym} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
        lines.append("    &END COORD")
        lines.append("  &END SUBSYS")
        lines.append("&END FORCE_EVAL\n")
        
        # MOTION section with MD settings
        lines.append("&MOTION")
        lines.append("  &MD")
        lines.append("    ENSEMBLE NVT")
        lines.append(f"    TEMPERATURE [K] {temperature}")
        lines.append("    TIMESTEP [fs] 0.5")
        lines.append("    STEPS 1000000")
        lines.append("    &THERMOSTAT")
        lines.append("      REGION MASSIVE")
        lines.append("      TYPE GLE")
        lines.append("      &GLE")
        lines.append("        NDIM 5")
        lines.append("        A_SCALE [ps^-1] 1.00")
        lines.append("        A_LIST    1.859575861256e+2   2.726385349840e-1   1.152610045461e+1  -3.641457826260e+1   2.317337581602e+2")
        lines.append("        A_LIST   -2.780952471206e-1   8.595159180871e-5   7.218904801765e-1  -1.984453934386e-1   4.240925758342e-1")
        lines.append("        A_LIST   -1.482580813121e+1  -7.218904801765e-1   1.359090212128e+0   5.149889628035e+0  -9.994926845099e+0")
        lines.append("        A_LIST   -1.037218912688e+1   1.984453934386e-1  -5.149889628035e+0   2.666191089117e+1   1.150771549531e+1")
        lines.append("        A_LIST    2.180134636042e+2  -4.240925758342e-1   9.994926845099e+0  -1.150771549531e+1   3.095839456559e+2")
        lines.append("      &END GLE")
        lines.append("    &END THERMOSTAT")
        lines.append("  &END MD")
        
        # Print settings for trajectory, forces, etc.
        lines.append("  &PRINT")
        lines.append("    &TRAJECTORY")
        lines.append("      FORMAT XYZ")
        lines.append("      UNIT angstrom")
        lines.append("      &EACH")
        lines.append("        MD 1")
        lines.append("      &END EACH")
        lines.append("    &END TRAJECTORY")
        lines.append("    &VELOCITIES OFF")
        lines.append("    &END VELOCITIES")
        lines.append("    &FORCES ON")
        lines.append("    &END FORCES")
        lines.append("    &RESTART_HISTORY")
        lines.append("      &EACH")
        lines.append("        MD 500")
        lines.append("      &END EACH")
        lines.append("    &END RESTART_HISTORY")
        lines.append("  &END PRINT")
        lines.append("&END MOTION")
        
        # Write the content to the file
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        
        return True
    except Exception as e:
        print(f"Error writing CP2K input file {output_path}: {str(e)}")
        return False


def process_json_files(json_dir: str, temperatures: List[float], output_dir: str) -> Dict[str, int]:
    """
    Process JSON files in the specified directory to generate CP2K AIMD input files.
    
    Args:
        json_dir: Directory containing JSON files
        temperatures: List of temperatures for AIMD simulations
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with statistics about the processing
    """
    if not check_requirements():
        return {"error": "Missing requirements"}
    
    # Find all JSON files
    json_pattern = os.path.join(json_dir, "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"[ERROR] No JSON files found in {json_dir}")
        return {"total": 0, "processed": 0, "failed": 0}
    
    print(f"\nFound {len(json_files)} JSON files in {json_dir}")
    print(f"Will generate CP2K input files for {len(temperatures)} temperatures: {temperatures}")
    print(f"Output will be saved in {output_dir}\n")
    
    # Process JSON files
    stats = {"total": len(json_files), "processed": 0, "failed": 0}
    
    for filepath in json_files:
        try:
            file_path = Path(filepath)
            base_name = file_path.stem
            
            print(f"Processing: {file_path.name}...", end=" ")
            
            # Extract final configuration
            final_config = extract_final_config(file_path)
            if not final_config:
                print("Failed (invalid configuration)")
                stats["failed"] += 1
                continue
            
            # Create ASE Atoms object
            atoms_obj = create_atoms_from_config(final_config)
            if not atoms_obj:
                print("Failed (could not create Atoms object)")
                stats["failed"] += 1
                continue
            
            # Define output directories
            base_out = Path(output_dir) / base_name
            xyz_out_dir = base_out / "XYZ"
            cp2k_out_dir = base_out / "CP2K"
            xyz_out_dir.mkdir(parents=True, exist_ok=True)
            cp2k_out_dir.mkdir(parents=True, exist_ok=True)
            
            # Write XYZ file
            xyz_path = xyz_out_dir / f"{base_name}_original.xyz"
            if not write_xyz_file(atoms_obj, xyz_path):
                print("Failed (could not write XYZ file)")
                stats["failed"] += 1
                continue
            
            # Write CP2K input files for each temperature
            cp2k_success = True
            for temp in temperatures:
                cp2k_path = cp2k_out_dir / f"T{temp}_{base_name}_cp2k_input.inp"
                if not write_cp2k_aimd_input(atoms_obj, cp2k_path, temperature=temp):
                    cp2k_success = False
                    break
            
            if not cp2k_success:
                print("Failed (could not write CP2K input files)")
                stats["failed"] += 1
                continue
            
            stats["processed"] += 1
            print("OK")
            
        except Exception as exc:
            stats["failed"] += 1
            print(f"ERROR: {exc}")
    
    print(f"\nProcessed {stats['processed']} out of {stats['total']} JSON files.")
    print(f"Generated {stats['processed'] * len(temperatures)} CP2K AIMD input files.")
    
    return stats


def main():
    """Command-line interface for processing AIMD JSON files."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process AIMD JSON files to generate CP2K input files for MD simulations."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Load and validate configuration
    try:
        config = load_config(args.config)
        is_valid, error_msg = validate_config(config)
        
        if not is_valid:
            print(f"Error in configuration: {error_msg}")
            return 1
            
        # Process the JSON files
        stats = process_json_files(
            json_dir=config['json_dir'], 
            temperatures=config['temperatures'], 
            output_dir=config['output_dir']
        )
        
        if stats["processed"] > 0:
            return 0
        else:
            return 1
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())