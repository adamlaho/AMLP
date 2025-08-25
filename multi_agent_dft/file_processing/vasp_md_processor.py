#!/usr/bin/env python3
"""
Module: multi_agent_dft.file_processing.vasp_md_processor
Purpose: Efficient VASP Molecular Dynamics input file generation and processing

This module provides core functions for:
1. VASP MD input file generation (INCAR, POSCAR, KPOINTS)
2. JSON trajectory processing for AIMD → VASP MD conversion
3. ASE integration for structure handling
4. Comprehensive thermostat and ensemble support
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# ASE imports with fallback
try:
    from ase import Atoms
    from ase.geometry import cellpar_to_cell, cell_to_cellpar
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


def check_ase_available() -> bool:
    """Check if ASE is available for VASP MD processing."""
    return ASE_AVAILABLE


# ============================================================================
# CORE VASP MD FILE GENERATORS
# ============================================================================

def generate_vasp_md_incar(md_config: Dict[str, Any], temperature: float) -> str:
    """
    Generate VASP INCAR file content for MD simulation.
    
    Args:
        md_config: MD configuration dictionary
        temperature: Simulation temperature in Kelvin
        
    Returns:
        INCAR file content as string
    """
    lines = [
        f"SYSTEM = VASP MD at {temperature}K",
        "",
        "# Basic DFT settings",
        "ISTART = 0",
        "ICHARG = 2",
        f"ENCUT = {md_config.get('encut', 400)}",
        f"PREC = {md_config.get('prec', 'Normal')}",
        f"LREAL = {md_config.get('lreal', 'Auto')}",
        ""
    ]
    
    # Exchange-correlation functional
    xc_func = md_config.get('xc_functional', 'PBE').upper()
    if xc_func != 'LDA':
        lines.extend([
            "# Exchange-correlation functional",
            f"GGA = {'PE' if xc_func == 'PBE' else 'RE' if xc_func == 'REVPBE' else 'PE'}",
            ""
        ])
    
    # vdW correction
    if md_config.get('vdw_correction', False):
        vdw_method = md_config.get('vdw_method', 'D3')
        ivdw_map = {'D2': 1, 'D3': 11, 'D3BJ': 12, 'TS': 2}
        lines.extend([
            "# van der Waals correction",
            "LVDW = .TRUE.",
            f"IVDW = {ivdw_map.get(vdw_method, 11)}",
            ""
        ])
    
    # Electronic settings
    lines.extend([
        "# Electronic convergence",
        f"EDIFF = {md_config.get('ediff', '1E-4')}",
        f"NELM = {md_config.get('nelm', 60)}",
        f"ALGO = {md_config.get('algo', 'Fast')}",
        f"ISMEAR = {md_config.get('ismear', 0)}",
        f"SIGMA = {md_config.get('sigma', 0.1)}",
        ""
    ])
    
    # MD core settings
    ensemble = md_config.get('ensemble', 'NVT')
    isif = 3 if ensemble == 'NPT' else 2
    
    lines.extend([
        "# Molecular Dynamics settings",
        "IBRION = 0",
        f"NSW = {md_config.get('steps', 50000)}",
        f"POTIM = {md_config.get('timestep', 1.0)}",
        f"ISIF = {isif}",
        f"TEBEG = {temperature}",
        f"TEEND = {temperature}",
        ""
    ])
    
    # Thermostat configuration
    lines.extend(_generate_thermostat_section(md_config, ensemble))
    
    # Pressure settings for NPT
    if ensemble == 'NPT':
        lines.extend([
            "# Pressure settings",
            f"PSTRESS = {md_config.get('pstress', 0.0)}",
            ""
        ])
    
    # Output settings
    lines.extend([
        "# Output settings",
        "LWAVE = .FALSE.",
        "LCHARG = .FALSE.",
        f"NBLOCK = {md_config.get('nblock', 1)}",
        f"KBLOCK = {md_config.get('kblock', 100)}"
    ])
    
    return "\n".join(lines)


def _generate_thermostat_section(md_config: Dict[str, Any], ensemble: str) -> List[str]:
    """Generate thermostat section for INCAR."""
    lines = ["# Thermostat settings"]
    
    if ensemble == 'NVE':
        lines.extend(["MDALGO = 1", "ANDERSEN_PROB = 0.0"])
    else:
        thermostat = md_config.get('thermostat', 'nose_hoover')
        
        if thermostat == 'nose_hoover':
            lines.extend([
                "MDALGO = 2",
                f"SMASS = {md_config.get('smass', 0.5)}"
            ])
        elif thermostat == 'langevin':
            lines.extend([
                "MDALGO = 3",
                f"LANGEVIN_GAMMA = {md_config.get('langevin_gamma', 10.0)}"
            ])
            if ensemble == 'NPT':
                lines.append(f"LANGEVIN_GAMMA_L = {md_config.get('langevin_gamma_l', 1.0)}")
        elif thermostat == 'andersen':
            lines.extend([
                "MDALGO = 1",
                f"ANDERSEN_PROB = {md_config.get('andersen_prob', 0.1)}"
            ])
        elif thermostat == 'csvr':
            lines.extend([
                "MDALGO = 5",
                f"CSVR_PERIOD = {md_config.get('csvr_period', 10)}"
            ])
        elif thermostat == 'nhc':
            lines.extend([
                "MDALGO = 4",
                f"NHC_NCHAINS = {md_config.get('nhc_nchains', 4)}",
                f"NHC_PERIOD = {md_config.get('nhc_period', 1)}"
            ])
        elif thermostat == 'multiple_andersen':
            lines.extend([
                "MDALGO = 13",
                f"ANDERSEN_PROB = {md_config.get('andersen_prob', 0.1)}"
            ])
    
    lines.append("")
    return lines


def generate_vasp_poscar(atoms: 'Atoms', comment: str = "VASP MD structure") -> str:
    """
    Generate VASP POSCAR from ASE Atoms object.
    
    Args:
        atoms: ASE Atoms object
        comment: Comment line for POSCAR
        
    Returns:
        POSCAR content as string
    """
    if not ASE_AVAILABLE:
        raise ImportError("ASE is required for POSCAR generation")
    
    lines = [comment, "1.0"]
    
    # Lattice vectors
    for vector in atoms.cell.array:
        lines.append(f"  {vector[0]:15.10f}  {vector[1]:15.10f}  {vector[2]:15.10f}")
    
    # Element symbols and counts
    symbols = atoms.get_chemical_symbols()
    unique_symbols = []
    symbol_counts = {}
    
    for symbol in symbols:
        if symbol not in symbol_counts:
            unique_symbols.append(symbol)
            symbol_counts[symbol] = 0
        symbol_counts[symbol] += 1
    
    lines.extend([
        "  " + "  ".join(unique_symbols),
        "  " + "  ".join(str(symbol_counts[symbol]) for symbol in unique_symbols),
        "Cartesian"
    ])
    
    # Atomic positions (grouped by element)
    positions = atoms.get_positions()
    for symbol in unique_symbols:
        for i, atom_symbol in enumerate(symbols):
            if atom_symbol == symbol:
                pos = positions[i]
                lines.append(f"  {pos[0]:15.10f}  {pos[1]:15.10f}  {pos[2]:15.10f}")
    
    return "\n".join(lines)


def generate_vasp_kpoints(grid: Optional[List[int]] = None) -> str:
    """
    Generate VASP KPOINTS file for MD (typically Gamma point).
    
    Args:
        grid: K-point grid [nx, ny, nz]. Defaults to [1, 1, 1]
        
    Returns:
        KPOINTS content as string
    """
    if grid is None:
        grid = [1, 1, 1]
    
    return "\n".join([
        "K-points for MD simulation",
        "0",
        "Gamma" if grid == [1, 1, 1] else "Monkhorst-Pack",
        f"{grid[0]} {grid[1]} {grid[2]}",
        "0 0 0"
    ])


def generate_potcar_info(atoms: 'Atoms') -> str:
    """Generate POTCAR information file."""
    if not ASE_AVAILABLE:
        raise ImportError("ASE is required for POTCAR info generation")
    
    symbols = atoms.get_chemical_symbols()
    unique_symbols = list(dict.fromkeys(symbols))  # Preserve order
    
    lines = [
        "POTCAR Information for VASP MD Calculation",
        "=" * 50,
        "",
        "Required POTCAR files in order:",
        ""
    ]
    
    for symbol in unique_symbols:
        lines.append(f"- {symbol}: Use appropriate pseudopotential (e.g., {symbol}_pv, {symbol}_sv, or {symbol})")
    
    lines.extend([
        "",
        "Concatenate POTCAR files in the order shown above:",
        f"cat {' '.join([f'{s}/POTCAR' for s in unique_symbols])} > POTCAR",
        "",
        "Use consistent pseudopotential type (PAW_PBE, PAW_GGA, etc.)"
    ])
    
    return "\n".join(lines)


# ============================================================================
# JSON TRAJECTORY PROCESSING
# ============================================================================

def extract_final_structure_from_json(json_file: Path) -> Optional['Atoms']:
    """
    Extract final structure from AIMD JSON trajectory as ASE Atoms object.
    
    Args:
        json_file: Path to JSON trajectory file
        
    Returns:
        ASE Atoms object or None if extraction fails
    """
    if not ASE_AVAILABLE:
        raise ImportError("ASE is required for JSON trajectory processing")
    
    try:
        with json_file.open('r') as f:
            data = json.load(f)
        
        if not data or not isinstance(data, list):
            return None
        
        final_config = data[-1]
        
        # Validate required fields
        required_fields = ["coordinates", "atom_types", "cell_lengths", "cell_angles"]
        if not all(field in final_config for field in required_fields):
            return None
        
        # Extract data
        coords = final_config["coordinates"]
        positions = np.array([[c["x"], c["y"], c["z"]] for c in coords])
        symbols = final_config["atom_types"]
        
        # Convert cell parameters to cell vectors
        cell_lengths = final_config["cell_lengths"]
        cell_angles = final_config["cell_angles"]
        cellpars = cell_lengths + cell_angles
        cell_vectors = cellpar_to_cell(cellpars)
        
        return Atoms(symbols=symbols, positions=positions, cell=cell_vectors, pbc=True)
        
    except Exception:
        return None


def write_complete_vasp_md_input(atoms: 'Atoms', output_dir: Path, 
                                temperature: float, md_config: Dict[str, Any]) -> bool:
    """
    Write complete VASP MD input set (INCAR, POSCAR, KPOINTS, POTCAR_INFO).
    
    Args:
        atoms: ASE Atoms object
        output_dir: Output directory
        temperature: MD temperature
        md_config: MD configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate and write files
        files_content = {
            "INCAR": generate_vasp_md_incar(md_config, temperature),
            "POSCAR": generate_vasp_poscar(atoms, f"MD structure at {temperature}K"),
            "KPOINTS": generate_vasp_kpoints(md_config.get('kpoint_grid')),
            "POTCAR_INFO.txt": generate_potcar_info(atoms)
        }
        
        for filename, content in files_content.items():
            (output_dir / filename).write_text(content, encoding='utf-8')
        
        return True
        
    except Exception:
        return False


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_json_trajectory_batch(json_dir: Path, output_dir: Path, 
                                 temperatures: List[float], 
                                 md_config: Dict[str, Any]) -> Dict[str, int]:
    """
    Process multiple JSON trajectory files for VASP MD input generation.
    
    Args:
        json_dir: Directory containing JSON files
        output_dir: Output directory for VASP inputs
        temperatures: List of temperatures for MD
        md_config: MD configuration dictionary
        
    Returns:
        Processing statistics dictionary
    """
    if not ASE_AVAILABLE:
        raise ImportError("ASE is required for batch processing")
    
    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        return {"total": 0, "processed": 0, "failed": 0}
    
    stats = {"total": len(json_files), "processed": 0, "failed": 0}
    
    for json_file in json_files:
        try:
            # Extract structure from JSON
            atoms = extract_final_structure_from_json(json_file)
            if atoms is None:
                stats["failed"] += 1
                continue
            
            # Generate inputs for each temperature
            base_name = json_file.stem
            success = True
            
            for temp in temperatures:
                temp_dir = output_dir / base_name / f"T{int(temp)}K"
                if not write_complete_vasp_md_input(atoms, temp_dir, temp, md_config):
                    success = False
                    break
            
            if success:
                stats["processed"] += 1
            else:
                stats["failed"] += 1
                
        except Exception:
            stats["failed"] += 1
    
    return stats


# ============================================================================
# STRUCTURE CONVERSION UTILITIES
# ============================================================================

def structure_file_to_atoms(file_path: Path) -> Optional['Atoms']:
    """
    Convert structure file (CIF/XYZ) to ASE Atoms object.
    
    Args:
        file_path: Path to structure file
        
    Returns:
        ASE Atoms object or None if conversion fails
    """
    if not ASE_AVAILABLE:
        raise ImportError("ASE is required for structure file conversion")
    
    try:
        from ase.io import read
        return read(str(file_path))
    except Exception:
        return None


def structure_data_to_atoms(structure_data: Dict[str, Any]) -> Optional['Atoms']:
    """
    Convert structure_data dictionary to ASE Atoms object.
    
    Args:
        structure_data: Dictionary with 'atoms' and 'cell' keys
        
    Returns:
        ASE Atoms object or None if conversion fails
    """
    if not ASE_AVAILABLE:
        raise ImportError("ASE is required for structure data conversion")
    
    try:
        symbols = [atom['symbol'] for atom in structure_data['atoms']]
        positions = np.array([atom['position'] for atom in structure_data['atoms']])
        cell = structure_data.get('cell')
        
        if cell is not None:
            return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        else:
            return Atoms(symbols=symbols, positions=positions)
    except Exception:
        return None


# ============================================================================
# TEMPLATE CONFIGURATIONS
# ============================================================================

def get_vasp_md_template(template_name: str) -> Dict[str, Any]:
    """
    Get predefined VASP MD configuration template.
    
    Args:
        template_name: Template identifier
        
    Returns:
        MD configuration dictionary
    """
    templates = {
        "nvt_standard": {
            "name": "Standard NVT",
            "temperatures": [300.0],
            "ensemble": "NVT",
            "thermostat": "nose_hoover",
            "timestep": 1.0,
            "steps": 50000,
            "smass": 0.5
        },
        "nvt_multi_temp": {
            "name": "Multi-temperature NVT",
            "temperatures": [200.0, 300.0, 400.0, 500.0],
            "ensemble": "NVT",
            "thermostat": "nose_hoover",
            "timestep": 1.0,
            "steps": 50000,
            "smass": 0.5
        },
        "npt_standard": {
            "name": "NPT ensemble",
            "temperatures": [300.0],
            "ensemble": "NPT",
            "thermostat": "langevin",
            "timestep": 1.0,
            "steps": 50000,
            "langevin_gamma": 10.0,
            "langevin_gamma_l": 1.0,
            "pstress": 0.0
        },
        "nve_microcanonical": {
            "name": "NVE ensemble",
            "temperatures": [300.0],
            "ensemble": "NVE",
            "timestep": 0.5,
            "steps": 100000
        },
        "high_temp_melting": {
            "name": "High-temperature melting",
            "temperatures": [1000.0, 1500.0, 2000.0],
            "ensemble": "NVT",
            "thermostat": "langevin",
            "timestep": 0.5,
            "steps": 100000,
            "langevin_gamma": 10.0
        }
    }
    
    return templates.get(template_name, templates["nvt_standard"])
