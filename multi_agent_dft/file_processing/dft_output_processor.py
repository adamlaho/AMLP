"""
Module: dft_output_processor.py
Purpose: Process output files from different DFT codes (CP2K, VASP, Gaussian)
         and extract information about forces, energies, and coordinates, storing
         the results in a standardized JSON format.
         
Updated: Enhanced VASP processing to handle all optimization steps and batch processing.
"""

import os
import json
import re
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import math
import numpy as np

# Conversion constants
HARTREE_TO_EV = 27.2114
HARTREE_PER_BOHR_TO_EV_ANG = 51.4221
BOHR_TO_ANGSTROM = 0.52917721
RY_TO_EV = 13.6057

# Base class for DFT output data
class DFTOutputData:
    def __init__(
        self,
        energy: float,
        forces: List[Tuple[str, float, float, float]],
        coordinates: List[Tuple[str, float, float, float]],
        cell_lengths: Optional[List[float]] = None,
        cell_angles: Optional[List[float]] = None,
        atom_types: Optional[List[str]] = None
    ):
        self.energy = energy
        self.forces = forces
        self.coordinates = coordinates
        self.cell_lengths = cell_lengths if cell_lengths is not None else []
        self.cell_angles = cell_angles if cell_angles is not None else []
        self.atom_types = atom_types if atom_types is not None else [c[0] for c in coordinates]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "energy": self.energy,
            "atom_types": self.atom_types,
            "forces": [
                {"element": f[0], "x": f[1], "y": f[2], "z": f[3]} for f in self.forces
            ],
            "coordinates": [
                {"element": c[0], "x": c[1], "y": c[2], "z": c[3]} for c in self.coordinates
            ],
            "cell_lengths": self.cell_lengths,
            "cell_angles": self.cell_angles,
        }

# ====================================================================== #
#                    ENHANCED VASP OUTPUT PROCESSOR                      #
# ====================================================================== #

import os
import math
import json
from pathlib import Path
from typing import Dict, Any, List

class VASPOutputProcessor:
    """Enhanced VASP output processor that handles all optimization steps with robust parsing."""
    
    def __init__(self):
        self.element_map = {
            1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
            11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
            21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
            31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
            41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
            51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd",
            61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb",
            71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
            81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th",
            91: "Pa", 92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm"
        }
    
    def safe_float_conversion(self, value_str: str) -> Optional[float]:
        """Safely convert a string to float, returning None if conversion fails."""
        try:
            # Remove common non-numeric suffixes and prefixes
            cleaned = value_str.strip()
            # Remove trailing characters that might be units or separators
            cleaned = re.sub(r'[^\d\.\-\+eE].*$', '', cleaned)
            if cleaned and cleaned not in ['', '-', '+']:
                return float(cleaned)
        except (ValueError, AttributeError):
            pass
        return None
    
    def extract_energy_robust(self, line: str) -> Optional[float]:
        """Robustly extract energy from a VASP energy line."""
        # Try different energy extraction patterns
        patterns = [
            r'free\s+energy\s+TOTEN\s*=\s*([-\d\.eE\+\-]+)',
            r'energy\(sigma->0\)\s*=\s*([-\d\.eE\+\-]+)',
            r'TOTEN\s*=\s*([-\d\.eE\+\-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                energy_val = self.safe_float_conversion(match.group(1))
                if energy_val is not None:
                    return energy_val
        
        # Fallback: try the original splitting method with error handling
        try:
            if "free  energy   TOTEN" in line and "=" in line:
                parts = line.split("=")
                if len(parts) > 1:
                    energy_part = parts[1].split()[0]
                    return self.safe_float_conversion(energy_part)
        except (IndexError, AttributeError):
            pass
        
        return None
    
    def parse_position_force_line(self, line: str) -> Optional[tuple]:
        """Robustly parse a position and force line."""
        try:
            parts = line.split()
            if len(parts) >= 6:
                # Try to convert the first 6 parts to floats
                nums = []
                for i in range(6):
                    val = self.safe_float_conversion(parts[i])
                    if val is not None:
                        nums.append(val)
                    else:
                        return None
                
                if len(nums) == 6:
                    positions = nums[0:3]
                    forces = nums[3:6]
                    return (positions, forces)
        except (IndexError, AttributeError):
            pass
        return None
    
    def parse_poscar(self, poscar_file: str) -> Dict[str, Any]:
        """Parse VASP POSCAR file to extract initial structure and cell parameters."""
        if not os.path.exists(poscar_file):
            raise FileNotFoundError(f"POSCAR file not found: {poscar_file}")
        
        with open(poscar_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Header
        comment = lines[0]
        scale = float(lines[1])
        
        # Lattice vectors
        lattice = []
        for i in range(3):
            vec = [float(x) * scale for x in lines[2 + i].split()]
            lattice.append(vec)
        
        # Compute lengths
        def norm(v): return math.sqrt(sum(vi*vi for vi in v))
        a, b, c = norm(lattice[0]), norm(lattice[1]), norm(lattice[2])
        
        # Compute angles
        def angle(u, v):
            dot_product = sum(ui*vi for ui, vi in zip(u, v))
            magnitude_product = norm(u) * norm(v)
            # Clamp to avoid numerical errors
            cos_angle = max(-1.0, min(1.0, dot_product / magnitude_product))
            return math.degrees(math.acos(cos_angle))
        
        alpha = angle(lattice[1], lattice[2])
        beta  = angle(lattice[0], lattice[2])
        gamma = angle(lattice[0], lattice[1])
        
        # Compute volume via triple product
        v1, v2, v3 = lattice
        volume = abs(
            v1[0]*(v2[1]*v3[2] - v2[2]*v3[1])
          - v1[1]*(v2[0]*v3[2] - v2[2]*v3[0])
          + v1[2]*(v2[0]*v3[1] - v2[1]*v3[0])
        )
        
        # Elements & counts
        elements    = lines[5].split()
        atom_counts = [int(x) for x in lines[6].split()]
        
        # Coordinates type
        coord_type   = lines[7].lower()
        is_cartesian = coord_type.startswith('c') or coord_type.startswith('k')
        
        # Read positions
        positions = []
        types     = []
        idx = 8
        for elem, count in zip(elements, atom_counts):
            for _ in range(count):
                pos = [float(x) for x in lines[idx].split()[:3]]
                if not is_cartesian:
                    # fractional → Cartesian
                    cart = [0.0, 0.0, 0.0]
                    for i in range(3):
                        for j in range(3):
                            cart[i] += pos[j] * lattice[j][i]
                    pos = cart
                positions.append(pos)
                types.append(elem)
                idx += 1
        
        return {
            "comment":      comment,
            "lattice":      lattice,
            "cell_lengths": [a, b, c],
            "cell_angles":  [alpha, beta, gamma],
            "cell_volume":  volume,
            "atom_types":   types,
            "atom_positions": positions,
            "num_atoms":    sum(atom_counts)
        }
    
    def extract_all_steps_from_outcar(self,
                                     outcar_file: str,
                                     initial_data: Dict[str, Any]
                                    ) -> List[Dict[str, Any]]:
        """Extract all ionic‐relaxation steps with robust parsing."""
        if not os.path.exists(outcar_file):
            raise FileNotFoundError(f"OUTCAR file not found: {outcar_file}")
        
        with open(outcar_file, 'r') as f:
            lines = f.readlines()
        
        atom_types = initial_data["atom_types"]
        num_atoms  = initial_data["num_atoms"]
        
        steps = []
        current_energy = None
        
        # start with POSCAR cell
        current_cell = [row[:] for row in initial_data["lattice"]]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # energy - use robust extraction
            if "free  energy   TOTEN" in line or "energy(sigma->0)" in line:
                energy = self.extract_energy_robust(line)
                if energy is not None:
                    current_energy = energy
            
            # updated cell
            if "VOLUME and BASIS-vectors are now" in line:
                # advance until 'direct lattice vectors' header
                while i < len(lines) and "direct lattice vectors" not in lines[i].lower():
                    i += 1
                
                if i < len(lines):
                    # next three lines are the direct lattice vectors
                    new_cell = []
                    for j in range(3):
                        if i + j + 1 < len(lines):
                            try:
                                parts = lines[i + j + 1].split()
                                if len(parts) >= 3:
                                    vec = [self.safe_float_conversion(parts[k]) for k in range(3)]
                                    if all(v is not None for v in vec):
                                        new_cell.append(vec)
                            except:
                                continue
                    
                    if len(new_cell) == 3:
                        current_cell = new_cell
                    i += 3  # Move past the lattice vectors
            
            # positions + forces
            if "POSITION" in line and "TOTAL-FORCE" in line:
                i += 1  # skip header separator
                
                # Skip any additional separator lines
                while i < len(lines) and ("---" in lines[i] or not lines[i].strip()):
                    i += 1
                
                poses, forces = [], []
                atoms_read = 0
                
                # Read the atom data
                while i < len(lines) and atoms_read < num_atoms:
                    line_data = self.parse_position_force_line(lines[i])
                    if line_data is not None:
                        pos, force = line_data
                        poses.append(pos)
                        forces.append(force)
                        atoms_read += 1
                    else:
                        # If we hit a non-data line, break
                        if "---" in lines[i] or not lines[i].strip():
                            break
                    i += 1
                
                # If we successfully read all atoms and have an energy
                if len(poses) == num_atoms and len(forces) == num_atoms and current_energy is not None:
                    # compute lengths & angles & volume
                    def norm(v): return math.sqrt(sum(vi*vi for vi in v))
                    def ang(u,v):
                        dot_product = sum(ui*vi for ui,vi in zip(u,v))
                        magnitude_product = norm(u) * norm(v)
                        if magnitude_product == 0:
                            return 0.0
                        cos_angle = max(-1.0, min(1.0, dot_product / magnitude_product))
                        return math.degrees(math.acos(cos_angle))
                    
                    try:
                        a_vec, b_vec, c_vec = current_cell
                        a, b, c = norm(a_vec), norm(b_vec), norm(c_vec)
                        alpha = ang(b_vec, c_vec)
                        beta  = ang(a_vec, c_vec)
                        gamma = ang(a_vec, b_vec)
                        
                        # triple‐product volume
                        v1, v2, v3 = a_vec, b_vec, c_vec
                        vol = abs(
                            v1[0]*(v2[1]*v3[2] - v2[2]*v3[1])
                          - v1[1]*(v2[0]*v3[2] - v2[2]*v3[0])
                          + v1[2]*(v2[0]*v3[1] - v2[1]*v3[0])
                        )
                        
                        # assemble step
                        step = {
                            "energy":       current_energy,
                            "atom_types":   atom_types,
                            "coordinates": [
                                {"element": atom_types[j], **dict(zip(("x","y","z"), pos))}
                                for j,pos in enumerate(poses)
                            ],
                            "forces": [
                                {"element": atom_types[j], **dict(zip(("x","y","z"), f))}
                                for j,f in enumerate(forces)
                            ],
                            "cell_lengths": [a, b, c],
                            "cell_angles":  [alpha, beta, gamma],
                            "cell_volume":  vol
                        }
                        steps.append(step)
                        
                    except Exception as e:
                        print(f"Warning: Error computing cell parameters for step: {e}")
                        continue
                
                # Continue from current position
                continue
            
            i += 1
        
        return steps
    
    def process_single_vasp_calculation(self, calc_dir: Path) -> Dict[str, Any]:
        """Process a single VASP calculation directory with error handling."""
        poscar = calc_dir / "POSCAR"
        outcar = calc_dir / "OUTCAR"
        
        if not poscar.exists():
            raise FileNotFoundError(f"POSCAR not found in {calc_dir}")
        if not outcar.exists():
            raise FileNotFoundError(f"OUTCAR not found in {calc_dir}")
        
        try:
            initial = self.parse_poscar(str(poscar))
            steps   = self.extract_all_steps_from_outcar(str(outcar), initial)
            
            return {
                "directory":          str(calc_dir),
                "num_steps":          len(steps),
                "initial_structure":  initial,
                "optimization_steps": steps
            }
        except Exception as e:
            raise Exception(f"Error processing {calc_dir}: {str(e)}")
    
    def process_directory_batch(self,
                                parent_dir: Path,
                                recursive: bool = True
                               ) -> Dict[str, Any]:
        """Process all VASP calculations in a parent directory with better error handling."""
        results = {}
        ok_count, err_count = 0, 0
        
        if recursive:
            vasp_dirs = {d.parent for d in parent_dir.rglob("OUTCAR") if d.is_file()}
        else:
            vasp_dirs = {d for d in parent_dir.iterdir()
                         if d.is_dir() and (d / "OUTCAR").exists()}
        
        print(f"Found {len(vasp_dirs)} VASP calculation directories to process")
        
        for d in sorted(vasp_dirs):
            try:
                print(f"Processing {d.name}...", end=" ")
                data = self.process_single_vasp_calculation(d)
                
                # write per‐step JSON
                out_json = d / f"{d.name}_vasp_output.json"
                with open(out_json, 'w') as fp:
                    json.dump(data["optimization_steps"], fp, indent=2)
                
                results[str(d)] = data
                ok_count += 1
                print(f"OK ({data['num_steps']} steps)")
                
            except Exception as e:
                print(f"ERROR: {e}")
                results[str(d)] = {"error": str(e)}
                err_count += 1
        
        summary = {
            "total_directories":      len(vasp_dirs),
            "processed_successfully": ok_count,
            "errors":                 err_count,
            "results":                results
        }
        
        summary_file = parent_dir / "vasp_processing_summary.json"
        with open(summary_file, 'w') as fp:
            json.dump(summary, fp, indent=2)
        
        print("\nProcessing complete:")
        print(f"  - Successfully processed: {ok_count}")
        print(f"  - Errors:                {err_count}")
        print(f"  - Summary saved to:     {summary_file}")
        
        return summary

# ====================================================================== #
#                    CP2K OUTPUT PROCESSING FUNCTIONS                    #
# ====================================================================== #

def extract_unit_cell_cp2k(input_file: str) -> Tuple[List[float], List[float]]:
    """
    Reads the &CELL section in the CP2K input file to extract:
      - ABC [angstrom] a b c
      - ALPHA_BETA_GAMMA [deg] alpha beta gamma

    Returns
    -------
    (cell_lengths, cell_angles) : ([float, float, float], [float, float, float])
    """
    cell_lengths = []
    cell_angles = []

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, 'r') as f:
        lines = f.readlines()

    cell_start = None
    for i, line in enumerate(lines):
        if "&CELL" in line.upper():
            cell_start = i
            break

    if cell_start is None:
        print(f"Warning: &CELL section not found in {input_file}")
        return (cell_lengths, cell_angles)

    # read lines after &CELL until &END CELL
    for line in lines[cell_start+1:]:
        if "&END CELL" in line.upper():
            break
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        if parts[0].upper() == "ABC" and parts[1].upper() == "[ANGSTROM]":
            # e.g. "ABC [angstrom] 11.25  5.95  13.60"
            cell_lengths = list(map(float, parts[2:5]))
        elif parts[0].upper() == "ALPHA_BETA_GAMMA" and parts[1].upper() == "[DEG]":
            # e.g. "ALPHA_BETA_GAMMA [deg]  90.0  99.5  90.0"
            cell_angles = list(map(float, parts[2:5]))

    return (cell_lengths, cell_angles)

def parse_frac_xyz_file_cp2k(xyz_file: str) -> List[Dict[str, Any]]:
    """
    Reads a multi-step fractional-coordinate .xyz file with format:
       num_atoms
       E = <energy in Hartree>
       Element frac_x frac_y frac_z
       ...
       num_atoms
       E = <energy in Hartree>
       ...

    Returns a list of geometry dicts:
       [{
          "energy": <float in eV or None>,
          "coordinates": [(element, fx, fy, fz), ...],
          "num_atoms": N
        }, ...]
    """
    if not os.path.exists(xyz_file):
        raise FileNotFoundError(f"XYZ file not found: {xyz_file}")

    geometries = []
    with open(xyz_file, 'r') as f:
        lines = [ln.strip() for ln in f]

    i = 0
    while i < len(lines):
        # skip empty lines
        if not lines[i]:
            i += 1
            continue

        # parse number of atoms
        try:
            num_atoms = int(lines[i])
        except ValueError:
            break

        # line i+1 may contain "E = ..."
        if i + 1 >= len(lines):
            break
        energy_line = lines[i+1]
        i += 2

        # find energy in Hartree and convert to eV
        match = re.search(r"E\s*=\s*([-\d.]+)", energy_line)
        if match:
            energy_hartree = float(match.group(1))
            energy = energy_hartree * HARTREE_TO_EV
        else:
            energy = None
            print(f"Warning: Could not find energy in line: {energy_line}")

        # read fractional coords
        coords = []
        for _ in range(num_atoms):
            if i >= len(lines):
                break
            parts = lines[i].split()
            i += 1
            if len(parts) == 4:
                element = parts[0]
                fx, fy, fz = map(float, parts[1:])
                coords.append((element, fx, fy, fz))

        if len(coords) == num_atoms:
            geometries.append({
                "energy": energy,
                "coordinates": coords,  # fractional coordinates
                "num_atoms": num_atoms
            })
        else:
            print(f"Warning: Not enough coordinates read for a frame in {xyz_file}")

    return geometries

def frac_to_cart_cp2k(frac_coords: List[Tuple[str, float, float, float]],
                 cell_lengths: List[float],
                 cell_angles: List[float]) -> List[Tuple[str, float, float, float]]:
    """
    Convert fractional coordinates to Cartesian coordinates using cell parameters.
    """
    if len(cell_lengths) < 3 or len(cell_angles) < 3:
        return frac_coords

    a, b, c = cell_lengths
    alpha, beta, gamma = cell_angles

    alpha_r = math.radians(alpha)
    beta_r  = math.radians(beta)
    gamma_r = math.radians(gamma)

    # Lattice vector a
    ax = a; ay = 0.0; az = 0.0
    # Lattice vector b
    bx = b * math.cos(gamma_r)
    by = b * math.sin(gamma_r)
    bz = 0.0
    # Lattice vector c
    cx = c * math.cos(beta_r)
    cy = c * (math.cos(alpha_r) - math.cos(beta_r)*math.cos(gamma_r)) / math.sin(gamma_r)
    cz = c * math.sqrt(1 - math.cos(alpha_r)**2 - math.cos(beta_r)**2 - math.cos(gamma_r)**2 + 2*math.cos(alpha_r)*math.cos(beta_r)*math.cos(gamma_r)) / math.sin(gamma_r)

    cell_matrix = [
        [ax, ay, az],
        [bx, by, bz],
        [cx, cy, cz],
    ]

    cart_coords = []
    for (el, fx, fy, fz) in frac_coords:
        x = fx*cell_matrix[0][0] + fy*cell_matrix[1][0] + fz*cell_matrix[2][0]
        y = fx*cell_matrix[0][1] + fy*cell_matrix[1][1] + fz*cell_matrix[2][1]
        z = fx*cell_matrix[0][2] + fy*cell_matrix[1][2] + fz*cell_matrix[2][2]
        cart_coords.append((el, x, y, z))
    return cart_coords

def extract_forces_per_geometry_cp2k(output_file: str, num_atoms: int) -> List[List[Tuple[str, float, float, float]]]:
    """
    Searches for blocks starting with "ATOMIC FORCES in [a.u.]" in the CP2K output file.
    Skips header lines and reads the next 'num_atoms' lines.
    Converts (fx, fy, fz) from Hartree/Bohr to eV/Å.
    """
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Output file not found: {output_file}")

    forces_per_geometry = []
    with open(output_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "ATOMIC FORCES in [a.u.]" in line:
            i += 3  # skip header lines
            current_forces = []
            for _ in range(num_atoms):
                if i >= len(lines):
                    break
                parts = lines[i].split()
                i += 1
                if len(parts) >= 6:
                    element = parts[2]
                    fx, fy, fz = map(float, parts[3:6])
                    # Convert forces from Hartree/Bohr to eV/Å
                    fx *= HARTREE_PER_BOHR_TO_EV_ANG
                    fy *= HARTREE_PER_BOHR_TO_EV_ANG
                    fz *= HARTREE_PER_BOHR_TO_EV_ANG
                    current_forces.append((element, fx, fy, fz))
            if len(current_forces) == num_atoms:
                forces_per_geometry.append(current_forces)
        else:
            i += 1
    return forces_per_geometry

def process_cp2k_output(input_file: str, output_file: str, frac_xyz_file: str, output_json: str, do_frac_to_cart: bool = True):
    """
    Process CP2K output files and save the extracted data to a JSON file.
    
    Parameters
    ----------
    input_file : str
        Path to the CP2K input file (.inp)
    output_file : str
        Path to the CP2K output file
    frac_xyz_file : str
        Path to the fractional coordinates XYZ file
    output_json : str
        Path to save the output JSON file
    do_frac_to_cart : bool, optional
        Whether to convert fractional coordinates to Cartesian, by default True
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing CP2K input file: {input_file}")
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Missing CP2K output file: {output_file}")
    if not os.path.exists(frac_xyz_file):
        raise FileNotFoundError(f"Missing fractional .xyz file: {frac_xyz_file}")

    # 1) Extract cell parameters
    cell_lengths, cell_angles = extract_unit_cell_cp2k(input_file)
    print(f"Extracted cell lengths: {cell_lengths}, cell angles: {cell_angles}")

    # 2) Parse geometries from the fractional .xyz file
    geometries = parse_frac_xyz_file_cp2k(frac_xyz_file)
    if not geometries:
        raise ValueError(f"No geometries found in {frac_xyz_file}")
    num_atoms = geometries[0]["num_atoms"]

    # 3) Extract forces for each geometry
    forces_list = extract_forces_per_geometry_cp2k(output_file, num_atoms)

    # 4) Combine data for each geometry
    all_data = []
    for idx, geo in enumerate(geometries):
        energy = geo["energy"]
        frac_coords = geo["coordinates"]
        atom_types = [atom[0] for atom in frac_coords]

        # Convert fractional to Cartesian if desired
        if do_frac_to_cart and cell_lengths and cell_angles:
            cart_coords = frac_to_cart_cp2k(frac_coords, cell_lengths, cell_angles)
        else:
            cart_coords = frac_coords

        # Get forces for this geometry
        forces = forces_list[idx] if idx < len(forces_list) else []

        entry = DFTOutputData(
            energy=energy,
            forces=forces,
            coordinates=cart_coords,
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
            atom_types=atom_types
        )
        all_data.append(entry.to_dict())

    # 5) Write data to JSON file
    with open(output_json, 'w') as jf:
        json.dump(all_data, jf, indent=2)

    print(f"\nWrote {len(all_data)} geometries to {output_json}")

# ====================================================================== #
#                    ORIGINAL VASP OUTPUT PROCESSING                     #
# ====================================================================== #

def parse_poscar(poscar_file: str) -> Dict[str, Any]:
    """
    Parse VASP POSCAR file to extract cell parameters and atom coordinates.
    
    Returns
    -------
    dict
        A dictionary containing cell parameters and atom coordinates.
    """
    if not os.path.exists(poscar_file):
        raise FileNotFoundError(f"POSCAR file not found: {poscar_file}")
    
    with open(poscar_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Parse header
    comment = lines[0]
    scaling_factor = float(lines[1])
    
    # Parse lattice vectors
    lattice = []
    for i in range(3):
        lattice.append([float(x) * scaling_factor for x in lines[2+i].split()])
    
    # Get lattice parameters (a, b, c, alpha, beta, gamma)
    a = math.sqrt(sum(x*x for x in lattice[0]))
    b = math.sqrt(sum(x*x for x in lattice[1]))
    c = math.sqrt(sum(x*x for x in lattice[2]))
    
    # Compute cell angles (in degrees)
    alpha = math.degrees(math.acos((lattice[1][0]*lattice[2][0] + 
                                  lattice[1][1]*lattice[2][1] + 
                                  lattice[1][2]*lattice[2][2]) / (b*c)))
    beta = math.degrees(math.acos((lattice[0][0]*lattice[2][0] + 
                                 lattice[0][1]*lattice[2][1] + 
                                 lattice[0][2]*lattice[2][2]) / (a*c)))
    gamma = math.degrees(math.acos((lattice[0][0]*lattice[1][0] + 
                                  lattice[0][1]*lattice[1][1] + 
                                  lattice[0][2]*lattice[1][2]) / (a*b)))
    
    # Parse element types and counts
    elements = lines[5].split()
    atom_counts = [int(x) for x in lines[6].split()]
    
    # Parse coordinate type
    coord_type = lines[7].lower()
    is_cartesian = coord_type.startswith('c') or coord_type.startswith('k')
    
    # Parse atomic positions
    atom_positions = []
    atom_types = []
    line_index = 8
    
    for element, count in zip(elements, atom_counts):
        for _ in range(count):
            position = [float(x) for x in lines[line_index].split()[:3]]
            if not is_cartesian:
                # Convert fractional to Cartesian
                cart = [0, 0, 0]
                for i in range(3):
                    for j in range(3):
                        cart[i] += position[j] * lattice[j][i]
                position = cart
            atom_positions.append(position)
            atom_types.append(element)
            line_index += 1
    
    # Create coordinate tuples
    coordinates = [(element, pos[0], pos[1], pos[2]) for element, pos in zip(atom_types, atom_positions)]
    
    return {
        "comment": comment,
        "lattice": lattice,
        "cell_lengths": [a, b, c],
        "cell_angles": [alpha, beta, gamma],
        "atom_types": atom_types,
        "coordinates": coordinates,
        "num_atoms": sum(atom_counts)
    }

def extract_energy_forces_vasp(outcar_file: str) -> Dict[str, Any]:
    """
    Extract energy and forces from VASP OUTCAR file (original - last step only).
    
    Returns
    -------
    dict
        A dictionary containing energy and forces.
    """
    if not os.path.exists(outcar_file):
        raise FileNotFoundError(f"OUTCAR file not found: {outcar_file}")
    
    energy = None
    forces = []
    atom_types = []
    
    with open(outcar_file, 'r') as f:
        lines = f.readlines()
    
    # Extract energy (find the last occurrence)
    for line in reversed(lines):
        if "free  energy   TOTEN" in line:
            energy = float(line.split("=")[1].split()[0])
            break
    
    # Extract forces and atom types
    force_block = False
    force_data = []
    
    for i, line in enumerate(lines):
        if "POSITION" in line and "TOTAL-FORCE" in line:
            force_block = True
            continue
        
        if force_block:
            if "-----------------------------------" in line:
                continue
            if not line.strip() or "------------------------------------------------------------" in line:
                force_block = False
                continue
            
            parts = line.split()
            if len(parts) >= 6:
                pos = [float(parts[0]), float(parts[1]), float(parts[2])]
                force = [float(parts[3]), float(parts[4]), float(parts[5])]
                force_data.append((pos, force))
    
    # Extract atom types (need to match them with forces)
    atom_types_section = False
    for i, line in enumerate(lines):
        if "VRHFIN" in line:
            element = line.split("=")[1].split(":")[0].strip()
            atom_types.append(element)
    
    # If we couldn't extract atom types directly, try to get them from POTCAR data in OUTCAR
    if not atom_types:
        poscar_section = False
        for i, line in enumerate(lines):
            if "POTCAR:" in line:
                element = line.split(":")[1].split()[0].strip()
                if element not in atom_types:
                    atom_types.append(element)
    
    # Try to match atom types with forces
    if atom_types and force_data:
        # We need to know how many of each atom type there are
        ion_types_count = []
        for i, line in enumerate(lines):
            if "ions per type =" in line:
                ion_types_count = [int(x) for x in line.split("=")[1].split()]
                break
        
        if ion_types_count:
            # Now we can construct the forces list with atom types
            forces = []
            atom_index = 0
            for type_index, count in enumerate(ion_types_count):
                element = atom_types[type_index] if type_index < len(atom_types) else "X"
                for _ in range(count):
                    if atom_index < len(force_data):
                        _, force = force_data[atom_index]
                        forces.append((element, force[0], force[1], force[2]))
                        atom_index += 1
    
    return {
        "energy": energy,
        "forces": forces
    }

def process_vasp_output(poscar_file: str, outcar_file: str, output_json: str):
    """
    Process VASP output files and save the extracted data to a JSON file (original - last step only).
    
    Parameters
    ----------
    poscar_file : str
        Path to the VASP POSCAR file
    outcar_file : str
        Path to the VASP OUTCAR file
    output_json : str
        Path to save the output JSON file
    """
    if not os.path.exists(poscar_file):
        raise FileNotFoundError(f"Missing POSCAR file: {poscar_file}")
    if not os.path.exists(outcar_file):
        raise FileNotFoundError(f"Missing OUTCAR file: {outcar_file}")
    
    # Parse POSCAR file
    poscar_data = parse_poscar(poscar_file)
    
    # Extract energy and forces from OUTCAR
    outcar_data = extract_energy_forces_vasp(outcar_file)
    
    # Create output data
    entry = DFTOutputData(
        energy=outcar_data["energy"],
        forces=outcar_data["forces"] if outcar_data["forces"] else [],
        coordinates=poscar_data["coordinates"],
        cell_lengths=poscar_data["cell_lengths"],
        cell_angles=poscar_data["cell_angles"],
        atom_types=poscar_data["atom_types"]
    )
    
    # Write data to JSON file
    with open(output_json, 'w') as jf:
        json.dump([entry.to_dict()], jf, indent=2)
    
    print(f"\nWrote geometry data to {output_json}")

# ====================================================================== #
#                  GAUSSIAN OUTPUT PROCESSING FUNCTIONS                  #
# ====================================================================== #

def extract_gaussian_xyz(log_file: str) -> List[Tuple[str, float, float, float]]:
    """
    Extract Cartesian coordinates from Gaussian log file (last geometry).
    
    Returns
    -------
    list
        A list of tuples (element, x, y, z) with coordinates in Angstroms.
    """
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    coords = []
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Find the last geometry block
    input_orientation_blocks = []
    for i, line in enumerate(lines):
        if "Input orientation:" in line or "Standard orientation:" in line:
            input_orientation_blocks.append(i)
    
    if not input_orientation_blocks:
        raise ValueError(f"No geometry blocks found in {log_file}")
    
    # Parse the last geometry block
    last_block = input_orientation_blocks[-1]
    i = last_block + 5  # Skip header
    
    while i < len(lines):
        line = lines[i].strip()
        if "---------------------------------------------------------------------" in line:
            break
        
        parts = line.split()
        if len(parts) >= 6:
            atom_num = int(parts[1])
            atom_type = int(parts[2])
            x, y, z = map(float, parts[3:6])
            
            # Convert atomic number to element symbol
            element = convert_atomic_num_to_element(atom_type)
            coords.append((element, x, y, z))
        
        i += 1
    
    return coords

def convert_atomic_num_to_element(atomic_num: int) -> str:
    """Convert atomic number to element symbol."""
    element_map = {
        1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
        11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
        21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
        31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
        41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
        51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd",
        61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb",
        71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
        81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th",
        91: "Pa", 92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm"
    }
    
    return element_map.get(atomic_num, f"X{atomic_num}")

def extract_energy_gaussian(log_file: str) -> Optional[float]:
    """
    Extract the final energy from a Gaussian log file (in eV).
    
    Returns
    -------
    float or None
        The energy in eV, or None if not found.
    """
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    energy = None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Try different energy lines, from most specific to least
    energy_patterns = [
        r"SCF Done:\s+E\([^)]+\)\s*=\s*([-\d.]+)",  # SCF energy
        r"EUMP2\s*=\s*([-\d.]+)",  # MP2 energy
        r"EUMP3\s*=\s*([-\d.]+)",  # MP3 energy
        r"EUMP4\(SDQ\)\s*=\s*([-\d.]+)",  # MP4 energy
        r"CCSD\(T\)\s*=\s*([-\d.]+)",  # CCSD(T) energy
    ]
    
    for pattern in energy_patterns:
        for line in reversed(lines):
            match = re.search(pattern, line)
            if match:
                energy_hartree = float(match.group(1))
                energy = energy_hartree * HARTREE_TO_EV
                break
        
        if energy is not None:
            break
    
    return energy

def extract_forces_gaussian(log_file: str) -> List[Tuple[str, float, float, float]]:
    """
    Extract forces from a Gaussian log file.
    
    Gaussian doesn't output forces directly, but we can extract them from the 
    optimization steps or from a force calculation.
    
    Returns
    -------
    list
        A list of tuples (element, fx, fy, fz) with forces in eV/Å.
    """
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    forces = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Look for force blocks (multiple patterns to try)
    force_patterns = [
        ("Forces (Hartree/Bohr)", 3, HARTREE_PER_BOHR_TO_EV_ANG),
        ("Forces (Hartrees/Bohr)", 3, HARTREE_PER_BOHR_TO_EV_ANG),
        ("Cartesian Gradient", 1, HARTREE_PER_BOHR_TO_EV_ANG),
    ]
    
    force_blocks = []
    for pattern, skip, _ in force_patterns:
        for i, line in enumerate(lines):
            if pattern in line:
                force_blocks.append((i, skip, pattern))
    
    if not force_blocks:
        print(f"Warning: No force blocks found in {log_file}")
        return forces
    
    # Get the last force block
    last_block = force_blocks[-1]
    i, skip, pattern = last_block
    i += skip
    
    conversion = next(conv for pat, _, conv in force_patterns if pat == pattern)
    
    # Format depends on the pattern
    if "Gradient" in pattern:
        # "Cartesian Gradient" format
        coords = extract_gaussian_xyz(log_file)
        element_idx = 0
        
        while i < len(lines):
            line = lines[i].strip()
            if not line or "---" in line:
                break
            
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                fx, fy, fz = map(float, parts[1:4])
                # Convert and negate to get forces (negative gradient)
                fx = -fx * conversion
                fy = -fy * conversion
                fz = -fz * conversion
                
                if element_idx < len(coords):
                    element = coords[element_idx][0]
                    forces.append((element, fx, fy, fz))
                    element_idx += 1
            
            i += 1
    else:
        # "Forces (Hartree/Bohr)" format
        while i < len(lines):
            line = lines[i].strip()
            if not line or "---" in line:
                break
            
            parts = line.split()
            if len(parts) >= 5 and parts[0].isdigit():
                element = parts[1]
                fx, fy, fz = map(float, parts[2:5])
                # Convert to eV/Å
                fx *= conversion
                fy *= conversion
                fz *= conversion
                forces.append((element, fx, fy, fz))
            
            i += 1
    
    return forces

def process_gaussian_output(log_file: str, output_json: str):
    """
    Process Gaussian output files and save the extracted data to a JSON file.
    
    Parameters
    ----------
    log_file : str
        Path to the Gaussian log file
    output_json : str
        Path to save the output JSON file
    """
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Missing Gaussian log file: {log_file}")
    
    # Extract coordinates
    coordinates = extract_gaussian_xyz(log_file)
    
    # Extract energy
    energy = extract_energy_gaussian(log_file)
    
    # Extract forces
    forces = extract_forces_gaussian(log_file)
    
    # Create output data
    entry = DFTOutputData(
        energy=energy,
        forces=forces,
        coordinates=coordinates,
        # Gaussian calculations often don't have cell parameters for molecular systems
        cell_lengths=None,
        cell_angles=None,
        atom_types=[c[0] for c in coordinates]
    )
    
    # Write data to JSON file
    with open(output_json, 'w') as jf:
        json.dump([entry.to_dict()], jf, indent=2)
    
    print(f"\nWrote geometry data to {output_json}")

# ====================================================================== #
#                         MAIN PROCESSING FUNCTION                       #
# ====================================================================== #

def process_dft_output(code_type: str, **kwargs):
    """
    Process DFT output files based on the code type.
    
    Parameters
    ----------
    code_type : str
        The type of DFT code ('cp2k', 'vasp', or 'gaussian').
    **kwargs
        Code-specific arguments for output processing.
    
    Notes
    -----
    For CP2K, expected kwargs:
        - input_file: str
        - output_file: str
        - frac_xyz_file: str
        - output_json: str
        - do_frac_to_cart: bool (optional)
    
    For VASP, expected kwargs:
        - poscar_file: str
        - outcar_file: str
        - output_json: str
    
    For Gaussian, expected kwargs:
        - log_file: str
        - output_json: str
    """
    if code_type.lower() == 'cp2k':
        required_kwargs = ['input_file', 'output_file', 'frac_xyz_file', 'output_json']
        for kwarg in required_kwargs:
            if kwarg not in kwargs:
                raise ValueError(f"Missing required parameter '{kwarg}' for CP2K output processing")
        process_cp2k_output(**kwargs)
    elif code_type.lower() == 'vasp':
        required_kwargs = ['poscar_file', 'outcar_file', 'output_json']
        for kwarg in required_kwargs:
            if kwarg not in kwargs:
                raise ValueError(f"Missing required parameter '{kwarg}' for VASP output processing")
        process_vasp_output(**kwargs)
    elif code_type.lower() == 'gaussian':
        required_kwargs = ['log_file', 'output_json']
        for kwarg in required_kwargs:
            if kwarg not in kwargs:
                raise ValueError(f"Missing required parameter '{kwarg}' for Gaussian output processing")
        process_gaussian_output(**kwargs)
    else:
        raise ValueError(f"Unsupported DFT code: {code_type}")
