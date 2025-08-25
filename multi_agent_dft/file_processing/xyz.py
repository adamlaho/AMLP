"""
XYZ file processing module for the Multi-Agent DFT Research System.
"""

import os
import re
import numpy as np
import logging
from pathlib import Path
import periodictable as pt
from collections import Counter

logger = logging.getLogger(__name__)


def parse_xyz(xyz_file):
    """
    Parse an XYZ file and extract structure information.
    
    Args:
        xyz_file (str or Path): Path to the XYZ file.
    
    Returns:
        dict: Parsed structure data.
    """
    structure_data = {
        'atoms': [],
        'meta': {
            'filename': str(Path(xyz_file).name),
            'source': 'XYZ',
        }
    }
    
    try:
        with open(xyz_file, 'r') as f:
            xyz_lines = f.readlines()
        
        # First line is the number of atoms
        try:
            num_atoms = int(xyz_lines[0].strip())
        except ValueError:
            logger.error(f"Invalid atom count in XYZ file {xyz_file}")
            return None
        
        # Second line is a comment
        comment = xyz_lines[1].strip()
        structure_data['meta']['comment'] = comment
        
        # Extract lattice vectors from comment if available
        # Format: Lattice="a11 a12 a13 a21 a22 a23 a31 a32 a33" or similar
        lattice_match = re.search(r'[Ll]attice="([^"]+)"', comment)
        if lattice_match:
            try:
                lattice_values = [float(x) for x in lattice_match.group(1).split()]
                if len(lattice_values) == 9:
                    cell = np.array(lattice_values).reshape(3, 3)
                    structure_data['cell'] = cell.tolist()
            except (ValueError, IndexError):
                logger.warning(f"Could not parse lattice vectors from XYZ comment")
        
        # Remaining lines are atoms with coordinates
        for i in range(2, min(2 + num_atoms, len(xyz_lines))):
            line = xyz_lines[i].strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 4:
                logger.warning(f"Skipping invalid atom line in XYZ: {line}")
                continue
            
            symbol = parts[0]
            
            # Standardize atom symbol
            symbol = symbol[0].upper() + symbol[1:].lower() if len(symbol) > 1 else symbol.upper()
            
            # Try to validate the element symbol
            try:
                element = getattr(pt, symbol)
                # If valid, use the proper symbol
                symbol = element.symbol
            except AttributeError:
                logger.warning(f"Unknown element symbol in XYZ: {symbol}")
            
            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                
                structure_data['atoms'].append({
                    'symbol': symbol,
                    'position': [x, y, z]
                })
            
            except ValueError:
                logger.warning(f"Error parsing coordinates in XYZ line: {line}")
        
        # Check if we found the expected number of atoms
        if len(structure_data['atoms']) != num_atoms:
            logger.warning(f"Expected {num_atoms} atoms but found {len(structure_data['atoms'])} in {xyz_file}")
        
        # If no cell was found in the comment, estimate a bounding box
        if 'cell' not in structure_data and structure_data['atoms']:
            # Get min and max coordinates
            min_coords = np.array([float('inf'), float('inf'), float('inf')])
            max_coords = np.array([float('-inf'), float('-inf'), float('-inf')])
            
            for atom in structure_data['atoms']:
                pos = np.array(atom['position'])
                min_coords = np.minimum(min_coords, pos)
                max_coords = np.maximum(max_coords, pos)
            
            # Add padding
            padding = 5.0  # Angstroms
            min_coords -= padding
            max_coords += padding
            
            # Create a diagonal cell matrix
            cell_dimensions = max_coords - min_coords
            cell = np.diag(cell_dimensions).tolist()
            structure_data['cell'] = cell
            
            # Shift all atoms by min_coords to have them inside the cell
            for atom in structure_data['atoms']:
                atom['position'] = (np.array(atom['position']) - min_coords).tolist()
        
        logger.info(f"Successfully parsed XYZ file {xyz_file}: {len(structure_data['atoms'])} atoms")
        return structure_data
    
    except Exception as e:
        logger.error(f"Error reading XYZ file {xyz_file}: {e}")
        return None


def xyz_to_mol_formula(structure_data):
    """
    Convert structure data to molecular formula.
    
    Args:
        structure_data (dict): Structure data from parse_xyz().
    
    Returns:
        str: Molecular formula.
    """
    if not structure_data or 'atoms' not in structure_data:
        return "Unknown"
    
    # Count atoms by element
    element_counts = Counter(atom['symbol'] for atom in structure_data['atoms'])
    
    # Sort elements by atomic number
    sorted_elements = sorted(element_counts.items(), 
                           key=lambda x: getattr(pt, x[0]).number)
    
    # Build formula
    formula_parts = []
    for symbol, count in sorted_elements:
        if count == 1:
            formula_parts.append(symbol)
        else:
            formula_parts.append(f"{symbol}{count}")
    
    return "".join(formula_parts)


def process_xyz_files(input_dir, output_dir=None, file_pattern="*.xyz", dft_code="cp2k", dft_config=None):
    """
    Process all XYZ files in a directory and generate DFT input files.
    
    Args:
        input_dir (str or Path): Directory containing XYZ files.
        output_dir (str or Path, optional): Directory to save output files.
        file_pattern (str, optional): Glob pattern for XYZ files.
        dft_code (str, optional): DFT code to generate input for ('cp2k', 'vasp', 'gaussian').
        dft_config (dict, optional): Configuration for the DFT input generation.
    
    Returns:
        list: List of paths to generated input files.
    """
    input_dir = Path(input_dir).expanduser().resolve()
    
    if output_dir is None:
        output_dir = input_dir / "dft_inputs"
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import the appropriate DFT module
    if dft_code.lower() == "cp2k":
        from ..dft.cp2k import save_cp2k_input
    elif dft_code.lower() == "vasp":
        from ..dft.vasp import save_vasp_inputs
    elif dft_code.lower() == "gaussian":
        from ..dft.gaussian import save_gaussian_input
    else:
        logger.error(f"Unsupported DFT code: {dft_code}")
        return []
    
    # Process each XYZ file
    xyz_files = list(input_dir.glob(file_pattern))
    if not xyz_files:
        logger.warning(f"No XYZ files found in {input_dir} with pattern {file_pattern}")
        return []
    
    logger.info(f"Found {len(xyz_files)} XYZ files to process")
    generated_files = []
    
    for xyz_file in xyz_files:
        try:
            # Parse the XYZ file
            structure_data = parse_xyz(xyz_file)
            if structure_data is None:
                logger.error(f"Failed to parse XYZ file: {xyz_file}")
                continue
            
            # Create structure-specific output directory
            struct_name = xyz_file.stem
            struct_output_dir = output_dir / struct_name
            struct_output_dir.mkdir(exist_ok=True)
            
            # Generate DFT input files
            if dft_code.lower() == "cp2k":
                input_file = struct_output_dir / f"{struct_name}.inp"
                success = save_cp2k_input(structure_data, input_file, dft_config)
                if success:
                    generated_files.append(input_file)
            
            elif dft_code.lower() == "vasp":
                # For VASP, we create a directory with all input files
                vasp_dir = struct_output_dir / "vasp"
                vasp_dir.mkdir(exist_ok=True)
                success = save_vasp_inputs(structure_data, vasp_dir, dft_config)
                if success:
                    generated_files.append(vasp_dir)
            
            elif dft_code.lower() == "gaussian":
                input_file = struct_output_dir / f"{struct_name}.gjf"
                success = save_gaussian_input(structure_data, input_file, dft_config)
                if success:
                    generated_files.append(input_file)
            
            # Create a metadata file with molecular formula
            mol_formula = xyz_to_mol_formula(structure_data)
            meta_file = struct_output_dir / "metadata.txt"
            with open(meta_file, 'w') as f:
                f.write(f"Structure: {struct_name}\n")
                f.write(f"Formula: {mol_formula}\n")
                f.write(f"Atoms: {len(structure_data['atoms'])}\n")
                f.write(f"Source: {xyz_file}\n")
                f.write(f"DFT Code: {dft_code}\n")
            
            generated_files.append(meta_file)
        
        except Exception as e:
            logger.error(f"Error processing XYZ file {xyz_file}: {e}")
    
    logger.info(f"Generated {len(generated_files)} output files")
    return generated_files