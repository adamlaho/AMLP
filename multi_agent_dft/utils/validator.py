"""
Enhanced structure file validation for the Multi-Agent DFT Research System.
"""

import os
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_structure(file_path):
    """
    Validate a structure file (CIF, XYZ, etc.)
    
    Args:
        file_path: Path to the structure file
        
    Returns:
        tuple: (valid, message, extra_info)
    """
    from pathlib import Path
    
    # Import the CIF fixing function
    try:
        from multi_agent_dft.utils.cif_fix import fix_cif_data_block
    except ImportError:
        # Define a fallback if the module isn't available
        def fix_cif_data_block(path):
            return path, False
    
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        return False, f"File does not exist: {file_path}", None
        
    # Special case: if it's a directory, consider it valid
    if file_path.is_dir():
        return True, "Directory is valid for batch processing", None
    
    # For CIF files, use our CIF fixing logic
    if file_path.suffix.lower() == '.cif':
        # Try to fix the CIF file if needed
        fixed_path, was_fixed = fix_cif_data_block(file_path)
        
        if was_fixed:
            return True, f"Fixed CIF file: added/corrected data_ block", {"fixed_path": fixed_path}
        else:
            # File was already valid
            return True, "Valid CIF file", None
        
    # For XYZ files, keep the original validation
    elif file_path.suffix.lower() == '.xyz':
        # Simple validation - check if file exists and has content
        if file_path.stat().st_size == 0:
            return False, "XYZ file is empty", None
            
        # Maybe add more XYZ validation here if needed
        return True, "Valid XYZ file", None
    
    # Unsupported file type
    else:
        return False, f"Unsupported file type: {file_path.suffix}", None

def validate_xyz_content(content, file_path):
    """
    Validate XYZ file content.
    
    Args:
        content (str): File content
        file_path (Path): Path to the file
    
    Returns:
        tuple: (is_valid, message, details)
    """
    lines = content.strip().split('\n')
    
    # XYZ must have at least 3 lines (atom count, comment, at least one atom)
    if len(lines) < 3:
        return False, "Invalid XYZ file: needs at least 3 lines.", {"error": "invalid_xyz_format"}
    
    # Try to parse the atom count
    try:
        atom_count = int(lines[0].strip())
    except ValueError:
        return False, "Invalid XYZ file: first line should be the number of atoms.", {"error": "invalid_atom_count"}
    
    # Check if we have the correct number of atom lines
    if len(lines) != atom_count + 2:
        return False, f"Invalid XYZ file: expected {atom_count} atoms but found {len(lines)-2}.", {
            "error": "atom_count_mismatch",
            "expected": atom_count,
            "found": len(lines)-2
        }
    
    # Parse and validate atom lines
    invalid_lines = []
    atoms = []
    
    for i in range(2, len(lines)):
        line = lines[i].strip()
        parts = line.split()
        
        if len(parts) < 4:
            invalid_lines.append((i+1, line, "Too few columns"))
            continue
        
        # Check if coordinates are numeric
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append({
                "element": parts[0],
                "position": [x, y, z]
            })
        except ValueError:
            invalid_lines.append((i+1, line, "Invalid coordinate format"))
    
    # If there are invalid lines, return error
    if invalid_lines:
        error_message = f"Invalid XYZ format - problems in {len(invalid_lines)} lines:\n"
        for line_num, content, reason in invalid_lines[:5]:  # Show first 5 errors
            error_message += f"Line {line_num}: {content} - {reason}\n"
        
        if len(invalid_lines) > 5:
            error_message += f"... and {len(invalid_lines) - 5} more errors."
            
        return False, error_message, {
            "error": "invalid_atom_lines",
            "problems": invalid_lines
        }
    
    # Check for reasonable coordinate ranges
    positions = np.array([atom["position"] for atom in atoms])
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    range_pos = max_pos - min_pos
    
    if np.any(range_pos > 1000.0):
        return False, "Suspicious XYZ coordinates: unusually large coordinate range.", {
            "error": "unusual_coordinate_range",
            "min": min_pos.tolist(),
            "max": max_pos.tolist()
        }
    
    # Count elements for summary
    element_counts = {}
    for atom in atoms:
        element = atom["element"]
        element_counts[element] = element_counts.get(element, 0) + 1
    
    # Success!
    return True, content, {
        "format": "xyz",
        "atom_count": atom_count,
        "elements": element_counts,
        "bounding_box": {
            "min": min_pos.tolist(),
            "max": max_pos.tolist()
        }
    }

def validate_cif_content(content, file_path):
    """
    Validate CIF file content.
    
    Args:
        content (str): File content
        file_path (Path): Path to the file
    
    Returns:
        tuple: (is_valid, message, details)
    """
    import re
    
    # Basic check for CIF format - should have data_ at the beginning
    if not re.search(r'data_\S+', content, re.IGNORECASE):
        return False, "Invalid CIF file: missing 'data_' block.", {"error": "missing_data_block"}
    
    # Check for cell parameters - a valid CIF should define the unit cell
    cell_params = {}
    for param in ['_cell_length_a', '_cell_length_b', '_cell_length_c',
                 '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma']:
        match = re.search(param + r'\s+([\d.]+)', content, re.IGNORECASE)
        if match:
            cell_params[param] = float(match.group(1))
    
    # At a minimum, we need the cell lengths
    required_params = ['_cell_length_a', '_cell_length_b', '_cell_length_c']
    if not all(param in cell_params for param in required_params):
        missing = [param for param in required_params if param not in cell_params]
        return False, f"Invalid CIF file: missing required cell parameters: {', '.join(missing)}", {
            "error": "missing_cell_params",
            "missing": missing
        }
    
    # Check for atom coordinates
    atom_loop = re.search(r'loop_\s+((?:_atom_site\.\S+\s+)+)', content, re.DOTALL)
    if not atom_loop:
        return False, "Invalid CIF file: missing atom site loop.", {"error": "missing_atom_loop"}
    
    # Check for minimal required atom site fields
    atom_fields = re.findall(r'_atom_site\.(\S+)', atom_loop.group(1))
    required_fields = ['label']
    coordinate_fields = ['fract_x', 'fract_y', 'fract_z']
    
    if not all(field in atom_fields for field in required_fields):
        return False, "Invalid CIF file: missing required atom site fields.", {
            "error": "missing_atom_fields",
            "missing": [field for field in required_fields if field not in atom_fields]
        }
    
    # Check either fractional or Cartesian coordinates are present
    has_coords = all(field in atom_fields for field in coordinate_fields)
    has_cartn = all(f'Cartn_{axis}' in atom_fields for axis in ['x', 'y', 'z'])
    
    if not (has_coords or has_cartn):
        return False, "Invalid CIF file: missing coordinate information for atoms.", {
            "error": "missing_coordinates"
        }
    
    # Extract space group info for validation result
    symmetry_info = {}
    for field in ['_symmetry_space_group_name_h-m', '_space_group_name_h-m', '_symmetry_int_tables_number']:
        match = re.search(field + r'\s+[\'"]?([\w\s\-/()]+)[\'"]?', content, re.IGNORECASE)
        if match:
            symmetry_info[field] = match.group(1).strip()
    
    # Try to count atoms
    atoms_data = re.search(r'loop_\s+((?:_atom_site\.\S+\s+)+)((?:.+\n)+?)(?:loop_|\Z)', content, re.DOTALL)
    atom_count = 0
    element_counts = {}
    
    if atoms_data:
        atom_lines = atoms_data.group(2).strip().split('\n')
        
        # Find index for label field
        try:
            label_idx = atom_fields.index('label')
            
            # Count atoms by line and extract elements
            for line in atom_lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) <= label_idx:
                    continue
                
                atom_count += 1
                
                # Extract element from label (common convention in CIF files)
                # Usually the first 1-2 letters before a number or special character
                element = re.match(r'([A-Za-z]+)', parts[label_idx])
                if element:
                    el = element.group(1)
                    # Standardize element symbol (first letter uppercase, rest lowercase)
                    el = el[0].upper() + el[1:].lower() if len(el) > 1 else el.upper()
                    element_counts[el] = element_counts.get(el, 0) + 1
                
        except ValueError:
            pass  # label field not found
        
    # Success! Return validation details
    return True, content, {
        "format": "cif",
        "cell_params": cell_params,
        "symmetry": symmetry_info,
        "atom_count": atom_count,
        "elements": element_counts
    }

def analyze_structure(file_path):
    """
    Perform a comprehensive analysis of a structure file.
    
    Args:
        file_path (str or Path): Path to the structure file.
    
    Returns:
        dict: Analysis results or None if validation fails.
    """
    valid, message, details = validate_structure(file_path)
    
    if not valid:
        logger.error(f"Structure validation failed: {message}")
        return None
    
    # If it's a directory, analyze each file
    file_path = Path(file_path)
    if file_path.is_dir():
        analyses = {}
        for ext in ['.cif', '.xyz']:
            for struct_file in file_path.glob(f"*{ext}"):
                file_analysis = analyze_structure(struct_file)
                if file_analysis:
                    analyses[struct_file.name] = file_analysis
        return analyses
    
    # Import the appropriate parser based on file type
    if file_path.name.lower().endswith('.cif'):
        from multi_agent_dft.file_processing.cif import parse_cif_file, detect_molecular_vs_periodic, suggest_dft_parameters
        
        # Parse the CIF file
        structure_data = parse_cif_file(file_path)
        if structure_data is None:
            logger.error(f"Failed to parse CIF file: {file_path}")
            return None
        
        # Analyze structure properties
        structure_type = detect_molecular_vs_periodic(structure_data)
        
        # Get DFT recommendations
        dft_params = suggest_dft_parameters(structure_data)
        
        # Return comprehensive analysis
        return {
            "file": str(file_path),
            "format": "cif",
            "structure_type": structure_type,
            "atom_count": len(structure_data['atoms']),
            "elements": structure_data['element_counts'],
            "cell_params": structure_data['cell_params'],
            "symmetry": structure_data['symmetry'],
            "dft_recommendations": dft_params
        }
        
    elif file_path.name.lower().endswith('.xyz'):
        from multi_agent_dft.file_processing.xyz import parse_xyz, xyz_to_mol_formula
        
        # Parse the XYZ file
        structure_data = parse_xyz(file_path)
        if structure_data is None:
            logger.error(f"Failed to parse XYZ file: {file_path}")
            return None
        
        # Get molecular formula
        formula = xyz_to_mol_formula(structure_data)
        
        # For XYZ, we assume it's a molecular system by default
        # But we can check for periodicity if cell info is available
        structure_type = "molecular"
        if 'cell' in structure_data:
            # If cell info is present, it might be periodic
            # Apply a similar detection logic as we did for CIF
            # This would be implemented in a full solution
            pass
        
        # Get basic recommendations
        dft_params = {
            "cp2k": {
                "global": {"RUN_TYPE": "GEO_OPT"},
                "dft": {"BASIS_SET_FILE_NAME": "BASIS_MOLOPT"},
                "mgrid": {"CUTOFF": 400}
            },
            "vasp": {"ENCUT": 400, "ISMEAR": 0},
            "gaussian": {"method": "B3LYP", "basis_set": "6-31G(d)"}
        }
        
        # Return analysis
        return {
            "file": str(file_path),
            "format": "xyz",
            "structure_type": structure_type,
            "formula": formula,
            "atom_count": len(structure_data['atoms']),
            "dft_recommendations": dft_params
        }
    
    return None