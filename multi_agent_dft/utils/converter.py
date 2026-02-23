"""
Conversion utilities for transforming between different DFT code formats.
"""

import os
import re
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def convert_input_format(source_file, target_format, output_file=None, parameters=None):
    """
    Convert a DFT input file from one format to another.
    
    Args:
        source_file (str or Path): Path to the source input file
        target_format (str): Target DFT code ('cp2k', 'vasp', 'gaussian')
        output_file (str or Path, optional): Path for the output file
        parameters (dict, optional): Additional parameters for the target format
        
    Returns:
        str or Path: Path to the generated output file
    """
    source_file = Path(source_file)
    
    # Determine source format
    source_format = detect_input_format(source_file)
    
    if source_format is None:
        logger.error(f"Could not determine source format for {source_file}")
        return None
    
    if source_format == target_format:
        logger.warning(f"Source and target formats are the same: {source_format}")
        return source_file
    
    logger.info(f"Converting from {source_format} to {target_format}")
    
    # Parse source file to extract structure and parameters
    structure_data, source_params = parse_input_file(source_file, source_format)
    
    if structure_data is None:
        logger.error(f"Failed to parse source file: {source_file}")
        return None
    
    # Determine output file name if not provided
    if output_file is None:
        if target_format == 'cp2k':
            output_file = source_file.with_suffix('.inp')
        elif target_format == 'vasp':
            # VASP uses multiple files, create a directory
            output_file = source_file.parent / f"{source_file.stem}_vasp"
            output_file.mkdir(exist_ok=True)
        elif target_format == 'gaussian':
            output_file = source_file.with_suffix('.gjf')
        else:
            output_file = source_file.with_suffix(f'.{target_format}')
    
    # Merge source parameters with provided parameters
    merged_params = {}
    if source_params:
        merged_params.update(source_params)
    if parameters:
        merged_params.update(parameters)
    
    # Generate target format input file
    if target_format.lower() == 'cp2k':
        from multi_agent_dft.dft.cp2k import save_cp2k_input
        success = save_cp2k_input(structure_data, output_file, merged_params)
    elif target_format.lower() == 'vasp':
        from multi_agent_dft.dft.vasp import save_vasp_inputs
        success = save_vasp_inputs(structure_data, output_file, merged_params)
    elif target_format.lower() == 'gaussian':
        from multi_agent_dft.dft.gaussian import save_gaussian_input
        success = save_gaussian_input(structure_data, output_file, merged_params) 
    else:
        logger.error(f"Unsupported target format: {target_format}")
        success = False
    
    if success:
        logger.info(f"Successfully converted to {target_format} format: {output_file}")
        return output_file
    else:
        logger.error(f"Failed to convert to {target_format} format")
        return None

def detect_input_format(file_path):
    """
    Detect the DFT code format of an input file.
    
    Args:
        file_path (str or Path): Path to the input file
        
    Returns:
        str or None: Detected format ('cp2k', 'vasp', 'gaussian') or None if unknown
    """
    file_path = Path(file_path)
    
    # Try to determine by file extension first
    if file_path.suffix.lower() in ['.inp', '.in']:
        # Could be CP2K, need to check content
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            if '&GLOBAL' in content and '&FORCE_EVAL' in content:
                return 'cp2k'
        except:
            pass
    
    elif file_path.name.upper() in ['INCAR', 'POSCAR', 'KPOINTS', 'POTCAR']:
        return 'vasp'
    
    elif file_path.suffix.lower() in ['.com', '.gjf', '.g09', '.g16']:
        return 'gaussian'
    
    # If extension is ambiguous, check file content
    try:
        with open(file_path, 'r') as f:
            content = f.read(4096)  # Read first 4KB for detection
        
        # CP2K signatures
        if '&GLOBAL' in content and '&FORCE_EVAL' in content:
            return 'cp2k'
        
        # VASP INCAR signatures
        if re.search(r'ISTART\s*=', content) or re.search(r'ENCUT\s*=', content):
            return 'vasp'
        
        # Gaussian signatures
        if re.search(r'^\s*#[PN]?', content, re.MULTILINE) and ('opt' in content.lower() or 'freq' in content.lower()):
            return 'gaussian'
        
        # Try to detect structure file formats
        if file_path.suffix.lower() == '.xyz':
            # XYZ file - not a DFT input but a structure file
            return 'xyz'
        elif file_path.suffix.lower() == '.cif':
            # CIF file - not a DFT input but a structure file
            return 'cif'
    
    except Exception as e:
        logger.error(f"Error reading file during format detection: {e}")
    
    return None

def parse_input_file(file_path, file_format=None):
    """
    Parse a DFT input file to extract structure data and parameters.
    
    Args:
        file_path (str or Path): Path to the input file
        file_format (str, optional): Format of the input file
        
    Returns:
        tuple: (structure_data, parameters) or (None, None) if parsing fails
    """
    file_path = Path(file_path)
    
    # Determine format if not provided
    if file_format is None:
        file_format = detect_input_format(file_path)
        
    if file_format is None:
        logger.error(f"Unknown file format: {file_path}")
        return None, None
    
    # For structure files, use the appropriate parser
    if file_format == 'cif':
        from multi_agent_dft.file_processing.cif import parse_cif_file
        structure_data = parse_cif_file(file_path)
        return structure_data, {}
    
    elif file_format == 'xyz':
        from multi_agent_dft.file_processing.xyz import parse_xyz
        structure_data = parse_xyz(file_path)
        return structure_data, {}
    
    # For DFT input files, extract both structure and parameters
    elif file_format == 'cp2k':
        return parse_cp2k_input(file_path)
    
    elif file_format == 'vasp':
        return parse_vasp_input(file_path)
    
    elif file_format == 'gaussian':
        return parse_gaussian_input(file_path)
    
    logger.error(f"Parsing not implemented for format: {file_format}")
    return None, None

def parse_cp2k_input(file_path):
    """
    Parse a CP2K input file to extract structure data and parameters.
    
    Args:
        file_path (str or Path): Path to the CP2K input file
        
    Returns:
        tuple: (structure_data, parameters) or (None, None) if parsing fails
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Initialize structure data and parameters
        structure_data = {
            'atoms': [],
            'cell': None,
            'meta': {
                'filename': str(Path(file_path).name),
                'source': 'CP2K'
            }
        }
        
        parameters = {
            'global': {},
            'dft': {},
            'scf': {},
            'xc': {},
            'kind_parameters': {}
        }
        
        # Extract global parameters
        global_match = re.search(r'&GLOBAL(.*?)&END\s+GLOBAL', content, re.DOTALL | re.IGNORECASE)
        if global_match:
            global_section = global_match.group(1)
            for line in global_section.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('!'):
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    parameters['global'][parts[0]] = parts[1]
        
        # Extract DFT parameters
        dft_match = re.search(r'&DFT(.*?)&END\s+DFT', content, re.DOTALL | re.IGNORECASE)
        if dft_match:
            dft_section = dft_match.group(1)
            for line in dft_section.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('!') or line.startswith('&'):
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    parameters['dft'][parts[0]] = parts[1]
        
        # Extract cell information
        cell_match = re.search(r'&CELL(.*?)&END\s+CELL', content, re.DOTALL | re.IGNORECASE)
        if cell_match:
            cell_section = cell_match.group(1)
            
            # Try to extract A, B, C vectors
            a_match = re.search(r'A\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', cell_section)
            b_match = re.search(r'B\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', cell_section)
            c_match = re.search(r'C\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', cell_section)
            
            if a_match and b_match and c_match:
                # We have explicit vectors
                a = [float(x) for x in a_match.groups()]
                b = [float(x) for x in b_match.groups()]
                c = [float(x) for x in c_match.groups()]
                structure_data['cell'] = [a, b, c]
            else:
                # Check for ABC format
                abc_match = re.search(r'ABC\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', cell_section)
                if abc_match:
                    a, b, c = [float(x) for x in abc_match.groups()]
                    # Simple orthogonal cell
                    structure_data['cell'] = [
                        [a, 0.0, 0.0],
                        [0.0, b, 0.0],
                        [0.0, 0.0, c]
                    ]
        
        # Extract atomic coordinates
        coord_match = re.search(r'&COORD(.*?)&END\s+COORD', content, re.DOTALL | re.IGNORECASE)
        if coord_match:
            coord_section = coord_match.group(1)
            for line in coord_section.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('!'):
                    continue
                
                parts = line.split()
                if len(parts) >= 4:
                    symbol = parts[0]
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        structure_data['atoms'].append({
                            'symbol': symbol,
                            'position': [x, y, z]
                        })
                    except ValueError:
                        continue
        
        # Extract element-specific parameters (KIND sections)
        kind_matches = re.finditer(r'&KIND\s+(\S+)(.*?)&END\s+KIND', content, re.DOTALL | re.IGNORECASE)
        for match in kind_matches:
            element = match.group(1).strip()
            kind_section = match.group(2)
            parameters['kind_parameters'][element] = {}
            
            for line in kind_section.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('!'):
                    continue
                
                parts = line.split(None, 1)
                if len(parts) == 2:
                    parameters['kind_parameters'][element][parts[0]] = parts[1]
        
        # Count element occurrences
        element_counts = {}
        for atom in structure_data['atoms']:
            symbol = atom['symbol']
            element_counts[symbol] = element_counts.get(symbol, 0) + 1
        
        structure_data['element_counts'] = element_counts
        
        return structure_data, parameters
    
    except Exception as e:
        logger.error(f"Error parsing CP2K input file: {e}")
        return None, None

def parse_vasp_input(file_path):
    """
    Parse VASP input files to extract structure data and parameters.
    
    Args:
        file_path (str or Path): Path to a VASP input file (POSCAR or INCAR)
        
    Returns:
        tuple: (structure_data, parameters) or (None, None) if parsing fails
    """
    file_path = Path(file_path)
    directory = file_path.parent
    
    structure_data = {
        'atoms': [],
        'cell': None,
        'meta': {
            'filename': str(file_path.name),
            'source': 'VASP'
        }
    }
    
    parameters = {}
    
    # Try to parse POSCAR for structure
    poscar_path = directory / "POSCAR" if file_path.name != "POSCAR" else file_path
    if poscar_path.exists():
        try:
            with open(poscar_path, 'r') as f:
                poscar_lines = f.readlines()
            
            # Parse POSCAR
            title = poscar_lines[0].strip()
            scale = float(poscar_lines[1].strip())
            
            # Parse lattice vectors
            lattice = []
            for i in range(2, 5):
                lattice.append([float(x) * scale for x in poscar_lines[i].split()])
            
            structure_data['cell'] = lattice
            
            # Parse element symbols and counts
            element_line = poscar_lines[5].strip()
            count_line = poscar_lines[6].strip()
            
            elements = element_line.split()
            counts = [int(x) for x in count_line.split()]
            
            element_counts = {}
            for element, count in zip(elements, counts):
                element_counts[element] = count
            
            structure_data['element_counts'] = element_counts
            
            # Check if coordinates are direct or cartesian
            coord_type = poscar_lines[7].strip().lower()
            is_direct = coord_type[0] in ['d', 's']  # Direct or Selective dynamics
            if is_direct and coord_type.startswith('s'):
                # Skip the selective dynamics line
                coord_start = 9
            else:
                coord_start = 8
            
            # Read atomic positions
            atom_index = 0
            for element, count in zip(elements, counts):
                for i in range(count):
                    if coord_start + atom_index < len(poscar_lines):
                        pos_line = poscar_lines[coord_start + atom_index].strip().split()
                        x, y, z = float(pos_line[0]), float(pos_line[1]), float(pos_line[2])
                        
                        if is_direct:
                            # Convert from fractional to cartesian
                            cart_pos = np.array([0.0, 0.0, 0.0])
                            for j in range(3):
                                cart_pos += np.array(lattice[j]) * [x, y, z][j]
                            position = cart_pos.tolist()
                            
                            structure_data['atoms'].append({
                                'symbol': element,
                                'position': position,
                                'fractional': [x, y, z]
                            })
                        else:
                            structure_data['atoms'].append({
                                'symbol': element,
                                'position': [x, y, z]
                            })
                    
                    atom_index += 1
        
        except Exception as e:
            logger.error(f"Error parsing POSCAR file: {e}")
    
    # Try to parse INCAR for parameters
    incar_path = directory / "INCAR" if file_path.name != "INCAR" else file_path
    if incar_path.exists():
        try:
            with open(incar_path, 'r') as f:
                incar_content = f.read()
            
            # Extract key-value pairs
            for line in incar_content.split('\n'):
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('!'):
                    continue
                
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Convert value to appropriate type
                    if value.replace('.', '', 1).isdigit():
                        # It's a number
                        if '.' in value:
                            parameters[key] = float(value)
                        else:
                            parameters[key] = int(value)
                    elif value.upper() in ['.TRUE.', '.FALSE.']:
                        # Boolean
                        parameters[key] = value.upper() == '.TRUE.'
                    else:
                        parameters[key] = value
        
        except Exception as e:
            logger.error(f"Error parsing INCAR file: {e}")
    
    # Try to parse KPOINTS for k-point settings
    kpoints_path = directory / "KPOINTS"
    if kpoints_path.exists():
        try:
            with open(kpoints_path, 'r') as f:
                kpoints_lines = f.readlines()
            
            # Extract k-point information
            if len(kpoints_lines) >= 4:
                mode = kpoints_lines[2].strip().lower()
                grid = kpoints_lines[3].strip().split()
                
                if len(grid) >= 3:
                    parameters['kpoints'] = {
                        'mode': mode,
                        'grid': [int(x) for x in grid[:3]]
                    }
                    
                    # Check for shift
                    if len(kpoints_lines) >= 5:
                        shift = kpoints_lines[4].strip().split()
                        if len(shift) >= 3:
                            parameters['kpoints']['shift'] = [float(x) for x in shift[:3]]
        
        except Exception as e:
            logger.error(f"Error parsing KPOINTS file: {e}")
    
    return structure_data, parameters

def batch_convert(input_dir, target_format, output_dir=None, recursive=False, file_pattern="*.*"):
    """
    Batch convert all supported structure files in a directory.
    
    Args:
        input_dir (str or Path): Input directory
        target_format (str): Target DFT code format
        output_dir (str or Path, optional): Output directory
        recursive (bool): Whether to search subdirectories
        file_pattern (str): File pattern to match
        
    Returns:
        list: Paths to the generated output files
    """
    input_dir = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_dir / f"{target_format}_outputs"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect input files
    input_files = []
    if recursive:
        # Find all files in directory and subdirectories
        for ext in ['.cif', '.xyz', '.inp', '.gjf', '.com']:
            input_files.extend(input_dir.glob(f"**/*{ext}"))
        
        # Add VASP files if target isn't VASP
        if target_format.lower() != 'vasp':
            for vasp_file in ['POSCAR', 'INCAR']:
                input_files.extend(input_dir.glob(f"**/{vasp_file}"))
    else:
        # Just search the current directory
        for ext in ['.cif', '.xyz', '.inp', '.gjf', '.com']:
            input_files.extend(input_dir.glob(f"*{ext}"))
            
        # Add VASP files if target isn't VASP
        if target_format.lower() != 'vasp':
            for vasp_file in ['POSCAR', 'INCAR']:
                input_files.extend(input_dir.glob(f"{vasp_file}"))
    
    # Filter by pattern
    if file_pattern != "*.*":
        from fnmatch import fnmatch
        input_files = [f for f in input_files if fnmatch(f.name, file_pattern)]
    
    # Convert each file
    output_files = []
    for input_file in input_files:
        try:
            rel_path = input_file.relative_to(input_dir)
            # Preserve subdirectory structure in output
            target_dir = output_dir / rel_path.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine output file name
            if target_format.lower() == 'cp2k':
                output_file = target_dir / f"{input_file.stem}.inp"
            elif target_format.lower() == 'vasp':
                output_file = target_dir / f"{input_file.stem}_vasp"
                output_file.mkdir(exist_ok=True)
            elif target_format.lower() == 'gaussian':
                output_file = target_dir / f"{input_file.stem}.gjf"
            else:
                output_file = target_dir / f"{input_file.stem}.{target_format}"
            
            # Perform conversion
            result = convert_input_format(input_file, target_format, output_file)
            if result:
                output_files.append(result)
                logger.info(f"Converted {input_file} to {result}")
            else:
                logger.error(f"Failed to convert {input_file}")
        
        except Exception as e:
            logger.error(f"Error converting {input_file}: {e}")
    
    logger.info(f"Converted {len(output_files)} files to {target_format} format")
    return output_files

def generate_optimal_cell(structure_data, padding=5.0):
    """
    Generate an optimal cell for a molecular structure.
    
    Args:
        structure_data (dict): Structure data dictionary
        padding (float): Padding around the molecule in Angstroms
        
    Returns:
        list: 3x3 cell matrix
    """
    if 'atoms' not in structure_data or not structure_data['atoms']:
        return [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    
    # Get atomic positions
    positions = np.array([atom['position'] for atom in structure_data['atoms']])
    
    # Calculate min and max coordinates
    min_coords = np.min(positions, axis=0)
    max_coords = np.max(positions, axis=0)
    
    # Add padding
    min_coords -= padding
    max_coords += padding
    
    # Create a diagonal cell matrix
    cell_lengths = max_coords - min_coords
    
    # Ensure minimum size
    cell_lengths = np.maximum(cell_lengths, 5.0)  # Minimum 5 Å in any dimension
    
    # Create cell matrix
    cell = [
        [cell_lengths[0], 0.0, 0.0],
        [0.0, cell_lengths[1], 0.0],
        [0.0, 0.0, cell_lengths[2]]
    ]
    
    # Shift atoms to cell origin
    for atom in structure_data['atoms']:
        atom['position'] = (np.array(atom['position']) - min_coords).tolist()
    
    return cell

def guess_element_from_mass(mass, tolerance=0.5):
    """
    Guess an element symbol based on atomic mass.
    
    Args:
        mass (float): Atomic mass
        tolerance (float): Tolerance in atomic mass units
        
    Returns:
        str or None: Element symbol or None if no match
    """
    # Dictionary of element symbols and masses
    element_masses = {
        'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811, 
        'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180, 
        'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974, 
        'S': 32.066, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078
    }
    
    # Find closest match
    best_match = None
    best_diff = float('inf')
    
    for element, element_mass in element_masses.items():
        diff = abs(mass - element_mass)
        if diff < best_diff:
            best_diff = diff
            best_match = element
    
    # Check if within tolerance
    if best_diff <= tolerance:
        return best_match
    
    return None

def parse_gaussian_input(file_path):
    """
    Parse a Gaussian input file to extract structure data and parameters.
    
    Args:
        file_path (str or Path): Path to the Gaussian input file
        
    Returns:
        tuple: (structure_data, parameters) or (None, None) if parsing fails
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Initialize structure data and parameters
        structure_data = {
            'atoms': [],
            'meta': {
                'filename': str(Path(file_path).name),
                'source': 'Gaussian'
            }
        }
        
        parameters = {
            'method': None,
            'basis_set': None,
            'job_type': None,
            'charge': 0,
            'multiplicity': 1
        }
        
        # Extract route section (starts with # and ends with empty line)
        route_match = re.search(r'(^|\n)#[pnPN]?(.*?)(\n\s*\n)', content, re.DOTALL)
        if route_match:
            route = route_match.group(2).strip()
            
            # Extract method and basis set
            # Common format: #p method/basis job-keywords
            method_basis_match = re.search(r'(\S+)/(\S+)', route)
            if method_basis_match:
                parameters['method'] = method_basis_match.group(1)
                parameters['basis_set'] = method_basis_match.group(2)
            
            # Extract job type
            job_keywords = ['opt', 'freq', 'sp', 'td', 'scan', 'irc', 'stable']
            for keyword in job_keywords:
                if re.search(r'\b' + keyword + r'\b', route, re.IGNORECASE):
                    if parameters['job_type']:
                        parameters['job_type'] += ' ' + keyword
                    else:
                        parameters['job_type'] = keyword
        
        # Extract charge and multiplicity
        # Usually found after two empty lines following the route section
        charge_mult_match = re.search(r'\n\s*\n\s*\n\s*(\-?\d+)\s+(\d+)\s*\n', content)
        if charge_mult_match:
            parameters['charge'] = int(charge_mult_match.group(1))
            parameters['multiplicity'] = int(charge_mult_match.group(2))
            
            # The atom coordinates follow the charge/multiplicity line
            atoms_section = content[charge_mult_match.end():].strip()
            
            # Parse atom coordinates (each line: Element X Y Z)
            for line in atoms_section.split('\n'):
                line = line.strip()
                if not line or line.startswith('!') or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        symbol = parts[0]
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        structure_data['atoms'].append({
                            'symbol': symbol,
                            'position': [x, y, z]
                        })
                    except ValueError:
                        continue
        
        # Count element occurrences
        element_counts = {}
        for atom in structure_data['atoms']:
            symbol = atom['symbol']
            element_counts[symbol] = element_counts.get(symbol, 0) + 1
        
        structure_data['element_counts'] = element_counts
        
        return structure_data, parameters
    
    except Exception as e:
        logger.error(f"Error parsing Gaussian input file: {e}")
        return None, None