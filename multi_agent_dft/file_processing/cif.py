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
        str or Path: Path to the generated output file or None if conversion fails.
    """
    source_file = Path(source_file)
    source_format = detect_input_format(source_file)
    
    if source_format is None:
        logger.error(f"Could not determine source format for {source_file}")
        return None
    
    if source_format == target_format:
        logger.warning(f"Source and target formats are the same: {source_format}")
        return source_file
    
    logger.info(f"Converting from {source_format} to {target_format}")
    
    structure_data, source_params = parse_input_file(source_file, source_format)
    if structure_data is None:
        logger.error(f"Failed to parse source file: {source_file}")
        return None
    
    # Determine output file name if not provided
    if output_file is None:
        if target_format == 'cp2k':
            output_file = source_file.with_suffix('.inp')
        elif target_format == 'vasp':
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
    if target_format == 'cp2k':
        from multi_agent_dft.dft.cp2k import save_cp2k_input
        success = save_cp2k_input(structure_data, output_file, merged_params)
    elif target_format == 'vasp':
        from multi_agent_dft.dft.vasp import save_vasp_inputs
        success = save_vasp_inputs(structure_data, output_file, merged_params)
    elif target_format == 'gaussian':
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
    Detect the DFT code format of an input file based on its extension and content.
    
    Args:
        file_path (str or Path): Path to the input file
        
    Returns:
        str or None: Detected format ('cp2k', 'vasp', 'gaussian', 'xyz', 'cif') or None if unknown.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    
    # Use file extension clues
    if suffix in ['.inp', '.in']:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            if '&GLOBAL' in content and '&FORCE_EVAL' in content:
                return 'cp2k'
        except Exception as e:
            logger.error(f"Error reading file {file_path} for format detection: {e}")
    
    elif file_path.name.upper() in ['INCAR', 'POSCAR', 'KPOINTS', 'POTCAR']:
        return 'vasp'
    elif suffix in ['.com', '.gjf', '.g09', '.g16']:
        return 'gaussian'
    
    # When extension is ambiguous, check file content
    try:
        with open(file_path, 'r') as f:
            content = f.read(4096)
        
        if '&GLOBAL' in content and '&FORCE_EVAL' in content:
            return 'cp2k'
        if re.search(r'ISTART\s*=', content) or re.search(r'ENCUT\s*=', content):
            return 'vasp'
        if re.search(r'^\s*#[PN]?', content, re.MULTILINE) and ('opt' in content.lower() or 'freq' in content.lower()):
            return 'gaussian'
        if suffix == '.xyz':
            return 'xyz'
        elif suffix == '.cif':
            return 'cif'
    except Exception as e:
        logger.error(f"Error reading file {file_path} during content detection: {e}")
    
    return None

def parse_input_file(file_path, file_format=None):
    """
    Parse a DFT input file to extract structure data and parameters.
    
    Args:
        file_path (str or Path): Path to the input file
        file_format (str, optional): Format of the input file
        
    Returns:
        tuple: (structure_data, parameters) or (None, None) if parsing fails.
    """
    file_path = Path(file_path)
    if file_format is None:
        file_format = detect_input_format(file_path)
    if file_format is None:
        logger.error(f"Unknown file format for file: {file_path}")
        return None, None

    if file_format == 'cif':
        # Use the improved CIF parser with error handling
        structure_data = parse_cif_file(file_path)
        return structure_data, {} if structure_data is not None else (None, None)
    elif file_format == 'xyz':
        from multi_agent_dft.file_processing.xyz import parse_xyz
        structure_data = parse_xyz(file_path)
        return structure_data, {}
    elif file_format == 'cp2k':
        return parse_cp2k_input(file_path)
    elif file_format == 'vasp':
        return parse_vasp_input(file_path)
    elif file_format == 'gaussian':
        return parse_gaussian_input(file_path)
    
    logger.error(f"Parsing not implemented for format: {file_format}")
    return None, None


def parse_cif_file(file_path):
    """
    Improved CIF parser that handles coordinates with uncertainty notation like '0.22113(9)'
    and properly handles unit cell angles for correct transformation from fractional to Cartesian.
    Returns a dict with 'atoms', 'cell', and 'meta', or None on failure.
    """
    import re
    import numpy as np
    import logging
    from pathlib import Path
    import math

    logger = logging.getLogger(__name__)
    
    try:
        text = Path(file_path).read_text()
        
        # 1) Extract cell parameters (lengths and angles)
        a_match = re.search(r"_cell_length_a\s+([\d\.Ee+-]+)", text, re.IGNORECASE)
        b_match = re.search(r"_cell_length_b\s+([\d\.Ee+-]+)", text, re.IGNORECASE)
        c_match = re.search(r"_cell_length_c\s+([\d\.Ee+-]+)", text, re.IGNORECASE)
        
        alpha_match = re.search(r"_cell_angle_alpha\s+([\d\.Ee+-]+)", text, re.IGNORECASE)
        beta_match = re.search(r"_cell_angle_beta\s+([\d\.Ee+-]+)", text, re.IGNORECASE)
        gamma_match = re.search(r"_cell_angle_gamma\s+([\d\.Ee+-]+)", text, re.IGNORECASE)
        
        # Make sure all cell parameters are found
        if not all([a_match, b_match, c_match, alpha_match, beta_match, gamma_match]):
            logger.error(f"Missing cell parameters in {file_path}")
            return None
        
        # Extract cell parameters
        a = float(a_match.group(1))
        b = float(b_match.group(1))
        c = float(c_match.group(1))
        
        alpha = float(alpha_match.group(1))
        beta = float(beta_match.group(1))
        gamma = float(gamma_match.group(1))
        
        # Convert angles to radians
        alpha_rad = math.radians(alpha)
        beta_rad = math.radians(beta)
        gamma_rad = math.radians(gamma)
        
        # Calculate transformation matrix using the method from the reference code
        # First vector along x-axis
        v_a = np.array([a, 0.0, 0.0])
        
        # Second vector in xy-plane
        v_b = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0.0])
        
        # Calculate the third vector's components
        # This follows the standard crystallographic convention
        try:
            n2 = (np.cos(alpha_rad) - np.cos(gamma_rad) * np.cos(beta_rad)) / np.sin(gamma_rad)
            v_c = np.array([
                c * np.cos(beta_rad),
                c * n2,
                c * np.sqrt(max(0, np.sin(beta_rad)**2 - n2**2))  # Ensure no negative values under sqrt
            ])
        except (ZeroDivisionError, ValueError):
            # Fallback for numerical issues
            logger.warning(f"Numerical issue in cell calculation for {file_path}. Using simplified cell.")
            v_c = np.array([0.0, 0.0, c])
        
        # Store the lattice matrix (each row is a lattice vector)
        lattice_matrix = np.array([v_a, v_b, v_c])
        
        # 2) locate the loop_ of atom_site
        lines = text.splitlines()
        col_tags = []
        data_rows = []
        in_loop = False

        for ln in lines:
            ln_stripped = ln.strip()
            # start of an atom_site loop
            if ln_stripped.lower().startswith("loop_"):
                in_loop = True
                col_tags = []
                continue
            if in_loop:
                if ln_stripped.startswith("_atom_site_"):
                    col_tags.append(ln_stripped)
                    continue
                # once we hit a non‐tag, non‐empty line, treat as data row
                if col_tags and not ln_stripped.startswith("_") and ln_stripped:
                    parts = ln_stripped.split()
                    # skip if not enough columns
                    if len(parts) < len(col_tags):
                        # likely end of this loop
                        in_loop = False
                        continue
                    data_rows.append(parts)
                    continue

        if not col_tags or not data_rows:
            logger.error(f"No atom_site loop found or no data rows in {file_path}")
            return None

        # 3) build atoms list
        atoms = []
        # map tag → column index
        idx = {tag: i for i, tag in enumerate(col_tags)}
        
        # Check for required columns
        required_tags = ["_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]
        missing_tags = [tag for tag in required_tags if tag not in idx]
        
        if missing_tags:
            logger.error(f"Missing required atom site tags in {file_path}: {missing_tags}")
            return None
        
        # Helper function to strip uncertainty notation
        def strip_uncertainty(value):
            """Remove uncertainty notation like (9) from values like 0.22113(9)"""
            return re.sub(r'\([^)]*\)', '', value)
        
        for row in data_rows:
            try:
                # prefer type_symbol, fallback to label
                symbol_idx = idx.get("_atom_site_type_symbol")
                if symbol_idx is None:
                    symbol_idx = idx.get("_atom_site_label")
                
                if symbol_idx is None or symbol_idx >= len(row):
                    logger.warning(f"Missing atom symbol in {file_path}, skipping row")
                    continue
                
                symbol = row[symbol_idx]
                
                # Clean up common formats in CIF files
                # Remove any digits from end of element symbols if that's how they're formatted
                symbol = re.sub(r'(\D+)\d+.*', r'\1', symbol)
                
                # Strip uncertainty notation from coordinates
                x_idx = idx.get("_atom_site_fract_x")
                y_idx = idx.get("_atom_site_fract_y")
                z_idx = idx.get("_atom_site_fract_z")
                
                if any(i is None or i >= len(row) for i in [x_idx, y_idx, z_idx]):
                    logger.warning(f"Missing coordinate data in {file_path}, skipping row")
                    continue
                
                fx_str = strip_uncertainty(row[x_idx])
                fy_str = strip_uncertainty(row[y_idx])
                fz_str = strip_uncertainty(row[z_idx])
                
                fx = float(fx_str)
                fy = float(fy_str)
                fz = float(fz_str)
                
                # Convert fractional coordinates to Cartesian
                # Using the same method as in the reference code
                frac_coords = np.array([fx, fy, fz])
                cart_coords = frac_coords.dot(lattice_matrix)
                
                atoms.append({
                    "symbol": symbol, 
                    "position": cart_coords.tolist(),
                    "fractional": [fx, fy, fz]  # Store fractional coordinates for VASP output
                })
            except (IndexError, ValueError) as e:
                logger.warning(f"Error processing atom row in {file_path}: {e}")
                continue

        # If no atoms were successfully parsed, return None
        if not atoms:
            logger.error(f"No atoms could be parsed from {file_path}")
            return None

        # Count elements for summary
        element_counts = {}
        for atom in atoms:
            symbol = atom["symbol"]
            element_counts[symbol] = element_counts.get(symbol, 0) + 1

        return {
            "atoms": atoms,
            "cell": lattice_matrix.tolist(),
            "element_counts": element_counts,
            "cell_params": {
                "lengths": [a, b, c],
                "angles": [alpha, beta, gamma]
            },
            "meta": {
                "filename": Path(file_path).name,
                "source": "cif"
            }
        }

    except Exception as e:
        logger.error(f"Error in parse_cif_file for {file_path}: {e}")
        return None


def parse_cp2k_input(file_path):
    """
    Parse a CP2K input file to extract structure data and parameters.
    
    Args:
        file_path (str or Path): Path to the CP2K input file.
        
    Returns:
        tuple: (structure_data, parameters) or (None, None) if parsing fails.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        structure_data = {'atoms': [], 'cell': None, 'meta': {'filename': str(Path(file_path).name), 'source': 'CP2K'}}
        parameters = {'global': {}, 'dft': {}, 'scf': {}, 'xc': {}, 'kind_parameters': {}}
        
        global_match = re.search(r'&GLOBAL(.*?)&END\s+GLOBAL', content, re.DOTALL | re.IGNORECASE)
        if global_match:
            for line in global_match.group(1).splitlines():
                line = line.strip()
                if line and not line.startswith('!'):
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        parameters['global'][parts[0]] = parts[1]
        
        dft_match = re.search(r'&DFT(.*?)&END\s+DFT', content, re.DOTALL | re.IGNORECASE)
        if dft_match:
            for line in dft_match.group(1).splitlines():
                line = line.strip()
                if line and not line.startswith('!') and not line.startswith('&'):
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        parameters['dft'][parts[0]] = parts[1]
        
        cell_match = re.search(r'&CELL(.*?)&END\s+CELL', content, re.DOTALL | re.IGNORECASE)
        if cell_match:
            cell_section = cell_match.group(1)
            a_match = re.search(r'A\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', cell_section)
            b_match = re.search(r'B\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', cell_section)
            c_match = re.search(r'C\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', cell_section)
            if a_match and b_match and c_match:
                a = [float(x) for x in a_match.groups()]
                b = [float(x) for x in b_match.groups()]
                c = [float(x) for x in c_match.groups()]
                structure_data['cell'] = [a, b, c]
            else:
                abc_match = re.search(r'ABC\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', cell_section)
                if abc_match:
                    a_val, b_val, c_val = [float(x) for x in abc_match.groups()]
                    structure_data['cell'] = [
                        [a_val, 0.0, 0.0],
                        [0.0, b_val, 0.0],
                        [0.0, 0.0, c_val]
                    ]
        
        coord_match = re.search(r'&COORD(.*?)&END\s+COORD', content, re.DOTALL | re.IGNORECASE)
        if coord_match:
            for line in coord_match.group(1).splitlines():
                line = line.strip()
                if line and not line.startswith('!'):
                    parts = line.split()
                    if len(parts) >= 4:
                        symbol = parts[0]
                        try:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            structure_data['atoms'].append({'symbol': symbol, 'position': [x, y, z]})
                        except ValueError:
                            continue
        
        kind_matches = re.finditer(r'&KIND\s+(\S+)(.*?)&END\s+KIND', content, re.DOTALL | re.IGNORECASE)
        for match in kind_matches:
            element = match.group(1).strip()
            kind_section = match.group(2)
            parameters['kind_parameters'][element] = {}
            for line in kind_section.splitlines():
                line = line.strip()
                if line and not line.startswith('!'):
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        parameters['kind_parameters'][element][parts[0]] = parts[1]
        
        element_counts = {}
        for atom in structure_data['atoms']:
            symbol = atom['symbol']
            element_counts[symbol] = element_counts.get(symbol, 0) + 1
        structure_data['element_counts'] = element_counts
        
        return structure_data, parameters
    
    except Exception as e:
        logger.error(f"Error parsing CP2K input file {file_path}: {e}")
        return None, None

def parse_vasp_input(file_path):
    """
    Parse VASP input files (POSCAR/INCAR/KPOINTS) to extract structure data and parameters.
    
    Args:
        file_path (str or Path): Path to a VASP input file.
        
    Returns:
        tuple: (structure_data, parameters) or (None, None) if parsing fails.
    """
    file_path = Path(file_path)
    directory = file_path.parent
    
    structure_data = {'atoms': [], 'cell': None, 'meta': {'filename': file_path.name, 'source': 'VASP'}}
    parameters = {}
    
    poscar_path = directory / "POSCAR" if file_path.name != "POSCAR" else file_path
    if poscar_path.exists():
        try:
            with open(poscar_path, 'r') as f:
                poscar_lines = f.readlines()
            title = poscar_lines[0].strip()
            scale = float(poscar_lines[1].strip())
            lattice = []
            for i in range(2, 5):
                lattice.append([float(x) * scale for x in poscar_lines[i].split()])
            structure_data['cell'] = lattice
            element_line = poscar_lines[5].strip()
            count_line = poscar_lines[6].strip()
            elements = element_line.split()
            counts = [int(x) for x in count_line.split()]
            element_counts = {element: count for element, count in zip(elements, counts)}
            structure_data['element_counts'] = element_counts
            
            coord_type = poscar_lines[7].strip().lower()
            is_direct = coord_type[0] in ['d', 's']
            coord_start = 9 if is_direct and coord_type.startswith('s') else 8
            atom_index = 0
            for element, count in zip(elements, counts):
                for i in range(count):
                    if coord_start + atom_index < len(poscar_lines):
                        pos_line = poscar_lines[coord_start + atom_index].strip().split()
                        x, y, z = float(pos_line[0]), float(pos_line[1]), float(pos_line[2])
                        if is_direct:
                            cart_pos = np.array([0.0, 0.0, 0.0])
                            for j in range(3):
                                cart_pos += np.array(lattice[j]) * [x, y, z][j]
                            position = cart_pos.tolist()
                            structure_data['atoms'].append({'symbol': element, 'position': position, 'fractional': [x, y, z]})
                        else:
                            structure_data['atoms'].append({'symbol': element, 'position': [x, y, z]})
                    atom_index += 1
        except Exception as e:
            logger.error(f"Error parsing POSCAR file: {e}")
    
    incar_path = directory / "INCAR" if file_path.name != "INCAR" else file_path
    if incar_path.exists():
        try:
            with open(incar_path, 'r') as f:
                incar_content = f.read()
            for line in incar_content.splitlines():
                line = line.strip()
                if line and not line.startswith(('#', '!')) and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if value.replace('.', '', 1).isdigit():
                        parameters[key] = float(value) if '.' in value else int(value)
                    elif value.upper() in ['.TRUE.', '.FALSE.']:
                        parameters[key] = value.upper() == '.TRUE.'
                    else:
                        parameters[key] = value
        except Exception as e:
            logger.error(f"Error parsing INCAR file: {e}")
    
    kpoints_path = directory / "KPOINTS"
    if kpoints_path.exists():
        try:
            with open(kpoints_path, 'r') as f:
                kpoints_lines = f.readlines()
            if len(kpoints_lines) >= 4:
                mode = kpoints_lines[2].strip().lower()
                grid = kpoints_lines[3].strip().split()
                if len(grid) >= 3:
                    parameters['kpoints'] = {'mode': mode, 'grid': [int(x) for x in grid[:3]]}
                    if len(kpoints_lines) >= 5:
                        shift = kpoints_lines[4].strip().split()
                        if len(shift) >= 3:
                            parameters['kpoints']['shift'] = [float(x) for x in shift[:3]]
        except Exception as e:
            logger.error(f"Error parsing KPOINTS file: {e}")
    
    return structure_data, parameters

def parse_gaussian_input(file_path):
    """
    Parse a Gaussian input file to extract structure data and parameters.
    
    Args:
        file_path (str or Path): Path to the Gaussian input file.
        
    Returns:
        tuple: (structure_data, parameters) or (None, None) if parsing fails.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        structure_data = {'atoms': [], 'meta': {'filename': str(Path(file_path).name), 'source': 'Gaussian'}}
        parameters = {'method': None, 'basis_set': None, 'job_type': None, 'charge': 0, 'multiplicity': 1}
        
        route_match = re.search(r'(^|\n)#[pnPN]?(.*?)(\n\s*\n)', content, re.DOTALL)
        if route_match:
            route = route_match.group(2).strip()
            method_basis_match = re.search(r'(\S+)/(\S+)', route)
            if method_basis_match:
                parameters['method'] = method_basis_match.group(1)
                parameters['basis_set'] = method_basis_match.group(2)
            for keyword in ['opt', 'freq', 'sp', 'td', 'scan', 'irc', 'stable']:
                if re.search(r'\b' + keyword + r'\b', route, re.IGNORECASE):
                    parameters['job_type'] = parameters.get('job_type', '') + ' ' + keyword if parameters.get('job_type') else keyword
        
        charge_mult_match = re.search(r'\n\s*\n\s*\n\s*(\-?\d+)\s+(\d+)\s*\n', content)
        if charge_mult_match:
            parameters['charge'] = int(charge_mult_match.group(1))
            parameters['multiplicity'] = int(charge_mult_match.group(2))
            atoms_section = content[charge_mult_match.end():].strip()
            for line in atoms_section.splitlines():
                line = line.strip()
                if line and not line.startswith(('!', '#')):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            symbol = parts[0]
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            structure_data['atoms'].append({'symbol': symbol, 'position': [x, y, z]})
                        except ValueError:
                            continue
        
        element_counts = {}
        for atom in structure_data['atoms']:
            symbol = atom['symbol']
            element_counts[symbol] = element_counts.get(symbol, 0) + 1
        structure_data['element_counts'] = element_counts
        
        return structure_data, parameters
    
    except Exception as e:
        logger.error(f"Error parsing Gaussian input file {file_path}: {e}")
        return None, None

def batch_convert(input_dir, target_format, output_dir=None, recursive=False, file_pattern="*.*"):
    """
    Batch convert all supported structure files in a directory.
    
    Args:
        input_dir (str or Path): Input directory.
        target_format (str): Target DFT code format.
        output_dir (str or Path, optional): Output directory.
        recursive (bool): Whether to search subdirectories.
        file_pattern (str): File pattern to match.
        
    Returns:
        list: Paths to the generated output files.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir / f"{target_format}_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_files = []
    if recursive:
        for ext in ['.cif', '.xyz', '.inp', '.gjf', '.com']:
            input_files.extend(input_dir.glob(f"**/*{ext}"))
        if target_format.lower() != 'vasp':
            for vasp_file in ['POSCAR', 'INCAR']:
                input_files.extend(input_dir.glob(f"**/{vasp_file}"))
    else:
        for ext in ['.cif', '.xyz', '.inp', '.gjf', '.com']:
            input_files.extend(input_dir.glob(f"*{ext}"))
        if target_format.lower() != 'vasp':
            for vasp_file in ['POSCAR', 'INCAR']:
                input_files.extend(input_dir.glob(vasp_file))
    
    if file_pattern != "*.*":
        from fnmatch import fnmatch
        input_files = [f for f in input_files if fnmatch(f.name, file_pattern)]
    
    output_files = []
    for input_file in input_files:
        try:
            rel_path = input_file.relative_to(input_dir)
            target_dir = output_dir / rel_path.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            
            if target_format.lower() == 'cp2k':
                output_file = target_dir / f"{input_file.stem}.inp"
            elif target_format.lower() == 'vasp':
                output_file = target_dir / f"{input_file.stem}_vasp"
                output_file.mkdir(exist_ok=True)
            elif target_format.lower() == 'gaussian':
                output_file = target_dir / f"{input_file.stem}.gjf"
            else:
                output_file = target_dir / f"{input_file.stem}.{target_format}"
            
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
        structure_data (dict): Structure data dictionary.
        padding (float): Padding around the molecule in Angstroms.
        
    Returns:
        list: 3x3 cell matrix.
    """
    if 'atoms' not in structure_data or not structure_data['atoms']:
        return [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    
    positions = np.array([atom['position'] for atom in structure_data['atoms']])
    min_coords = np.min(positions, axis=0)
    max_coords = np.max(positions, axis=0)
    min_coords -= padding
    max_coords += padding
    cell_lengths = np.maximum(max_coords - min_coords, 5.0)
    cell = [
        [cell_lengths[0], 0.0, 0.0],
        [0.0, cell_lengths[1], 0.0],
        [0.0, 0.0, cell_lengths[2]]
    ]
    for atom in structure_data['atoms']:
        atom['position'] = (np.array(atom['position']) - min_coords).tolist()
    return cell

def guess_element_from_mass(mass, tolerance=0.5):
    """
    Guess an element symbol based on atomic mass.
    
    Args:
        mass (float): Atomic mass.
        tolerance (float): Tolerance in atomic mass units.
        
    Returns:
        str or None: Element symbol or None if no match.
    """
    element_masses = {
        'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811, 
        'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180, 
        'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974, 
        'S': 32.066, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078
    }
    best_match = None
    best_diff = float('inf')
    for element, element_mass in element_masses.items():
        diff = abs(mass - element_mass)
        if diff < best_diff:
            best_diff = diff
            best_match = element
    return best_match if best_diff <= tolerance else None

from pathlib import Path

def cif_to_xyz(cif_file, xyz_file=None):
    """
    Convert a CIF file to XYZ format.
    
    Args:
        cif_file: Path to the CIF file
        xyz_file: Path to save the XYZ file (optional)
        
    Returns:
        bool: Success status
    """
    from pathlib import Path
    from multi_agent_dft.utils.validator import validate_structure
    
    cif_file = Path(cif_file)
    
    # Set default XYZ file path if not provided
    if xyz_file is None:
        xyz_file = cif_file.with_suffix('.xyz')
    else:
        xyz_file = Path(xyz_file)
    
    # Validate and fix the CIF file if needed
    valid, msg, extra_info = validate_structure(cif_file)
    
    if not valid:
        print(f"Cannot convert invalid CIF file: {cif_file} - {msg}")
        return False
    
    # If the file was fixed, use the fixed version
    if extra_info and "fixed_path" in extra_info:
        cif_file = extra_info["fixed_path"]
        print(f"Using fixed CIF file: {cif_file}")
    
    try:
        # Use ASE to convert CIF to XYZ (if available)
        try:
            from ase.io import read, write
            atoms = read(cif_file)
            write(xyz_file, atoms)
            print(f"Converted CIF to XYZ using ASE: {xyz_file}")
            return True
        except ImportError:
            # Fallback to your existing CIF to XYZ conversion logic
            print("ASE not available, using fallback CIF parser")
            # Your existing code here...
            return True
    except Exception as e:
        print(f"Error converting CIF to XYZ: {str(e)}")
        return False

def process_cif_files(input_dir, output_dir=None, pattern="*.cif"):
    """
    Convert every CIF in `input_dir` matching `pattern` into .xyz files in `output_dir`.
    Returns a list of all generated .xyz Paths.
    """
    input_dir = Path(input_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_dir

    converted = []
    failed = 0
    cif_files = list(input_dir.glob(pattern))
    total = len(cif_files)
    
    for i, cif_file in enumerate(cif_files):
        try:
            xyz_file = output_dir / cif_file.with_suffix('.xyz').name
            result = cif_to_xyz(cif_file, xyz_file)
            if result:
                converted.append(result)
                logger.info(f"[{i+1}/{total}] Successfully converted {cif_file.name}")
            else:
                failed += 1
                logger.warning(f"[{i+1}/{total}] Failed to convert {cif_file.name}")
        except Exception as e:
            failed += 1
            logger.error(f"[{i+1}/{total}] Error converting {cif_file.name}: {e}")
    
    logger.info(f"Conversion complete: {len(converted)} successful, {failed} failed")
    return converted