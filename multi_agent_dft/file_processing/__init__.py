"""
File processing module for the Multi-Agent DFT Research System.
"""

from .cif import parse_cif_file, cif_to_xyz, process_cif_files
from .xyz import parse_xyz, xyz_to_mol_formula, process_xyz_files

import logging


logger = logging.getLogger(__name__)


def validate_structure(file_path):
    """
    Validate that the structure file exists, has a correct extension (.xyz or .cif), and is non-empty.
    
    Args:
        file_path (str): Path to the structure file.
    
    Returns:
        tuple: (is_valid, content_or_error_message)
    """
    import os
    
    if not os.path.exists(file_path):
        return False, "File does not exist."
    
    valid_extensions = ['.xyz', '.cif']
    if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
        return False, "Invalid file extension. Expected .xyz or .cif"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
        
        if not content:
            return False, "File is empty."
        
        return True, content
    
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def process_structure_file(file_path, output_dir=None, dft_code="cp2k", dft_config=None):
    """
    Process a structure file and generate DFT input.
    
    Args:
        file_path (str): Path to the structure file (.xyz or .cif).
        output_dir (str, optional): Directory to save the output files.
        dft_code (str, optional): DFT code to use ('cp2k', 'vasp', 'gaussian').
        dft_config (dict, optional): Configuration for the DFT input.
    
    Returns:
        str or None: Path to the generated input file, or None if failed.
    """
    from pathlib import Path
    
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File does not exist: {file_path}")
        return None
    
    # Determine file type
    if file_path.suffix.lower() == '.xyz':
        structure_data = parse_xyz(file_path)
        if structure_data is None:
            return None
    
    elif file_path.suffix.lower() == '.cif':
        structure_data = parse_cif(file_path)
        if structure_data is None:
            return None
    
    else:
        logger.error(f"Unsupported file extension: {file_path.suffix}")
        return None
    
    # Set up output directory
    if output_dir is None:
        output_dir = file_path.parent / "dft_inputs" / file_path.stem
    else:
        output_dir = Path(output_dir) / file_path.stem
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate DFT input
    if dft_code.lower() == "cp2k":
        from ..dft.cp2k import save_cp2k_input
        output_file = output_dir / f"{file_path.stem}.inp"
        success = save_cp2k_input(structure_data, output_file, dft_config)
    
    elif dft_code.lower() == "vasp":
        from ..dft.vasp import save_vasp_inputs
        vasp_dir = output_dir / "vasp"
        vasp_dir.mkdir(exist_ok=True)
        success = save_vasp_inputs(structure_data, vasp_dir, dft_config)
        output_file = vasp_dir / "POSCAR"  # Return POSCAR path as the main output
    
    elif dft_code.lower() == "gaussian":
        from ..dft.gaussian import save_gaussian_input
        output_file = output_dir / f"{file_path.stem}.gjf"
        success = save_gaussian_input(structure_data, output_file, dft_config)
    
    else:
        logger.error(f"Unsupported DFT code: {dft_code}")
        return None
    
    if success:
        logger.info(f"Successfully generated {dft_code} input file: {output_file}")
        return str(output_file)
    else:
        logger.error(f"Failed to generate {dft_code} input file")
        return None


def process_structure_files(input_dir, output_dir=None, file_pattern="*.*", dft_code="cp2k", dft_config=None):
    """
    Process all structure files in a directory and generate DFT inputs.
    
    Args:
        input_dir (str): Directory containing structure files.
        output_dir (str, optional): Directory to save the output files.
        file_pattern (str, optional): Glob pattern for structure files.
        dft_code (str, optional): DFT code to use ('cp2k', 'vasp', 'gaussian').
        dft_config (dict, optional): Configuration for the DFT input.
    
    Returns:
        list: Paths to the generated input files.
    """
    from pathlib import Path
    import glob
    
    input_dir = Path(input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory does not exist or is not a directory: {input_dir}")
        return []
    
    # Set up output directory
    if output_dir is None:
        output_dir = input_dir / "dft_inputs"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all structure files
    pattern = str(input_dir / file_pattern)
    files = [Path(f) for f in glob.glob(pattern) if Path(f).is_file()]
    
    xyz_files = [f for f in files if f.suffix.lower() == '.xyz']
    cif_files = [f for f in files if f.suffix.lower() == '.cif']
    
    logger.info(f"Found {len(xyz_files)} XYZ files and {len(cif_files)} CIF files")
    
    # Process files by type
    generated_files = []
    
    if xyz_files:
        xyz_output = process_xyz_files(input_dir, output_dir, "*.xyz", dft_code, dft_config)
        generated_files.extend(xyz_output)
    
    if cif_files:
        cif_output = process_cif_files(input_dir, output_dir, "*.cif", False, dft_code, dft_config)
        generated_files.extend(cif_output)
    
    return generated_files