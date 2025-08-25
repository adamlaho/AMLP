# multi_agent_dft/utils/validation.py
import os
from pathlib import Path
from typing import Tuple, Union, List, Dict, Any

from multi_agent_dft.utils.logging import get_logger

logger = get_logger(__name__)

def validate_structure(file_path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Validate a structure file or directory of structure files.
    
    Args:
        file_path: Path to the structure file or directory
        
    Returns:
        Tuple containing:
            - Boolean indicating whether the file/directory is valid
            - Content of the file if valid, directory path if a valid directory, or error message if invalid
    """
    file_path = Path(file_path).expanduser().resolve()
    
    # Check if path exists
    if not file_path.exists():
        logger.error(f"Path does not exist: {file_path}")
        return False, "Path does not exist."
    
    # If it's a directory, check if it contains valid structure files
    if file_path.is_dir():
        cif_files = list(file_path.glob("*.cif"))
        xyz_files = list(file_path.glob("*.xyz"))
        
        if cif_files or xyz_files:
            num_files = len(cif_files) + len(xyz_files)
            logger.info(f"Found {len(cif_files)} CIF files and {len(xyz_files)} XYZ files in {file_path}")
            return True, str(file_path)
        else:
            logger.error(f"No CIF or XYZ files found in directory: {file_path}")
            return False, "No CIF or XYZ files found in directory."
    
    # If it's a file, proceed with file validation
    # Check file size
    max_size_mb = 10  # Default max file size in MB
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        logger.error(f"File too large: {file_path} ({file_size_mb:.2f} MB > {max_size_mb} MB)")
        return False, f"File is too large ({file_size_mb:.2f} MB). Maximum size is {max_size_mb} MB."
    
    # Check file extension
    allowed_extensions = ['.xyz', '.cif']
    if not any(file_path.name.lower().endswith(ext) for ext in allowed_extensions):
        logger.error(f"Invalid file extension: {file_path}")
        return False, f"Invalid file extension. Expected: {', '.join(allowed_extensions)}"
    
    # Check if file is not empty
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
        
        if not content:
            logger.error(f"File is empty: {file_path}")
            return False, "File is empty."
        
        # Basic format validation based on file extension
        if file_path.name.lower().endswith('.xyz'):
            return validate_xyz_format(content, file_path)
        elif file_path.name.lower().endswith('.cif'):
            return validate_cif_format(content, file_path)
        else:
            # Should not reach here due to extension check above
            return False, "Unsupported file format."
            
    except UnicodeDecodeError:
        logger.error(f"File is not a text file: {file_path}")
        return False, "File is not a text file. Only ASCII/UTF-8 files are supported."
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return False, f"Error reading file: {str(e)}"

def validate_xyz_format(content: str, file_path: Path) -> Tuple[bool, str]:
    """
    Validate the format of an XYZ file.
    
    Args:
        content: Content of the XYZ file
        file_path: Path to the XYZ file
        
    Returns:
        Tuple containing:
            - Boolean indicating whether the file format is valid
            - Content if valid, error message if invalid
    """
    lines = content.strip().split('\n')
    
    # XYZ format should have at least 3 lines
    if len(lines) < 3:
        logger.error(f"Invalid XYZ format (too few lines): {file_path}")
        return False, "Invalid XYZ format: file has too few lines."
    
    # First line should be a number
    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        logger.error(f"Invalid XYZ format (first line not a number): {file_path}")
        return False, "Invalid XYZ format: first line should be the number of atoms."
    
    # Check if the number of atoms matches the actual content
    if len(lines) - 2 != num_atoms:
        logger.error(f"Invalid XYZ format (atom count mismatch): {file_path}")
        return False, f"Invalid XYZ format: expected {num_atoms} atoms, but found {len(lines) - 2}."
    
    # Check if atom lines have the correct format
    for i in range(2, len(lines)):
        parts = lines[i].strip().split()
        if len(parts) < 4:
            logger.error(f"Invalid XYZ format (line {i+1} missing coordinates): {file_path}")
            return False, f"Invalid XYZ format: line {i+1} is missing coordinates."
        
        # Try to parse coordinates as floats
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            logger.error(f"Invalid XYZ format (line {i+1} has invalid coordinates): {file_path}")
            return False, f"Invalid XYZ format: line {i+1} has invalid coordinates."
    
    return True, content

def validate_cif_format(content: str, file_path: Path) -> Tuple[bool, str]:
    """
    Validate the format of a CIF file.
    
    Args:
        content: Content of the CIF file
        file_path: Path to the CIF file
        
    Returns:
        Tuple containing:
            - Boolean indicating whether the file format is valid
            - Content if valid, error message if invalid
    """
    # Basic CIF validation - check for essential data blocks
    required_fields = [
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma"
    ]
    
    # Check for at least some required fields
    missing_fields = []
    for field in required_fields:
        if field.lower() not in content.lower():
            missing_fields.append(field)
    
    if len(missing_fields) > 3:  # Allow some flexibility
        logger.error(f"Invalid CIF format (missing essential fields): {file_path}")
        return False, f"Invalid CIF format: missing essential fields {', '.join(missing_fields[:3])}..."
    
    # Check for atomic positions
    if "_atom_site_" not in content.lower():
        logger.error(f"Invalid CIF format (no atomic positions): {file_path}")
        return False, "Invalid CIF format: no atomic position data found."
    
    return True, content