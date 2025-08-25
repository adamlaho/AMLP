"""
CIF validation and fixing utilities.
"""
from pathlib import Path

def fix_cif_data_block(cif_file_path):
    """
    Fix CIF files by ensuring they have a valid data_ block with an identifier.
    
    Args:
        cif_file_path: Path to the CIF file
        
    Returns:
        tuple: (fixed_path, was_modified)
            fixed_path: Path to the fixed file (may be the original if no changes were needed)
            was_modified: Boolean indicating if the file was modified
    """
    cif_path = Path(cif_file_path)
    
    # Read the file content
    with open(cif_path, 'r') as f:
        content = f.read()
    
    # Case 1: File has no data_ block at all
    if 'data_' not in content:
        # Create a new file with the data_ block added
        fixed_path = cif_path.with_suffix('.fixed.cif')
        fixed_content = f'data_{cif_path.stem}\n' + content
        with open(fixed_path, 'w') as f:
            f.write(fixed_content)
        return fixed_path, True
    
    # Case 2: File has an empty data_ block (data_ with no identifier)
    first_line = content.splitlines()[0] if content.splitlines() else ""
    if first_line.strip() == 'data_' or 'data_\n' in content[:20]:
        # Replace the empty data_ with a named one
        fixed_path = cif_path.with_suffix('.fixed.cif')
        fixed_content = content.replace('data_', f'data_{cif_path.stem}', 1)
        with open(fixed_path, 'w') as f:
            f.write(fixed_content)
        return fixed_path, True
    
    # Case 3: File already has a valid data_ block with identifier
    return cif_path, False