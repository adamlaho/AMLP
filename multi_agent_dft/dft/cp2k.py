"""
CP2K-specific functionality for the Multi-Agent DFT Research System.
"""

import os
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CP2KInterface:
    """Interface for generating CP2K input files and analyzing CP2K outputs."""
    
    def __init__(self, config=None):
        """
        Initialize the CP2K interface with optional configuration.
        
        Args:
            config (dict, optional): Configuration parameters for CP2K calculations.
        """
        self.config = config or {}
        self.default_config = {
            'global': {
                'RUN_TYPE': 'ENERGY',
                'PRINT_LEVEL': 'MEDIUM',
                'PROJECT': 'cp2k_project'
            },
            'dft': {
                'BASIS_SET_FILE_NAME': 'BASIS_MOLOPT',
                'POTENTIAL_FILE_NAME': 'GTH_POTENTIALS',
                'xc': {'XC_FUNCTIONAL': 'PBE'}
            },
            'mgrid': {
                'CUTOFF': 400,
                'NGRIDS': 4
            },
            'qs': {
                'EPS_DEFAULT': 1.0E-10,
                'METHOD': 'GPW'
            },
            'scf': {
                'SCF_GUESS': 'ATOMIC',
                'EPS_SCF': 1.0E-6,
                'MAX_SCF': 50,
                'ADDED_MOS': 20
            },
            'outer_scf': {
                'MAX_SCF': 10,
                'EPS_SCF': 1.0E-6
            },
            'kind_parameters': {}
        }
        
    def get_config(self):
        """
        Get the effective configuration by merging default with user config.
        
        Returns:
            dict: Effective configuration.
        """
        import copy
        effective_config = copy.deepcopy(self.default_config)
        
        # Update nested dictionaries
        for key, value in self.config.items():
            if isinstance(value, dict) and key in effective_config and isinstance(effective_config[key], dict):
                effective_config[key].update(value)
            else:
                effective_config[key] = value
                
        return effective_config
    
    def generate_input(self, structure_data, output_file=None, custom_config=None):
        """
        Generate a CP2K input file from structure data.
        
        Args:
            structure_data (dict): Structure data with atom coordinates and metadata.
            output_file (str or Path, optional): Path to save the input file.
            custom_config (dict, optional): Custom configuration for this specific job.
            
        Returns:
            str or bool: CP2K input file content if output_file is None, 
                         otherwise returns True if successful, False if failed.
        """
        # Merge configs
        config = self.get_config()
        if custom_config:
            for key, value in custom_config.items():
                if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    config[key].update(value)
                else:
                    config[key] = value
        
        # Generate input content
        input_content = build_cp2k_input(structure_data, config)
        
        # Save to file if requested
        if output_file:
            return save_cp2k_input(structure_data, output_file, config)
        else:
            return input_content
    
    def parse_output(self, output_file):
        """
        Parse a CP2K output file.
        
        Args:
            output_file (str or Path): Path to the CP2K output file.
            
        Returns:
            dict: Parsed output data.
        """
        return parse_cp2k_output(output_file)


def build_cp2k_input(structure_data, cp2k_config):
    """
    Build a CP2K input file from structure data and configuration.
    
    Args:
        structure_data (dict): Structure data (coordinates, cell, atoms).
        cp2k_config (dict): CP2K configuration.
    
    Returns:
        str: CP2K input file content.
    """
    input_sections = []
    
    # GLOBAL section
    global_section = ["&GLOBAL"]
    for key, value in cp2k_config.get('global', {}).items():
        global_section.append(f"  {key} {value}")
    global_section.append("&END GLOBAL")
    input_sections.append("\n".join(global_section))
    
    # FORCE_EVAL section
    force_eval = ["&FORCE_EVAL", "  METHOD Quickstep"]
    
    # DFT subsection
    dft = ["  &DFT"]
    for key, value in cp2k_config.get('dft', {}).items():
        dft.append(f"    {key} {value}")
    
    # MGRID subsection
    dft.append("    &MGRID")
    for key, value in cp2k_config.get('mgrid', {}).items():
        dft.append(f"      {key} {value}")
    dft.append("    &END MGRID")
    
    # QS subsection
    dft.append("    &QS")
    for key, value in cp2k_config.get('qs', {}).items():
        dft.append(f"      {key} {value}")
    dft.append("    &END QS")
    
    # SCF subsection
    dft.append("    &SCF")
    for key, value in cp2k_config.get('scf', {}).items():
        dft.append(f"      {key} {value}")
    
    # OUTER_SCF subsection
    dft.append("      &OUTER_SCF")
    for key, value in cp2k_config.get('outer_scf', {}).items():
        dft.append(f"        {key} {value}")
    dft.append("      &END OUTER_SCF")
    dft.append("    &END SCF")
    
    # XC subsection
    dft.append("    &XC")
    
    # XC_GRID subsection
    if 'xc_grid' in cp2k_config.get('xc', {}):
        dft.append("      &XC_GRID")
        for key, value in cp2k_config.get('xc', {}).get('xc_grid', {}).items():
            dft.append(f"        {key} {value}")
        dft.append("      &END XC_GRID")
    
    # XC_FUNCTIONAL subsection
    if 'xc_functional' in cp2k_config.get('xc', {}):
        dft.append("      &XC_FUNCTIONAL")
        for key, value in cp2k_config.get('xc', {}).get('xc_functional', {}).items():
            dft.append(f"        {key} {value}")
        dft.append("      &END XC_FUNCTIONAL")
    
    # VDW_POTENTIAL subsection
    if 'vdw_potential' in cp2k_config.get('xc', {}):
        dft.append("      &VDW_POTENTIAL")
        for key, value in cp2k_config.get('xc', {}).get('vdw_potential', {}).items():
            if key != 'pair_potential':
                dft.append(f"        {key} {value}")
        
        # PAIR_POTENTIAL subsection
        if 'pair_potential' in cp2k_config.get('xc', {}).get('vdw_potential', {}):
            dft.append("        &PAIR_POTENTIAL")
            for key, value in cp2k_config.get('xc', {}).get('vdw_potential', {}).get('pair_potential', {}).items():
                dft.append(f"          {key} {value}")
            dft.append("        &END PAIR_POTENTIAL")
        
        dft.append("      &END VDW_POTENTIAL")
    
    dft.append("    &END XC")
    dft.append("  &END DFT")
    
    # SUBSYS subsection
    subsys = ["  &SUBSYS"]
    
    # CELL subsection
    subsys.append("    &CELL")
    if 'cell' in structure_data:
        cell = structure_data['cell']
        subsys.append(f"      A {cell[0][0]} {cell[0][1]} {cell[0][2]}")
        subsys.append(f"      B {cell[1][0]} {cell[1][1]} {cell[1][2]}")
        subsys.append(f"      C {cell[2][0]} {cell[2][1]} {cell[2][2]}")
    else:
        # Default to a 10x10x10 Å box if no cell is provided
        subsys.append("      ABC 10.0 10.0 10.0")
    subsys.append("    &END CELL")
    
    # COORD subsection
    subsys.append("    &COORD")
    for atom in structure_data.get('atoms', []):
        x, y, z = atom['position']
        subsys.append(f"      {atom['symbol']} {x:.6f} {y:.6f} {z:.6f}")
    subsys.append("    &END COORD")
    
    # KIND subsection for each element
    kind_parameters = cp2k_config.get('kind_parameters', {})
    atom_symbols = {atom['symbol'] for atom in structure_data.get('atoms', [])}
    
    for symbol in atom_symbols:
        subsys.append(f"    &KIND {symbol}")
        if symbol in kind_parameters:
            for key, value in kind_parameters[symbol].items():
                subsys.append(f"      {key} {value}")
        else:
            # Default parameters if not specified
            subsys.append("      BASIS_SET DZVP-MOLOPT-SR-GTH")
            subsys.append("      POTENTIAL GTH-PBE")
        subsys.append("    &END KIND")
    
    subsys.append("  &END SUBSYS")
    
    force_eval.extend(dft)
    force_eval.extend(subsys)
    force_eval.append("&END FORCE_EVAL")
    
    input_sections.append("\n".join(force_eval))
    
    return "\n\n".join(input_sections)


def save_cp2k_input(structure_data, output_path, cp2k_config):
    """
    Save a CP2K input file.
    
    Args:
        structure_data (dict): Structure data (coordinates, cell, atoms).
        output_path (str or Path): Output file path.
        cp2k_config (dict): CP2K configuration.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        input_content = build_cp2k_input(structure_data, cp2k_config)
        
        with open(output_path, 'w') as f:
            f.write(input_content)
        
        logger.info(f"CP2K input file saved to {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving CP2K input file: {e}")
        return False


def interactive_cp2k_config():
    """
    Interactively ask the user for CP2K parameters.
    
    Returns:
        dict: CP2K configuration.
    """
    from ..config import load_config
    
    # Start with default CP2K configuration
    cp2k_config = load_config('cp2k_defaults.yaml')
    
    print("\nPlease enter CP2K parameters (press Enter to use defaults):")
    
    # Project name
    project = input(f"Project name (default '{cp2k_config['global']['PROJECT']}'): ")
    if project:
        cp2k_config['global']['PROJECT'] = project
    
    # Run type
    run_type = input(f"Run type (default '{cp2k_config['global']['RUN_TYPE']}'): ")
    if run_type:
        cp2k_config['global']['RUN_TYPE'] = run_type
    
    # Basis set file
    basis_set = input(f"Basis set file name (default '{cp2k_config['dft']['BASIS_SET_FILE_NAME']}'): ")
    if basis_set:
        cp2k_config['dft']['BASIS_SET_FILE_NAME'] = basis_set
    
    # Potential file
    potential_file = input(f"Potential file name (default '{cp2k_config['dft']['POTENTIAL_FILE_NAME']}'): ")
    if potential_file:
        cp2k_config['dft']['POTENTIAL_FILE_NAME'] = potential_file
    
    # MGRID parameters
    ngrids = input(f"MGRID NGRIDS (default {cp2k_config['mgrid']['NGRIDS']}): ")
    if ngrids:
        cp2k_config['mgrid']['NGRIDS'] = int(ngrids)
    
    cutoff = input(f"MGRID CUTOFF (default {cp2k_config['mgrid']['CUTOFF']}): ")
    if cutoff:
        cp2k_config['mgrid']['CUTOFF'] = float(cutoff)
    
    # SCF parameters
    max_scf = input(f"MAX_SCF (default {cp2k_config['scf']['MAX_SCF']}): ")
    if max_scf:
        cp2k_config['scf']['MAX_SCF'] = int(max_scf)
    
    eps_scf = input(f"EPS_SCF (default {cp2k_config['scf']['EPS_SCF']}): ")
    if eps_scf:
        cp2k_config['scf']['EPS_SCF'] = float(eps_scf)
    
    # Ask about custom elements
    custom_elements = input("Do you want to specify custom basis sets for any elements? (y/n, default n): ")
    if custom_elements.lower() == 'y':
        while True:
            element = input("Enter element symbol (or press Enter to finish): ")
            if not element:
                break
            
            basis_set = input(f"Basis set for {element} (default 'DZVP-MOLOPT-SR-GTH'): ")
            potential = input(f"Potential for {element} (default 'GTH-PBE'): ")
            
            cp2k_config['kind_parameters'][element] = {
                "BASIS_SET": basis_set if basis_set else "DZVP-MOLOPT-SR-GTH",
                "POTENTIAL": potential if potential else "GTH-PBE"
            }
    
    return cp2k_config


def parse_cp2k_output(output_file):
    """
    Parse a CP2K output file and extract key information.
    
    Args:
        output_file (str or Path): Path to the CP2K output file.
    
    Returns:
        dict: Extracted information from the output file.
    """
    # Placeholder implementation
    logger.warning("CP2K output parsing not yet fully implemented")
    
    result = {
        "energy": None,
        "converged": False,
        "errors": []
    }
    
    try:
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Check for convergence
        if "SCF run converged" in content:
            result["converged"] = True
        
        # Try to extract final energy
        energy_match = re.search(r"ENERGY\|.*?:\s+([-\d.]+)", content)
        if energy_match:
            result["energy"] = float(energy_match.group(1))
        
        # Check for errors
        errors = []
        error_lines = re.findall(r"ERROR.*", content)
        for error in error_lines:
            errors.append(error.strip())
        
        result["errors"] = errors
    
    except Exception as e:
        logger.error(f"Error parsing CP2K output: {e}")
        result["errors"].append(str(e))
    
    return result