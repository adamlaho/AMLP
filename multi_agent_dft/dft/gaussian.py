"""
Gaussian interface module for the Multi-Agent DFT Research System.
"""

import os
import re
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class GaussianInterface:
    """Interface for generating Gaussian input files and analyzing Gaussian outputs."""
    
    def __init__(self, config=None):
        """
        Initialize the Gaussian interface with optional configuration.
        
        Args:
            config (dict, optional): Configuration parameters for Gaussian calculations.
        """
        self.config = config or {}
        self.default_config = {
            'method': 'B3LYP',
            'basis_set': '6-31G(d)',
            'job_type': 'Opt',
            'charge': 0,
            'multiplicity': 1,
            'memory': '4GB',
            'nproc': 4
        }
    
    def get_config(self):
        """
        Get the effective configuration by merging default with user config.
        
        Returns:
            dict: Effective configuration.
        """
        effective_config = self.default_config.copy()
        effective_config.update(self.config)
        return effective_config
    
    def generate_input(self, structure_data, output_file=None, custom_config=None):
        """
        Generate a Gaussian input file from structure data.
        
        Args:
            structure_data (dict): Structure data with atom coordinates and metadata.
            output_file (str or Path, optional): Path to save the input file.
            custom_config (dict, optional): Custom configuration for this specific job.
            
        Returns:
            str: Gaussian input file content.
        """
        # Merge configs
        config = self.get_config()
        if custom_config:
            config.update(custom_config)
        
        # Build the Gaussian input file
        gjf_lines = []
        
        # Add checkpoint file specification if requested
        if 'save_chk' in config:
            gjf_lines.append(f"%chk={config['save_chk']}")
        
        # Memory and processor specification
        gjf_lines.append(f"%mem={config.get('memory', '4GB')}")
        gjf_lines.append(f"%nprocshared={config.get('nproc', 4)}")
        
        # Route section
        method = config.get('method', 'B3LYP')
        basis = config.get('basis_set', '6-31G(d)')
        job = config.get('job_type', 'Opt')
        
        route = f"#p {method}/{basis} {job}"
        
        # Add additional keywords for SCF
        if 'scf' in config and config['scf'] == 'tight':
            route += " SCF=Tight"
        
        # Add population analysis
        if 'pop' in config and config['pop'] != 'None':
            route += f" Pop={config['pop']}"
        
        # Add solvent if specified
        if 'solvent' in config:
            solvent_model = config.get('solvent_model', 'PCM')
            route += f" SCRF=({solvent_model},Solvent={config['solvent']})"
        
        # Add dispersion correction if specified
        if 'dispersion' in config:
            route += f" EmpiricalDispersion={config['dispersion']}"
        
        # Add any additional route options (including force printing)
        if 'route_additions' in config and isinstance(config['route_additions'], list):
            for addition in config['route_additions']:
                route += f" {addition}"
        
        gjf_lines.append(route)
        gjf_lines.append("")  # Empty line
        
        # Title section
        filename = structure_data['meta'].get('filename', 'structure')
        gjf_lines.append(f"Gaussian calculation for {filename}")
        gjf_lines.append("")  # Empty line
        
        # Charge and multiplicity
        charge = config.get('charge', 0)
        multiplicity = config.get('multiplicity', 1)
        gjf_lines.append(f"{charge} {multiplicity}")
        
        # Atomic coordinates
        for atom in structure_data['atoms']:
            pos = atom['position']
            gjf_lines.append(f"{atom['symbol']} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
        
        gjf_lines.append("")  # Empty line
        
        input_content = "\n".join(gjf_lines)
        
        # Save to file if requested
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(input_content)
            logger.info(f"Saved Gaussian input file to {output_file}")
        
        return input_content
    
    
    def parse_output(self, output_file):
        """
        Parse a Gaussian output file.
        
        Args:
            output_file (str or Path): Path to the Gaussian output file.
            
        Returns:
            dict: Parsed output data.
        """
        output_file = Path(output_file)
        if not output_file.exists():
            logger.error(f"Gaussian output file not found: {output_file}")
            return None
        
        # Initialize results dictionary
        results = {
            'converged': False,
            'energy': None,
            'gradient': None,
            'charges': {},
            'dipole': None,
            'frequencies': [],
            'meta': {
                'filename': str(output_file.name)
            }
        }
        
        try:
            with open(output_file, 'r') as f:
                content = f.read()
                
            # Check if calculation converged
            if "Normal termination" in content:
                results['converged'] = True
            
            # Extract SCF energy
            energy_match = re.search(r'SCF Done:.*?=\s+([-\d.]+)', content)
            if energy_match:
                results['energy'] = float(energy_match.group(1))
            
            # Extract dipole moment
            dipole_match = re.search(r'Dipole moment \(Debye\):\s*X=\s*([-\d.]+)\s*Y=\s*([-\d.]+)\s*Z=\s*([-\d.]+)', content)
            if dipole_match:
                results['dipole'] = [float(dipole_match.group(1)), float(dipole_match.group(2)), float(dipole_match.group(3))]
            
            # Extract frequencies if available
            freq_matches = re.findall(r'Frequencies --\s+([-\d.]+)', content)
            if freq_matches:
                results['frequencies'] = [float(f) for f in freq_matches]
            
            logger.info(f"Successfully parsed Gaussian output file: {output_file}")
            return results
            
        except Exception as e:
            logger.error(f"Error parsing Gaussian output file {output_file}: {e}")
            return None

def save_gaussian_input(structure_data, output_file, config=None):
    """
    Save a Gaussian input file based on structure data.
    
    Args:
        structure_data (dict): Structure data from parse_cif() or other sources.
        output_file (str or Path): Path to save the input file.
        config (dict, optional): Configuration for the calculation.
        
    Returns:
        bool: Success status.
    """
    try:
        interface = GaussianInterface(config)
        interface.generate_input(structure_data, output_file)
        logger.info(f"Generated Gaussian input file: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error generating Gaussian input: {e}")
        return False