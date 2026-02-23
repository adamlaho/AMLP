"""
Psi4-specific functionality for the Multi-Agent DFT Research System.
"""

import os
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class Psi4Interface:
    """Interface for generating Psi4 input files and analyzing Psi4 outputs."""

    def __init__(self, config=None):
        """
        Initialize the Psi4 interface with optional configuration.

        Args:
            config (dict, optional): Configuration parameters for Psi4 calculations.
        """
        self.config = config or {}
        self.default_config = {
            'method': 'B3LYP',
            'basis_set': 'def2-TZVP',
            'job_type': 'optimize',
            'charge': 0,
            'multiplicity': 1,
            'memory': '4 GB',
            'nthreads': 4,
            'scf_type': 'DF',  # Density fitting
            'reference': 'RHF',
            'freeze_core': True,
            'dft_spherical_points': 302,
            'dft_radial_points': 75
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
        Generate a Psi4 input file from structure data.

        Args:
            structure_data (dict): Structure data with atom coordinates and metadata.
            output_file (str or Path, optional): Path to save the input file.
            custom_config (dict, optional): Custom configuration for this specific job.

        Returns:
            str: Psi4 input file content.
        """
        # Merge configs
        config = self.get_config()
        if custom_config:
            config.update(custom_config)

        # Build the Psi4 input file
        psi4_lines = []

        # Memory and thread specification
        psi4_lines.append(f"memory {config.get('memory', '4 GB')}")
        psi4_lines.append(f"set_num_threads({config.get('nthreads', 4)})")
        psi4_lines.append("")

        # Molecule specification
        charge = config.get('charge', 0)
        mult = config.get('multiplicity', 1)

        psi4_lines.append("molecule {")
        psi4_lines.append(f"  {charge} {mult}")

        # Add atomic coordinates
        if 'coordinates' in structure_data:
            for coord in structure_data['coordinates']:
                element = coord.get('element', coord.get('atom_type', 'X'))
                x = coord.get('x', 0.0)
                y = coord.get('y', 0.0)
                z = coord.get('z', 0.0)
                psi4_lines.append(f"  {element:2s}  {x:12.8f}  {y:12.8f}  {z:12.8f}")
        elif 'atoms' in structure_data:
            for atom in structure_data['atoms']:
                element = atom.get('element', atom.get('symbol', 'X'))
                pos = atom.get('position', [0.0, 0.0, 0.0])
                psi4_lines.append(f"  {element:2s}  {pos[0]:12.8f}  {pos[1]:12.8f}  {pos[2]:12.8f}")

        # Handle units
        units = structure_data.get('units', 'angstrom')
        psi4_lines.append(f"  units {units}")

        # Add symmetry if specified
        symmetry = config.get('symmetry', 'c1')
        psi4_lines.append(f"  symmetry {symmetry}")

        psi4_lines.append("}")
        psi4_lines.append("")

        # Set calculation options
        psi4_lines.append("set {")
        psi4_lines.append(f"  basis {config.get('basis_set', 'def2-TZVP')}")
        psi4_lines.append(f"  scf_type {config.get('scf_type', 'DF')}")
        psi4_lines.append(f"  reference {config.get('reference', 'RHF')}")

        # Add freeze_core if applicable
        if config.get('freeze_core', True):
            psi4_lines.append("  freeze_core True")

        # DFT grid specification
        if 'dft' in config.get('method', '').lower() or any(x in config.get('method', '').upper() for x in ['B3LYP', 'PBE', 'BLYP']):
            psi4_lines.append(f"  dft_spherical_points {config.get('dft_spherical_points', 302)}")
            psi4_lines.append(f"  dft_radial_points {config.get('dft_radial_points', 75)}")

        # Add dispersion correction if specified
        if 'dispersion' in config:
            psi4_lines.append(f"  dft_dispersion_parameters [{config['dispersion']}]")

        # SCF convergence
        if 'scf_convergence' in config:
            psi4_lines.append(f"  e_convergence {config['scf_convergence']}")
            psi4_lines.append(f"  d_convergence {config.get('density_convergence', config['scf_convergence'] * 10)}")

        # Add geometry optimization options if needed
        if config.get('job_type', 'optimize') in ['optimize', 'opt']:
            if 'opt_convergence' in config:
                psi4_lines.append(f"  geom_maxiter {config.get('max_iterations', 50)}")
                psi4_lines.append(f"  g_convergence {config['opt_convergence']}")

        psi4_lines.append("}")
        psi4_lines.append("")

        # Job execution command
        job_type = config.get('job_type', 'optimize')
        method = config.get('method', 'B3LYP')

        if job_type in ['optimize', 'opt']:
            psi4_lines.append(f"optimize('{method}')")
        elif job_type == 'energy':
            psi4_lines.append(f"energy('{method}')")
        elif job_type == 'frequency':
            psi4_lines.append(f"freq, wfn = frequency('{method}', return_wfn=True)")
        elif job_type == 'gradient':
            psi4_lines.append(f"grad, wfn = gradient('{method}', return_wfn=True)")
        else:
            psi4_lines.append(f"{job_type}('{method}')")

        psi4_lines.append("")

        input_content = "\n".join(psi4_lines)

        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(input_content)
            logger.info(f"Psi4 input file written to {output_path}")
            return True

        return input_content


def save_psi4_input(structure_data, output_file, config=None):
    """
    Convenience function to save a Psi4 input file.

    Args:
        structure_data (dict): Structure data with atom coordinates and metadata.
        output_file (str or Path): Path to save the input file.
        config (dict, optional): Configuration parameters for Psi4.

    Returns:
        bool: Success status.
    """
    interface = Psi4Interface(config)
    return interface.generate_input(structure_data, output_file)
