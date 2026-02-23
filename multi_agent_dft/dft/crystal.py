"""
CRYSTAL-specific functionality for the Multi-Agent DFT Research System.
CRYSTAL is a periodic quantum chemistry code for solid-state calculations.
"""

import os
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class CRYSTALInterface:
    """Interface for generating CRYSTAL input files and analyzing CRYSTAL outputs."""

    def __init__(self, config=None):
        """
        Initialize the CRYSTAL interface with optional configuration.

        Args:
            config (dict, optional): Configuration parameters for CRYSTAL calculations.
        """
        self.config = config or {}
        self.default_config = {
            'title': 'CRYSTAL calculation',
            'dimensionality': 3,  # 0=molecule, 1=polymer, 2=slab, 3=crystal
            'crystal_type': 1,  # Crystal type (space group)
            'hamiltonian': 'DFT',
            'functional': 'PBE',
            'basis_set': 'pob-TZVP',
            'shrink': [8, 8],  # Shrinking factors for k-point grid
            'tolinteg': [7, 7, 7, 7, 14],  # Tolerance for Coulomb and exchange integrals
            'scf_convergence': 7,  # 10^-7
            'max_scf_cycles': 50,
            'mixing': 30,  # Percentage of Fock/KS matrix mixing
            'levelshift': [2, 1],  # Level shifting for SCF convergence
            'optimize_geometry': False,
            'calculate_band_structure': False
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
        Generate a CRYSTAL input file from structure data.

        Args:
            structure_data (dict): Structure data with atom coordinates and metadata.
            output_file (str or Path, optional): Path to save the input file.
            custom_config (dict, optional): Custom configuration for this specific job.

        Returns:
            str: CRYSTAL input file content.
        """
        # Merge configs
        config = self.get_config()
        if custom_config:
            for key, value in custom_config.items():
                if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    config[key].update(value)
                else:
                    config[key] = value

        # Build the CRYSTAL input file
        crystal_lines = []

        # Title line
        title = config.get('title', 'CRYSTAL calculation')
        crystal_lines.append(title)

        # Dimensionality and crystal type
        dimensionality = config.get('dimensionality', 3)
        crystal_type = config.get('crystal_type', 1)
        crystal_lines.append(f"{dimensionality} {crystal_type}")

        # Cell parameters (for periodic systems)
        if dimensionality > 0 and 'cell_parameters' in structure_data:
            cell_params = structure_data['cell_parameters']
            # Format: a b c alpha beta gamma for 3D
            if dimensionality == 3:
                crystal_lines.append(f"{cell_params['a']:.6f} {cell_params['b']:.6f} {cell_params['c']:.6f}")
                crystal_lines.append(f"{cell_params['alpha']:.6f} {cell_params['beta']:.6f} {cell_params['gamma']:.6f}")
            elif dimensionality == 2:
                crystal_lines.append(f"{cell_params['a']:.6f} {cell_params['b']:.6f}")
                crystal_lines.append(f"{cell_params.get('gamma', 90.0):.6f}")
            elif dimensionality == 1:
                crystal_lines.append(f"{cell_params['a']:.6f}")
        elif dimensionality > 0 and 'cell_lengths' in structure_data:
            # Alternative format
            lengths = structure_data['cell_lengths']
            angles = structure_data.get('cell_angles', [90.0, 90.0, 90.0])
            if dimensionality == 3:
                crystal_lines.append(f"{lengths[0]:.6f} {lengths[1]:.6f} {lengths[2]:.6f}")
                crystal_lines.append(f"{angles[0]:.6f} {angles[1]:.6f} {angles[2]:.6f}")
            elif dimensionality == 2:
                crystal_lines.append(f"{lengths[0]:.6f} {lengths[1]:.6f}")
                crystal_lines.append(f"{angles[2]:.6f}")
            elif dimensionality == 1:
                crystal_lines.append(f"{lengths[0]:.6f}")

        # Number of irreducible atoms
        if 'coordinates' in structure_data:
            n_atoms = len(structure_data['coordinates'])
        elif 'atoms' in structure_data:
            n_atoms = len(structure_data['atoms'])
        else:
            n_atoms = 0

        crystal_lines.append(str(n_atoms))

        # Atomic coordinates
        if 'coordinates' in structure_data:
            for coord in structure_data['coordinates']:
                atomic_number = coord.get('atomic_number', self._element_to_z(coord.get('element', 'X')))
                x = coord.get('x', 0.0)
                y = coord.get('y', 0.0)
                z = coord.get('z', 0.0)
                crystal_lines.append(f"{atomic_number} {x:.8f} {y:.8f} {z:.8f}")
        elif 'atoms' in structure_data:
            for atom in structure_data['atoms']:
                atomic_number = atom.get('atomic_number', self._element_to_z(atom.get('element', 'X')))
                pos = atom.get('position', [0.0, 0.0, 0.0])
                crystal_lines.append(f"{atomic_number} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}")

        # OPTGEOM keyword for geometry optimization
        if config.get('optimize_geometry', False):
            crystal_lines.append("OPTGEOM")
            opt_config = config.get('optimization', {})
            if opt_config:
                crystal_lines.append(f"MAXCYCLE {opt_config.get('max_cycles', 50)}")
                crystal_lines.append(f"TOLDEG {opt_config.get('gradient_threshold', 0.0003)}")
                crystal_lines.append("END")

        # End of geometry section
        crystal_lines.append("END")

        # Hamiltonian specification
        hamiltonian = config.get('hamiltonian', 'DFT')
        crystal_lines.append(hamiltonian)

        if hamiltonian.upper() == 'DFT':
            functional = config.get('functional', 'PBE')
            crystal_lines.append(functional)

            # DFT grid specifications
            if 'dft_grid' in config:
                grid = config['dft_grid']
                crystal_lines.append(f"GRID {grid.get('type', 'XLGRID')}")

            # Dispersion correction
            if 'dispersion' in config:
                disp = config['dispersion']
                crystal_lines.append(f"GRIMME {disp}")

        # Basis set specification
        basis_set = config.get('basis_set', 'pob-TZVP')
        if basis_set in ['sto-3g', 'sto-6g', '6-21g', '6-31g', 'pob-tzvp', 'pob-dzvp']:
            crystal_lines.append(basis_set.upper())
        else:
            # Custom basis set file
            crystal_lines.append(f"BASISSET")
            crystal_lines.append(basis_set)

        # Shrinking factors (k-point mesh)
        shrink = config.get('shrink', [8, 8])
        crystal_lines.append(f"SHRINK {shrink[0]} {shrink[1]}")

        # Integral tolerance
        tolinteg = config.get('tolinteg', [7, 7, 7, 7, 14])
        crystal_lines.append(f"TOLINTEG")
        crystal_lines.append(" ".join(map(str, tolinteg)))

        # SCF options
        crystal_lines.append(f"MAXCYCLE {config.get('max_scf_cycles', 50)}")
        crystal_lines.append(f"TOLDEE {config.get('scf_convergence', 7)}")

        # Fock/KS matrix mixing
        mixing = config.get('mixing', 30)
        crystal_lines.append(f"FMIXING {mixing}")

        # Level shifting
        if 'levelshift' in config:
            levelshift = config['levelshift']
            crystal_lines.append(f"LEVSHIFT {levelshift[0]} {levelshift[1]}")

        # Properties to calculate
        if config.get('calculate_band_structure', False):
            crystal_lines.append("BAND")
            band_config = config.get('band_structure', {})
            # Add band structure path here

        if config.get('calculate_dos', False):
            crystal_lines.append("DOSS")

        # End of input
        crystal_lines.append("END")

        input_content = "\n".join(crystal_lines)

        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(input_content)
            logger.info(f"CRYSTAL input file written to {output_path}")
            return True

        return input_content

    def _element_to_z(self, element):
        """Convert element symbol to atomic number."""
        element_map = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
            'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
            'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
            'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
            'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56
        }
        return element_map.get(element, 0)


def save_crystal_input(structure_data, output_file, config=None):
    """
    Convenience function to save a CRYSTAL input file.

    Args:
        structure_data (dict): Structure data with atom coordinates and metadata.
        output_file (str or Path): Path to save the input file.
        config (dict, optional): Configuration parameters for CRYSTAL.

    Returns:
        bool: Success status.
    """
    interface = CRYSTALInterface(config)
    return interface.generate_input(structure_data, output_file)
