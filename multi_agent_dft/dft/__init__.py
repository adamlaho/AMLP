# multi_agent_dft/dft/__init__.py
"""
DFT simulation modules for the Multi-Agent DFT Research System.

This package contains modules for different DFT code interfaces, including
Gaussian, VASP, CP2K, Psi4, ORCA, and CRYSTAL, providing functionality to
generate input files and process output files.
"""

from multi_agent_dft.dft.gaussian import GaussianInterface
from multi_agent_dft.dft.vasp import VASPInterface
from multi_agent_dft.dft.cp2k import CP2KInterface
from multi_agent_dft.dft.psi4 import Psi4Interface
from multi_agent_dft.dft.orca import ORCAInterface
from multi_agent_dft.dft.crystal import CRYSTALInterface

from .cp2k import save_cp2k_input
from .vasp import save_vasp_inputs
from .gaussian import save_gaussian_input
from .psi4 import save_psi4_input
from .orca import save_orca_input
from .crystal import save_crystal_input

__all__ = [
    'GaussianInterface',
    'VASPInterface',
    'CP2KInterface',
    'Psi4Interface',
    'ORCAInterface',
    'CRYSTALInterface',
    'save_cp2k_input',
    'save_vasp_inputs',
    'save_gaussian_input',
    'save_psi4_input',
    'save_orca_input',
    'save_crystal_input'
]

