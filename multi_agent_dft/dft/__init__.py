# multi_agent_dft/dft/__init__.py
"""
DFT simulation modules for the Multi-Agent DFT Research System.

This package contains modules for different DFT code interfaces, including
Gaussian, VASP, and CP2K, providing functionality to generate input files
and process output files.
"""

from multi_agent_dft.dft.gaussian import GaussianInterface
from multi_agent_dft.dft.vasp import VASPInterface
from multi_agent_dft.dft.cp2k import CP2KInterface
from .cp2k import save_cp2k_input
from .vasp import save_vasp_inputs
from .gaussian import save_gaussian_input

__all__ = [
    'GaussianInterface',
    'VASPInterface',
    'CP2KInterface'
]

