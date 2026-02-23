"""
Analysis module for AMLP

This module contains various analysis tools for molecular dynamics and quantum chemistry:
- Free energy calculations from MD trajectories
- Structural analysis tools
- Thermodynamic property calculations

Author: Adam Lahouari
"""

from .free_energy import MDFreeEnergy

__all__ = ['MDFreeEnergy']
