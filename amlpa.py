"""
Enhanced Molecular Dynamics Analysis Tool
Description:
    A comprehensive tool for MD analysis including:
    - Geometry optimization
    - Cell optimization
    - Phonon calculations  
    - MD simulations (NVE, NVT with Langevin/Berendsen/Nosé-Hoover)
    - RDF analysis with proper normalization (converges to 1)
    - Energy drift analysis (cumulative and instantaneous)
    - RMSD analysis
    - Coordination analysis
    - Caching system
    - Enhanced plotting capabilities
"""

import os
import sys
import re
import time
import random
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist

from ase.io import read, write
from ase import units, Atoms
from ase.geometry.cell import Cell
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.npt import NPT
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import BFGS, LBFGS, FIRE
from ase.filters import StrainFilter
from ase.build import make_supercell
from ase.geometry import find_mic

# Import analysis modules
from multi_agent_dft.analysis import MDFreeEnergy

# Set random seeds for reproducibility
np.random.seed(701)
random.seed(701)

class BerendsenThermostat:
    """
    Simple Berendsen thermostat implementation for NVT simulations
    """
    def __init__(self, atoms, timestep, temperature, taut=100*units.fs):
        """
        atoms: ASE Atoms object
        timestep: MD timestep in ASE units
        temperature: Target temperature in Kelvin
        taut: Time constant for temperature coupling (default 100 fs)
        """
        self.atoms = atoms
        self.timestep = timestep
        self.temperature = temperature
        self.taut = taut
        self.dt = timestep
        
    def scale_velocities(self):
        """Scale velocities according to Berendsen thermostat"""
        current_temp = self.atoms.get_temperature()
        if current_temp > 0:
            lambda_factor = np.sqrt(1.0 + (self.dt / self.taut) * (self.temperature / current_temp - 1.0))
            velocities = self.atoms.get_velocities()
            self.atoms.set_velocities(velocities * lambda_factor)

class CacheManager:
    """Manage caching of simulation results to avoid recomputation"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_writes = 0
        
    def get_cache_key(self, config: dict, identifier: str) -> str:
        """Generate unique cache key from config and identifier"""
        config_str = str(sorted(config.items()))
        key_data = f"{config_str}_{identifier}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_md_cache_key(self, config: dict, temperature: float) -> str:
        """Generate cache key for MD trajectory"""
        # Include all parameters that affect MD trajectory
        relevant_params = {
            'temperature': temperature,
            'md_steps': config.get('md_steps', 10000),
            'timestep': config.get('timestep', 0.5),
            'thermostat': config.get('thermostat', 'langevin'),
            'friction': config.get('friction', 0.01),
            'taut': config.get('taut', 100.0),
            'nh_tdamp': config.get('nh_tdamp', 25.0),
            'nh_tchain': config.get('nh_tchain', 3),
            'nh_tloop': config.get('nh_tloop', 1),
            'save_interval': config.get('save_interval', 100),
            # Include structure info to invalidate if structure changes
            'n_atoms': config.get('_n_atoms', 0),
            'cell': str(config.get('_cell', [])),
            'replicate_initial': config.get('replicate_initial', False),
            'replicate_dims': str(config.get('replicate_dims', None))
        }
        return self.get_cache_key(relevant_params, f"md_trajectory_{temperature}K")
    
    def get_energy_drift_cache_key(self, config: dict, temperature: float) -> str:
        """Generate cache key for energy drift analysis"""
        relevant_params = {
            'timestep': config.get('timestep', 0.5),
            'energy_drift_start_time_ps': config.get('energy_drift_start_time_ps', 0.0),
            'temperature': temperature
        }
        return self.get_cache_key(relevant_params, f"energy_drift_{temperature}K")
    
    def save_cache(self, key: str, data: Any, description: str = "") -> None:
        """Save data to cache"""
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            self.cache_writes += 1
            print(f"✓ Cached {description}: {key[:8]}...")
        except Exception as e:
            print(f"✗ Warning: Could not save cache {key[:8]}...: {e}")
    
    def load_cache(self, key: str) -> Any:
        """Load data from cache"""
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                self.cache_hits += 1
                return data
            except Exception as e:
                print(f"✗ Warning: Could not load cache {key[:8]}...: {e}")
                self.cache_misses += 1
        else:
            self.cache_misses += 1
        return None
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_writes': self.cache_writes,
            'total_cached_files': len(cache_files),
            'total_cache_size_mb': total_size / (1024 * 1024)
        }
    
    def print_cache_summary(self, logger=None):
        """Print cache usage summary"""
        stats = self.get_cache_stats()
        
        lines = [
            "Cache Statistics:",
            f"  Cache hits: {stats['cache_hits']}",
            f"  Cache misses: {stats['cache_misses']}",
            f"  Cache writes: {stats['cache_writes']}",
            f"  Total cached files: {stats['total_cached_files']}",
            f"  Total cache size: {stats['total_cache_size_mb']:.2f} MB"
        ]
        
        if logger:
            for line in lines:
                logger.log(line)
        else:
            for line in lines:
                print(line)

class Logger:
    """Simple logging utility"""
    
    def __init__(self, log_file: str = None):
        self.log_file = log_file
        
    def log(self, message: str, level: str = 'INFO'):
        """Log message to console and optionally to file"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {level}: {message}"
        print(log_message)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_message + '\n')
    
    def log_section(self, title: str):
        """Log a section header"""
        separator = "=" * 60
        self.log(separator)
        self.log(f" {title} ")
        self.log(separator)

class MDAnalyzer:
    """Main class for molecular dynamics analysis"""
    
    def __init__(self, config_file: str):
        self.start_time = time.time()
        self.config = self._load_config(config_file)
        self.setup_directories()
        self.cache = CacheManager(self.output_dir / 'cache')
        self.logger = Logger(self.output_dir / 'analysis.log')
        self.results = {}
        self.trajectories = {}
        self.atoms = None
        self.initial_atoms = None
        self.simulation_cell_replicated = False
        
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_directories(self):
        """Create output directories"""
        self.output_dir = Path(self.config.get('output_dir', 'results'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        for subdir in ['plots', 'trajectories', 'cache']:
            (self.output_dir / subdir).mkdir(exist_ok=True)
    
    def load_structure(self, xyz_file: str):
        """Load initial structure from XYZ file with robust error handling"""
        self.logger.log_section("STRUCTURE LOADING")

        energy_from_file = None
        cell_params = None

        # Parse cell parameters and energy from header if available
        if self.config.get('readcell_info', True):
            try:
                energy_from_file, cell_params = self._parse_xyz_header(xyz_file)

                if energy_from_file is not None:
                    self.logger.log(f"Energy from file: {energy_from_file:.6f} eV")
                else:
                    self.logger.log("No energy information found in XYZ header", 'INFO')

                if cell_params is not None:
                    self.logger.log(f"Cell from file: {cell_params}")
                else:
                    self.logger.log("No cell information found in XYZ header", 'INFO')

            except Exception as e:
                self.logger.log(f"Error reading XYZ header: {e}", 'WARNING')
                cell_params = None

        # Fall back to config if cell not found in file
        if cell_params is None:
            cell_params = self.config.get('cell_params')
            if cell_params is not None:
                self.logger.log(f"Using cell from config: {cell_params}")
            else:
                self.logger.log("No cell parameters available (file or config)", 'WARNING')

        # Load atoms
        self.atoms = read(xyz_file, index=0)

        # Set cell if available
        if cell_params is not None:
            try:
                self.atoms.set_cell(Cell.fromcellpar(cell_params))
                self.logger.log("Cell parameters set successfully")
            except Exception as e:
                self.logger.log(f"Error setting cell run_cell_optimization: {e}", 'ERROR')
                if self.config.get('pbc', True):
                    self.logger.log("WARNING: PBC enabled but no valid cell set!", 'WARNING')
        else:
            if self.config.get('pbc', True):
                self.logger.log("WARNING: PBC enabled but no cell parameters available!", 'WARNING')

        # FIX: Convert PBC to native Python bool to avoid numpy.bool_ issues
        pbc_config = self.config.get('pbc', True)
        if isinstance(pbc_config, (list, tuple)):
            # Convert each element to native Python bool
            pbc_list = [bool(x) for x in pbc_config]
        else:
            # Single bool value - apply to all dimensions
            pbc_list = [bool(pbc_config)] * 3
        self.atoms.set_pbc(pbc_list)

        # Setup calculator
        self._setup_calculator()

        self.logger.log(f"Loaded structure: {len(self.atoms)} atoms")
        self.logger.log(f"Cell: {self.atoms.get_cell()}")
        self.logger.log(f"PBC: {self.atoms.get_pbc()}")

        # Store initial structure (before replication)
        self.initial_atoms = self.atoms.copy()

        # Store structure info in config for cache key generation
        self.config['_n_atoms'] = len(self.atoms)
        self.config['_cell'] = self.atoms.get_cell().tolist()

        self.logger.log(f"Initial structure stored: {len(self.atoms)} atoms")   


    def replicate_simulation_cell(self):
        """
        Replicate the cell for simulation purposes BEFORE any calculations.
        This is done once and affects all subsequent simulations and optimizations.
        This replication is NOT used for RDF calculations.
        """
        replicate_initial = self.config.get('replicate_initial', False)
        replicate_dims = self.config.get('replicate_dims', None)
        
        if not replicate_initial or replicate_dims is None:
            self.logger.log("No cell replication requested for simulation")
            self.simulation_cell_replicated = False
            return
        
        if self.simulation_cell_replicated:
            self.logger.log("Cell already replicated for simulation", 'WARNING')
            return
        
        self.logger.log_section("CELL REPLICATION FOR SIMULATION")
        self.logger.log("Replicating cell for larger supercell energy evaluation")
        
        original_atoms = len(self.atoms)
        
        # Use integer mode for cell replication
        if all(isinstance(x, bool) for x in replicate_dims):
            # Convert booleans to integers (True->2, False->1)
            replicate_factors = [2 if x else 1 for x in replicate_dims]
        else:
            replicate_factors = replicate_dims[:3]
        
        transform_matrix = np.diag(replicate_factors)
        self.atoms = make_supercell(self.atoms, transform_matrix)
        
        # Re-attach calculator to replicated structure
        self.atoms.calc = self._calculator
        
        self.simulation_cell_replicated = True
        
        self.logger.log(f"Cell replicated with factors: {replicate_factors}")
        self.logger.log(f"Atoms: {original_atoms} -> {len(self.atoms)}")
        self.logger.log(f"New cell: {self.atoms.get_cell()}")
        self.logger.log("All subsequent simulations will use this replicated cell")
    
    def _parse_xyz_header(self, filename: str) -> Tuple[Optional[float], Optional[List[float]]]:
        """
        Parse energy and cell parameters from XYZ header with flexible format support.

        Supports various formats:
        - Energy: Energy=value, energy=value, E=value, e=value
        - Cell: Cell:values, cell:values, Lattice:values, lattice:values
        - Cell can be in cellpar format (6 values: a, b, c, alpha, beta, gamma)
        - Cell can be in matrix format (9 values: 3x3 matrix in row-major order)

        Returns (energy, cell) where either can be None if not found.
        """
        with open(filename, 'r') as f:
            f.readline()  # Skip atom count
            comment_line = f.readline().strip()

        energy = None
        cell = None

        # Try multiple energy patterns (case-insensitive variations)
        energy_patterns = [
            r'Energy[=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
            r'energy[=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
            r'E[=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
            r'e[=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
            r'ENERGY[=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
        ]

        for pattern in energy_patterns:
            energy_match = re.search(pattern, comment_line)
            if energy_match:
                try:
                    energy = float(energy_match.group(1))
                    break
                except (ValueError, IndexError):
                    continue

        # Try to parse cell/lattice information
        # First, try to find the Lattice or Cell keyword and extract quoted or unquoted values
        lattice_patterns = [
            r'Lattice[=:]\s*"([^"]+)"',  # Quoted format: Lattice="..."
            r'lattice[=:]\s*"([^"]+)"',
            r'LATTICE[=:]\s*"([^"]+)"',
            r'Cell[=:]\s*"([^"]+)"',
            r'cell[=:]\s*"([^"]+)"',
            r'CELL[=:]\s*"([^"]+)"',
        ]

        lattice_string = None
        for pattern in lattice_patterns:
            match = re.search(pattern, comment_line)
            if match:
                lattice_string = match.group(1)
                break

        # If no quoted format found, try unquoted format
        if lattice_string is None:
            unquoted_patterns = [
                r'Lattice[=:]\s*((?:[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*)+)',
                r'lattice[=:]\s*((?:[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*)+)',
                r'LATTICE[=:]\s*((?:[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*)+)',
                r'Cell[=:]\s*((?:[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*)+)',
                r'cell[=:]\s*((?:[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*)+)',
                r'CELL[=:]\s*((?:[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*)+)',
            ]

            for pattern in unquoted_patterns:
                match = re.search(pattern, comment_line)
                if match:
                    # Extract up to the next keyword or end of string
                    potential_string = match.group(1)
                    # Stop at next keyword (Properties, Energy, etc.)
                    next_keyword = re.search(r'(Properties|Energy|energy|E=|e=)', potential_string)
                    if next_keyword:
                        lattice_string = potential_string[:next_keyword.start()].strip()
                    else:
                        lattice_string = potential_string.strip()
                    break

        # Parse the lattice string
        if lattice_string:
            try:
                # Extract all numeric values
                values = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', lattice_string)
                values = [float(v) for v in values]

                if len(values) == 9:
                    # Matrix format: 9 values representing 3x3 cell matrix
                    # Convert to ASE Cell object and then to cellpar
                    cell_matrix = np.array(values).reshape(3, 3)
                    from ase.geometry.cell import Cell as ASECell
                    ase_cell = ASECell(cell_matrix)
                    cell = ase_cell.cellpar().tolist()

                elif len(values) == 6:
                    # Cellpar format: a, b, c, alpha, beta, gamma
                    cell = values

                elif len(values) > 9:
                    # Too many values, try to take first 9 for matrix format
                    cell_matrix = np.array(values[:9]).reshape(3, 3)
                    from ase.geometry.cell import Cell as ASECell
                    ase_cell = ASECell(cell_matrix)
                    cell = ase_cell.cellpar().tolist()

            except (ValueError, IndexError) as e:
                # If parsing fails, cell remains None
                pass

        return energy, cell
    
    def _setup_calculator(self):
        """Setup MACE calculator once and store it with proper type handling"""
        if hasattr(self, '_calculator'):
            self.atoms.calc = self._calculator
            return
            
        try:
            from mace.calculators import MACECalculator
            
            device_option = self.config.get('device', 'gpu').lower()
            if device_option == 'cpu':
                device_str = 'cpu'
            else:
                gpu_devices = self.config.get('gpus', ['cuda:0'])
                device_str = ','.join(gpu_devices)
            
            model_paths = self.config.get('model_paths', ['/path/to/your/model'])
            self._calculator = MACECalculator(
                model_paths=model_paths, 
                device=device_str,
                default_dtype='float64'  # Explicit dtype
            )
            
            # CRITICAL FIX: Convert all numpy types to Python native types
            # This fixes the numpy.bool_ compatibility issue
            self._fix_atoms_types()
            
            self.atoms.calc = self._calculator
            
            self.logger.log(f"MACE calculator setup: {device_str}")
            
        except ImportError:
            self.logger.log("MACE not available, using EMT calculator", 'WARNING')
            from ase.calculators.emt import EMT
            self._calculator = EMT()
            self.atoms.calc = self._calculator
    
    def run_single_point(self) -> bool:
        """Run single point energy calculation"""
        if not self.config.get('single_point', False):
            return False
            
        self.logger.log_section("SINGLE POINT ENERGY")
        
        try:
            energy = self.atoms.get_potential_energy()
            forces = self.atoms.get_forces()
            max_force = np.max(np.linalg.norm(forces, axis=1))
            
            self.logger.log(f"Energy: {energy:.6f} eV")
            self.logger.log(f"Energy per atom: {energy/len(self.atoms):.6f} eV")
            self.logger.log(f"Maximum force: {max_force:.6f} eV/Å")
            
            self.results['single_point'] = {
                'energy': energy,
                'energy_per_atom': energy/len(self.atoms),
                'max_force': max_force
            }
            
            return True
            
        except Exception as e:
            self.logger.log(f"Single point calculation failed: {e}", 'ERROR')
            return False
    
    def run_geometry_optimization(self) -> bool:
        """Run geometry optimization"""
        if not self.config.get('geo_opt', False):
            return False
            
        self.logger.log_section("GEOMETRY OPTIMIZATION")
        
        # Test calculator before optimization
        try:
            self.logger.log("Testing calculator...")
            initial_energy = self.atoms.get_potential_energy()
            self.logger.log(f"Initial energy: {initial_energy:.6f} eV")
            forces = self.atoms.get_forces()
            max_force = np.max(np.linalg.norm(forces, axis=1))
            self.logger.log(f"Maximum force: {max_force:.6f} eV/Å")
        except Exception as e:
            self.logger.log(f"Calculator test failed: {e}", 'ERROR')
            return False
        
        optimizer_name = self.config.get('optimizer', 'BFGS').upper()
        fmax = self.config.get('fmax', 0.001)
        
        # Setup optimizer
        if optimizer_name == 'BFGS':
            opt = BFGS(self.atoms)
        elif optimizer_name == 'LBFGS':
            opt = LBFGS(self.atoms)
        elif optimizer_name == 'FIRE':
            opt = FIRE(self.atoms)
        else:
            self.logger.log(f"Unknown optimizer {optimizer_name}, using BFGS", 'WARNING')
            opt = BFGS(self.atoms)
        
        self.logger.log(f"Starting optimization with {optimizer_name}, fmax={fmax}")
        
        # Add callback for progress
        step_count = [0]
        def simple_callback():
            step_count[0] += 1
            if step_count[0] % 10 == 0:
                current_energy = self.atoms.get_potential_energy()
                forces = self.atoms.get_forces()
                max_force = np.max(np.linalg.norm(forces, axis=1))
                self.logger.log(f"Step {step_count[0]}: E={current_energy:.6f} eV, Fmax={max_force:.6f} eV/Å")
        
        opt.attach(simple_callback, interval=1)
        
        try:
            opt.run(fmax=fmax)
            
            final_energy = self.atoms.get_potential_energy()
            self.logger.log(f"Optimization completed in {step_count[0]} steps")
            self.logger.log(f"Final energy: {final_energy:.6f} eV")
            
            # Save optimized structure
            opt_file = self.output_dir / f"{self.config['base_name']}_optimized.xyz"
            write(str(opt_file), self.atoms)
            self.logger.log(f"Optimized structure saved: {opt_file}")
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            self.logger.log(f"Optimization failed: {e}", 'ERROR')
            import traceback
            self.logger.log(traceback.format_exc(), 'ERROR')
            return False

    def _fix_atoms_types(self):
        """
        Fix numpy type incompatibilities by converting to native Python types.
        This resolves 'numpy.bool' object cannot be interpreted as an integer errors.
        """
        # Fix PBC - convert numpy.bool_ to Python bool
        pbc = self.atoms.get_pbc()
        if hasattr(pbc, 'tolist'):
            pbc_list = pbc.tolist()
            self.atoms.set_pbc(pbc_list)
        
        # Ensure positions are standard numpy array (not structured)
        positions = self.atoms.get_positions()
        self.atoms.set_positions(positions.astype(np.float64))
        
        # Ensure cell is proper type
        cell = self.atoms.get_cell()
        if hasattr(cell, 'array'):
            self.atoms.set_cell(cell.array.astype(np.float64))


    def test_calculator_capabilities(self) -> Dict[str, bool]:
        """
        Test what the calculator can actually compute.
        Returns dict with capabilities.
        """
        self.logger.log("Testing calculator capabilities...")
        
        capabilities = {
            'energy': False,
            'forces': False,
            'stress': False
        }
        
        # Fix types before testing
        self._fix_atoms_types()
        
        try:
            # Test energy
            energy = self.atoms.get_potential_energy()
            capabilities['energy'] = True
            self.logger.log(f"✓ Energy calculation works: {energy:.6f} eV")
        except Exception as e:
            self.logger.log(f"✗ Energy calculation failed: {e}", 'ERROR')
            import traceback
            self.logger.log(traceback.format_exc(), 'ERROR')
            return capabilities
        
        try:
            # Test forces
            forces = self.atoms.get_forces()
            max_force = np.max(np.linalg.norm(forces, axis=1))
            capabilities['forces'] = True
            self.logger.log(f"✓ Force calculation works: max force = {max_force:.6f} eV/Å")
        except Exception as e:
            self.logger.log(f"✗ Force calculation failed: {e}", 'ERROR')
        
        try:
            # Test stress - CRITICAL for cell optimization
            stress = self.atoms.get_stress()
            capabilities['stress'] = True
            self.logger.log(f"✓ Stress calculation works: {stress}")
        except Exception as e:
            self.logger.log(f"✗ Stress calculation failed: {e}", 'WARNING')
            self.logger.log("  Cell optimization requires stress tensor - will be skipped", 'WARNING')
        
        return capabilities




    def run_cell_optimization(self) -> bool:
        """Run cell optimization (variable cell shape relaxation)"""
        if not self.config.get('cell_opt', False):
            return False
            
        self.logger.log_section("CELL OPTIMIZATION")
        
        # Clear GPU cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        try:
            initial_energy = self.atoms.get_potential_energy()
            initial_cell = self.atoms.get_cell()
            self.logger.log(f"Initial energy: {initial_energy:.6f} eV")
            self.logger.log(f"Initial cell:\n{initial_cell}")
        except Exception as e:
            self.logger.log(f"Calculator test failed: {e}", 'ERROR')
            return False
        
        optimizer_name = self.config.get('cell_optimizer', 'BFGS').upper()
        fmax = self.config.get('cell_fmax', 0.01)
        scalar_pressure = self.config.get('scalar_pressure', 0.0)  # GPa
        
        # Convert pressure to eV/Å³
        pressure_ev = scalar_pressure * units.GPa
        
        # Use StrainFilter for variable cell optimization
        try:
            strain_filter = StrainFilter(self.atoms, scalar_pressure=pressure_ev)
            self.logger.log("Using StrainFilter with scalar_pressure parameter")
        except TypeError:
            strain_filter = StrainFilter(self.atoms)
            if scalar_pressure != 0.0:
                self.logger.log(f"WARNING: scalar_pressure={scalar_pressure} GPa requested but not supported by this ASE version", 'WARNING')
            self.logger.log("Using StrainFilter without scalar_pressure parameter")
        
        # Setup optimizer
        if optimizer_name == 'BFGS':
            opt = BFGS(strain_filter)
        elif optimizer_name == 'LBFGS':
            opt = LBFGS(strain_filter)
        elif optimizer_name == 'FIRE':
            opt = FIRE(strain_filter)
        else:
            self.logger.log(f"Unknown optimizer {optimizer_name}, using BFGS", 'WARNING')
            opt = BFGS(strain_filter)
        
        self.logger.log(f"Starting cell optimization with {optimizer_name}, fmax={fmax}")
        if scalar_pressure != 0.0:
            self.logger.log(f"Target pressure: {scalar_pressure} GPa")
        
        # Add callback with aggressive memory management
        step_count = [0]
        def cell_callback():
            step_count[0] += 1
            
            # Clear GPU cache every step to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if step_count[0] % 10 == 0:
                current_energy = self.atoms.get_potential_energy()
                current_cell = self.atoms.get_cell()
                cell_volume = self.atoms.get_volume()
                self.logger.log(f"Step {step_count[0]}: E={current_energy:.6f} eV, V={cell_volume:.3f} Å³")
        
        opt.attach(cell_callback, interval=1)
        
        try:
            opt.run(fmax=fmax)
            
            final_energy = self.atoms.get_potential_energy()
            final_cell = self.atoms.get_cell()
            final_volume = self.atoms.get_volume()
            
            self.logger.log(f"Cell optimization completed in {step_count[0]} steps")
            self.logger.log(f"Final energy: {final_energy:.6f} eV")
            self.logger.log(f"Final volume: {final_volume:.3f} Å³")
            self.logger.log(f"Final cell:\n{final_cell}")
            
            # Save optimized structure
            opt_file = self.output_dir / f"{self.config['base_name']}_cell_optimized.xyz"
            write(str(opt_file), self.atoms)
            self.logger.log(f"Cell-optimized structure saved: {opt_file}")
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            self.logger.log(f"Cell optimization failed: {e}", 'ERROR')
            import traceback
            self.logger.log(traceback.format_exc(), 'ERROR')
            
            # Clear CUDA cache even on failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return False



    def analyze_rmsd(self, temperature: float = None) -> Dict[str, Any]:
        """Calculate RMSD between initial and current structure"""
        if not self.config.get('run_rmsd', False):
            return {}
            
        self.logger.log_section("RMSD ANALYSIS")
        
        if self.initial_atoms is None:
            self.logger.log("No initial structure available for RMSD", 'WARNING')
            return {}
        
        current_atoms = self.atoms
        initial_pos = self.initial_atoms.get_positions()
        current_pos = current_atoms.get_positions()
        initial_cell = self.initial_atoms.get_cell()
        current_cell = current_atoms.get_cell()
        
        if len(initial_pos) != len(current_pos):
            self.logger.log(f"Structure size mismatch: {len(initial_pos)} vs {len(current_pos)}", 'ERROR')
            return {}
        
        # Handle PBC for basic RMSD calculations
        if self.atoms.get_pbc().any():
            displacement = current_pos - initial_pos
            displacement = find_mic(displacement, current_cell)[0]
            current_pos_corrected = initial_pos + displacement
        else:
            current_pos_corrected = current_pos
        
        # Calculate standard RMSD metrics (unit cell, Cartesian)
        rmsd_simple = self._calculate_rmsd_simple(initial_pos, current_pos_corrected)
        rmsd_centered = self._calculate_rmsd_centered(initial_pos, current_pos_corrected)
        rmsd_aligned = self._calculate_rmsd_aligned(initial_pos, current_pos_corrected)
        
        # Calculate fractional coordinate RMSD (accounts for cell changes)
        rmsd_fractional = self._calculate_rmsd_fractional(
            initial_pos, current_pos, initial_cell, current_cell
        )
        
        # Calculate supercell RMSD (like RMSD15, captures lattice parameter errors)
        supercell_dims = self.config.get('rmsd_supercell_dims', [3, 3, 3])
        rmsd_supercell = self._calculate_rmsd_supercell(
            self.initial_atoms, current_atoms, supercell_dims
        )
        
        # Calculate cell parameter deviations
        cell_deviations = self._calculate_cell_deviation(initial_cell, current_cell)
        
        # Per-atom displacement analysis
        displacement = current_pos - initial_pos
        if self.atoms.get_pbc().any():
            displacement = find_mic(displacement, current_cell)[0]
        
        per_atom_displacements = np.sqrt(np.sum(displacement**2, axis=1))
        
        analysis = {
            # Unit cell Cartesian RMSD (existing metrics)
            'rmsd_simple': rmsd_simple,
            'rmsd_centered': rmsd_centered, 
            'rmsd_aligned': rmsd_aligned,
            
            # New metrics that account for cell changes
            'rmsd_fractional': rmsd_fractional,
            'rmsd_supercell': rmsd_supercell,
            'supercell_dims': supercell_dims,
            
            # Cell parameter deviations
            'cell_deviations': cell_deviations,
            
            # Per-atom statistics
            'n_atoms': len(initial_pos),
            'max_displacement': np.max(per_atom_displacements),
            'mean_displacement': np.mean(per_atom_displacements),
            'std_displacement': np.std(per_atom_displacements),
            'per_atom_displacements': per_atom_displacements
        }
        
        # Logging
        self.logger.log(f"RMSD Analysis:")
        self.logger.log(f"  Unit Cell (Cartesian):")
        self.logger.log(f"    Simple RMSD: {rmsd_simple:.4f} Å")
        self.logger.log(f"    Centered RMSD: {rmsd_centered:.4f} Å")
        self.logger.log(f"    Aligned RMSD (Kabsch): {rmsd_aligned:.4f} Å")
        self.logger.log(f"  Fractional Coordinates:")
        self.logger.log(f"    RMSD (fractional): {rmsd_fractional:.6f}")
        self.logger.log(f"  Supercell {supercell_dims}:")
        self.logger.log(f"    RMSD (supercell): {rmsd_supercell:.4f} Å")
        self.logger.log(f"  Cell Parameters:")
        self.logger.log(f"    Δa: {cell_deviations['delta_a']:.4f} Å")
        self.logger.log(f"    Δb: {cell_deviations['delta_b']:.4f} Å")
        self.logger.log(f"    Δc: {cell_deviations['delta_c']:.4f} Å")
        self.logger.log(f"    ΔV: {cell_deviations['volume_change_percent']:.2f}%")
        self.logger.log(f"  Per-atom Displacement:")
        self.logger.log(f"    Max: {analysis['max_displacement']:.4f} Å")
        self.logger.log(f"    Mean: {analysis['mean_displacement']:.4f} Å")
        
        return analysis


    def _calculate_rmsd_simple(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Simple RMSD calculation"""
        return np.sqrt(np.mean(np.sum((pos1 - pos2)**2, axis=1)))
    
    def _calculate_rmsd_centered(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """RMSD after centering structures"""
        pos1_centered = pos1 - np.mean(pos1, axis=0)
        pos2_centered = pos2 - np.mean(pos2, axis=0)
        return self._calculate_rmsd_simple(pos1_centered, pos2_centered)

    def _calculate_rmsd_aligned(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """RMSD after optimal alignment (Kabsch algorithm)"""
        try:
            pos1_centered = pos1 - np.mean(pos1, axis=0)
            pos2_centered = pos2 - np.mean(pos2, axis=0)
            
            H = pos1_centered.T @ pos2_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            pos1_aligned = pos1_centered @ R
            
            return self._calculate_rmsd_simple(pos1_aligned, pos2_centered)
            
        except Exception as e:
            self.logger.log(f"Alignment failed, using centered RMSD: {e}", 'WARNING')
            return self._calculate_rmsd_centered(pos1, pos2)

    def _calculate_rmsd_fractional(self, pos1: np.ndarray, pos2: np.ndarray, 
                                    cell1: Cell, cell2: Cell) -> float:
        """RMSD in fractional coordinates (accounts for cell changes)"""
        # Convert Cartesian positions to fractional coordinates
        frac1 = cell1.scaled_positions(pos1)
        frac2 = cell2.scaled_positions(pos2)
        
        # Handle PBC wrapping in fractional space
        frac_diff = frac2 - frac1
        frac_diff = frac_diff - np.round(frac_diff)
        
        return np.sqrt(np.mean(np.sum(frac_diff**2, axis=1)))

    def _calculate_rmsd_supercell(self, atoms1: Atoms, atoms2: Atoms, 
                                replicate: List[int] = [3, 3, 3]) -> float:
        """RMSD on replicated supercell (like RMSD15, captures lattice errors)"""
        from ase.build import make_supercell
        
        # Replicate both structures
        transform = np.diag(replicate)
        super1 = make_supercell(atoms1.copy(), transform)
        super2 = make_supercell(atoms2.copy(), transform)
        
        pos1 = super1.get_positions()
        pos2 = super2.get_positions()
        
        # Use Kabsch alignment on the supercell
        return self._calculate_rmsd_aligned(pos1, pos2)

    def _calculate_cell_deviation(self, cell1: Cell, cell2: Cell) -> Dict[str, float]:
        """Calculate cell parameter deviations"""
        params1 = cell1.cellpar()
        params2 = cell2.cellpar()
        
        return {
            'delta_a': abs(params1[0] - params2[0]),
            'delta_b': abs(params1[1] - params2[1]),
            'delta_c': abs(params1[2] - params2[2]),
            'delta_alpha': abs(params1[3] - params2[3]),
            'delta_beta': abs(params1[4] - params2[4]),
            'delta_gamma': abs(params1[5] - params2[5]),
            'delta_volume': abs(cell1.volume - cell2.volume),
            'volume_change_percent': 100 * abs(cell1.volume - cell2.volume) / cell1.volume if cell1.volume > 0 else 0
        }



    def calculate_energy_drift_statistics(self, times: np.ndarray, energies: np.ndarray, 
                                        temperatures: np.ndarray, start_time_ps: float = 0.0, 
                                        timestep_fs: float = 0.5) -> Dict[str, Any]:
        """
        Calculate energy drift statistics:
        - Cumulative drift: ΔE_cum(t) = (1/N) * Σ |E(t_i) - E(0)| / |E(0)|
        - Instantaneous drift: ΔE_inst(t) = |E(t) - E(0)| / |E(0)|
        """
        times = np.array(times)
        energies = np.array(energies)
        temperatures = np.array(temperatures)
        
        # Convert start_time from ps to fs
        start_time_fs = start_time_ps * 1000
        
        # Find starting index
        start_idx = np.argmax(times >= start_time_fs)
        if start_idx == 0 and times[0] < start_time_fs:
            start_idx = int(0.1 * len(times))
            actual_start_time_ps = times[start_idx] / 1000
            self.logger.log(f"Start time {start_time_ps}ps not found, using {actual_start_time_ps:.2f}ps")
        else:
            actual_start_time_ps = times[start_idx] / 1000
        
        # Trim data
        times_trimmed = times[start_idx:]
        energies_trimmed = energies[start_idx:]
        temperatures_trimmed = temperatures[start_idx:]
        
        # Reference energy E0
        E0 = energies_trimmed[0]
        
        # Instantaneous drift (simultaneous)
        instantaneous_drift = np.abs(energies_trimmed - E0) / np.abs(E0)
        
        # Cumulative drift
        cumulative_drift = np.zeros_like(energies_trimmed)
        for i in range(len(energies_trimmed)):
            if i == 0:
                cumulative_drift[i] = 0.0
            else:
                nsteps = i + 1
                cumulative_drift[i] = np.sum(instantaneous_drift[:i+1]) / nsteps
        
        # Statistics
        final_drift_cumulative = cumulative_drift[-1]
        final_drift_instantaneous = instantaneous_drift[-1]
        max_drift = np.max(cumulative_drift)
        mean_drift = np.mean(cumulative_drift)
        std_drift = np.std(cumulative_drift)
        
        # Linear trend on cumulative drift
        step_numbers = np.arange(len(times_trimmed))
        drift_rate_per_step = 0
        r_squared = 0
        if len(step_numbers) > 10:
            coeffs = np.polyfit(step_numbers, cumulative_drift, 1)
            drift_rate_per_step = coeffs[0]
            
            fitted_drift = coeffs[0] * step_numbers + coeffs[1]
            ss_res = np.sum((cumulative_drift - fitted_drift) ** 2)
            ss_tot = np.sum((cumulative_drift - np.mean(cumulative_drift)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Quality assessment
        if final_drift_cumulative < 0.0001:
            drift_quality = "Excellent"
        elif final_drift_cumulative < 0.001:
            drift_quality = "Good"
        elif final_drift_cumulative < 0.01:
            drift_quality = "Acceptable"
        else:
            drift_quality = "Poor"
        
        return {
            'reference_energy_E0': E0,
            'final_energy': energies_trimmed[-1],
            'final_drift_cumulative': final_drift_cumulative,
            'final_drift_instantaneous': final_drift_instantaneous,
            'max_drift': max_drift,
            'mean_drift': mean_drift,
            'std_drift': std_drift,
            'drift_rate_per_step': drift_rate_per_step,
            'linear_fit_r_squared': r_squared,
            'avg_temperature': np.mean(temperatures_trimmed),
            'temperature_std': np.std(temperatures_trimmed),
            'temperature_drift': temperatures_trimmed[-1] - temperatures_trimmed[0],
            'drift_quality': drift_quality,
            'simulation_time_ps': (times_trimmed[-1] - times_trimmed[0]) / 1000,
            'n_points': len(times_trimmed),
            'start_time_ps': actual_start_time_ps,
            'times': times_trimmed,
            'times_ps': times_trimmed / 1000.0,
            'energies': energies_trimmed,
            'temperatures_array': temperatures_trimmed,
            'instantaneous_drift_values': instantaneous_drift,
            'cumulative_drift_values': cumulative_drift,
            'step_numbers': step_numbers
        }
    
    def analyze_energy_drift(self, temperature: float) -> Dict[str, Any]:
        """Analyze energy drift for NVE simulations only"""
        if not self.config.get('run_energy_drift', False):
            return {}
        
        thermostat_type = self.config.get('thermostat', 'nve').lower()
        if thermostat_type != 'nve':
            self.logger.log(f"Energy drift analysis skipped for {thermostat_type} (NVE only)")
            return {}
        
        cache_key = self.cache.get_energy_drift_cache_key(self.config, temperature)
        
        cached_drift = self.cache.load_cache(cache_key)
        if cached_drift is not None:
            self.logger.log(f"Loading energy drift analysis from cache for {temperature}K")
            # Still write the log file from cached data
            self._write_energy_drift_log(cached_drift, temperature)
            return cached_drift
        
        self.logger.log_section(f"ENERGY DRIFT ANALYSIS - {temperature}K (NVE)")
        
        traj_key = f'md_{temperature}K'
        if traj_key not in self.trajectories:
            self.logger.log(f"No trajectory data found for {temperature}K", 'WARNING')
            return {}
        
        traj_data = self.trajectories[traj_key]
        energies = np.array(traj_data['energies'])
        times = np.array(traj_data['times'])
        temperatures = np.array(traj_data['temperatures'])
        
        if len(energies) < 2:
            self.logger.log("Insufficient data for energy drift analysis", 'WARNING')
            return {}
        
        start_time_ps = self.config.get('energy_drift_start_time_ps', 0.0)
        timestep_fs = self.config.get('timestep', 0.5)
        
        statistics = self.calculate_energy_drift_statistics(
            times, energies, temperatures, start_time_ps, timestep_fs
        )
        
        self.cache.save_cache(cache_key, statistics, f"Energy drift analysis {temperature}K")
        
        # Write detailed energy drift log file
        self._write_energy_drift_log(statistics, temperature)
        
        self.logger.log(f"Energy drift analysis completed:")
        self.logger.log(f"  Final cumulative drift: {statistics['final_drift_cumulative']:.8f}")
        self.logger.log(f"  Final instantaneous drift: {statistics['final_drift_instantaneous']:.8f}")
        self.logger.log(f"  Quality: {statistics['drift_quality']}")
        
        return statistics
    
    def _write_energy_drift_log(self, statistics: Dict[str, Any], temperature: float) -> None:
        """Write detailed energy drift log file with both cumulative and instantaneous values"""
        log_file = self.output_dir / f'energy_drift_{temperature}K.log'
        
        self.logger.log(f"Writing energy drift log: {log_file}")
        
        with open(log_file, 'w') as f:
            # Write header
            f.write(f"# Energy Drift Analysis Log - NVE Ensemble\n")
            f.write(f"# Temperature: {temperature} K (initial)\n")
            f.write(f"# Analysis start time: {statistics['start_time_ps']:.3f} ps\n")
            f.write(f"# Simulation time: {statistics['simulation_time_ps']:.3f} ps\n")
            f.write(f"# Number of data points: {statistics['n_points']}\n")
            f.write(f"# Reference energy E(0): {statistics['reference_energy_E0']:.6f} eV\n")
            f.write(f"#\n")
            f.write(f"# Formulas:\n")
            f.write(f"#   Instantaneous drift: ΔE_inst(t) = |E(t) - E(0)| / |E(0)|\n")
            f.write(f"#   Cumulative drift:    ΔE_cum(t)  = (1/N_step) × Σ|E_k - E(0)| / |E(0)|\n")
            f.write(f"#\n")
            f.write(f"# Final Statistics:\n")
            f.write(f"#   Final cumulative drift: {statistics['final_drift_cumulative']:.8e}\n")
            f.write(f"#   Final instantaneous drift: {statistics['final_drift_instantaneous']:.8e}\n")
            f.write(f"#   Max cumulative drift: {statistics['max_drift']:.8e}\n")
            f.write(f"#   Mean cumulative drift: {statistics['mean_drift']:.8e}\n")
            f.write(f"#   Std cumulative drift: {statistics['std_drift']:.8e}\n")
            f.write(f"#   Drift rate per step: {statistics['drift_rate_per_step']:.8e}\n")
            f.write(f"#   Linear fit R²: {statistics['linear_fit_r_squared']:.6f}\n")
            f.write(f"#   Quality assessment: {statistics['drift_quality']}\n")
            f.write(f"#   Stability assessment: {statistics['stability_assessment']}\n")
            f.write(f"#\n")
            f.write(f"# Temperature Statistics:\n")
            f.write(f"#   Average temperature: {statistics['avg_temperature']:.2f} K\n")
            f.write(f"#   Temperature std: {statistics['temperature_std']:.2f} K\n")
            f.write(f"#   Temperature drift: {statistics['temperature_drift']:.2f} K\n")
            f.write(f"#\n")
            f.write(f"# {'Step':>10} {'Time(ps)':>12} {'Energy(eV)':>14} {'Temp(K)':>10} {'ΔE_inst':>14} {'ΔE_cum':>14}\n")
            
            # Write data
            times_ps = statistics['times_ps']
            energies = statistics['energies']
            temps = statistics['temperatures_array']
            inst_drift = statistics['instantaneous_drift_values']
            cum_drift = statistics['cumulative_drift_values']
            step_numbers = statistics['step_numbers']
            
            # Adjust time to start from 0 for clarity
            adjusted_times = times_ps - statistics['start_time_ps']
            
            for i in range(len(times_ps)):
                f.write(f"  {step_numbers[i]:>10d} "
                       f"{adjusted_times[i]:>12.4f} "
                       f"{energies[i]:>14.6f} "
                       f"{temps[i]:>10.2f} "
                       f"{inst_drift[i]:>14.8e} "
                       f"{cum_drift[i]:>14.8e}\n")
            
            # Write summary at the end
            f.write(f"\n# ============================================================\n")
            f.write(f"# SUMMARY\n")
            f.write(f"# ============================================================\n")
            f.write(f"# Quality: {statistics['drift_quality']}\n")
            f.write(f"# Stability: {statistics['stability_assessment']}\n")
            f.write(f"#\n")
            f.write(f"# Interpretation:\n")
            if statistics['final_drift_cumulative'] < 0.0001:
                f.write(f"# EXCELLENT energy conservation - perfect NVE simulation\n")
            elif statistics['final_drift_cumulative'] < 0.001:
                f.write(f"# GOOD energy conservation - acceptable for production\n")
            elif statistics['final_drift_cumulative'] < 0.01:
                f.write(f"# ACCEPTABLE energy conservation - consider reducing timestep\n")
            else:
                f.write(f"# POOR energy conservation - reduce timestep significantly\n")
            f.write(f"#\n")
            f.write(f"# Instantaneous drift shows point-by-point fluctuations\n")
            f.write(f"# Cumulative drift shows time-averaged absolute deviation\n")
            f.write(f"# For good NVE, cumulative drift should plateau at low value\n")
        
        self.logger.log(f"Energy drift log written with {len(times_ps)} data points")
    
    def compute_neighbors_for_original_atoms(self, original_atoms: Atoms, cutoff: float = 12.0, 
                                            atom_types: Union[str, List[str]] = "all") -> List[int]:
        """
        Count neighbors for atoms in the original cell (no replication for RDF).
        We use the original cell only, respecting PBC.
        """
        center_positions = []
        if atom_types == "all":
            center_positions = [atom.position for atom in original_atoms]
        else:
            if isinstance(atom_types, str):
                atom_types = [atom_types]
            for atom in original_atoms:
                if atom.symbol in atom_types:
                    center_positions.append(atom.position)

        neighbor_counts = []
        all_positions = original_atoms.get_positions()
        cell = original_atoms.cell
        pbc = original_atoms.pbc

        for cpos in center_positions:
            delta = all_positions - cpos
            mic_vecs, dist = find_mic(delta, cell=cell, pbc=pbc)
            same_mask = dist < 1e-6
            is_neighbor = (dist < cutoff) & (~same_mask)
            neighbor_counts.append(np.sum(is_neighbor))

        return neighbor_counts

    def analyze_coordination_numbers(self, temperature: float = None, label: str = "") -> Dict[str, Any]:
        """Analyze coordination numbers"""
        if not self.config.get('run_coordination', False):
            return {}
            
        self.logger.log_section(f"COORDINATION ANALYSIS - {label}")
        
        cutoff = self.config.get('coordination_cutoff', 3.0)
        atom_types = self.config.get('coordination_atom_types', 'all')

        neighbor_counts = self.compute_neighbors_for_original_atoms(
            original_atoms=self.atoms,
            cutoff=cutoff,
            atom_types=atom_types
        )

        if len(neighbor_counts) == 0:
            self.logger.log("No center atoms found for specified types", 'WARNING')
            return {}

        analysis = {
            'neighbor_counts': neighbor_counts,
            'mean_coordination': np.mean(neighbor_counts),
            'std_coordination': np.std(neighbor_counts),
            'min_coordination': np.min(neighbor_counts),
            'max_coordination': np.max(neighbor_counts),
            'cutoff': cutoff,
            'atom_types': atom_types,
            'n_atoms': len(neighbor_counts)
        }

        self.logger.log(f"Coordination analysis for {atom_types} atoms:")
        self.logger.log(f"  Cutoff: {cutoff} Å")
        self.logger.log(f"  Mean coordination: {analysis['mean_coordination']:.2f}")
        self.logger.log(f"  Std deviation: {analysis['std_coordination']:.2f}")
        self.logger.log(f"  Range: {analysis['min_coordination']}-{analysis['max_coordination']}")

        return analysis

    def plot_coordination_histogram(self, temperature: float = None, label: str = "") -> None:
        """Plot histogram of coordination numbers"""
        coordination_key = f'coordination_{label}' if label else 'coordination'
        if coordination_key not in self.results:
            return

        analysis = self.results[coordination_key]
        neighbor_counts = analysis['neighbor_counts']
        atom_types = analysis['atom_types']
        cutoff = analysis['cutoff']

        fig, ax = plt.subplots(figsize=(8, 8))
        
        bins = range(min(neighbor_counts), max(neighbor_counts) + 2)
        ax.hist(neighbor_counts, bins=bins, align='left', edgecolor='black', alpha=0.7)
        
        ax.set_xlabel('Number of neighbors', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.set_title(f'Coordination Numbers - {label}\nAtom types: {atom_types}, Cutoff: {cutoff} Å', fontsize=14)
        ax.grid(True, alpha=0.3)

        mean_coord = analysis['mean_coordination']
        std_coord = analysis['std_coordination']
        ax.text(0.7, 0.9, f'Mean: {mean_coord:.2f}\nStd: {std_coord:.2f}', 
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        
        safe_label = label.replace(' ', '_').replace('-', '_')
        plot_file = self.output_dir / 'plots' / f'coordination_{safe_label}_histogram.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.log(f"Coordination histogram saved: {plot_file}")
    
    def validate_and_adjust_rdf_cutoff(self, atoms: Atoms, rmax: float) -> float:
        """
        Validate RDF cutoff and automatically adjust if needed.
        Returns the safe cutoff value (either original or adjusted).
        """
        cell_lengths = atoms.cell.lengths()
        min_cell_length = np.min(cell_lengths)
        max_allowed_rmax = min_cell_length / 2.0
        
        if rmax > max_allowed_rmax:
            self.logger.log(
                f"WARNING: Requested RDF cutoff {rmax:.3f} Å exceeds maximum allowed {max_allowed_rmax:.3f} Å",
                'WARNING'
            )
            self.logger.log(f"Cell lengths: {cell_lengths}", 'WARNING')
            self.logger.log(
                f"Automatically adjusting RDF cutoff to {max_allowed_rmax:.3f} Å (0.5 × min cell length)",
                'WARNING'
            )
            return max_allowed_rmax
        
        self.logger.log(f"RDF cutoff validation passed: {rmax:.3f} Å <= {max_allowed_rmax:.3f} Å")
        return rmax
    
    def compute_rdf_normalized(self, atoms: Atoms, rmin: float = 0.0, rmax: float = 10.0, 
                              nbins: int = 100, atom_pairs: List[Tuple[str, str]] = None) -> Dict[str, Dict]:
        """
        Compute RDF with proper normalization that converges to 1 at long range.
        NO CELL REPLICATION - uses only the simulation cell with PBC.
        Automatically adjusts rmax if it exceeds the safe limit.
        """
        # Validate and adjust cutoff if needed
        adjusted_rmax = self.validate_and_adjust_rdf_cutoff(atoms, rmax)
        if adjusted_rmax != rmax:
            self.logger.log(f"RDF calculation will use adjusted cutoff: {adjusted_rmax:.3f} Å", 'INFO')
        
        if atom_pairs is None:
            symbols = set(atoms.get_chemical_symbols())
            atom_pairs = [(s1, s2) for s1 in symbols for s2 in symbols if s1 <= s2]
        else:
            if atom_pairs and isinstance(atom_pairs[0], list):
                atom_pairs = [(pair[0], pair[1]) for pair in atom_pairs]
        
        self.logger.log(f"Computing normalized RDF for pairs: {atom_pairs}")
        self.logger.log("Using original simulation cell only (no replication)")
        
        volume = atoms.get_volume()
        results = {}
        
        for pair in atom_pairs:
            symbol1, symbol2 = pair
            pair_name = f"{symbol1}-{symbol2}"
            
            indices1 = [i for i, sym in enumerate(atoms.get_chemical_symbols()) if sym == symbol1]
            indices2 = [i for i, sym in enumerate(atoms.get_chemical_symbols()) if sym == symbol2]
            
            if not indices1 or not indices2:
                self.logger.log(f"No atoms found for pair {pair_name}", 'WARNING')
                continue
            
            self.logger.log(f"Analyzing {pair_name}: {len(indices1)} x {len(indices2)} atoms")
            
            r, g_r = self._compute_pair_rdf_normalized(
                atoms, indices1, indices2, rmin, adjusted_rmax, nbins, volume
            )
            
            results[pair_name] = {
                'r': r,
                'g_r': g_r,
                'normalization': 'converges_to_1',
                'rmax_used': adjusted_rmax,
                'rmax_requested': rmax
            }
        
        return results
    
    def _compute_pair_rdf_normalized(self, atoms: Atoms, indices1: List[int], indices2: List[int],
                                    rmin: float, rmax: float, nbins: int, volume: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute RDF for a specific atom pair with proper normalization.
        Uses minimum image convention with PBC.
        """
        positions = atoms.get_positions()
        pos1 = positions[indices1]
        pos2 = positions[indices2]
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        
        distances = []
        
        # Compute all pairwise distances using MIC
        for i, p1 in enumerate(pos1):
            for j, p2 in enumerate(pos2):
                # Skip self-pairs for same species
                if indices1 == indices2 and i == j:
                    continue
                
                delta = p2 - p1
                mic_vec, dist = find_mic(delta.reshape(1, -1), cell=cell, pbc=pbc)
                distances.append(dist[0])
        
        distances = np.array(distances)
        
        # Create histogram
        bins = np.linspace(rmin, rmax, nbins + 1)
        hist, edges = np.histogram(distances, bins=bins)
        r = 0.5 * (edges[1:] + edges[:-1])
        dr = edges[1] - edges[0]
        
        # Normalization for proper RDF
        # g(r) = (V / N_pairs) * (n(r) / (4πr²dr))
        # where n(r) is the count in shell at r
        
        shell_volumes = 4.0 * np.pi * r**2 * dr
        
        # Number of pairs and normalization
        if indices1 == indices2:
            # Same species: each atom sees N-1 others, but we count each pair once
            N = len(indices1)
            n_pairs = N * (N - 1) / 2
            # Ideal gas density for this species
            rho = N / volume
        else:
            # Different species
            N1 = len(indices1)
            N2 = len(indices2)
            n_pairs = N1 * N2
            # Ideal gas density for species 2
            rho = N2 / volume
        
        # For same species, we've undercounted by factor of 2 in pair counting
        # Correct normalization:
        if indices1 == indices2:
            # We have N atoms, each can form pairs with N-1 others
            # Total directed pairs = N*(N-1), but we want undirected pairs
            # Our histogram counts each pair once (due to i != j check)
            # Expected pairs in shell for uniform density:
            # Each of N atoms sees rho * 4πr²dr atoms in the shell
            ideal_count = len(indices1) * rho * shell_volumes
        else:
            # Different species: N1 atoms each see rho2 * 4πr²dr atoms of species 2
            ideal_count = len(indices1) * rho * shell_volumes
        
        # Avoid division by zero
        g_r = np.where(ideal_count > 0, hist / ideal_count, 0)
        
        return r, g_r
    
    def run_md_simulation(self, temperature: float, steps: int) -> Dict[str, List]:
        """Run MD simulation with various thermostats (with caching)"""
        
        # Check cache first
        md_cache_key = self.cache.get_md_cache_key(self.config, temperature)
        cached_trajectory = self.cache.load_cache(md_cache_key)
        
        if cached_trajectory is not None:
            self.logger.log(f"Loading MD trajectory from cache for {temperature}K")
            self.logger.log(f"  Cached trajectory: {len(cached_trajectory['times'])} frames")
            
            # Store in trajectories dict
            self.trajectories[f'md_{temperature}K'] = cached_trajectory
            
            # Restore the trajectory file
            traj_file = self.output_dir / 'trajectories' / f'md_{temperature}K.xyz'
            if not traj_file.exists() and 'positions' in cached_trajectory:
                self.logger.log(f"Restoring trajectory file from cache...")
                temp_atoms = self.atoms.copy()
                for positions in cached_trajectory['positions']:
                    temp_atoms.set_positions(positions)
                    write(str(traj_file), temp_atoms, append=True)
                self.logger.log(f"Trajectory file restored: {traj_file}")
            
            # IMPORTANT: Create MD log file from cached data
            self._write_md_log_from_trajectory(cached_trajectory, temperature)
            
            return cached_trajectory
        
        self.logger.log(f"Running MD simulation at {temperature}K for {steps} steps")
        
        timestep = self.config.get('timestep', 0.5) * units.fs
        save_interval = self.config.get('save_interval', 100)
        log_interval = self.config.get('log_interval', 100)
        thermostat = self.config.get('thermostat', 'langevin').lower()
        
        # Create MD log file for this temperature
        md_log_file = self.output_dir / f'md_{temperature}K.log'
        md_log = open(md_log_file, 'w')
        
        # Write header
        md_log.write(f"# Molecular Dynamics Simulation Log\n")
        md_log.write(f"# Temperature: {temperature} K\n")
        md_log.write(f"# Thermostat: {thermostat}\n")
        md_log.write(f"# Timestep: {timestep/units.fs:.3f} fs\n")
        md_log.write(f"# Total steps: {steps}\n")
        md_log.write(f"# Number of atoms: {len(self.atoms)}\n")
        md_log.write(f"# Cell: {self.atoms.get_cell()}\n")
        md_log.write(f"#\n")
        md_log.write(f"# {'Step':>10} {'Time(ps)':>12} {'Epot(eV)':>14} {'Ekin(eV)':>14} {'Etot(eV)':>14} {'Temp(K)':>12}\n")
        md_log.flush()
        
        self.logger.log(f"MD log file created: {md_log_file}")
        
        # Initialize velocities
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature)
        Stationary(self.atoms)
        ZeroRotation(self.atoms)
        
        # Setup dynamics based on thermostat
        if thermostat == 'nve':
            dyn = VelocityVerlet(self.atoms, timestep=timestep)
            self.logger.log(f"Using VelocityVerlet for NVE ensemble")
        elif thermostat == 'langevin':
            friction = self.config.get('friction', 0.01)
            dyn = Langevin(self.atoms, timestep=timestep, temperature_K=temperature, friction=friction)
            self.logger.log(f"Using Langevin thermostat with friction={friction}")
        elif thermostat == 'berendsen':
            taut = self.config.get('taut', 100.0) * units.fs
            # Use VelocityVerlet with Berendsen rescaling
            dyn = VelocityVerlet(self.atoms, timestep=timestep)
            berendsen = BerendsenThermostat(self.atoms, timestep, temperature, taut)
            self.logger.log(f"Using Berendsen thermostat with taut={taut/units.fs:.1f} fs")
        elif thermostat in ['nose-hoover', 'nosehover', 'nh']:
            # Nosé-Hoover chain thermostat
            # tdamp: characteristic time scale (typically 100 × timestep)
            # tchain: number of thermostat variables (default 3)
            # tloop: number of sub-steps in integration (default 1)
            nh_tdamp = self.config.get('nh_tdamp', 100.0 * timestep / units.fs) * units.fs
            nh_tchain = self.config.get('nh_tchain', 3)
            nh_tloop = self.config.get('nh_tloop', 1)
            
            dyn = NoseHooverChainNVT(
                self.atoms, 
                timestep=timestep, 
                temperature_K=temperature,
                tdamp=nh_tdamp,
                tchain=nh_tchain,
                tloop=nh_tloop
            )
            self.logger.log(f"Using Nosé-Hoover chain thermostat with tdamp={nh_tdamp/units.fs:.1f} fs, "
                          f"chain={nh_tchain}, loop={nh_tloop}")
        else:
            self.logger.log(f"Unknown thermostat {thermostat}, using Langevin", 'WARNING')
            dyn = Langevin(self.atoms, timestep=timestep, temperature_K=temperature, friction=0.01)
        
        # Trajectory data
        trajectory_data = {
            'times': [],
            'energies': [],
            'temperatures': [],
            'positions': [],
            'potential_energies': [],
            'kinetic_energies': []
        }
        
        traj_file = self.output_dir / 'trajectories' / f'md_{temperature}K.xyz'
        if traj_file.exists():
            traj_file.unlink()
        
        progress_bar = tqdm(total=steps, desc=f"MD {temperature}K", unit="steps")
        step_count = [0]  # Use list for mutable counter
        
        def log_trajectory():
            # Apply Berendsen rescaling if using that thermostat
            if thermostat == 'berendsen':
                berendsen.scale_velocities()
            
            current_time = dyn.get_time() / units.fs
            epot = self.atoms.get_potential_energy()
            ekin = self.atoms.get_kinetic_energy()
            current_energy = epot + ekin
            current_temp = self.atoms.get_temperature()
            
            trajectory_data['times'].append(current_time)
            trajectory_data['energies'].append(current_energy)
            trajectory_data['temperatures'].append(current_temp)
            trajectory_data['potential_energies'].append(epot)
            trajectory_data['kinetic_energies'].append(ekin)
            trajectory_data['positions'].append(self.atoms.get_positions().copy())
            
            write(str(traj_file), self.atoms, append=True)
            
            step_count[0] += save_interval
            progress_bar.update(save_interval)
        
        def log_md_data():
            # Log thermodynamic data at specified interval
            current_time = dyn.get_time() / units.fs
            epot = self.atoms.get_potential_energy()
            ekin = self.atoms.get_kinetic_energy()
            etot = epot + ekin
            current_temp = self.atoms.get_temperature()
            current_step = step_count[0]
            
            md_log.write(f"  {current_step:>10d} {current_time/1000:>12.4f} {epot:>14.6f} {ekin:>14.6f} {etot:>14.6f} {current_temp:>12.2f}\n")
            md_log.flush()  # Ensure data is written immediately
        
        dyn.attach(log_trajectory, interval=save_interval)
        dyn.attach(log_md_data, interval=log_interval)
        
        try:
            dyn.run(steps)
            progress_bar.close()
            
            # Write final statistics to MD log
            md_log.write(f"\n# Final Statistics\n")
            md_log.write(f"# Total frames saved: {len(trajectory_data['times'])}\n")
            if trajectory_data['energies']:
                avg_energy = np.mean(trajectory_data['energies'])
                std_energy = np.std(trajectory_data['energies'])
                avg_temp = np.mean(trajectory_data['temperatures'])
                std_temp = np.std(trajectory_data['temperatures'])
                md_log.write(f"# Average total energy: {avg_energy:.6f} ± {std_energy:.6f} eV\n")
                md_log.write(f"# Average temperature: {avg_temp:.2f} ± {std_temp:.2f} K\n")
                md_log.write(f"# Target temperature: {temperature} K\n")
                md_log.write(f"# Temperature deviation: {avg_temp - temperature:.2f} K\n")
            md_log.close()
            
            self.logger.log(f"MD simulation completed: {len(trajectory_data['times'])} frames saved")
            self.logger.log(f"MD log saved to: {md_log_file}")
            
            # Cache the trajectory data
            self.cache.save_cache(md_cache_key, trajectory_data, f"MD trajectory {temperature}K")
            
            self.trajectories[f'md_{temperature}K'] = trajectory_data
            
            return trajectory_data
            
        except Exception as e:
            self.logger.log(f"MD simulation failed: {e}", 'ERROR')
            md_log.write(f"\n# ERROR: Simulation failed with exception: {e}\n")
            md_log.close()
            progress_bar.close()
            return {}
    
    def _write_md_log_from_trajectory(self, trajectory_data: Dict[str, List], temperature: float) -> None:
        """Write MD log file from cached trajectory data"""
        md_log_file = self.output_dir / f'md_{temperature}K.log'
        
        self.logger.log(f"Writing MD log from cached data: {md_log_file}")
        
        timestep = self.config.get('timestep', 0.5)
        thermostat = self.config.get('thermostat', 'langevin').lower()
        
        with open(md_log_file, 'w') as f:
            # Write header
            f.write(f"# Molecular Dynamics Simulation Log (from cache)\n")
            f.write(f"# Temperature: {temperature} K\n")
            f.write(f"# Thermostat: {thermostat}\n")
            f.write(f"# Timestep: {timestep:.3f} fs\n")
            f.write(f"# Total frames: {len(trajectory_data['times'])}\n")
            f.write(f"# Number of atoms: {len(self.atoms)}\n")
            f.write(f"# Cell: {self.atoms.get_cell()}\n")
            f.write(f"#\n")
            f.write(f"# {'Step':>10} {'Time(ps)':>12} {'Epot(eV)':>14} {'Ekin(eV)':>14} {'Etot(eV)':>14} {'Temp(K)':>12}\n")
            
            # Write data
            times = trajectory_data['times']
            energies = trajectory_data['energies']
            temps = trajectory_data['temperatures']
            epots = trajectory_data.get('potential_energies', [])
            ekins = trajectory_data.get('kinetic_energies', [])
            
            # If potential/kinetic not in cache, compute from total and temperature
            if not epots or not ekins:
                epots = []
                ekins = []
                for i, (etot, temp) in enumerate(zip(energies, temps)):
                    # Approximate ekin from temperature
                    ekin_approx = (3.0 / 2.0) * len(self.atoms) * units.kB * temp
                    epot_approx = etot - ekin_approx
                    ekins.append(ekin_approx)
                    epots.append(epot_approx)
            
            save_interval = self.config.get('save_interval', 100)
            
            for i, (time, epot, ekin, etot, temp) in enumerate(zip(times, epots, ekins, energies, temps)):
                step = i * save_interval
                f.write(f"  {step:>10d} {time/1000:>12.4f} {epot:>14.6f} {ekin:>14.6f} {etot:>14.6f} {temp:>12.2f}\n")
            
            # Write final statistics
            f.write(f"\n# Final Statistics\n")
            f.write(f"# Total frames saved: {len(times)}\n")
            if energies:
                avg_energy = np.mean(energies)
                std_energy = np.std(energies)
                avg_temp = np.mean(temps)
                std_temp = np.std(temps)
                f.write(f"# Average total energy: {avg_energy:.6f} ± {std_energy:.6f} eV\n")
                f.write(f"# Average temperature: {avg_temp:.2f} ± {std_temp:.2f} K\n")
                f.write(f"# Target temperature: {temperature} K\n")
                f.write(f"# Temperature deviation: {avg_temp - temperature:.2f} K\n")
        
        self.logger.log(f"MD log written from cache with {len(times)} frames")
    
    def analyze_rdf_from_trajectory(self, temperature: float) -> Dict[str, Dict]:
        """Analyze RDF from MD trajectory with proper normalization"""
        traj_key = f'md_{temperature}K'
        if traj_key not in self.trajectories:
            self.logger.log(f"No trajectory data for {temperature}K", 'WARNING')
            return {}
        
        # Check cache first
        rdf_cache_key = self.cache.get_cache_key(
            {
                'temperature': temperature,
                'rdf_rmin': self.config.get('rdf_rmin', 0.0),
                'rdf_rmax': self.config.get('rdf_rmax', 10.0),
                'rdf_nbins': self.config.get('rdf_nbins', 100),
                'atom_pairs': str(self.config.get('atom_pairs', None)),
                'rdf_nframes': self.config.get('rdf_nframes', None)
            },
            f"rdf_{temperature}K"
        )
        
        cached_rdf = self.cache.load_cache(rdf_cache_key)
        if cached_rdf is not None:
            self.logger.log(f"Loading RDF analysis from cache for {temperature}K")
            return cached_rdf
        
        self.logger.log(f"Analyzing RDF from MD trajectory at {temperature}K")
        
        traj_data = self.trajectories[traj_key]
        positions_list = traj_data['positions']
        
        if len(positions_list) == 0:
            return {}
        
        rmin = self.config.get('rdf_rmin', 0.0)
        rmax = self.config.get('rdf_rmax', 10.0)
        nbins = self.config.get('rdf_nbins', 100)
        atom_pairs = self.config.get('atom_pairs', None)
        
        if atom_pairs is None:
            self.logger.log("No atom_pairs specified, auto-detecting all unique pairs")
        
        # Sample frames
        n_frames_limit = self.config.get('rdf_nframes', None)
        if n_frames_limit and n_frames_limit < len(positions_list):
            frame_indices = np.random.choice(len(positions_list), n_frames_limit, replace=False)
            selected_positions = [positions_list[i] for i in sorted(frame_indices)]
        else:
            selected_positions = positions_list
        
        self.logger.log(f"Analyzing RDF from {len(selected_positions)} frames")
        
        # Collect RDFs
        all_rdfs = {}
        
        for frame_idx, positions in enumerate(tqdm(selected_positions, desc="RDF frames")):
            temp_atoms = self.atoms.copy()
            temp_atoms.set_positions(positions)
            
            frame_rdf = self.compute_rdf_normalized(
                temp_atoms, rmin=rmin, rmax=rmax, nbins=nbins, atom_pairs=atom_pairs
            )
            
            for pair_name, rdf_data in frame_rdf.items():
                if pair_name not in all_rdfs:
                    all_rdfs[pair_name] = {'r_values': [], 'g_values': []}
                
                all_rdfs[pair_name]['r_values'].append(rdf_data['r'])
                all_rdfs[pair_name]['g_values'].append(rdf_data['g_r'])
        
        # Average RDFs
        averaged_rdfs = {}
        for pair_name, data in all_rdfs.items():
            if data['g_values']:
                r_avg = np.mean(data['r_values'], axis=0)
                g_avg = np.mean(data['g_values'], axis=0)
                g_std = np.std(data['g_values'], axis=0)
                
                averaged_rdfs[pair_name] = {
                    'r': r_avg,
                    'g_r': g_avg,
                    'g_r_std': g_std,
                    'n_frames': len(data['g_values']),
                    'temperature': temperature,
                    'normalization': 'converges_to_1'
                }
        
        # Cache the results
        self.cache.save_cache(rdf_cache_key, averaged_rdfs, f"RDF analysis {temperature}K")
        
        self.logger.log(f"RDF analysis complete for pairs: {list(averaged_rdfs.keys())}")
        
        return averaged_rdfs
    

    def plot_rdf_comparison(self) -> None:
        """Plot RDF comparisons across temperatures"""
        self.logger.log("Plotting RDF comparisons...")
        
        all_pairs = set()
        temp_data = {}
        
        for key in self.results:
            if key.startswith('rdf_') and key.endswith('K'):
                temp_str = key[4:]
                try:
                    temp = int(temp_str.replace('K', ''))
                    temp_data[temp] = self.results[key]
                    all_pairs.update(self.results[key].keys())
                except ValueError:
                    continue
        
        if not all_pairs:
            self.logger.log("No RDF data found for plotting", 'WARNING')
            return
        
        sigma = self.config.get('rdf_smoothing_sigma', 0)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for pair_name in sorted(all_pairs):
            fig, ax = plt.subplots(figsize=(8, 8))
            
            lines_plotted = 0
            rmax_info = None
            
            for i, temp in enumerate(sorted(temp_data.keys())):
                if pair_name in temp_data[temp]:
                    rdf_data = temp_data[temp][pair_name]
                    r = rdf_data['r']
                    g_r = rdf_data['g_r']
                    
                    # Get rmax info for title
                    if rmax_info is None and 'rmax_used' in rdf_data:
                        rmax_used = rdf_data['rmax_used']
                        rmax_requested = rdf_data.get('rmax_requested', rmax_used)
                        if abs(rmax_used - rmax_requested) > 0.01:
                            rmax_info = f"rmax adjusted: {rmax_requested:.1f}→{rmax_used:.1f} Å"
                    
                    if sigma > 0:
                        g_r_smoothed = gaussian_filter1d(g_r, sigma=sigma)
                    else:
                        g_r_smoothed = g_r
                    
                    if 'g_r_std' in rdf_data:
                        g_r_std = rdf_data['g_r_std']
                        if sigma > 0:
                            g_r_std = gaussian_filter1d(g_r_std, sigma=sigma)
                        ax.errorbar(r, g_r_smoothed, yerr=g_r_std, 
                                  label=f"{temp} K", linewidth=2.5, alpha=0.8,
                                  color=colors[i % len(colors)], errorevery=5)
                    else:
                        ax.plot(r, g_r_smoothed, label=f"{temp} K", linewidth=3.0, alpha=0.8,
                               color=colors[i % len(colors)])
                    
                    lines_plotted += 1
            
            if lines_plotted > 0:
                ax.set_xlabel('Distance (Å)', fontsize=14)
                ax.set_ylabel('g(r)', fontsize=14)
                title = f'RDF: {pair_name} (normalized to 1 at long range)'
                if sigma > 0:
                    title += f' (σ={sigma})'
                if rmax_info:
                    title += f'\n{rmax_info}'
                ax.set_title(title, fontsize=16)
                ax.legend(frameon=True, fancybox=False, framealpha=0.9, fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)
                ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='g(r)=1')
                
                safe_pair_name = pair_name.replace('-', '_')
                plot_file = self.output_dir / 'plots' / f'rdf_{safe_pair_name}_comparison.png'
                plt.tight_layout()
                plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                self.logger.log(f"RDF comparison plot saved: {plot_file}")
            else:
                plt.close()
    
    def plot_energy_drift(self, temperature: float) -> None:
        """
        Plot energy drift analysis showing both cumulative and instantaneous drift.
        
        Cumulative drift: ΔE = (1/N_step) × Σ |E_k - E(0)| / |E(0)|
        Instantaneous drift: ΔE_inst = |E(t) - E(0)| / |E(0)|
        """
        analysis_key = f'energy_drift_{temperature}K'
        if analysis_key not in self.results:
            return
        
        analysis = self.results[analysis_key]
        times_ps = analysis['times_ps']
        instantaneous_drift = analysis['instantaneous_drift_values']
        cumulative_drift = analysis['cumulative_drift_values']
        
        # Adjust time to start from 0
        start_time_ps = analysis['start_time_ps']
        adjusted_times_ps = times_ps - start_time_ps
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot instantaneous drift (fluctuating)
        ax.plot(adjusted_times_ps, instantaneous_drift, 
                color='lightblue', linewidth=2.0, alpha=0.5, 
                label='Instantaneous: |E(t)-E(0)|/|E(0)|')
        
        # Plot cumulative drift (averaged)
        ax.plot(adjusted_times_ps, cumulative_drift, 
                color='red', linewidth=3.0, alpha=0.9,
                label='Cumulative: (1/N)Σ|E_k-E(0)|/|E(0)|')
        
        ax.set_xlabel('Analysis Time (ps)', fontsize=14)
        ax.set_ylabel('Energy Drift ΔE', fontsize=14)
        ax.set_title(f'Energy Conservation Analysis (NVE) - {temperature}K\nCumulative vs Instantaneous Drift', fontsize=16)
        ax.set_xlim(0, adjusted_times_ps[-1])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        
        # Add statistics box
        stats_text = f"Final cumulative: {analysis['final_drift_cumulative']:.2e}\n"
        stats_text += f"Final instantaneous: {analysis['final_drift_instantaneous']:.2e}\n"
        stats_text += f"Quality: {analysis['drift_quality']}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        plot_file = self.output_dir / 'plots' / f'energy_drift_{temperature}K.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.log(f"Energy drift plot saved: {plot_file}")

    def analyze_free_energy_from_trajectory(self, temperature: float) -> Dict[str, Any]:
        """
        Analyze vibrational free energy from MD trajectory using VACF method.

        Parameters:
        -----------
        temperature : float
            Simulation temperature (K)

        Returns:
        --------
        results : dict
            Dictionary containing free energy analysis results
        """
        if not self.config.get('run_free_energy', False):
            return None

        self.logger.log_section(f"FREE ENERGY ANALYSIS FROM MD TRAJECTORY ({temperature}K)")

        # Get trajectory file
        traj_file = self.output_dir / 'trajectories' / f'md_{temperature}K.xyz'

        if not traj_file.exists():
            self.logger.log(f"Trajectory file not found: {traj_file}", 'WARNING')
            self.logger.log("Run MD simulation first to generate trajectory", 'INFO')
            return None

        try:
            # Initialize MDFreeEnergy calculator
            timestep = self.config.get('timestep', 0.5)  # fs

            fe_calc = MDFreeEnergy(
                trajectory_file=str(traj_file),
                temperature=temperature,
                timestep=timestep,
                logger=self.logger
            )

            # Calculate VACF
            max_lag = self.config.get('free_energy_max_lag', None)
            window = self.config.get('free_energy_window', 'hann')
            fe_calc.calculate_vacf(max_lag=max_lag, window=window)

            # Calculate VDOS
            omega_max = self.config.get('free_energy_omega_max', 4000)
            n_points = self.config.get('free_energy_n_points', 8192)
            fe_calc.calculate_vdos(omega_max=omega_max, n_points=n_points)

            # Temperature array for thermodynamic calculations
            T_min = self.config.get('free_energy_T_min', 0)
            T_max = self.config.get('free_energy_T_max', 500)
            n_temps = self.config.get('free_energy_n_temps', 100)
            T_array = np.linspace(T_min, T_max, n_temps)

            # Calculate thermodynamic properties
            F_vib = fe_calc.calculate_free_energy(T_array)
            S_vib = fe_calc.calculate_entropy(T_array)
            C_v = fe_calc.calculate_heat_capacity(T_array)
            E_ZPE = fe_calc.calculate_zero_point_energy()

            # Run convergence analysis if requested
            convergence = None
            if self.config.get('free_energy_convergence', True):
                fractions = self.config.get('free_energy_convergence_fractions',
                                           [0.2, 0.4, 0.6, 0.8, 1.0])
                convergence = fe_calc.convergence_analysis(max_time_fractions=fractions)

                # Plot convergence
                conv_plot = self.output_dir / 'plots' / f'free_energy_convergence_{temperature}K.png'
                fe_calc.plot_convergence(convergence, output_file=str(conv_plot))

            # Compare with phonopy if available
            phonopy_file = self.config.get('phonopy_file', None)
            comparison = None
            if phonopy_file and Path(phonopy_file).exists():
                comparison = fe_calc.compare_with_phonopy(phonopy_file, T_array=T_array)

            # Export results
            fe_output_dir = self.output_dir / f'free_energy_{temperature}K'
            fe_calc.export_results(str(fe_output_dir), T_array=T_array)

            # Prepare results dictionary
            results = {
                'temperature': temperature,
                'timestep_fs': timestep,
                'n_frames': fe_calc.n_frames,
                'n_atoms': fe_calc.n_atoms,
                'total_time_ps': fe_calc.n_frames * timestep / 1000,
                'E_ZPE_eV_per_atom': E_ZPE,
                'F_vib_at_sim_T_eV_per_atom': F_vib[np.argmin(np.abs(T_array - temperature))],
                'S_vib_at_sim_T_eV_per_K_per_atom': S_vib[np.argmin(np.abs(T_array - temperature))],
                'C_v_at_sim_T_eV_per_K_per_atom': C_v[np.argmin(np.abs(T_array - temperature))],
                'VDOS_integral': np.trapz(fe_calc.vdos, fe_calc.omega),
                'VACF_final_value': fe_calc.vacf[-1],
                'convergence_data': convergence,
                'phonopy_comparison': comparison
            }

            self.logger.log(f"Free energy analysis complete for {temperature}K")
            self.logger.log(f"Results exported to: {fe_output_dir}")

            return results

        except Exception as e:
            self.logger.log(f"Error in free energy analysis: {e}", 'ERROR')
            import traceback
            self.logger.log(traceback.format_exc(), 'ERROR')
            return None

    def run_full_analysis(self, xyz_file: str) -> None:
        """Run complete analysis workflow"""
        self.logger.log_section("STARTING FULL MD ANALYSIS")
        
        # Load structure
        self.load_structure(xyz_file)
        
        # IMPORTANT: Replicate cell for simulation BEFORE any calculations
        self.replicate_simulation_cell()
        
        # Single point energy
        self.run_single_point()
        
        # Geometry optimization
        if self.run_geometry_optimization():
            rmsd_results = self.analyze_rmsd()
            if rmsd_results:
                self.results['rmsd_optimization'] = rmsd_results
            
            coord_results = self.analyze_coordination_numbers(label="optimized")
            if coord_results:
                self.results['coordination_optimized'] = coord_results
                self.plot_coordination_histogram(label="optimized")
        
        # Cell optimization
        if self.run_cell_optimization():
            rmsd_cell = self.analyze_rmsd()
            if rmsd_cell:
                self.results['rmsd_cell_optimization'] = rmsd_cell
        
        # Initial structure coordination
        initial_coord = self.analyze_coordination_numbers(label="initial")
        if initial_coord:
            self.results['coordination_initial'] = initial_coord
            self.plot_coordination_histogram(label="initial")
        
        # MD simulations
        # Support both 'run_md' and 'md_simu' for consistency (run_md preferred)
        run_md = self.config.get('run_md', self.config.get('md_simu', True))  # Default to True
        if run_md:
            temperatures = self.config.get('temperatures', [300])
            md_steps = self.config.get('md_steps', 10000)
            
            for temp in temperatures:
                self.logger.log_section(f"MD ANALYSIS AT {temp}K")
                
                traj_data = self.run_md_simulation(temp, md_steps)
                
                if traj_data:
                    # Energy drift (NVE only)
                    drift_analysis = self.analyze_energy_drift(temp)
                    if drift_analysis:
                        self.results[f'energy_drift_{temp}K'] = drift_analysis
                        self.plot_energy_drift(temp)
                    
                    # RDF analysis
                    if self.config.get('run_rdf', False):  # ← ADD THIS CHECK
                        rdf_results = self.analyze_rdf_from_trajectory(temp)
                        if rdf_results:
                            self.results[f'rdf_{temp}K'] = rdf_results

                    # Free energy analysis
                    fe_results = self.analyze_free_energy_from_trajectory(temp)
                    if fe_results:
                        self.results[f'free_energy_{temp}K'] = fe_results
            # Generate comparison plots
            if self.config.get('run_rdf', False):  # ← ADD THIS CHECK
                self.plot_rdf_comparison()
        else:
            self.logger.log_section("MD SIMULATIONS SKIPPED")
            self.logger.log("MD simulations disabled in configuration (run_md: false)")
        # Print cache statistics
        self.logger.log_section("CACHE STATISTICS")
        self.cache.print_cache_summary(self.logger)
        
        # Save final results
        results_file = self.output_dir / 'analysis_results.yaml'
        with open(results_file, 'w') as f:
            yaml_results = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    yaml_results[key] = self._convert_numpy_for_yaml(value)
                else:
                    yaml_results[key] = value
            
            # Add cache stats to results
            yaml_results['cache_statistics'] = self.cache.get_cache_stats()
            
            yaml.dump(yaml_results, f, default_flow_style=False)
        
        self.logger.log_section("ANALYSIS COMPLETE")
        self.logger.log(f"Results saved to: {results_file}")
        self.logger.log(f"Total runtime: {time.time() - self.start_time:.2f} seconds")
    
    def _convert_numpy_for_yaml(self, obj):
        """Convert numpy arrays to lists for YAML"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_for_yaml(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_for_yaml(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

def main():
    """Main execution function"""
    if len(sys.argv) < 3:
        print("Usage: python enhanced_md_analysis.py <input.xyz> <config.yaml>")
        sys.exit(1)

    xyz_file = sys.argv[1]
    config_file = sys.argv[2]

    analyzer = MDAnalyzer(config_file)
    analyzer.run_full_analysis(xyz_file)

if __name__ == "__main__":
    main()