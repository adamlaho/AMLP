# Analysis Tools

This directory contains analysis tools for molecular dynamics simulations.

## Structure

```
multi_agent_dft/analysis/
├── __init__.py              # Module exports
├── free_energy.py           # MDFreeEnergy class for vibrational free energy calculations
├── cli/                     # Command-line interface tools
│   ├── __init__.py
│   ├── run_free_energy_analysis.py       # Single trajectory analysis
│   └── analyze_existing_trajectories.py  # Batch trajectory analysis
└── README.md                # This file
```

## Free Energy Analysis

The free energy module provides MD-based calculation of temperature-dependent thermodynamic properties using the velocity autocorrelation function (VACF) method.

### Features

- Velocity Autocorrelation Function (VACF) calculation
- Vibrational Density of States (VDOS) from Fourier transform
- Temperature-dependent free energy, entropy, and heat capacity
- Convergence analysis
- Phonopy comparison (harmonic vs anharmonic)
- Publication-quality plots

### Usage

#### 1. Integrated with AMLPA

```python
from amlpa import MDAnalyzer

# In your config.yaml, set:
# run_free_energy: true

analyzer = MDAnalyzer('config.yaml')
analyzer.run_full_analysis('structure.xyz')
```

#### 2. Standalone CLI Tool (Single Trajectory)

```bash
cd multi_agent_dft/analysis/cli
python run_free_energy_analysis.py trajectory.xyz --temperature 300 --timestep 0.5
```

#### 3. Batch Analysis (Multiple Trajectories)

```bash
cd multi_agent_dft/analysis/cli
python analyze_existing_trajectories.py config_batch.yaml
```

#### 4. Python API

```python
from multi_agent_dft.analysis import MDFreeEnergy
import numpy as np

# Load and analyze trajectory
fe = MDFreeEnergy('trajectory.xyz', temperature=300, timestep=0.5)
fe.calculate_vacf()
fe.calculate_vdos()

# Calculate properties
T_array = np.linspace(0, 500, 100)
F_vib = fe.calculate_free_energy(T_array)
S_vib = fe.calculate_entropy(T_array)
C_v = fe.calculate_heat_capacity(T_array)
E_ZPE = fe.calculate_zero_point_energy()

# Export results
fe.export_results('output_directory/', T_array=T_array)
```

## Documentation

See the root-level documentation files:
- `FREE_ENERGY_DOCUMENTATION.md` - Complete theory and implementation details
- `QUICK_START_FREE_ENERGY.md` - Quick start guide
- `STANDALONE_ANALYSIS_GUIDE.md` - Standalone tool usage examples
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation summary

## Configuration Examples

Example configuration files are provided in the root directory:
- `config_free_energy_example.yaml` - Integrated workflow configuration
- `config_batch_analysis_example.yaml` - Batch analysis configuration

## Requirements

- ASE (Atomic Simulation Environment)
- NumPy
- SciPy
- Matplotlib
- PyYAML

## Citation

If you use this module in your research, please cite:

> Lahouari, A. (2025). AMLP: Automated Machine Learning Potentials Framework.
> Free Energy Module: MD-based thermodynamic properties from VACF method.
