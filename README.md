# Automated Machine Learning Potential with Multi-Agent DFT Research System

![AMLP Logo](amlp_logo_git.png)

A comprehensive framework that combines AI-assisted computational chemistry research with automated machine learning analysis for atomic structures. This integrated system orchestrates literature analysis, DFT code selection, structure processing for quantum chemistry simulations, and machine learning analysis of atomic structures.

## Table of Contents
- [Multi-Agent DFT Research System](#multi-agent-dft-research-system)
  - [Overview](#overview)
  - [Features](#features)
  - [Installation](#installation)
  - [API Configuration](#api-configuration)
  - [Usage](#usage)
- [AML-analysis: Automated Machine Learning Analysis](#aml-analysis-automated-machine-learning-analysis)
  - [What Can AMLa Do?](#what-can-amla-do)
  - [Getting Started with AMLa](#getting-started-with-amla)
  - [How to Use AMLa](#how-to-use-amla)
  - [Configuration Guide](#configuration-guide)

## Multi-Agent DFT Research System

### Overview

The Multi-Agent DFT Research System is an integrated framework that combines:

1. **AI-driven research analysis** - Uses specialized AI agents to analyze research topics and generate summaries
2. **DFT code expertise** - Provides expert recommendations for Gaussian, VASP, and CP2K simulations
3. **Input file generation** - Efficiently processes crystallographic structures for DFT calculations
4. **Output data processing** - Extracts and formats simulation results for analysis or ML model training

### Features

#### AI-Agent Research Assistance
- **Experimental Chemist Agent**: Provides summaries focusing on experimental aspects of research topics
- **Theoretical Chemist Agent**: Analyzes theoretical foundations and computational approaches
- **DFT Expert Agents**: Specialized agents for Gaussian, VASP, and CP2K provide code-specific recommendations
- **Supervisor Agents**: Integrate information and provide comprehensive reports

#### Input Generation
- **Multi-code support**: Generate inputs for CP2K, VASP, and Gaussian
- **Batch processing**: Convert multiple structure files automatically
- **Format conversion**: Process CIF and XYZ files with validation
- **Supercell creation**: Build supercells with custom dimensions
- **Interactive guidance**: Step-by-step parameter selection for DFT calculations

#### Output Processing
- **DFT output extraction**: Extract energies, forces, and coordinates from simulation results
- **ML-ready dataset creation**: Convert DFT outputs to HDF5 format for machine learning potentials
- **AIMD processing**: Generate AIMD inputs from optimized structures at multiple temperatures

### Installation

#### Requirements
- Python 3.8+
- Required Python packages:
  - NumPy
  - PyYAML
  - ASE (Atomic Simulation Environment, optional but recommended)
  - openai (for AI agent functionality)
  - requests

#### Setup
1. Clone the repository:
```
git clone https://github.com/adamlaho/AMLP.git
cd AMLP
```

2. Install dependencies:
```
pip install -r requirements.txt
```

### API Configuration

The AI agents in this system use OpenAI's API for text generation. Follow these steps to configure API access:

1. **Get API Key**:
   - Sign up for an account at [OpenAI Platform](https://platform.openai.com/)
   - Navigate to the API keys section and create a new secret key
   - Copy the key (you will not be able to view it again)

2. **Set Environment Variable**:
   
   The system looks for the API key in the `OPENAI_API_KEY` environment variable:
   ```bash
   # On Linux/macOS
   export OPENAI_API_KEY="your-api-key-here"
   
   # On Windows
   set OPENAI_API_KEY=your-api-key-here
   ```

3. **Model Configuration**:
   
   The system uses "gpt-4" by default, but you can specify a different model when initializing agents:
   
   ```python
   # In your code or custom scripts
   from multi_agent_dft.agents.base import Agent
   
   # Initialize with a specific model
   agent = Agent(name="custom_agent", model="gpt-4-1106-preview")
   ```

4. **Usage Monitoring**:
   - Be aware of your [OpenAI API usage limits](https://platform.openai.com/account/limits)
   - The AI agent functionality will consume tokens based on the length of inputs and outputs
   - The system implements basic retry logic for API rate limiting (3 attempts with exponential backoff)

### Usage

#### Basic Usage
Run the main script to start the system:
```
python3 main_amlpt.py
```

The system will present a menu with five operation modes:
1. AI-agent feedback (research summaries & reports)
2. Input generation (CP2K/VASP/Gaussian)
3. Output processing (extract forces, energies, coordinates)
4. ML potential dataset creation (JSON to MACE HDF5)
5. AIMD processing (JSON to CP2K AIMD inputs)

#### AI-Assisted Research Workflow

This mode helps you explore research topics with AI assistance:

1. Enter a research topic or question
2. The system will refine your query and analyze literature
3. Review reports from Experimental and Theoretical Chemist agents
4. Examine DFT-specific recommendations from expert agents
5. Use the generated reports to guide your computational research

Example:
```
Enter your research topic or question: Metal oxide catalysts for water splitting
```

#### Input Generation

Generate input files for DFT calculations using either batch mode or guided mode:

##### Batch Mode
Automatically convert all supported files using default templates:
```
Batch-mode: which DFT code? (CP2K/VASP/Gaussian): cp2k
Path to file or directory: ./structures
Output directory: ./cp2k_inputs
```

##### Guided Mode
Step through detailed parameter selection for your DFT calculation:
```
Which DFT code? (CP2K/VASP/Gaussian): VASP
```

##### Supercell Creation
Build supercells with custom dimensions during input generation:
```
Create a supercell? (y/n) [n]: y
Multiplier for x-axis [1]: 2
Multiplier for y-axis [1]: 2
Multiplier for z-axis [1]: 2
```

#### Output Processing

Extract data from DFT calculation outputs:
```
Select DFT code (1/2/3): 1
Path to CP2K input file (.inp): ./cp2k_calcs/input.inp
Path to CP2K output file: ./cp2k_calcs/output.out
Path for output JSON file [output_data.json]: results.json
```

#### ML Dataset Creation

Convert DFT outputs to machine learning potential training data:
```
Full path to JSON file containing DFT data: ./results/dft_data.json
Output directory for HDF5 datasets [current directory]: ./ml_datasets
Dataset base name [dft_data]: water_system
```

#### AIMD Processing

Generate AIMD inputs from optimized structures at multiple temperatures:
```
Path to your JSON file or directory: ./optimized_structures
Output directory for generated files: ./aimd_inputs
Select template (1-5) [1]: 2
```

#### Structure File Support

The system supports the following structure file formats:
- **CIF** (Crystallographic Information File)
- **XYZ** (Cartesian coordinates)

#### Output Files

Depending on the mode, the system generates:
- **CP2K**: .inp input files
- **VASP**: INCAR, POSCAR, KPOINTS, and POTCAR files in subdirectories
- **Gaussian**: .com 
- **Research Reports**: .txt 
- **Processed Data**: .json and .h5 data files

#### Advanced Usage

##### Command Line Options

Fix CIF files with structural issues:
```
python main_amlpt.py --fix-cif
```

##### Programmatic Usage

The system can be imported and used in other Python scripts:

```python
from amlp_t.main_amlpt import MultiAgentSystem

# Initialize the system
system = MultiAgentSystem()

# Process a specific research query non-interactively
system.process_query("Metal oxide catalysts for water splitting")

# Generate inputs for a structure file
system._process_single_cp2k_file(Path("structure.cif"), Path("./outputs"))
```

#### Troubleshooting

##### API-Related Issues

1. **Authentication Errors**: Verify your API key is correct and properly set in the environment or config file
2. **Rate Limiting**: If you see `RateLimitError`, the system will automatically retry with exponential backoff
3. **Connection Issues**: Check your internet connection and proxy settings if applicable
4. **Model Not Available**: Ensure you're using a model that's available to your API key level

##### Common Issues

1. **File validation errors**: Check if your CIF or XYZ files follow standard format
2. **Missing cell parameters**: Ensure cell information is properly defined for periodic systems
3. **ASE import errors**: Install ASE for full functionality: `pip install ase`

## AML-analysis: Automated Machine Learning Analysis

AMLa is a tool that helps you analyze atomic structures using machine learning. It combines several analysis methods into one easy workflow:
- Geometry optimization
- Vibrational analysis
- Molecular dynamics simulations
- Structural analysis (RDF and coordination)

### What Can AMLa Do?

- **Use Pre-trained Models**: Works with MACE machine learning potentials
- **Run Multiple Analyses**: Perform different analyses in a single workflow
- **Easy Configuration**: Change simulation settings using a simple YAML file
- **Reproducible Research**: Get consistent results for scientific work

### Getting Started with AMLa

#### System Requirements
- Python 3.7 or newer (Python 3.9 recommended)
- Required packages: numpy, matplotlib, pyyaml, torch, tqdm, scipy, ase, mace-torch

### How to Use AMLa

Run the main analysis script:
```bash
python analyser.py <input_file.xyz> config_file="config.yaml"
```

### Configuration Guide

Create a `config.yaml` file to customize your analysis. Here's what you can configure:

#### Basic Settings
```yaml
output_dir: "./results"           # Where to save results
plot_dpi: 300                     # Image quality for plots
```

#### Structure Settings
```yaml
# Cell information
readcell_info: False              # Read cell from XYZ file?
cell_params: [5.0, 5.0, 10.0, 90.0, 90.0, 120.0]  # Manual cell parameters

# System settings
pbc: True                         # Use periodic boundaries?
```

#### Analysis Options

##### Single-Point Calculation
```yaml
single_point: True                # Calculate energy and forces
```

##### Geometry Optimization
```yaml
geo_opt: True                     # Run geometry optimization
optimizer: "BFGS"                 # Optimization algorithm
fmax: 0.001                       # Force convergence criterion
```

##### Vibrational Analysis
```yaml
vib_input: True                   # Run phonon calculations
phonon_delta: 0.01                # Displacement step size
phonon_supercell: [3, 3, 3]       # Supercell for phonons
phonon_plot: True                 # Create phonon plots
```
More features can be found in the config.yaml example in the repository.

##### Molecular Dynamics
```yaml
MD_run: True                      # Run MD simulation
Temp_initial: 50                  # Starting temperature (K)
Temp_final: 350                   # Final temperature (K)
Temp_step: 25                     # Temperature increment
Step: 2000000                     # Number of MD steps
timestep: 0.5                     # Time step (fs)
```

##### Structural Analysis
```yaml
# RDF settings
rdf_rmax: 10.0                    # Maximum distance for RDF
rdf_nbins: 100                    # Number of RDF bins
rdf_atom_types: "all"             # Atoms to include in RDF

# Replication for analysis
replicate_dims: [True, True, False]  # Which dimensions to replicate
```

#### Hardware Configuration
```yaml
device: "gpu"                     # Use GPU or CPU
gpus: ["cuda:0"]                  # Which GPU to use
model_paths:
  - "/path/to/your/model1"        # Path to ML model
```
Use your own MACE potential or you can also use the foundation models from MACE on this website https://github.com/ACEsuit/mace-mp

## Contributing

Contributions are welcome! Fork the repository and submit pull requests with improvements or bug fixes. For major changes, please open an issue first to discuss your ideas.

## License

[MIT License](LICENSE)

## Acknowledgments

- This project utilizes several open-source packages including ASE, NumPy, PyYAML, and MACE
- DFT code parameters are based on best practices from the computational chemistry community
- The AI agent system utilizes OpenAI's GPT models for text generation

**Disclaimer:** Parts of this project are currently under active development. All features, APIs, and documentation may change as new functionalities are implemented and improvements are made.
