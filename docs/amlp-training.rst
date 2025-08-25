AMLP-training
=============

Overview
--------

The AMLP-training system (implemented in ``amlpt.py``) is a Multi-Agent DFT Research System that provides several core functionalities:

1. Research assistance through AI agents
2. Automated input file generation for DFT codes
3. Processing of calculation outputs
4. Creation of machine learning datasets
5. Generation of AIMD input files

The system operates in multiple modes that can be selected through an interactive menu.

AI Agent System
--------------

The AI agent system consists of specialized agents that analyze research topics and provide insights:

Agent Types
^^^^^^^^^^

- **Experimental Chemist Agent**: Focuses on experimental aspects of research
- **Theoretical Chemist Agent**: Analyzes theoretical foundations
- **DFT Expert Agents**: Specialized for Gaussian, VASP, and CP2K
- **Supervisor Agents**: Integrate information and provide comprehensive reports

Research Workflow
^^^^^^^^^^^^^^^^

1. User enters a research query
2. The system refines the query
3. Specialized agents analyze the literature
4. Comprehensive reports are generated from multiple perspectives

Sample Usage
^^^^^^^^^^^

.. code-block:: python

   from amlpt import MultiAgentSystem

   system = MultiAgentSystem()
   system.process_query("Metal oxide catalysts for water splitting")

Input Generation
---------------

AMLP-training can generate input files for multiple DFT codes from structure files:

Supported DFT Codes
^^^^^^^^^^^^^^^^^^

- **CP2K**: Generates ``.inp`` input files
- **VASP**: Creates INCAR, POSCAR, KPOINTS files
- **Gaussian**: Produces ``.com`` input files

Input Generation Modes
^^^^^^^^^^^^^^^^^^^^^

- **Batch Mode**: Automatically process multiple files with default templates
- **Guided Mode**: Interactive parameter selection with expert guidance

Structure Manipulation
^^^^^^^^^^^^^^^^^^^^^

- Supercell creation with custom dimensions
- Structure validation and format conversion
- Cell parameter handling

CP2K Input Generation
^^^^^^^^^^^^^^^^^^^^

The system provides detailed options for CP2K input files:

* Project name and run type
* DFT settings (functionals, basis sets, vdW corrections)
* SCF convergence parameters
* Grid settings for plane wave calculations
* Geometry optimization parameters
* Cell optimization settings
* Molecular dynamics parameters
* Output and printing options

VASP Input Generation
^^^^^^^^^^^^^^^^^^^^

For VASP, the system can configure:

* Exchange-correlation functionals
* Plane wave cutoffs
* Electronic smearing
* K-point sampling
* Structural optimization settings
* Molecular dynamics parameters
* Output settings

Gaussian Input Generation
^^^^^^^^^^^^^^^^^^^^^^^

Gaussian input generation includes:

* Computational methods and basis sets
* Job types (optimization, frequency calculation, etc.)
* Molecular properties (charge, multiplicity)
* Solvent effects
* Advanced SCF options
* Population analysis options

Batch Processing
^^^^^^^^^^^^^^^

For processing multiple structure files:

.. code-block:: python

   # Generate CP2K inputs for all CIF files in a directory
   system._batch_input_generation(code="cp2k", path="/path/to/structures", out="/path/to/outputs")

Output Processing
----------------

The system can process output files from DFT calculations to extract useful data:

Features
^^^^^^^

- Extract energies, forces, and coordinates from calculation outputs
- Convert between different coordinate systems
- Format data for further analysis or machine learning

Supported Output Formats
^^^^^^^^^^^^^^^^^^^^^^^

- CP2K output files
- VASP OUTCAR files
- Gaussian log files

Output Format
^^^^^^^^^^^^

- JSON files with structured data

Example
^^^^^^^

.. code-block:: python

   from multi_agent_dft.file_processing.dft_output_processor import process_dft_output
   
   # Process a CP2K calculation
   process_dft_output(
       code_type="cp2k",
       input_file="input.inp",
       output_file="output.out",
       frac_xyz_file="coords.xyz",
       output_json="results.json"
   )

ML Dataset Creation
------------------

AMLP-training can convert processed DFT data into formats suitable for machine learning potential training:

Features
^^^^^^^

- Convert JSON data to HDF5 format for ML training
- Split data into training and validation sets
- Apply force thresholds and unit conversions
- Configure periodic boundary conditions

Customization Options
^^^^^^^^^^^^^^^^^^^^

- Training/validation ratio
- Batch size for HDF5 files
- Force thresholds
- Unit conversion factors
- PBC handling

Example
^^^^^^^

.. code-block:: python

   from multi_agent_dft.file_processing.ml_dataset_converter import create_mace_h5_dataset
   
   # Create ML datasets
   train_h5, valid_h5 = create_mace_h5_dataset(
       json_file="dft_results.json",
       output_dir="ml_data",
       dataset_name="my_dataset",
       train_ratio=0.8,
       batch_size=4,
       max_force_threshold=300.0
   )

AIMD Processing
--------------

The system can generate input files for ab initio molecular dynamics simulations:

Features
^^^^^^^

- Create AIMD inputs for multiple temperatures
- Support for different ensembles (NVE, NVT, NPT)
- Configure thermostats and barostats
- Set up trajectory output options

Template Options
^^^^^^^^^^^^^^^

- Standard NVT simulations
- Multi-temperature studies
- High-temperature melting simulations
- Low-temperature glass transition simulations

Example
^^^^^^^

.. code-block:: python

   # Using the interactive interface
   system._handle_aimd_processing()
   
   # Or programmatically
   system._write_cp2k_aimd_input(
       atoms=atoms_object,
       output_path="aimd_300K.inp",
       temperature=300.0,
       config={
           "ensemble": "NVT",
           "thermostat": "NOSE",
           "timestep": 0.5,
           "steps": 1000000
       }
   )

Configuration
------------

AMLP-training can be configured through both interactive prompts and configuration files:

Configuration Methods
^^^^^^^^^^^^^^^^^^^^

- Interactive command-line interface
- YAML configuration files

Configurable Parameters
^^^^^^^^^^^^^^^^^^^^^^

- API settings for AI agents
- DFT calculation parameters
- Output processing options
- Dataset creation settings
- AIMD simulation parameters

Example Configuration
^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # Sample config.yaml for AMLP-training
   
   # API settings
   api:
     model: "gpt-4"
     max_tokens: 4000
   
   # Default DFT parameters
   dft:
     cp2k:
       cutoff: 400
       functional: "PBE"
       basis_set: "DZVP-MOLOPT-SR-GTH"
     vasp:
       encut: 400
       kpoints: [3, 3, 3]
       ismear: 0
     gaussian:
       method: "B3LYP"
       basis_set: "6-31G(d)"
