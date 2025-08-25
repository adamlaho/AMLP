File Processing
==============

The AMLP system includes powerful file processing capabilities to handle various aspects of computational chemistry data workflows. These modules provide a comprehensive set of tools for working with different file formats, converting between them, and preparing data for simulations and analysis.

.. contents:: Table of Contents
   :local:
   :depth: 2

Structure File Processing
-----------------------

CIF Processing
^^^^^^^^^^^^^

The CIF (Crystallographic Information File) processing module provides robust tools for working with crystallographic structure files.

Key Features
"""""""""""

* Parsing CIF files with robust error handling
* Extracting cell parameters and atomic coordinates
* Converting coordinates between fractional and Cartesian
* Converting CIF files to XYZ format
* Batch processing of multiple CIF files

Core Functions
"""""""""""

.. code-block:: python

   def parse_cif_file(file_path)
   def cif_to_xyz(cif_file, xyz_file=None)
   def process_cif_files(input_dir, output_dir=None, pattern="*.cif")

The CIF parser is specially designed to handle common issues in CIF files:

* Uncertainty notation in coordinates (e.g., '0.22113(9)')
* Mixed element symbol formats
* Proper transformation from fractional to Cartesian coordinates

XYZ Processing
^^^^^^^^^^^^

The XYZ file processing module provides functionality for working with the simple yet widely used XYZ format.

Key Features
"""""""""""

* Reading and parsing XYZ files with coordinate data
* Extracting lattice vectors from comments
* Generating 3D cells for non-periodic structures
* Converting structure data to molecular formulas
* Batch processing of multiple XYZ files

Core Functions
"""""""""""

.. code-block:: python

   def parse_xyz(xyz_file)
   def xyz_to_mol_format(structure_data)
   def process_xyz_files(input_dir, output_dir=None, file_pattern="*.xyz", dft_code="cp2k")

The XYZ parser includes several advanced features:

* Extracting embedded cell information from comment lines
* Standardizing element symbols
* Building bounding box cells for molecular systems
* Handling various XYZ file formats and variations

Format Conversion
^^^^^^^^^^^^^^^^^^^

The system includes powerful functions for converting between different structure file formats:

.. code-block:: python

   def convert_input_format(source_file, target_format, output_file=None, parameters=None)
   def detect_input_format(file_path)
   def parse_input_file(file_path, file_format=None)
   def batch_convert(input_dir, target_format, output_dir=None, recursive=False)

These functions enable seamless conversion between formats like CIF, XYZ, CP2K, VASP, and Gaussian, preserving all relevant structural information.

DFT Output Processing
-------------------

The DFT output processing module extracts useful data from quantum chemistry output files and converts them to standardized formats.

Key Features
^^^^^^^^^^^

* Extract energies, forces, and coordinates from DFT outputs
* Support for multiple DFT codes (CP2K, VASP, Gaussian)
* Conversion between different unit systems
* Cell parameter processing
* Standardized JSON output format

Core Functions
^^^^^^^^^^^

.. code-block:: python

   def process_dft_output(code_type, **kwargs)

For CP2K outputs:

.. code-block:: python

   def process_cp2k_output(input_file, output_file, frac_xyz_file, output_json, do_frac_to_cart=True)
   def extract_unit_cell_cp2k(input_file)
   def parse_frac_xyz_file_cp2k(xyz_file)
   def frac_to_cart_cp2k(frac_coords, cell_lengths, cell_angles)
   def extract_forces_per_geometry_cp2k(output_file, num_atoms)

For VASP outputs:

.. code-block:: python

   def process_vasp_output(poscar_file, outcar_file, output_json)
   def parse_poscar(poscar_file)
   def extract_energy_forces_vasp(outcar_file)

For Gaussian outputs:

.. code-block:: python

   def process_gaussian_output(log_file, output_json)
   def extract_gaussian_xyz(log_file)
   def extract_energy_gaussian(log_file)
   def extract_forces_gaussian(log_file)

The output processor can handle various unit conversions:

* Hartree to eV (energy)
* Hartree/Bohr to eV/Å (forces)
* Fractional to Cartesian coordinates

ML Dataset Creation
-----------------

The ML dataset conversion module transforms processed DFT data into formats suitable for machine learning potential training.

Key Features
^^^^^^^^^^^

* Convert JSON data to HDF5 format for ML training
* Split data into training and validation sets
* Filter configurations based on force thresholds
* Handle periodic boundary conditions
* Support for common ML potential frameworks like MACE

Core Functions
^^^^^^^^^^^

.. code-block:: python

   def create_mace_h5_dataset(json_file, output_dir=None, dataset_name=None, train_ratio=0.85, 
                             batch_size=4, max_force_threshold=300.0, conversion_factor=1.0, 
                             pbc_handling="auto")

This function performs several essential steps:

1. Loads JSON data containing DFT results
2. Splits data into training and validation sets
3. Filters out configurations with forces exceeding the threshold
4. Converts units if necessary
5. Determines periodic boundary conditions
6. Creates batched HDF5 files in MACE-compatible format

PBC Handling Options
^^^^^^^^^^^^^^^^^^

The ML dataset converter supports flexible handling of periodic boundary conditions:

* **"auto"**: Use PBC based on the presence of cell parameters
* **"always"**: Always use PBC [True, True, True] for configurations with cells
* **"never"**: Always use non-PBC [False, False, False]
* **Directional**: Specify PBC only in certain directions, e.g., "xy" → [True, True, False]

AIMD Processing
------------

The AIMD (Ab Initio Molecular Dynamics) processing module extracts final geometries from trajectories and generates new simulation inputs.

Key Features
^^^^^^^^^^^

* Extract final configurations from AIMD trajectories
* Convert atomic coordinates and cell parameters
* Generate CP2K input files for MD simulations at different temperatures
* Write XYZ files for visualization and further processing

Core Functions
^^^^^^^^^^^

.. code-block:: python

   def process_json_files(json_dir, temperatures, output_dir)
   def extract_final_config(json_file)
   def create_atoms_from_config(config)
   def write_xyz_file(atoms, output_path, comment="XYZ from AIMD processing")
   def write_cp2k_aimd_input(atoms, output_path, temperature)

The module can process multiple AIMD trajectories and generate CP2K input files for simulations at different temperatures, making it easy to study temperature-dependent properties of materials.

Configuration File
^^^^^^^^^^^^^^^^

The AIMD processor uses a YAML configuration file with parameters like:

.. code-block:: yaml

   json_dir: "/path/to/json_files"
   output_dir: "/path/to/aimd_outputs"
   melting_point: 300          # in Kelvin (optional)
   temperatures: [175, 200, 225, 250, 300, 350]

Usage Examples
-----------

CIF to XYZ Conversion
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from multi_agent_dft.file_processing.cif import cif_to_xyz, process_cif_files
   
   # Convert a single CIF file
   cif_to_xyz("structure.cif", "structure.xyz")
   
   # Process all CIF files in a directory
   xyz_files = process_cif_files("structures/", "converted/")

DFT Output Processing
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from multi_agent_dft.file_processing.dft_output_processor import process_dft_output
   
   # Process CP2K output
   process_dft_output(
       code_type="cp2k",
       input_file="input.inp",
       output_file="output.out",
       frac_xyz_file="trajectory.xyz",
       output_json="results.json"
   )
   
   # Process VASP output
   process_dft_output(
       code_type="vasp",
       poscar_file="POSCAR",
       outcar_file="OUTCAR",
       output_json="results.json"
   )

ML Dataset Creation
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from multi_agent_dft.file_processing.ml_dataset_converter import create_mace_h5_dataset
   
   # Create datasets for ML training
   train_h5, valid_h5 = create_mace_h5_dataset(
       json_file="dft_results.json",
       output_dir="ml_data",
       dataset_name="water_system",
       train_ratio=0.8,
       max_force_threshold=200.0,
       pbc_handling="xy"  # 2D periodic system
   )

AIMD Processing
^^^^^^^^^^^^

.. code-block:: python

   from multi_agent_dft.file_processing.aimd_processor import process_json_files
   
   # Process AIMD trajectories and generate CP2K inputs for different temperatures
   process_json_files(
       json_dir="aimd_trajectories/",
       temperatures=[200, 300, 400, 500],
       output_dir="aimd_inputs/"
   )

Integration with Other Components
-------------------------------

The file processing modules integrate seamlessly with other components of the AMLP system:

* **AI-Assisted Research**: Research outputs can guide file processing parameters
* **DFT Input Generation**: Processed structures can be used to generate input files
* **Analysis Pipeline**: Processed outputs can be further analyzed with AMLP-analysis

The modular design ensures that each component can be used independently or as part of a larger workflow, providing flexibility for various research needs.
