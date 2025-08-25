AMLP-analysis
=============

Overview
--------

The AMLP-analysis system (implemented in ``amlpa.py``) provides tools for analyzing atomic structures using machine learning potentials:

Key Features
^^^^^^^^^^^

- Single-point calculations
- Geometry optimization
- Vibrational/phonon analysis
- Molecular dynamics simulations
- Structural analysis (RDF and coordination)

The system uses pre-trained machine learning potentials (like MACE) to perform calculations efficiently.

Basic Usage
^^^^^^^^^^

.. code-block:: bash

   python amlpa.py input.xyz config_file="config.yaml"

Geometry Optimization
--------------------

AMLP-analysis can optimize molecular structures to find minimum energy configurations:

Features
^^^^^^^

- Multiple optimization algorithms (BFGS, LBFGS, FIRE)
- Convergence criteria customization
- Progress tracking
- Energy and force monitoring

Optimizer Options
^^^^^^^^^^^^^^^^

- **BFGS**: Robust general-purpose optimizer
- **LBFGS**: Limited-memory variant for large systems
- **FIRE**: Fast Inertial Relaxation Engine for efficient minimization

Configuration
^^^^^^^^^^^^

.. code-block:: yaml

   # Geometry optimization settings
   geo_opt: True
   optimizer: "BFGS"  # Options: BFGS, LBFGS, FIRE
   fmax: 0.001  # Force convergence criterion
   optimizer_trajectory: "opt.traj"  # Trajectory file name
   optimizer_restart: "restart.pkl"  # Optional restart file

Example Output
^^^^^^^^^^^^^

After optimization, the system generates:

- Optimized structure in XYZ format
- Energy and force information
- Optimization trajectory
- RDF analysis of the optimized structure

Vibrational Analysis
-------------------

The system can calculate vibrational properties and phonon spectra:

Features
^^^^^^^

- Frequency calculations
- Phonon band structure calculation
- Phonon density of states (DOS)
- Full phonon dispersion visualization

Analysis Options
^^^^^^^^^^^^^^^

- Displacement magnitude control
- Supercell size configuration
- Custom k-path selection
- Phonon grid density settings

Configuration
^^^^^^^^^^^^

.. code-block:: yaml

   # Phonon calculation settings
   vib_input: True
   phonon: True
   phonon_delta: 0.01  # Displacement step size
   phonon_supercell: [3, 3, 3]  # Supercell size
   phonon_plot: True
   phonon_grid: [20, 20, 20]  # Grid for DOS
   phonon_npts: 200  # Number of points for DOS
   phonon_width: 1e-3  # Broadening factor

Example Output
^^^^^^^^^^^^^

The vibrational analysis produces:

- Frequency data
- Phonon band structure plots
- Phonon density of states plots
- Full phonon dispersion maps

Molecular Dynamics
-----------------

AMLP-analysis can run molecular dynamics simulations:

Features
^^^^^^^

- Temperature range scanning
- Langevin dynamics
- Trajectory analysis
- Energy and temperature monitoring
- Performance metrics (simulation speed)

MD Parameters
^^^^^^^^^^^^

- Temperature range and step size
- Time step control
- Friction coefficient
- Trajectory saving frequency

Configuration
^^^^^^^^^^^^

.. code-block:: yaml

   # MD simulation settings
   MD_run: True
   Temp_initial: 50  # Starting temperature in K
   Temp_final: 350  # Final temperature in K
   Temp_step: 25  # Temperature increment
   Step: 2000000  # Number of MD steps
   timestep: 0.5  # Time step in fs
   md_friction: 0.01  # Friction coefficient
   MD_save_interval: 5000  # Save trajectory every N steps

Example Output
^^^^^^^^^^^^^

MD simulations generate:

- Trajectory files in XYZ format
- Energy and temperature logs
- RDF analysis for each temperature
- Performance metrics (simulation speed in ns/day)

Structural Analysis
------------------

The system provides tools for analyzing structural properties:

Radial Distribution Function (RDF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RDF calculations can be customized with:

.. code-block:: yaml

   # RDF settings
   rdf_min: 0.0  # Minimum distance
   rdf_rmax: 10.0  # Maximum distance
   rdf_nbins: 100  # Number of bins
   rdf_atom_types: "all"  # Specific atom types or "all"
   rdf_average: True  # Average RDF over the trajectory
   rdf_nframes: 100  # Number of frames to use for averaging
   rdf_smoothing_sigma: 2  # Gaussian smoothing parameter

The RDF calculation supports:

- Custom distance range and binning
- Atom type filtering
- Partial RDFs
- Trajectory-averaged RDFs
- Gaussian smoothing for cleaner plots

Coordination Analysis
^^^^^^^^^^^^^^^^^^^^

Configuration for coordination analysis:

.. code-block:: yaml

   # Coordination analysis settings
   coordination_cutoff: 3.0  # Cutoff distance for neighbors

The coordination analysis provides:

- Distance-based neighbor counting
- Atom type filtering
- Coordination number histograms
- Customizable cutoff distances

Replication Settings
^^^^^^^^^^^^^^^^^^^

For proper structural analysis of periodic systems:

.. code-block:: yaml

   # Cell replication settings
   replicate_dims: [True, True, False]  # Which dimensions to replicate

Features:

- Selective cell replication for 2D systems
- Dimension-specific replication factors
- Margin control for boundary effects

Helper Functions
----------------

AMLP-analysis includes several helper functions for common tasks:

Cell Parameter Handling
^^^^^^^^^^^^^^^^^^^^^^

Functions to:

- Parse XYZ headers for cell information
- Select appropriate band paths for different lattice types
- Apply proper periodic boundary conditions

Spinner for Long Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^

A spinner to provide visual feedback during long-running operations:

.. code-block:: python

   from amlpa import run_with_spinner
   
   # Run a function with a spinner
   result = run_with_spinner(some_long_function, *args, **kwargs)

Configuration
------------

AMLP-analysis is configured through a YAML file:

Basic Settings
^^^^^^^^^^^^^

.. code-block:: yaml

   # Basic settings
   output_dir: "./results"
   plot_dpi: 300
   md_trajectory_dir: "md"
   plot_dir: "plots"
   phonon_dir: "phonon"

Structure Settings
^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # Structure settings
   readcell_info: True  # Read cell from XYZ file header
   cell_params: [5.0, 5.0, 10.0, 90.0, 90.0, 120.0]  # Manual cell parameters if needed
   pbc: True  # Use periodic boundary conditions

Hardware Configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # Hardware settings
   device: "gpu"  # "cpu" or "gpu"
   gpus: ["cuda:0"]  # GPU devices to use
   model_paths:
     - "/path/to/your/model"  # MACE model path

Complete Example
^^^^^^^^^^^^^^^

Here's a complete example configuration file:

.. code-block:: yaml

   # Output directories
   output_dir: "./results"
   plot_dpi: 300
   md_trajectory_dir: "md"
   plot_dir: "plots"
   phonon_dir: "phonon"

   # Structure settings
   readcell_info: True
   pbc: True

   # Hardware settings
   device: "gpu"
   gpus: ["cuda:0"]
   model_paths:
     - "/path/to/your/model"

   # Analysis options
   single_point: True
   geo_opt: True
   optimizer: "BFGS"
   fmax: 0.001

   # Phonon settings
   vib_input: True
   phonon: True
   phonon_delta: 0.01
   phonon_supercell: [3, 3, 3]
   phonon_plot: True

   # MD settings
   MD_run: True
   Temp_initial: 50
   Temp_final: 350
   Temp_step: 25
   Step: 2000000
   timestep: 0.5
   md_friction: 0.01
   MD_save_interval: 5000

   # RDF settings
   rdf_rmax: 10.0
   rdf_nbins: 100
   rdf_atom_types: "all"
   rdf_compare: True
   replicate_dims: [True, True, True]
   rdf_smoothing_sigma: 2

   # Coordination analysis
   coordination_cutoff: 3.0
