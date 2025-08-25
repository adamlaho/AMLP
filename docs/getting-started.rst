Getting Started
===============

Quick Start
----------

AMLP-training
^^^^^^^^^^^^^

To launch the Multi-Agent DFT Research System:

.. code-block:: bash

   python amlpt.py

This will present a menu with five operation modes:

1. AI-agent feedback (research summaries & reports)
2. Input generation (CP2K/VASP/Gaussian)
3. Output processing (extract forces, energies, coordinates)
4. ML potential dataset creation (JSON to MACE HDF5)
5. AIMD processing (JSON to CP2K AIMD inputs)

Follow the interactive prompts to configure your workflow.

AMLP-analysis
^^^^^^^^^^^^

To analyze a structure using ML potentials:

.. code-block:: bash

   python amlpa.py input.xyz config_file="config.yaml"

Make sure your configuration file is properly set up for your analysis needs.

Example Workflows
----------------

Research to Simulation
^^^^^^^^^^^^^^^^^^^^^

1. Use AMLP-training to research a topic:

   .. code-block:: bash

      python amlpt.py
      # Select option 1: AI-agent feedback

2. Generate DFT inputs based on recommendations:

   .. code-block:: bash

      python amlpt.py
      # Select option 2: Input generation

3. Run DFT calculations (using external software)

4. Process outputs with AMLP-training:

   .. code-block:: bash

      python amlpt.py
      # Select option 3: Output processing

5. Create ML datasets:

   .. code-block:: bash

      python amlpt.py
      # Select option 4: ML potential dataset creation

Structure Analysis
^^^^^^^^^^^^^^^^^

1. Prepare a configuration file (config.yaml):

   .. code-block:: yaml

      output_dir: "./results"
      plot_dpi: 300
      
      # Structure settings
      readcell_info: True
      pbc: True
      
      # Analysis options
      geo_opt: True
      optimizer: "BFGS"
      
      # ML model
      device: "gpu"
      model_paths:
        - "/path/to/your/model"

2. Run AMLP-analysis with your structure:

   .. code-block:: bash

      python amlpa.py structure.xyz config_file="config.yaml"

3. Examine the results in the output directory

Using API Integration
^^^^^^^^^^^^^^^^^^^^

For programmatic usage in your own Python scripts:

.. code-block:: python

   from amlpt import MultiAgentSystem

   # Initialize the system
   system = MultiAgentSystem()

   # Process a specific research query
   system.process_query("Metal oxide catalysts for water splitting")

   # Generate inputs for a structure file
   from pathlib import Path
   system._process_single_cp2k_file(Path("structure.cif"), Path("./outputs"))
