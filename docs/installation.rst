Installation
============

The AMLP package is available on GitHub at `https://github.com/adamlaho/AMLP <https://github.com/adamlaho/AMLP>`_.

Requirements
-----------

AMLP requires:

* Python 3.7+ (Python 3.9 recommended)
* CUDA-compatible GPU (recommended for ML potentials)

Core Dependencies
----------------

* numpy
* matplotlib
* pyyaml
* torch
* tqdm
* scipy
* ase (Atomic Simulation Environment)
* mace-torch (for ML potentials)
* openai (for AI agent functionality)

Installation Steps
-----------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/adamlaho/AMLP
      cd AMLP


2. Install dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

3. (Optional) Set up OpenAI API key for AI agent functionality:

   .. code-block:: bash

      # On Linux/macOS
      export OPENAI_API_KEY="your-api-key-here"
      
      # On Windows
      set OPENAI_API_KEY=your-api-key-here

Development Installation
-----------------------

For development purposes, you can install the package in editable mode:

.. code-block:: bash

   git clone https://github.com/username/AMLP.git
   cd AMLP
   pip install -e .

Verifying the Installation
-------------------------

To verify that the installation works correctly, run:

.. code-block:: bash

   # For AMLP-training
   python amlpt.py --help
   
   # For AMLP-analysis (requires a structure file and config)
   python amlpa.py --help

Getting Help
-----------

If you encounter any issues during installation:

* Check the `GitHub Issues <https://github.com/adamlaho/AMLP/issues>`_ for known problems
* Open a new issue if your problem hasn't been reported
* Consult the :doc:`getting-started` guide for additional help
