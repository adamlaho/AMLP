Welcome to AMLP's documentation!
===============================


AMLP (Automated Machine Learning Potential) is a comprehensive AI-enhanced computational chemistry platform that seamlessly integrates multi-agent artificial intelligence with advanced machine learning for atomic structure analysis. By combining the Multi-Agent DFT Research System with the AML-analysis framework, we provide a complete workflow for computational chemists - from literature review to structure analysis. The platform features specialized AI agents that analyze research literature, provide expert recommendations for DFT calculations using various codes (Gaussian, VASP, CP2K), and automate the generation of input files. The integrated AML-analysis component leverages machine learning potentials to perform rapid and accurate atomic structure analysis, including geometry optimization, vibrational analysis, molecular dynamics simulations, and structural property calculations. This end-to-end solution allows initial research exploration and structural analysis, enabling researchers to explore new materials and chemical systems with computational methods backed by artificial intelligence.


.. image:: https://img.shields.io/badge/python-3.7+-blue.svg
   :target: https://www.python.org/downloads/

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   getting-started
   amlp-training
   amlp-analysis
   agents
   publication-api
   file-processing

Key Tools & Features
-------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Tool
     - Description
   * - **AI Research Agents**
     - Specialized AI agents to analyze research topics and generate summaries
   * - **DFT Input Generation**
     - Automated input file creation for CP2K, VASP, and Gaussian
   * - **Structure Conversion**
     - Tools for converting between different structure file formats
   * - **Supercell Creation**
     - Build supercells with custom dimensions for periodic systems
   * - **MD Simulation**
     - Run molecular dynamics simulations with ML potentials
   * - **Vibrational Analysis**
     - Calculate vibrational properties and phonon spectra
   * - **Structural Analysis**
     - Analyze RDF and coordination environments
   * - **ML Dataset Creation**
     - Convert DFT data to machine learning training formats
   * - **AIMD Processing**
     - Generate AIMD inputs for multiple temperatures

Supported Software
-----------------

.. grid:: 3

    .. grid-item-card:: CP2K
        :img-top: _static/cp2k_logo.png
        :text-align: center
        
        Input generation for the CP2K quantum chemistry package
        
    .. grid-item-card:: VASP
        :img-top: _static/vasp_logo.png
        :text-align: center
        
        Input generation for the Vienna Ab initio Simulation Package
        
    .. grid-item-card:: Gaussian
        :img-top: _static/gaussian_logo.png
        :text-align: center
        
        Input generation for Gaussian quantum chemistry software

.. grid:: 3

    .. grid-item-card:: MACE
        :img-top: _static/mace_logo.png
        :text-align: center
        
        Integration with MACE machine learning potentials
        
    .. grid-item-card:: ASE
        :img-top: _static/ase_logo.png
        :text-align: center
        
        Built on the Atomic Simulation Environment
        
    .. grid-item-card:: OpenAI
        :img-top: _static/openai_logo.png
        :text-align: center
        
        Powered by OpenAI's language models

Quick Links
----------

* `GitHub Repository <https://github.com/adamlaho/AMLP/>`_
* `Report Issues <https://github.com/adamlaho/AMLP/issues>`_
* `Change Log <https://github.com/adamlaho/AMLP/blob/main/CHANGELOG.md>`_

Indices and tables
=================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

