Publication API
==============

The Publication API is a core component of the AMLP-training system that enables agents to search for and analyze scientific literature. This API provides functionality for retrieving publications, extracting relevant information, and generating reports.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The ``PublicationAPI`` provides methods for:

1. Searching scientific literature based on queries
2. Retrieving full-text content when available
3. Analyzing publications for trends and key information
4. Extracting parameters from publications
5. Generating formatted reports

This API is used by the Chemistry Agents and DFT Expert Agents to provide research-backed insights and recommendations.

Key Features
-----------

Publication Search
^^^^^^^^^^^^^^^^^

The API can search for publications using various backend services:

.. code-block:: python

   def search(self, query, max_results=10, service="default")

This method returns a list of publication dictionaries with fields like:

- ``title``: Publication title
- ``authors``: List of authors
- ``journal``: Journal or conference name
- ``year``: Publication year
- ``abstract``: Abstract text
- ``url``: Link to the publication
- ``doi``: Digital Object Identifier
- ``id``: Unique identifier

Example:

.. code-block:: python

   from multi_agent_dft.api.publication import PublicationAPI
   
   pub_api = PublicationAPI()
   publications = pub_api.search("metal oxide catalysts water splitting", max_results=5)
   for pub in publications:
       print(f"Title: {pub['title']}")
       print(f"Authors: {', '.join(pub['authors'])}")
       print(f"Year: {pub['year']}")
       print("---")

Full-Text Retrieval
^^^^^^^^^^^^^^^^^

For more detailed analysis, the API can retrieve full text when available:

.. code-block:: python

   def get_full_text(self, publication)

This method attempts to retrieve the full text of a publication from various sources. If successful, it returns the text content; otherwise, it returns ``None``.

Publication Analysis
^^^^^^^^^^^^^^^^^

The API provides methods for analyzing sets of publications:

.. code-block:: python

   def analyze_publications(self, publications, focus_keywords=None)

This method analyzes publications for trends including:

- Most common keywords
- Temporal trends (publication years)
- Citation patterns
- Focus keyword relevance
- Topic modeling

Example:

.. code-block:: python

   publications = pub_api.search("DFT metal oxide catalysts", max_results=10)
   analysis = pub_api.analyze_publications(
       publications, 
       focus_keywords=["water splitting", "photocatalysis", "oxygen evolution"]
   )
   
   # Access analysis results
   print(f"Most common keywords: {analysis['keyword_analysis']['most_common']}")
   print(f"Year distribution: {analysis['temporal_analysis']['year_distribution']}")

Parameter Extraction
^^^^^^^^^^^^^^^^^

A powerful feature of the API is the ability to extract parameters from publications:

.. code-block:: python

   def extract_parameters(self, publication, code_name=None, system_type=None)

This method identifies and extracts computational parameters from publication text, including:

- Functional types (PBE, B3LYP, etc.)
- Basis sets
- Energy cutoffs
- k-point meshes
- Convergence criteria
- System-specific parameters

Example:

.. code-block:: python

   # Extract DFT parameters related to water splitting
   publications = pub_api.search("water splitting DFT VASP", max_results=5)
   for pub in publications:
       params = pub_api.extract_parameters(pub, code_name="vasp", system_type="water")
       if params:
           print(f"Publication: {pub['title']}")
           for param in params:
               print(f"  • {param['param_name']}: {param['param_value']}")

Report Generation
^^^^^^^^^^^^^^^

The API can generate formatted reports from publications and analysis:

.. code-block:: python

   def generate_report(self, publications, analysis=None)

This method creates a structured report with sections for:

- Summary of search results
- Key publications (sorted by relevance)
- Publication trends
- Key findings
- Recommendations based on literature

Example:

.. code-block:: python

   publications = pub_api.search("metal oxide catalysts water splitting", max_results=10)
   analysis = pub_api.analyze_publications(publications)
   report = pub_api.generate_report(publications, analysis)
   
   # Print or save the report
   print(report)
   with open("literature_report.md", "w") as f:
       f.write(report)

Implementation Details
--------------------

Search Backends
^^^^^^^^^^^^^

The PublicationAPI supports multiple search backends with graceful fallback:

1. **Academic APIs**: Integration with academic search APIs when credentials are provided
2. **Open Sources**: Access to open access databases and repositories
3. **Web Scraping**: Limited scraping capabilities for public information
4. **Synthetic Data**: Generation of synthetic literature data for testing or when other methods fail

The search backend selection is determined by the available credentials and the ``service`` parameter.

Text Processing
^^^^^^^^^^^^^

The API includes sophisticated text processing capabilities:

1. **Named Entity Recognition**: Identifying chemical compounds, methods, and parameters
2. **Relation Extraction**: Connecting parameters with their values
3. **Topic Modeling**: Identifying key research themes
4. **Tokenization**: Breaking text into meaningful units for analysis

These capabilities enable the API to extract structured information from unstructured text.

Parameter Discovery
^^^^^^^^^^^^^^^^^

The parameter extraction capability uses a combination of:

1. **Pattern Matching**: Regular expressions for common parameter formats
2. **Contextual Analysis**: Understanding parameters in the context of methods
3. **Knowledge Base**: Domain-specific knowledge of typical parameters
4. **Semantic Analysis**: Understanding parameter relationships

This multi-strategy approach improves the accuracy of parameter extraction.

Configuration
-----------

The PublicationAPI can be configured with various options:

.. code-block:: python

   config = {
       "publication_api": {
           "search_services": {
               "default": "open_sources",
               "fallbacks": ["web_scraping", "synthetic"]
           },
           "credentials": {
               "academic_api_key": "your_api_key_here"
           },
           "cache_enabled": True,
           "cache_dir": "./cache",
           "text_processing": {
               "max_tokens": 1000,
               "quality": "high"
           }
       }
   }
   
   pub_api = PublicationAPI(config=config)

Integration with Agents
---------------------

The PublicationAPI is used by several agent types:

**Chemistry Agents**:
   - Search for publications related to research queries
   - Analyze publication trends
   - Generate comprehensive summaries

**DFT Expert Agents**:
   - Extract computational parameters from literature
   - Identify best practices for specific systems
   - Generate parameter recommendations based on published work

**Supervisor Agent**:
   - Use publication data to generate follow-up questions
   - Integrate findings from literature into recommendations

Example Workflow
^^^^^^^^^^^^^

Here's a typical workflow using the PublicationAPI within the agent system:

1. User submits a research query about metal oxide catalysts
2. Chemistry agents use PublicationAPI to search for relevant literature
3. PublicationAPI retrieves and analyzes publications
4. Chemistry agents generate summaries based on the analysis
5. DFT Expert agents use PublicationAPI to extract computational parameters
6. DFT Expert agents recommend simulation approaches based on literature
7. Supervisor agent integrates all information into a comprehensive report

This workflow demonstrates how the PublicationAPI enables evidence-based recommendations throughout the system.
