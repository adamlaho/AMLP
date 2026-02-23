Agent System
============

The AMLP system uses a sophisticated multi-agent architecture to provide AI-assisted research capabilities. The agent system is organized hierarchically with specialized agents for different aspects of computational chemistry research.

.. contents:: Table of Contents
   :local:
   :depth: 2

Base Agent
---------

The base ``Agent`` class provides the foundation for all specialized agents in the system. It handles communication with large language models (via OpenAI's API) and maintains conversation memory.

Key Features
^^^^^^^^^^^

* API integration with retry logic for resilience
* Conversation memory management
* Flexible configuration options
* Serialization of agent memory

Core Methods
^^^^^^^^^^^

.. code-block:: python

   def chat(self, messages, temperature=None, top_p=None)
   def query(self, user_content, system_content=None)
   def reset_memory()
   def save_memory(file_path)
   def load_memory(file_path)

Usage Example
^^^^^^^^^^^^

.. code-block:: python

   from multi_agent_dft.agents.base import Agent
   
   # Create a basic agent
   agent = Agent(name="ResearchAssistant", model="gpt-4")
   
   # Simple query
   response = agent.query(
       user_content="What are hybrid functionals in DFT?",
       system_content="You are a computational chemistry expert."
   )
   print(response)

Chemistry Agents
--------------

The chemistry agents specialize in analyzing research literature and providing domain-specific insights. Two complementary agents offer different perspectives:

1. ``ExperimentalChemistAgent``: Focuses on laboratory techniques, synthesis, and characterization
2. ``TheoreticalChemistAgent``: Focuses on computational methods and theoretical frameworks

Key Features
^^^^^^^^^^^

* Publication search and retrieval
* Ranking algorithms for relevance
* Enhanced search strategies using multiple queries
* Publication analysis with domain-specific focus
* Citation tracking in generated summaries

Research Workflow
^^^^^^^^^^^^^^^^

The chemistry agents follow a sophisticated research workflow:

1. Extract key terms from the research query
2. Generate multiple targeted search queries
3. Retrieve and deduplicate publications
4. Rank publications by relevance using custom scoring
5. Analyze top publications, focusing on domain-specific aspects
6. Generate comprehensive reports with proper citations

Key Methods
^^^^^^^^^^^

.. code-block:: python

   def summarize(self, user_query, additional_context="", max_results=15)
   def get_detailed_information(self, specific_topic, max_results=10)
   def _rank_publications_by_relevance(self, publications, query)
   def _generate_enhanced_summary(self, user_query, additional_context, report)

Experimental Chemist Agent
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``ExperimentalChemistAgent`` specializes in experimental aspects of chemistry research:

.. code-block:: python

   from multi_agent_dft.agents.chemistry_agents import ExperimentalChemistAgent
   
   # Initialize the agent
   exp_agent = ExperimentalChemistAgent()
   
   # Generate a summary of experimental aspects
   summary = exp_agent.summarize("Metal oxide catalysts for water splitting")
   print(summary)

This agent focuses on keywords like "synthesis," "characterization," "spectroscopy," and other experimental techniques. Its summaries emphasize laboratory procedures, experimental conditions, and characterization methods.

Theoretical Chemist Agent
^^^^^^^^^^^^^^^^^^^^^^^^

The ``TheoreticalChemistAgent`` specializes in computational and theoretical aspects:

.. code-block:: python

   from multi_agent_dft.agents.chemistry_agents import TheoreticalChemistAgent
   
   # Initialize the agent
   theo_agent = TheoreticalChemistAgent()
   
   # Generate a summary of theoretical aspects
   summary = theo_agent.summarize("Metal oxide catalysts for water splitting")
   print(summary)

This agent focuses on keywords like "DFT," "simulation," "calculation," and other computational methods. Its summaries emphasize theoretical frameworks, computational methods, and predictive modeling approaches.

DFT Expert Agents
---------------

The DFT expert agents provide specialized knowledge about specific quantum chemistry software packages. Three expert agents are implemented:

1. ``GaussianExpertAgent``: Expert in Gaussian software
2. ``VASPExpertAgent``: Expert in Vienna Ab initio Simulation Package (VASP)
3. ``CP2KExpertAgent``: Expert in CP2K software

These agents draw on software documentation and published literature to provide actionable recommendations for DFT calculations.

Key Features
^^^^^^^^^^^

* Software-specific knowledge base
* Documentation retrieval and caching
* Publication search with software-specific focus
* Parameter extraction from literature
* Technical analysis generation

Base DFT Expert Agent
^^^^^^^^^^^^^^^^^^^

The ``DFTExpertAgent`` class provides common functionality for all DFT expert agents:

.. code-block:: python

   def analyze(self, refined_query)
   def get_documentation()
   def _fetch_literature_parameters(self, system)

These methods enable the agents to:
1. Retrieve software documentation
2. Search for relevant publications
3. Extract parameters from scientific literature
4. Generate comprehensive technical analyses

Specialized DFT Agents
^^^^^^^^^^^^^^^^^^^^

Each specialized agent focuses on a particular DFT code:

**Gaussian Expert Agent**

.. code-block:: python

   from multi_agent_dft.agents.dft_agents import GaussianExpertAgent
   
   # Initialize the agent
   gaussian_agent = GaussianExpertAgent()
   
   # Generate analysis for a specific research question
   analysis = gaussian_agent.analyze("Reaction mechanism of CO2 reduction")
   print(analysis)

The Gaussian expert focuses on molecular systems, focusing on methods, basis sets, and keywords for specific chemical properties.

**VASP Expert Agent**

.. code-block:: python

   from multi_agent_dft.agents.dft_agents import VASPExpertAgent
   
   # Initialize the agent
   vasp_agent = VASPExpertAgent()
   
   # Generate analysis for a specific research question
   analysis = vasp_agent.analyze("Band structure of 2D materials")
   print(analysis)

The VASP expert focuses on periodic systems, with emphasis on INCAR parameters, pseudopotentials, and k-point sampling.

**CP2K Expert Agent**

.. code-block:: python

   from multi_agent_dft.agents.dft_agents import CP2KExpertAgent
   
   # Initialize the agent
   cp2k_agent = CP2KExpertAgent()
   
   # Generate analysis for a specific research question
   analysis = cp2k_agent.analyze("Ab initio molecular dynamics of water")
   print(analysis)

The CP2K expert focuses on mixed Gaussian and plane wave methods, with emphasis on input file structure and convergence strategies.

Supervisor Agent
--------------

The ``SupervisorAgent`` integrates information from multiple sources and provides synthesized recommendations. It can serve different roles based on its configuration:

1. **Integration Supervisor**: Synthesizes information from different expert agents
2. **DFT Recommendation Supervisor**: Evaluates input from DFT experts and provides software recommendations

Key Features
^^^^^^^^^^^

* Flexible role-based functionality
* Enhanced model capabilities (uses premium models when available)
* Integration of diverse expert perspectives
* Follow-up question generation for research refinement

Key Methods
^^^^^^^^^^^

.. code-block:: python

   def integrate(self, content, additional_context="")
   def generate_followup_question(self, research_query, max_results=5)

Usage Example
^^^^^^^^^^^^

.. code-block:: python

   from multi_agent_dft.agents.supervisor import SupervisorAgent
   
   # Create integration supervisor
   integration_supervisor = SupervisorAgent(role="Integration")
   
   # Integrate content from multiple experts
   integrated_report = integration_supervisor.integrate(
       f"Experimental Report:\n{exp_content}\n\nTheoretical Report:\n{theo_content}"
   )
   print(integrated_report)
   
   # Create DFT recommendation supervisor
   dft_supervisor = SupervisorAgent(role="DFT_Recommendation")
   
   # Get DFT recommendations
   dft_recommendations = dft_supervisor.integrate(
       f"Gaussian Analysis:\n{gaussian_content}\n\nVASP Analysis:\n{vasp_content}\n\nCP2K Analysis:\n{cp2k_content}"
   )
   print(dft_recommendations)

Advanced Features
--------------

Agent Communication
^^^^^^^^^^^^^^^^^

Agents can communicate with each other through the supervisor agents, which integrate and synthesize information from multiple sources:

.. code-block:: python

   # Multi-agent workflow
   query = "Metal oxide catalysts for water splitting"
   exp_summary = exp_agent.summarize(query)
   theo_summary = theo_agent.summarize(query)
   
   integrated_content = f"Experimental:\n{exp_summary}\n\nTheoretical:\n{theo_summary}"
   integrated_report = integration_supervisor.integrate(integrated_content)
   
   gaussian_analysis = gaussian_agent.analyze(query)
   vasp_analysis = vasp_agent.analyze(query)
   cp2k_analysis = cp2k_agent.analyze(query)
   
   dft_content = f"Gaussian:\n{gaussian_analysis}\n\nVASP:\n{vasp_analysis}\n\nCP2K:\n{cp2k_analysis}"
   dft_recommendations = dft_supervisor.integrate(dft_content)

Memory Management
^^^^^^^^^^^^^^^

Each agent maintains a memory of conversation history, which can be saved and loaded:

.. code-block:: python

   # Save agent memory
   exp_agent.save_memory("experimental_agent_memory.json")
   
   # Load agent memory
   new_exp_agent = ExperimentalChemistAgent()
   new_exp_agent.load_memory("experimental_agent_memory.json")

Configuration
^^^^^^^^^^^

Agents can be configured with custom parameters:

.. code-block:: python

   config = {
       "agents": {
           "models": {
               "default": "gpt-4",
               "premium": "gpt-4-turbo"
           },
           "experimental_chemist": {
               "focus_keywords": ["synthesis", "characterization", "experiment"]
           },
           "theoretical_chemist": {
               "focus_keywords": ["calculation", "DFT", "simulation"]
           },
           "dft_experts": {
               "gaussian": {
                   "doc_url": "https://gaussian.com/capabilities/",
                   "keywords": ["gaussian", "g16", "g09"]
               }
           }
       }
   }
   
   # Create agent with custom config
   custom_exp_agent = ExperimentalChemistAgent(config=config)
