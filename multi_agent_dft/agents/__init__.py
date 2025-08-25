"""
Agent modules for the Multi-Agent DFT Research System.

This package contains various agent implementations for different roles
in the research workflow, including experimental chemists, theoretical chemists,
DFT experts, and supervisor agents.
"""

# Import the base agent directly to avoid circular imports
from multi_agent_dft.agents.base import Agent

# Uncomment these imports when the modules are properly implemented
from multi_agent_dft.agents.chemistry_agents import ExperimentalChemistAgent, TheoreticalChemistAgent
from multi_agent_dft.agents.dft_agents import DFTExpertAgent
from multi_agent_dft.agents.supervisor import SupervisorAgent

__all__ = [
    'Agent',
    'ExperimentalChemistAgent',
    'TheoreticalChemistAgent',
    'DFTExpertAgent',
    'SupervisorAgent'
]