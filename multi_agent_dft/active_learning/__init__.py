# multi_agent_dft/active_learning/__init__.py
"""
Active Learning module for the Multi-Agent DFT Research System.

This package provides functionality for:
- Active Learning loop orchestration
- Foundation model management
- Dataset quality analysis
- Model validation and testing
- Failed configuration tracking
"""

from multi_agent_dft.active_learning.workflow_manager import ActiveLearningWorkflow
from multi_agent_dft.active_learning.foundation_models import FoundationModelManager
from multi_agent_dft.active_learning.dataset_analyzer import DatasetQualityAnalyzer
from multi_agent_dft.active_learning.model_validator import ModelValidator
from multi_agent_dft.active_learning.config_tracker import ConfigurationTracker

__all__ = [
    'ActiveLearningWorkflow',
    'FoundationModelManager',
    'DatasetQualityAnalyzer',
    'ModelValidator',
    'ConfigurationTracker'
]
