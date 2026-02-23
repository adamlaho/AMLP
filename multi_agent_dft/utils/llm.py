import logging
from typing import Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

class LLMModel:
    """Base class for LLM models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM model.
        
        Args:
            config: Configuration dictionary for the model
        """
        self.config = config or {}
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate text from the model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        raise NotImplementedError("Subclasses must implement this method")


class DummyLLMModel(LLMModel):
    """
    Dummy LLM model for testing purposes.
    
    This model returns a fixed response for parameter extraction.
    """
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate a dummy response.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        logger.info(f"DummyLLM received prompt: {prompt[:100]}...")
        
        # Check if this is a parameter extraction prompt
        if "Extract DFT" in prompt and "parameters" in prompt:
            # Determine the DFT code type from the prompt
            if "GAUSSIAN" in prompt.upper():
                return """
                [
                    {
                        "param_name": "functional",
                        "param_value": "B3LYP",
                        "context": "geometry optimization"
                    },
                    {
                        "param_name": "basis set",
                        "param_value": "6-31G(d)",
                        "context": "for non-metal atoms"
                    }
                ]
                """
            elif "VASP" in prompt.upper():
                return """
                [
                    {
                        "param_name": "ENCUT",
                        "param_value": "500 eV",
                        "context": "plane wave cutoff"
                    },
                    {
                        "param_name": "KPOINTS",
                        "param_value": "4x4x4",
                        "context": "Monkhorst-Pack grid"
                    }
                ]
                """
            elif "CP2K" in prompt.upper():
                return """
                [
                    {
                        "param_name": "basis set",
                        "param_value": "DZVP-MOLOPT-SR-GTH",
                        "context": "for all atoms"
                    },
                    {
                        "param_name": "functional",
                        "param_value": "PBE",
                        "context": "exchange-correlation"
                    }
                ]
                """
            else:
                return "[]"  # Empty parameters if code not recognized
        
        # Default response for other types of prompts
        return "This is a placeholder response from the dummy LLM model."


def get_llm_model(model_name: str = "default_model") -> LLMModel:
    """
    Get a LLM model instance by name.
    
    Args:
        model_name: Name of the model to get
        
    Returns:
        LLM model instance
    """
    # For now, just return the dummy model regardless of the name
    logger.info(f"Creating LLM model: {model_name}")
    return DummyLLMModel()