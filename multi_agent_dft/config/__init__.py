# multi_agent_dft/config/__init__.py
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """Manages configuration settings for the multi-agent DFT system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to a YAML configuration file. If None, the default configuration is used.
        """
        self.config_dir = Path(__file__).parent.resolve()
        self.default_config_path = self.config_dir / "default_config.yaml"
        self.user_config_path = Path(config_path) if config_path else None
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from default and user-specified files."""
        # Load default configuration
        try:
            if self.default_config_path.exists():
                with open(self.default_config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                # If default config doesn't exist, create a minimal default config
                config = {
                    "api": {
                        "openai": {
                            "api_key": "",
                            "model": "gpt-3.5-turbo",
                            "temperature": 0.7,
                            "max_tokens": 1000
                        },
                        "publication": {
                            "api_key": "",
                            "cache_ttl": 86400
                        }
                    },
                    "agents": {
                        "base": {
                            "memory_size": 10
                        },
                        "experimental_chemist": {
                            "focus_keywords": ["experiment", "synthesis", "measurement"]
                        },
                        "theoretical_chemist": {
                            "focus_keywords": ["theory", "simulation", "DFT", "model"]
                        },
                        "dft_experts": {
                            "gaussian": {
                                "doc_url": "https://gaussian.com/man/",
                                "keywords": ["Gaussian", "DFT", "quantum chemistry"]
                            },
                            "vasp": {
                                "doc_url": "https://www.vasp.at/wiki/index.php/The_VASP_Manual",
                                "keywords": ["VASP", "DFT", "periodic", "crystal"]
                            },
                            "cp2k": {
                                "doc_url": "https://www.cp2k.org/documentation",
                                "keywords": ["CP2K", "DFT", "AIMD", "mixed basis"]
                            }
                        }
                    },
                    "dft": {
                        "default_output_dir": "./output",
                        "structure_validation": {
                            "allowed_extensions": [".xyz", ".cif"]
                        }
                    },
                    "logging": {
                        "level": "INFO",
                        "console": True
                    }
                }
                
                # Create the config directory if it doesn't exist
                self.default_config_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write the default config
                with open(self.default_config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            print(f"Error loading default configuration: {str(e)}")
            config = {}
        
        # Override with user configuration if provided
        if self.user_config_path and self.user_config_path.exists():
            try:
                with open(self.user_config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    self._deep_update(config, user_config)
            except Exception as e:
                print(f"Error loading user configuration: {str(e)}")
        
        # Override with environment variables
        self._update_from_env(config)
        
        return config
    
    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        """Recursively update a dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
    
    def _update_from_env(self, config: Dict[str, Any]) -> None:
        """Update configuration with environment variables."""
        # Ensure the config has the required structure
        if 'api' not in config:
            config['api'] = {}
        if 'openai' not in config['api']:
            config['api']['openai'] = {}
        if 'publication' not in config['api']:
            config['api']['publication'] = {}
            
        # API keys
        if 'OPENAI_API_KEY' in os.environ:
            config['api']['openai']['api_key'] = os.environ['OPENAI_API_KEY']
        
        if 'PUBLICATION_API_KEY' in os.environ:
            config['api']['publication']['api_key'] = os.environ['PUBLICATION_API_KEY']
        
        # Other environment variables can be added here
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated path to the configuration key (e.g., 'api.openai.model')
            default: Default value to return if the key is not found
            
        Returns:
            The configuration value or the default value if the key is not found
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a complete configuration section.
        
        Args:
            section: Top-level section name
            
        Returns:
            The configuration section as a dictionary
        """
        return self.config.get(section, {})

# Create a singleton instance
config_manager = ConfigManager()

# Helper function to get configuration values
def get_config(key_path: str, default: Any = None) -> Any:
    """Get a configuration value by dot-separated path."""
    return config_manager.get(key_path, default)