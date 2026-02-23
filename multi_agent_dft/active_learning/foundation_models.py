"""
Foundation Model Manager for MACE-MP and other foundation models.

This module handles:
- Downloading and caching foundation models
- Model selection and configuration
- Running MD/AIMD with foundation models
- Extracting configurations from MD trajectories
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FoundationModel:
    """Data class for foundation model information."""
    name: str
    url: str
    description: str
    elements: List[str]
    size: str  # small, medium, large
    architecture: str  # MACE, etc.

class FoundationModelManager:
    """
    Manager for foundation models from ACEsuit/mace-foundations.

    Handles downloading, caching, and configuring foundation models
    for use in Active Learning workflows.
    """

    MACE_FOUNDATION_MODELS = {
        'mace-mp-0-small': {
            'url': 'https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model',
            'description': 'MACE-MP-0 Small (128 hidden dims, Materials Project trained)',
            'elements': 'All elements up to Z=94',
            'size': 'small',
            'architecture': 'MACE'
        },
        'mace-mp-0-medium': {
            'url': 'https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model',
            'description': 'MACE-MP-0 Medium (128 hidden dims, L1, Materials Project trained)',
            'elements': 'All elements up to Z=94',
            'size': 'medium',
            'architecture': 'MACE'
        },
        'mace-mp-0-large': {
            'url': 'https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-256-L1_epoch-199.model',
            'description': 'MACE-MP-0 Large (256 hidden dims, L1, Materials Project trained)',
            'elements': 'All elements up to Z=94',
            'size': 'large',
            'architecture': 'MACE'
        },
        'mace-mp-0-medium-mpa': {
            'url': 'https://github.com/ACEsuit/mace-mp/releases/download/mace-universal-finetuned/2024-08-10-mace-universal.model',
            'description': 'MACE-MP-0 Medium MPA (Multi-purpose architecture, universal)',
            'elements': 'All elements up to Z=94',
            'size': 'medium',
            'architecture': 'MACE'
        },
        'mace-off-2023': {
            'url': 'https://github.com/ACEsuit/mace-off/releases/download/mace_off_2023/MACE-OFF23_medium.model',
            'description': 'MACE-OFF-2023 Medium (Organic molecules, H, C, N, O)',
            'elements': ['H', 'C', 'N', 'O'],
            'size': 'medium',
            'architecture': 'MACE'
        },
        'mace-off-2023-small': {
            'url': 'https://github.com/ACEsuit/mace-off/releases/download/mace_off_2023/MACE-OFF23_small.model',
            'description': 'MACE-OFF-2023 Small (Organic molecules, H, C, N, O)',
            'elements': ['H', 'C', 'N', 'O'],
            'size': 'small',
            'architecture': 'MACE'
        },
        'mace-off-2023-large': {
            'url': 'https://github.com/ACEsuit/mace-off/releases/download/mace_off_2023/MACE-OFF23_large.model',
            'description': 'MACE-OFF-2023 Large (Organic molecules, H, C, N, O)',
            'elements': ['H', 'C', 'N', 'O'],
            'size': 'large',
            'architecture': 'MACE'
        }
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the Foundation Model Manager.

        Args:
            cache_dir: Directory to cache downloaded models.
                      Defaults to ~/.cache/mace-foundations/
        """
        if cache_dir is None:
            cache_dir = Path.home() / '.cache' / 'mace-foundations'

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.models_info = self.MACE_FOUNDATION_MODELS.copy()
        self.downloaded_models = {}

        self._scan_cache()

    def _scan_cache(self):
        """Scan cache directory for already downloaded models."""
        for model_name in self.models_info.keys():
            model_path = self.cache_dir / f"{model_name}.model"
            if model_path.exists():
                self.downloaded_models[model_name] = model_path
                logger.info(f"Found cached model: {model_name}")

    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available foundation models.

        Returns:
            Dictionary of model names to their information.
        """
        return self.models_info.copy()

    def print_available_models(self):
        """Print formatted list of available models."""
        print("\n" + "="*80)
        print("Available Foundation Models".center(80))
        print("="*80 + "\n")

        for idx, (name, info) in enumerate(self.models_info.items(), 1):
            cached = " [CACHED]" if name in self.downloaded_models else ""
            print(f"{idx}. {name}{cached}")
            print(f"   Description: {info['description']}")
            print(f"   Size: {info['size']}")
            print(f"   Elements: {info['elements']}")
            print()

    def download_model(self, model_name: str, force: bool = False) -> Path:
        """
        Download a foundation model if not already cached.

        Args:
            model_name: Name of the model to download
            force: Force re-download even if cached

        Returns:
            Path to the downloaded model file
        """
        if model_name not in self.models_info:
            raise ValueError(f"Unknown model: {model_name}. Use list_available_models() to see options.")

        model_path = self.cache_dir / f"{model_name}.model"

        # Check if already downloaded
        if model_path.exists() and not force:
            logger.info(f"Model {model_name} already cached at {model_path}")
            self.downloaded_models[model_name] = model_path
            return model_path

        # Download the model
        url = self.models_info[model_name]['url']
        logger.info(f"Downloading {model_name} from {url}...")
        print(f"Downloading {model_name}...")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0

            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rProgress: {progress:.1f}%", end='', flush=True)

            print("\n✓ Download complete!")
            self.downloaded_models[model_name] = model_path
            return model_path

        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            if model_path.exists():
                model_path.unlink()  # Remove partial download
            raise

    def get_model_path(self, model_name: str, auto_download: bool = True) -> Path:
        """
        Get the path to a model, downloading if necessary.

        Args:
            model_name: Name of the model
            auto_download: Automatically download if not cached

        Returns:
            Path to the model file
        """
        if model_name in self.downloaded_models:
            return self.downloaded_models[model_name]

        if auto_download:
            return self.download_model(model_name)
        else:
            raise FileNotFoundError(f"Model {model_name} not in cache. Use download_model() first.")

    def select_models_interactive(self) -> List[str]:
        """
        Interactive selection of foundation models.

        Returns:
            List of selected model names
        """
        self.print_available_models()

        print("="*80)
        print("Select foundation model(s) for Active Learning")
        print("Enter model numbers separated by commas (e.g., 1,3,5)")
        print("Or enter model names separated by commas")
        print("="*80)

        selection = input("\nYour selection: ").strip()

        selected_models = []
        model_list = list(self.models_info.keys())

        for item in selection.split(','):
            item = item.strip()

            # Try to parse as number
            try:
                idx = int(item) - 1
                if 0 <= idx < len(model_list):
                    selected_models.append(model_list[idx])
                else:
                    print(f"Warning: Invalid index {item}")
            except ValueError:
                # Try as model name
                if item in self.models_info:
                    selected_models.append(item)
                else:
                    print(f"Warning: Unknown model {item}")

        if not selected_models:
            print("No models selected. Using default: mace-mp-0-medium")
            selected_models = ['mace-mp-0-medium']

        print(f"\nSelected models: {', '.join(selected_models)}")

        # Offer to download if not cached
        for model in selected_models:
            if model not in self.downloaded_models:
                download = input(f"\nModel {model} not cached. Download now? (y/n) [y]: ").strip().lower()
                if download != 'n':
                    try:
                        self.download_model(model)
                    except Exception as e:
                        print(f"Error downloading {model}: {e}")
                        selected_models.remove(model)

        return selected_models

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for running MD with a specific model.

        Args:
            model_name: Name of the foundation model

        Returns:
            Configuration dictionary for MACE calculator
        """
        model_path = self.get_model_path(model_name)

        config = {
            'model_path': str(model_path),
            'model_name': model_name,
            'device': 'cuda',  # Use GPU if available
            'default_dtype': 'float64',
            'model_type': 'MACE'
        }

        return config
