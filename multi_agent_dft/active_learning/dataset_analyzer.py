"""
Dataset Quality Analyzer for Active Learning.

Analyzes the quality and coverage of DFT datasets to determine if additional
sampling is needed before model training.
"""

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class DatasetStatistics:
    """Statistics for a dataset."""
    n_configurations: int
    energy_mean: float
    energy_std: float
    energy_min: float
    energy_max: float
    force_mean: float
    force_std: float
    force_max: float
    n_high_force_configs: int
    n_elements: int
    element_counts: Dict[str, int]
    has_stress: bool
    recommendations: List[str]

class DatasetQualityAnalyzer:
    """
    Analyzer for dataset quality and coverage assessment.

    Evaluates whether a dataset is sufficient for training or if
    additional sampling is needed.
    """

    def __init__(self, force_threshold: float = 10.0, energy_outlier_std: float = 3.0):
        """
        Initialize the analyzer.

        Args:
            force_threshold: Threshold for identifying high-force configurations (eV/Å)
            energy_outlier_std: Number of standard deviations for energy outliers
        """
        self.force_threshold = force_threshold
        self.energy_outlier_std = energy_outlier_std

    def analyze_json_dataset(self, json_file: Path) -> DatasetStatistics:
        """
        Analyze a JSON dataset file.

        Args:
            json_file: Path to JSON file containing configurations

        Returns:
            DatasetStatistics object
        """
        logger.info(f"Analyzing dataset: {json_file}")

        with open(json_file, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Expected JSON to contain a list of configurations")

        return self._compute_statistics(data)

    def _compute_statistics(self, configurations: List[Dict]) -> DatasetStatistics:
        """Compute statistics from configuration list."""
        energies = []
        forces_all = []
        max_forces = []
        element_counts = defaultdict(int)
        has_stress = False

        for config in configurations:
            # Energy
            if 'energy' in config:
                energies.append(config['energy'])

            # Forces
            if 'forces' in config:
                forces = config['forces']
                for force in forces:
                    fx, fy, fz = force['x'], force['y'], force['z']
                    force_mag = np.sqrt(fx**2 + fy**2 + fz**2)
                    forces_all.append(force_mag)
                max_forces.append(max([np.sqrt(f['x']**2 + f['y']**2 + f['z']**2) for f in forces]))

            # Elements
            if 'atom_types' in config:
                for element in config['atom_types']:
                    element_counts[element] += 1

            # Stress
            if 'stress' in config and config['stress'] is not None:
                has_stress = True

        # Compute statistics
        energies = np.array(energies)
        forces_all = np.array(forces_all)
        max_forces = np.array(max_forces)

        n_high_force = np.sum(max_forces > self.force_threshold)

        stats = DatasetStatistics(
            n_configurations=len(configurations),
            energy_mean=float(np.mean(energies)) if len(energies) > 0 else 0.0,
            energy_std=float(np.std(energies)) if len(energies) > 0 else 0.0,
            energy_min=float(np.min(energies)) if len(energies) > 0 else 0.0,
            energy_max=float(np.max(energies)) if len(energies) > 0 else 0.0,
            force_mean=float(np.mean(forces_all)) if len(forces_all) > 0 else 0.0,
            force_std=float(np.std(forces_all)) if len(forces_all) > 0 else 0.0,
            force_max=float(np.max(forces_all)) if len(forces_all) > 0 else 0.0,
            n_high_force_configs=int(n_high_force),
            n_elements=len(element_counts),
            element_counts=dict(element_counts),
            has_stress=has_stress,
            recommendations=[]
        )

        # Generate recommendations
        stats.recommendations = self._generate_recommendations(stats, energies, forces_all)

        return stats

    def _generate_recommendations(self, stats: DatasetStatistics,
                                  energies: np.ndarray,
                                  forces: np.ndarray) -> List[str]:
        """Generate recommendations based on statistics."""
        recommendations = []

        # Check dataset size
        if stats.n_configurations < 100:
            recommendations.append("⚠️  Dataset is very small (<100 configs). Consider more sampling.")
        elif stats.n_configurations < 500:
            recommendations.append("⚠️  Dataset is small (<500 configs). More data may improve model quality.")
        else:
            recommendations.append("✓ Dataset size is adequate for initial training.")

        # Check force distribution
        if stats.n_high_force_configs > stats.n_configurations * 0.1:
            recommendations.append(f"⚠️  {stats.n_high_force_configs} configs have forces >{self.force_threshold} eV/Å. Consider filtering or more equilibrated sampling.")

        # Check energy outliers
        if len(energies) > 0:
            energy_outliers = np.abs(energies - stats.energy_mean) > self.energy_outlier_std * stats.energy_std
            n_outliers = np.sum(energy_outliers)
            if n_outliers > stats.n_configurations * 0.05:
                recommendations.append(f"⚠️  {n_outliers} energy outliers detected (>{self.energy_outlier_std}σ). Review these configurations.")

        # Check force coverage
        if len(forces) > 0:
            if stats.force_max < 1.0:
                recommendations.append("⚠️  Maximum force is low (<1 eV/Å). Dataset may lack high-energy configurations.")
            elif stats.force_max > 20.0:
                recommendations.append("⚠️  Very high forces detected (>20 eV/Å). May indicate numerical issues or need filtering.")

        # Check stress
        if not stats.has_stress:
            recommendations.append("ℹ️  No stress data found. If training with stress, add stress calculations.")

        # Overall assessment
        if stats.n_configurations >= 500 and stats.n_high_force_configs < stats.n_configurations * 0.05:
            recommendations.append("✓ Dataset appears ready for model training.")
        else:
            recommendations.append("⚠️  Consider additional sampling before training.")

        return recommendations

    def print_statistics(self, stats: DatasetStatistics):
        """Print formatted statistics."""
        print("\n" + "="*80)
        print("Dataset Quality Analysis".center(80))
        print("="*80 + "\n")

        print(f"Number of configurations: {stats.n_configurations}")
        print(f"Number of elements: {stats.n_elements}")
        print(f"Elements: {', '.join(stats.element_counts.keys())}")
        print()

        print("Energy Statistics (eV):")
        print(f"  Mean: {stats.energy_mean:.4f}")
        print(f"  Std:  {stats.energy_std:.4f}")
        print(f"  Min:  {stats.energy_min:.4f}")
        print(f"  Max:  {stats.energy_max:.4f}")
        print()

        print("Force Statistics (eV/Å):")
        print(f"  Mean: {stats.force_mean:.4f}")
        print(f"  Std:  {stats.force_std:.4f}")
        print(f"  Max:  {stats.force_max:.4f}")
        print(f"  High-force configs (>{self.force_threshold} eV/Å): {stats.n_high_force_configs}")
        print()

        print("Stress data:", "Present" if stats.has_stress else "Not found")
        print()

        print("="*80)
        print("Recommendations:")
        print("="*80)
        for rec in stats.recommendations:
            print(f"  {rec}")
        print()

    def plot_distributions(self, json_file: Path, output_dir: Optional[Path] = None):
        """
        Generate distribution plots for energies and forces.

        Args:
            json_file: Path to JSON dataset
            output_dir: Directory to save plots (optional)
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
        except ImportError:
            logger.warning("Matplotlib not available. Skipping plots.")
            return

        with open(json_file, 'r') as f:
            data = json.load(f)

        energies = [c['energy'] for c in data if 'energy' in c]
        forces = []
        for config in data:
            if 'forces' in config:
                for force in config['forces']:
                    fx, fy, fz = force['x'], force['y'], force['z']
                    force_mag = np.sqrt(fx**2 + fy**2 + fz**2)
                    forces.append(force_mag)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Energy distribution
        ax1.hist(energies, bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Energy (eV)')
        ax1.set_ylabel('Count')
        ax1.set_title('Energy Distribution')
        ax1.axvline(np.mean(energies), color='r', linestyle='--', label='Mean')
        ax1.legend()

        # Force distribution
        ax2.hist(forces, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Force magnitude (eV/Å)')
        ax2.set_ylabel('Count')
        ax2.set_title('Force Distribution')
        ax2.axvline(self.force_threshold, color='r', linestyle='--', label=f'Threshold ({self.force_threshold})')
        ax2.legend()

        plt.tight_layout()

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / 'dataset_distributions.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved distribution plots to {output_file}")
        else:
            plt.show()

        plt.close()

    def assess_need_for_more_sampling(self, stats: DatasetStatistics) -> Tuple[bool, str]:
        """
        Assess whether more sampling is needed.

        Args:
            stats: Dataset statistics

        Returns:
            Tuple of (needs_more_sampling, reason)
        """
        if stats.n_configurations < 500:
            return True, "Dataset size is too small (<500 configurations)"

        if stats.n_high_force_configs > stats.n_configurations * 0.1:
            return True, f"Too many high-force configurations ({stats.n_high_force_configs})"

        if stats.force_max < 1.0:
            return True, "Insufficient force diversity (max force <1 eV/Å)"

        return False, "Dataset appears sufficient for training"
