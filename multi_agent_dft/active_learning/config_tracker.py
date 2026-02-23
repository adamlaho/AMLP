"""
Configuration Tracker for Active Learning.

Tracks failed or problematic configurations during MD simulations
for recomputation with DFT in the Active Learning loop.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class FailedConfiguration:
    """Data class for a failed configuration."""
    timestamp: str
    frame_index: int
    source_trajectory: str
    failure_reason: str
    atomic_numbers: List[int]
    positions: List[List[float]]
    cell: Optional[List[List[float]]]
    pbc: List[bool]
    energy: Optional[float] = None
    forces: Optional[List[List[float]]] = None
    max_force: Optional[float] = None
    recomputed: bool = False

class ConfigurationTracker:
    """
    Tracker for failed or problematic configurations in Active Learning.

    Monitors MD simulations for failures and tracks configurations
    that need DFT recomputation.
    """

    def __init__(self, working_dir: Path):
        """
        Initialize the tracker.

        Args:
            working_dir: Working directory for saving tracker data
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.failed_configs: List[FailedConfiguration] = []
        self.recomputed_indices: Set[int] = set()

        self.tracker_file = self.working_dir / 'failed_configs.json'
        self.load()

    def add_failed_config(self, frame_index: int, trajectory_file: str,
                         atoms_dict: Dict[str, Any], reason: str):
        """
        Add a failed configuration to the tracker.

        Args:
            frame_index: Frame index in the trajectory
            trajectory_file: Source trajectory file
            atoms_dict: Dictionary with atomic data (positions, cell, etc.)
            reason: Reason for failure
        """
        config = FailedConfiguration(
            timestamp=datetime.now().isoformat(),
            frame_index=frame_index,
            source_trajectory=trajectory_file,
            failure_reason=reason,
            atomic_numbers=atoms_dict['atomic_numbers'],
            positions=atoms_dict['positions'],
            cell=atoms_dict.get('cell'),
            pbc=atoms_dict.get('pbc', [True, True, True]),
            energy=atoms_dict.get('energy'),
            forces=atoms_dict.get('forces'),
            max_force=atoms_dict.get('max_force')
        )

        self.failed_configs.append(config)
        logger.info(f"Added failed config from frame {frame_index}: {reason}")

        self.save()

    def mark_as_recomputed(self, indices: List[int]):
        """
        Mark configurations as recomputed with DFT.

        Args:
            indices: List of indices in failed_configs to mark
        """
        for idx in indices:
            if 0 <= idx < len(self.failed_configs):
                self.failed_configs[idx].recomputed = True
                self.recomputed_indices.add(idx)

        logger.info(f"Marked {len(indices)} configurations as recomputed")
        self.save()

    def get_pending_configs(self) -> List[FailedConfiguration]:
        """
        Get configurations that haven't been recomputed yet.

        Returns:
            List of pending configurations
        """
        return [c for c in self.failed_configs if not c.recomputed]

    def get_all_configs(self) -> List[FailedConfiguration]:
        """Get all tracked configurations."""
        return self.failed_configs.copy()

    def save(self):
        """Save tracker data to file."""
        data = {
            'failed_configs': [asdict(c) for c in self.failed_configs],
            'recomputed_indices': list(self.recomputed_indices)
        }

        with open(self.tracker_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved tracker data to {self.tracker_file}")

    def load(self):
        """Load tracker data from file."""
        if not self.tracker_file.exists():
            logger.debug("No existing tracker file found")
            return

        try:
            with open(self.tracker_file, 'r') as f:
                data = json.load(f)

            self.failed_configs = [FailedConfiguration(**c) for c in data.get('failed_configs', [])]
            self.recomputed_indices = set(data.get('recomputed_indices', []))

            logger.info(f"Loaded {len(self.failed_configs)} tracked configurations")

        except Exception as e:
            logger.error(f"Error loading tracker file: {e}")

    def export_to_xyz(self, output_file: Path, include_recomputed: bool = False):
        """
        Export failed configurations to XYZ format for visualization.

        Args:
            output_file: Output XYZ file
            include_recomputed: Include already recomputed configs
        """
        configs = self.failed_configs if include_recomputed else self.get_pending_configs()

        if not configs:
            logger.warning("No configurations to export")
            return

        from ase import Atoms
        from ase.io import write

        structures = []
        for config in configs:
            atoms = Atoms(
                numbers=config.atomic_numbers,
                positions=config.positions,
                cell=config.cell if config.cell else None,
                pbc=config.pbc
            )
            atoms.info['frame_index'] = config.frame_index
            atoms.info['failure_reason'] = config.failure_reason
            atoms.info['source'] = config.source_trajectory
            structures.append(atoms)

        write(output_file, structures)
        logger.info(f"Exported {len(structures)} configurations to {output_file}")

    def print_summary(self):
        """Print summary of tracked configurations."""
        total = len(self.failed_configs)
        pending = len(self.get_pending_configs())
        recomputed = total - pending

        print("\n" + "="*80)
        print("Failed Configuration Tracker Summary".center(80))
        print("="*80 + "\n")

        print(f"Total failed configurations: {total}")
        print(f"  - Pending DFT recomputation: {pending}")
        print(f"  - Already recomputed: {recomputed}")
        print()

        if pending > 0:
            print("Pending configurations:")
            for i, config in enumerate(self.get_pending_configs()[:10]):
                print(f"  {i+1}. Frame {config.frame_index} from {Path(config.source_trajectory).name}")
                print(f"     Reason: {config.failure_reason}")

            if pending > 10:
                print(f"  ... and {pending - 10} more")

        print("\n" + "="*80 + "\n")

    def analyze_failure_reasons(self) -> Dict[str, int]:
        """
        Analyze and count failure reasons.

        Returns:
            Dictionary mapping reasons to counts
        """
        from collections import Counter
        reasons = [c.failure_reason for c in self.failed_configs]
        return dict(Counter(reasons))

    def clear_recomputed(self):
        """Remove configurations that have been recomputed."""
        self.failed_configs = self.get_pending_configs()
        self.recomputed_indices.clear()
        self.save()
        logger.info("Cleared recomputed configurations from tracker")
