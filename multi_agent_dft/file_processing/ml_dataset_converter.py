#!/usr/bin/env python3
"""
Enhanced ML Dataset Converter with AMLP Format Support, Intelligent Splitting,
and Interactive Cluster Analysis for Outlier Detection

This script converts molecular dynamics trajectory data from AMLP JSON format 
to HDF5 format suitable for training machine learning interatomic potentials (MLIPs).

Features:
- AMLP format conversion (dict-based coordinates -> arrays)
- Intelligent hybrid splitting (temporal blocking + energy stratification)
- Interactive species detection and E0s configuration
- Cluster analysis with visualization for outlier detection
- Interactive outlier removal based on energy-force clustering
- Multiple energy filtering methods (absolute, statistical, positive-only)
- Preview statistics to guide filtering decisions
- Comprehensive data validation and error checking
- Support for periodic boundary conditions
- Train/validation split with configurable ratios
"""

import json
import h5py
import numpy as np
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter, defaultdict

# Optional imports for cluster analysis
try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
#                      AMLP FORMAT CONVERSION FUNCTIONS
# ============================================================================

def convert_dict_coords_to_array(coord_list: List[Dict]) -> np.ndarray:
    """
    Convert coordinate list from AMLP dict format to numpy array.
    
    Input:  [{'element': 'C', 'x': 1.0, 'y': 2.0, 'z': 3.0}, ...]
    Output: [[1.0, 2.0, 3.0], ...]
    """
    coords = []
    for coord_dict in coord_list:
        coords.append([coord_dict['x'], coord_dict['y'], coord_dict['z']])
    return np.array(coords, dtype=np.float64)


def cell_parameters_to_matrix(lengths: List[float], angles: List[float]) -> np.ndarray:
    """
    Convert cell parameters (a, b, c, alpha, beta, gamma) to 3x3 cell matrix.
    """
    a, b, c = lengths
    alpha, beta, gamma = np.radians(angles)
    
    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    
    val = 1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma
    
    if val < 0:
        raise ValueError(f"Invalid cell angles lead to negative volume: {angles}")
    
    vol_factor = np.sqrt(val)
    
    cell = np.array([
        [a, 0, 0],
        [b * cos_gamma, b * sin_gamma, 0],
        [c * cos_beta, 
         c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
         c * vol_factor / sin_gamma]
    ], dtype=np.float64)
    
    return cell


def normalize_config_from_amlp(config: Dict) -> Dict:
    normalized = {}

    for field in ['atom_types', 'energy', 'cell_volume', 'source_file']:
        if field in config:
            normalized[field] = config[field]

    # positions: already a flat array (XYZ) or AMLP dict format
    if 'positions' in config:
        normalized['positions'] = np.array(config['positions'], dtype=np.float64)
    elif 'coordinates' in config:
        try:
            normalized['positions'] = convert_dict_coords_to_array(config['coordinates'])
        except (KeyError, ValueError, TypeError):
            pass

    # forces: already a flat array (XYZ) or AMLP dict format
    if 'forces' in config:
        try:
            f = config['forces']
            if f and isinstance(f[0], dict):
                normalized['forces'] = convert_dict_coords_to_array(f)
            else:
                normalized['forces'] = np.array(f, dtype=np.float64)
        except (KeyError, ValueError, TypeError):
            pass

    # cell: already a 3x3 array (XYZ) or AMLP cell_lengths/cell_angles
    if 'cell' in config:
        try:
            normalized['cell'] = np.array(config['cell'], dtype=np.float64)
        except (ValueError, TypeError):
            pass
    elif 'cell_lengths' in config and 'cell_angles' in config:
        try:
            normalized['cell'] = cell_parameters_to_matrix(
                config['cell_lengths'],
                config['cell_angles']
            )
        except (ValueError, ZeroDivisionError):
            pass

    if 'stress' in config:
        try:
            stress = np.array(config['stress'], dtype=np.float64)
            if stress.shape == (3, 3):
                normalized['stress'] = stress
        except (ValueError, TypeError):
            pass

    return normalized
# ============================================================================
#                           SPECIES DETECTION
# ============================================================================

def detect_all_species_in_json(json_data: List[Dict]) -> Dict[str, int]:
    """
    Scan entire JSON dataset to detect all unique atomic species.
    """
    species_counter = Counter()
    
    print("\nScanning dataset for atomic species...")
    
    for i, config in enumerate(json_data):
        atom_types = config.get("atom_types", [])
        species_counter.update(atom_types)
        
        if (i + 1) % 1000 == 0:
            print(f"  Scanned {i + 1}/{len(json_data)} configurations...", end='\r')
    
    print(f"  Scanned {len(json_data)} configurations" + " " * 30)
    
    return dict(species_counter)


def display_species_statistics(species_info: Dict[str, int], num_configs: int):
    """
    Display formatted statistics about atomic species in dataset.
    """
    total_atoms = sum(species_info.values())
    
    print("\n" + "="*80)
    print("DATASET COMPOSITION ANALYSIS".center(80))
    print("="*80)
    print(f"\nTotal configurations: {num_configs}")
    print(f"Total atoms:          {total_atoms}")
    print(f"Unique species:       {len(species_info)}")
    
    print("\n" + "-"*80)
    print(f"{'Element':<10} {'Count':<12} {'Percentage':<12} {'Avg per config':<15}")
    print("-"*80)
    
    for element in sorted(species_info.keys()):
        count = species_info[element]
        percentage = (count / total_atoms) * 100
        avg_per_config = count / num_configs
        print(f"{element:<10} {count:<12} {percentage:>5.2f}%      {avg_per_config:>6.1f}")
    
    print("-"*80)


# ============================================================================
#                           E0s CONFIGURATION
# ============================================================================

def get_E0s_from_user(species_info: Dict[str, int]) -> Optional[Dict[str, float]]:
    """
    Interactively collect isolated atom energies (E0s) from user.
    """
    print("\n" + "="*80)
    print("E0s CONFIGURATION".center(80))
    print("="*80)
    
    print("\nIsolated atom energies (E0s) are used to calculate atomization energies:")
    print("  E_atomization = E_total - Sum(E0_i)")
    print("\nThese values should come from single-point calculations of isolated atoms")
    print("using the same DFT method, functional, and basis set as your dataset.")
    
    print("\n" + "-"*80)
    skip = input("\nDo you want to configure E0s for energy-based filtering? (y/n) [y]: ").strip().lower()
    if skip == 'n':
        print("\n  Skipping E0s configuration. Only force-based filtering will be applied.")
        return None
    
    print("\n" + "-"*80)
    print("ENTER E0s FOR EACH ELEMENT".center(80))
    print("-"*80)
    
    E0s_dict = {}
    species_list = sorted(species_info.keys())
    
    for i, element in enumerate(species_list, 1):
        count = species_info[element]
        
        while True:
            try:
                prompt = f"[{i}/{len(species_list)}] E0 for {element} ({count} atoms) [eV]: "
                E0_input = input(prompt).strip()
                
                if not E0_input:
                    print(f"  Error: E0 for {element} is required. Please enter a value.")
                    continue
                
                E0_value = float(E0_input)
                E0s_dict[element] = E0_value
                print(f"  E0({element}) = {E0_value} eV")
                break
                
            except ValueError:
                print(f"  Error: Invalid number. Please enter a valid floating-point value.")
    
    print("\n" + "-"*80)
    print("E0s SUMMARY".center(80))
    print("-"*80)
    print(f"{'Element':<10} {'E0 (eV)':<15} {'Atom Count':<12}")
    print("-"*80)
    for element in sorted(E0s_dict.keys()):
        print(f"{element:<10} {E0s_dict[element]:<15.8f} {species_info[element]:<12}")
    print("-"*80)
    
    confirm = input("\nConfirm E0s configuration? (y/n) [y]: ").strip().lower() or 'y'
    if confirm != 'y':
        print("\n  Restarting E0s configuration...\n")
        return get_E0s_from_user(species_info)
    
    print("\n  E0s configuration complete")
    return E0s_dict


# ============================================================================
#                     ATOMIZATION ENERGY CALCULATIONS
# ============================================================================

def calculate_atomization_energy_per_atom(energy: float, 
                                         atom_types: List[str], 
                                         E0s_dict: Dict[str, float]) -> float:
    """
    Calculate atomization energy per atom.
    """
    E0_sum = sum(E0s_dict[symbol] for symbol in atom_types)
    E_atomization = energy - E0_sum
    E_atomization_per_atom = E_atomization / len(atom_types)
    return E_atomization_per_atom


def calculate_energy_per_atom(energy: float, n_atoms: int) -> float:
    """
    Calculate energy per atom (without E0s subtraction).
    """
    return energy / n_atoms


# ============================================================================
#                     CLUSTER ANALYSIS AND OUTLIER DETECTION
# ============================================================================

def extract_dataset_features(data: List[Dict], 
                            E0s_dict: Optional[Dict[str, float]] = None,
                            conversion_factor: float = 1.0) -> Dict:
    """
    Extract energy and force features from dataset for cluster analysis.
    
    Parameters
    ----------
    data : list of dict
        Configuration dictionaries
    E0s_dict : dict, optional
        Isolated atom energies for atomization energy calculation
    conversion_factor : float
        Energy conversion factor
        
    Returns
    -------
    features : dict
        Dictionary containing arrays of:
        - energies_per_atom: Energy per atom for each config
        - mean_forces: Mean force magnitude for each config
        - max_forces: Max force magnitude for each config
        - n_atoms: Number of atoms for each config
        - valid_indices: Original indices of valid configs
    """
    energies_per_atom = []
    mean_forces = []
    max_forces = []
    n_atoms_list = []
    valid_indices = []
    
    print("\nExtracting features for cluster analysis...")
    
    for idx, config in enumerate(data):
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(data)} configurations...", end='\r')
        
        config_norm = normalize_config_from_amlp(config)
        
        atom_types = config_norm.get('atom_types', [])
        if not atom_types or 'energy' not in config_norm:
            continue
        
        # Check E0s availability if provided
        if E0s_dict is not None:
            if not all(symbol in E0s_dict for symbol in atom_types):
                continue
        
        try:
            energy = float(config_norm['energy']) * conversion_factor
            n_atoms = len(atom_types)
            
            # Calculate energy per atom
            if E0s_dict is not None:
                e_per_atom = calculate_atomization_energy_per_atom(energy, atom_types, E0s_dict)
            else:
                e_per_atom = calculate_energy_per_atom(energy, n_atoms)
            
            energies_per_atom.append(e_per_atom)
            n_atoms_list.append(n_atoms)
            valid_indices.append(idx)
            
            # Calculate force statistics
            if 'forces' in config_norm:
                forces = np.array(config_norm['forces'])
                force_norms = np.linalg.norm(forces, axis=1)
                mean_forces.append(np.mean(force_norms))
                max_forces.append(np.max(force_norms))
            else:
                mean_forces.append(0.0)
                max_forces.append(0.0)
                
        except (KeyError, ValueError, TypeError):
            continue
    
    print(f"  Extracted features from {len(valid_indices)} configurations" + " " * 20)
    
    return {
        'energies_per_atom': np.array(energies_per_atom),
        'mean_forces': np.array(mean_forces),
        'max_forces': np.array(max_forces),
        'n_atoms': np.array(n_atoms_list),
        'valid_indices': np.array(valid_indices)
    }


def detect_clusters(features: Dict, eps: float = 0.5, min_samples: int = 50) -> Tuple[np.ndarray, int, int]:
    """
    Detect clusters using DBSCAN on energy-force space.
    
    Parameters
    ----------
    features : dict
        Features dictionary from extract_dataset_features
    eps : float
        DBSCAN epsilon parameter
    min_samples : int
        DBSCAN minimum samples parameter
        
    Returns
    -------
    labels : np.ndarray
        Cluster labels for each point (-1 = noise)
    n_clusters : int
        Number of clusters found
    n_noise : int
        Number of noise points
    """
    if not HAS_SKLEARN:
        print("  Warning: scikit-learn not available. Skipping cluster detection.")
        return np.zeros(len(features['energies_per_atom']), dtype=int), 1, 0
    
    # Normalize features for clustering
    X = np.column_stack([features['energies_per_atom'], features['mean_forces']])
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X_norm)
    labels = clustering.labels_
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    return labels, n_clusters, n_noise


def analyze_clusters(features: Dict, labels: np.ndarray) -> Dict:
    """
    Analyze statistics for each cluster.
    
    Returns
    -------
    cluster_stats : dict
        Statistics for each cluster including energy range, mean, count, etc.
    """
    cluster_stats = {}
    unique_labels = sorted(set(labels))
    
    for label in unique_labels:
        mask = labels == label
        
        cluster_stats[label] = {
            'count': int(mask.sum()),
            'e_min': float(features['energies_per_atom'][mask].min()),
            'e_max': float(features['energies_per_atom'][mask].max()),
            'e_mean': float(features['energies_per_atom'][mask].mean()),
            'e_std': float(features['energies_per_atom'][mask].std()),
            'f_mean': float(features['mean_forces'][mask].mean()),
            'f_max': float(features['max_forces'][mask].max()),
            'n_atoms_unique': sorted(set(features['n_atoms'][mask]))
        }
    
    return cluster_stats


def plot_cluster_analysis(features: Dict, 
                         labels: np.ndarray, 
                         cluster_stats: Dict,
                         dataset_name: str = "Dataset",
                         output_file: Optional[str] = None,
                         energy_threshold: Optional[float] = None):
    """
    Create visualization of cluster analysis.
    """
    if not HAS_MATPLOTLIB:
        print("  Warning: matplotlib not available. Skipping visualization.")
        return
    
    # Set font sizes to 22 (user preference)
    plt.rcParams.update({
        'font.size': 22,
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 18,
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Energy vs Mean Force colored by cluster
    ax1 = axes[0]
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            ax1.scatter(features['mean_forces'][mask], features['energies_per_atom'][mask],
                       c='gray', alpha=0.3, s=20, label=f'Noise ({mask.sum()})')
        else:
            ax1.scatter(features['mean_forces'][mask], features['energies_per_atom'][mask],
                       c=[colors[i]], alpha=0.5, s=20, 
                       label=f'Cluster {label} ({mask.sum()})')
    
    # Add energy threshold line if provided
    if energy_threshold is not None:
        ax1.axhline(y=energy_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold: {energy_threshold:.4f}')
    
    ax1.set_xlabel('Mean Force (eV/A)')
    ax1.set_ylabel('Energy per atom (eV)')
    ax1.set_title(f'{dataset_name}\n{len(unique_labels)} clusters found')
    ax1.legend(loc='best')
    
    # Plot 2: Energy distribution histogram
    ax2 = axes[1]
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            ax2.hist(features['energies_per_atom'][mask], bins=50, alpha=0.5, 
                    color='gray', label=f'Noise ({mask.sum()})')
        else:
            ax2.hist(features['energies_per_atom'][mask], bins=50, alpha=0.5,
                    color=colors[i], label=f'Cluster {label} ({mask.sum()})')
    
    if energy_threshold is not None:
        ax2.axvline(x=energy_threshold, color='red', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('Energy per atom (eV)')
    ax2.set_ylabel('Count')
    ax2.set_title('Energy Distribution by Cluster')
    ax2.legend(loc='best')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"\n  Plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def display_cluster_analysis(cluster_stats: Dict, features: Dict):
    """
    Display cluster analysis results in formatted table.
    """
    print("\n" + "="*100)
    print("CLUSTER ANALYSIS RESULTS".center(100))
    print("="*100)
    
    n_total = len(features['energies_per_atom'])
    
    print(f"\n{'Cluster':<10} {'Count':<10} {'%':<8} {'E_min':<12} {'E_max':<12} {'E_mean':<12} {'E_std':<10} {'F_max':<10} {'N_atoms':<15}")
    print("-"*100)
    
    for label in sorted(cluster_stats.keys()):
        stats = cluster_stats[label]
        pct = (stats['count'] / n_total) * 100
        label_str = 'Noise' if label == -1 else f'Cluster {label}'
        n_atoms_str = str(stats['n_atoms_unique']) if len(stats['n_atoms_unique']) <= 3 else f"{stats['n_atoms_unique'][0]}..{stats['n_atoms_unique'][-1]}"
        
        print(f"{label_str:<10} {stats['count']:<10} {pct:>5.1f}%  {stats['e_min']:>+10.4f}  {stats['e_max']:>+10.4f}  "
              f"{stats['e_mean']:>+10.4f}  {stats['e_std']:>8.4f}  {stats['f_max']:>8.2f}  {n_atoms_str:<15}")
    
    print("-"*100)
    
    # Overall statistics
    print(f"\nOverall dataset:")
    print(f"  Total configurations: {n_total}")
    print(f"  Energy range: [{features['energies_per_atom'].min():.4f}, {features['energies_per_atom'].max():.4f}] eV/atom")
    print(f"  Energy span: {(features['energies_per_atom'].max() - features['energies_per_atom'].min())*1000:.1f} meV/atom")


def interactive_outlier_removal(data: List[Dict],
                               features: Dict,
                               labels: np.ndarray,
                               cluster_stats: Dict,
                               dataset_name: str = "Dataset") -> Tuple[List[Dict], Dict]:
    """
    Interactively guide user through outlier removal process with Before/After plotting.
    """
    print("\n" + "="*80)
    print("OUTLIER REMOVAL OPTIONS".center(80))
    print("="*80)

    # --- 1. PLOT BEFORE REMOVAL ---
    if HAS_MATPLOTLIB:
        before_plot_name = f"{dataset_name}_analysis_original.png"
        print(f"\n  Generating 'Before' plot: {before_plot_name}...")
        plot_cluster_analysis(features, labels, cluster_stats, 
                             dataset_name=f"{dataset_name} (Original)",
                             output_file=before_plot_name)
    # -------------------------------
    
    n_clusters = len([l for l in set(labels) if l >= 0])
    n_noise = (labels == -1).sum()
    
    # Check if there are potential issues
    has_multiple_clusters = n_clusters > 1
    has_noise = n_noise > 0
    
    # Check for energy gaps between clusters
    energy_gaps = []
    cluster_energies = []
    if n_clusters > 1:
        for label in sorted(cluster_stats.keys()):
            if label >= 0:
                cluster_energies.append((label, cluster_stats[label]['e_min'], cluster_stats[label]['e_max']))
        
        cluster_energies.sort(key=lambda x: x[1])  # Sort by min energy
        
        for i in range(len(cluster_energies) - 1):
            gap = cluster_energies[i+1][1] - cluster_energies[i][2]
            if gap > 0.01:  # Gap > 10 meV
                energy_gaps.append({
                    'cluster_low': cluster_energies[i][0],
                    'cluster_high': cluster_energies[i+1][0],
                    'gap_meV': gap * 1000
                })
    
    # Display findings
    if has_multiple_clusters:
        print(f"\n  Found {n_clusters} distinct clusters in the data!")
        if energy_gaps:
            print(f"\n  Energy gaps detected between clusters:")
            for gap_info in energy_gaps:
                print(f"    Cluster {gap_info['cluster_low']} <-> Cluster {gap_info['cluster_high']}: "
                      f"{gap_info['gap_meV']:.1f} meV gap")
    
    if has_noise:
        print(f"\n  {n_noise} noise points detected (don't fit any cluster)")
    
    if not has_multiple_clusters and not has_noise:
        print("\n  Dataset looks clean - single cluster, no noise points.")
        skip = input("\nProceed without removing any structures? (y/n) [y]: ").strip().lower() or 'y'
        if skip == 'y':
            return data, {'method': 'none', 'removed': 0}
    
    # Show options
    print("\n" + "-"*80)
    print("REMOVAL OPTIONS".center(80))
    print("-"*80)
    
    print("\n1. Remove by energy threshold")
    print("   Remove all structures with energy per atom below a threshold")
    
    print("\n2. Remove specific cluster(s)")
    print("   Select which cluster(s) to remove entirely")

    print("\n3. Remove specific cluster(s) AND noise")
    print("   Select cluster(s) to remove + automatically remove all noise")
    
    print("\n4. Remove noise points only")
    print(f"   Remove {n_noise} structures that don't fit any cluster")
    
    print("\n5. Remove by energy threshold AND noise")
    print("   Combine energy threshold with noise removal")
    
    print("\n6. No removal")
    print("   Keep all structures")
    
    print("\n" + "-"*80)
    choice = input("\nSelect option (1-6) [6]: ").strip() or "6"
    
    valid_indices = features['valid_indices']
    removal_mask = np.zeros(len(data), dtype=bool)
    removal_config = {'method': 'none', 'removed': 0}
    
    if choice == "1":
        # Remove by energy threshold
        print("\n--- Energy Threshold Removal ---")
        print(f"\nCurrent energy range: [{features['energies_per_atom'].min():.4f}, {features['energies_per_atom'].max():.4f}] eV/atom")
        
        if energy_gaps and cluster_energies:
            suggested = (cluster_energies[0][2] + cluster_energies[1][1]) / 2
            print(f"Suggested threshold (midpoint of gap): {suggested:.4f} eV/atom")
        
        while True:
            try:
                threshold_input = input("\nEnter energy threshold (structures BELOW this will be removed): ").strip()
                threshold = float(threshold_input)
                
                to_remove = features['energies_per_atom'] < threshold
                n_remove = to_remove.sum()
                
                print(f"\n  This would remove {n_remove} structures ({n_remove/len(features['energies_per_atom'])*100:.1f}%)")
                
                confirm = input("  Confirm? (y/n): ").strip().lower()
                if confirm == 'y':
                    for i, idx in enumerate(valid_indices):
                        if to_remove[i]:
                            removal_mask[idx] = True
                    removal_config = {'method': 'energy_threshold', 'threshold': threshold, 'removed': n_remove}
                    break
            except ValueError:
                print("  Invalid number. Please try again.")
    
    elif choice == "2":
        # Remove specific clusters
        print("\n--- Remove Specific Cluster(s) ---")
        print("\nAvailable clusters:")
        for label in sorted(cluster_stats.keys()):
            if label >= 0:
                stats = cluster_stats[label]
                print(f"  Cluster {label}: {stats['count']} structures, "
                      f"E = [{stats['e_min']:.4f}, {stats['e_max']:.4f}] eV/atom")
        
        clusters_input = input("\nEnter cluster number(s) to remove (comma-separated): ").strip()
        try:
            clusters_to_remove = [int(c.strip()) for c in clusters_input.split(',')]
            n_remove = 0
            for i, idx in enumerate(valid_indices):
                if labels[i] in clusters_to_remove:
                    removal_mask[idx] = True
                    n_remove += 1
            print(f"\n  Marked {n_remove} structures for removal")
            removal_config = {'method': 'cluster_removal', 'clusters_removed': clusters_to_remove, 'removed': n_remove}
        except ValueError:
            print("  Invalid input. No clusters removed.")

    elif choice == "3":
        # Remove specific clusters AND noise
        print("\n--- Remove Specific Cluster(s) AND Noise ---")
        print("\nAvailable clusters:")
        for label in sorted(cluster_stats.keys()):
            if label >= 0:
                stats = cluster_stats[label]
                print(f"  Cluster {label}: {stats['count']} structures, "
                      f"E = [{stats['e_min']:.4f}, {stats['e_max']:.4f}] eV/atom")
        
        clusters_input = input("\nEnter cluster number(s) to remove (comma-separated): ").strip()
        try:
            clusters_to_remove = [int(c.strip()) for c in clusters_input.split(',')]
            n_remove = 0
            for i, idx in enumerate(valid_indices):
                if labels[i] in clusters_to_remove or labels[i] == -1:
                    removal_mask[idx] = True
                    n_remove += 1
            print(f"\n  Marked {n_remove} structures for removal (clusters + noise)")
            removal_config = {'method': 'cluster_noise_removal', 'clusters_removed': clusters_to_remove, 'noise_removed': True, 'removed': n_remove}
        except ValueError:
            print("  Invalid input. No clusters removed.")
    
    elif choice == "4":
        # Remove noise only
        print("\n--- Remove Noise Points ---")
        n_remove = 0
        for i, idx in enumerate(valid_indices):
            if labels[i] == -1:
                removal_mask[idx] = True
                n_remove += 1
        print(f"  Marked {n_remove} noise points for removal")
        removal_config = {'method': 'noise_removal', 'removed': n_remove}
    
    elif choice == "5":
        # Combined removal (Threshold + Noise)
        print("\n--- Combined Removal (Energy Threshold + Noise) ---")
        threshold_input = input("Enter energy threshold (structures BELOW this will be removed): ").strip()
        try:
            threshold = float(threshold_input)
            n_remove = 0
            for i, idx in enumerate(valid_indices):
                if features['energies_per_atom'][i] < threshold or labels[i] == -1:
                    removal_mask[idx] = True
                    n_remove += 1
            print(f"  Marked {n_remove} structures for removal (threshold + noise)")
            removal_config = {'method': 'combined', 'threshold': threshold, 'removed': n_remove}
        except ValueError:
            print("  Invalid threshold. No structures removed.")
    
    else:
        print("\n  No structures will be removed based on cluster analysis.")
    
    # Apply removal
    if removal_config['removed'] > 0:
        filtered_data = [config for i, config in enumerate(data) if not removal_mask[i]]
        print(f"\n  Removed {removal_config['removed']} structures")
        print(f"  Remaining: {len(filtered_data)} structures")
        
        # --- 2. PLOT AFTER REMOVAL ---
        if HAS_MATPLOTLIB:
            try:
                print("\n  Generating 'After' plot...")
                # We need to slice the features to plot only the remaining points
                keep_mask = ~removal_mask
                # Since features arrays are aligned with valid_indices, we need a mask 
                # that matches the length of the features arrays, not the full data length.
                # Construct mask for feature arrays:
                feature_keep_mask = []
                valid_set = set(valid_indices)
                
                # Iterate through valid_indices to see if they were marked for removal
                feature_keep_mask = np.array([not removal_mask[idx] for idx in valid_indices])
                
                filtered_features = {
                    'energies_per_atom': features['energies_per_atom'][feature_keep_mask],
                    'mean_forces': features['mean_forces'][feature_keep_mask],
                    'max_forces': features['max_forces'][feature_keep_mask],
                    'n_atoms': features['n_atoms'][feature_keep_mask],
                    'valid_indices': features['valid_indices'][feature_keep_mask]
                }
                filtered_labels = labels[feature_keep_mask]
                
                # Recalculate stats for the plot
                filtered_stats = analyze_clusters(filtered_features, filtered_labels)
                
                after_plot_name = f"{dataset_name}_analysis_cleaned.png"
                plot_cluster_analysis(filtered_features, filtered_labels, filtered_stats,
                                     dataset_name=f"{dataset_name} (Cleaned)",
                                     output_file=after_plot_name)
                                     
            except Exception as e:
                print(f"  Warning: Could not generate 'After' plot: {e}")
        # -----------------------------
        
        return filtered_data, removal_config
    else:
        return data, removal_config

def run_cluster_analysis(data: List[Dict],
                        E0s_dict: Optional[Dict[str, float]] = None,
                        conversion_factor: float = 1.0,
                        dataset_name: str = "Dataset",
                        interactive: bool = True) -> Tuple[List[Dict], Dict]:
    """
    Run complete cluster analysis workflow.
    
    Parameters
    ----------
    data : list of dict
        Configuration data
    E0s_dict : dict, optional
        Isolated atom energies
    conversion_factor : float
        Energy conversion factor
    dataset_name : str
        Name for display/plots
    interactive : bool
        If True, prompt user for outlier removal decisions
        
    Returns
    -------
    filtered_data : list of dict
        Data with outliers removed (if any)
    analysis_results : dict
        Analysis results including cluster stats, removal config
    """
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS".center(80))
    print("="*80)
    
    # Extract features
    features = extract_dataset_features(data, E0s_dict, conversion_factor)
    
    if len(features['valid_indices']) == 0:
        print("  Warning: No valid configurations found for analysis.")
        return data, {'error': 'no_valid_configs'}
    
    # Detect clusters
    print("\nRunning DBSCAN clustering...")
    labels, n_clusters, n_noise = detect_clusters(features)
    print(f"  Found {n_clusters} cluster(s) and {n_noise} noise point(s)")
    
    # Analyze clusters
    cluster_stats = analyze_clusters(features, labels)
    
    # Display results
    display_cluster_analysis(cluster_stats, features)
    
    # Create visualization
    if HAS_MATPLOTLIB:
        show_plot = input("\nShow cluster analysis plot? (y/n) [y]: ").strip().lower() or 'y'
        if show_plot == 'y':
            plot_cluster_analysis(features, labels, cluster_stats, dataset_name)
    
    # Interactive outlier removal
    if interactive:
        filtered_data, removal_config = interactive_outlier_removal(
            data, features, labels, cluster_stats, dataset_name
        )
    else:
        filtered_data = data
        removal_config = {'method': 'none', 'removed': 0}
    
    analysis_results = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'cluster_stats': cluster_stats,
        'removal_config': removal_config,
        'original_count': len(data),
        'final_count': len(filtered_data)
    }
    
    return filtered_data, analysis_results


# ============================================================================
#                        ENERGY FILTERING CONFIGURATION
# ============================================================================

def calculate_atomization_preview(data_sample: List[Dict], 
                                 E0s_dict: Dict[str, float], 
                                 conversion_factor: float = 1.0) -> Optional[Dict]:
    """
    Calculate preview statistics for atomization energies from a sample of data.
    """
    atomization_energies = []
    
    for config in data_sample:
        config = normalize_config_from_amlp(config)
        
        atom_types = config.get("atom_types", [])
        if not atom_types or "energy" not in config:
            continue
        
        if not all(symbol in E0s_dict for symbol in atom_types):
            continue
        
        try:
            energy_eV = float(config["energy"]) * conversion_factor
            E_atom = calculate_atomization_energy_per_atom(energy_eV, atom_types, E0s_dict)
            atomization_energies.append(E_atom)
        except (KeyError, ValueError, TypeError):
            continue
    
    if not atomization_energies:
        return None
    
    atomization_energies = np.array(atomization_energies)
    
    return {
        'mean': float(np.mean(atomization_energies)),
        'std': float(np.std(atomization_energies)),
        'min': float(np.min(atomization_energies)),
        'max': float(np.max(atomization_energies)),
        'positive_count': int(np.sum(atomization_energies > 0)),
        'total_count': len(atomization_energies)
    }


def get_energy_filter_threshold(atomization_preview: Optional[Dict] = None) -> Dict:
    """
    Ask user for the atomization energy threshold for filtering with multiple options.
    """
    print("\n" + "="*80)
    print("ADDITIONAL ENERGY FILTERING CONFIGURATION".center(80))
    print("="*80)
    
    print("\nAfter cluster-based outlier removal, you can apply additional energy filtering.")
    print("\nPhysics reminder:")
    print("  - E_atomization should be NEGATIVE for stable structures")
    print("  - Positive values = unphysical (atoms want to separate)")
    
    if atomization_preview:
        print("\n" + "-"*80)
        print("CURRENT STATISTICS".center(80))
        print("-"*80)
        print(f"  Mean:                    {atomization_preview['mean']:+.3f} eV/atom")
        print(f"  Standard deviation:      {atomization_preview['std']:+.3f} eV/atom")
        print(f"  Range:                   [{atomization_preview['min']:+.3f}, {atomization_preview['max']:+.3f}] eV/atom")
        if atomization_preview['positive_count'] > 0:
            pct = (atomization_preview['positive_count'] / atomization_preview['total_count']) * 100
            print(f"  Positive E_atom:         {atomization_preview['positive_count']} ({pct:.1f}%)")
    
    print("\n" + "="*80)
    print("FILTERING OPTIONS".center(80))
    print("="*80)
    
    print("\n1. Absolute threshold (eV/atom)")
    print("2. Statistical threshold (N standard deviations)")
    print("3. Remove positive only (E_atom > 0)")
    print("4. No additional filtering")
    
    method_choice = input("\nSelect filtering method (1-4) [4]: ").strip() or "4"
    
    if method_choice == "1":
        while True:
            try:
                threshold_input = input("\nMaximum E_atom (eV/atom) [0.5]: ").strip() or "0.5"
                threshold = float(threshold_input)
                if threshold <= 0:
                    print("  Threshold must be positive.")
                    continue
                return {'enabled': True, 'method': 'absolute', 'threshold': threshold}
            except ValueError:
                print("  Invalid number.")
    
    elif method_choice == "2":
        while True:
            try:
                sigma_input = input("\nNumber of standard deviations [5.0]: ").strip() or "5.0"
                n_sigma = float(sigma_input)
                if n_sigma <= 0:
                    print("  Sigma must be positive.")
                    continue
                return {'enabled': True, 'method': 'sigma', 'threshold': n_sigma}
            except ValueError:
                print("  Invalid number.")
    
    elif method_choice == "3":
        return {'enabled': True, 'method': 'positive_only', 'threshold': 0.0}
    
    else:
        return {'enabled': False, 'method': None, 'threshold': None}


def should_remove_by_energy(atomization_energy_per_atom: float, 
                           filter_config: Dict,
                           mean_energy: Optional[float] = None,
                           std_energy: Optional[float] = None) -> bool:
    """
    Determine if a structure should be removed based on its atomization energy.
    """
    if not filter_config['enabled']:
        return False
    
    method = filter_config['method']
    threshold = filter_config['threshold']
    
    if method == 'absolute':
        return atomization_energy_per_atom > threshold
    elif method == 'sigma':
        if mean_energy is None or std_energy is None:
            raise ValueError("Sigma method requires mean_energy and std_energy")
        cutoff = mean_energy + threshold * std_energy
        return atomization_energy_per_atom > cutoff
    elif method == 'positive_only':
        return atomization_energy_per_atom > 0.0
    else:
        return False


# ============================================================================
#                     INTELLIGENT DATA SPLITTING
# ============================================================================

def stratified_split_by_source(data: List[Dict], 
                               train_ratio: float = 0.85,
                               random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Split data by source files to avoid temporal correlation.
    """
    file_groups = defaultdict(list)
    for config in data:
        source = config.get('source_file', 'unknown')
        file_groups[source].append(config)
    
    file_names = list(file_groups.keys())
    random.seed(random_seed)
    random.shuffle(file_names)
    
    train_data = []
    valid_data = []
    train_count = 0
    total_target = int(len(data) * train_ratio)
    
    for file_name in file_names:
        configs = file_groups[file_name]
        if train_count < total_target:
            train_data.extend(configs)
            train_count += len(configs)
        else:
            valid_data.extend(configs)
    
    return train_data, valid_data


def hybrid_split(data: List[Dict],
                 E0s_dict: Dict[str, float],
                 train_ratio: float = 0.85,
                 block_size: int = 100,
                 n_strata: int = 10,
                 random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Enhanced hybrid split stratifying by BOTH energy and forces.
    """
    print("\n  Creating temporal blocks from source files...")
    
    file_groups = defaultdict(list)
    for config in data:
        source = config.get('source_file', 'unknown')
        file_groups[source].append(config)
    
    all_blocks = []
    for source_file, configs in file_groups.items():
        for i in range(0, len(configs), block_size):
            block = configs[i:i+block_size]
            
            block_energies = []
            block_forces = []
            for config in block:
                config_norm = normalize_config_from_amlp(config)
                atom_types = config_norm.get('atom_types', [])
                
                if atom_types and 'energy' in config_norm:
                    try:
                        energy = float(config_norm['energy'])
                        E_atom = calculate_atomization_energy_per_atom(energy, atom_types, E0s_dict)
                        block_energies.append(E_atom)
                    except (KeyError, ValueError, TypeError):
                        pass
                
                if 'forces' in config_norm:
                    try:
                        forces_arr = np.array(config_norm['forces'])
                        max_force = np.max(np.abs(forces_arr))
                        block_forces.append(max_force)
                    except (ValueError, TypeError):
                        pass
            
            if block_energies and block_forces:
                avg_energy = np.mean(block_energies)
                avg_force = np.mean(block_forces)
                all_blocks.append((avg_energy, avg_force, source_file, block))
    
    print(f"    Created {len(all_blocks)} blocks from {len(file_groups)} source files")
    
    print(f"  Stratifying blocks by energy AND forces...")
    
    n_energy_bins = n_strata
    n_force_bins = n_strata
    
    energies = [b[0] for b in all_blocks]
    forces = [b[1] for b in all_blocks]
    
    energy_min, energy_max = min(energies), max(energies)
    force_min, force_max = min(forces), max(forces)
    
    bins = defaultdict(list)
    for block in all_blocks:
        avg_energy, avg_force, source_file, configs = block
        
        e_bin = int((avg_energy - energy_min) / (energy_max - energy_min + 1e-10) * n_energy_bins)
        f_bin = int((avg_force - force_min) / (force_max - force_min + 1e-10) * n_force_bins)
        
        e_bin = min(e_bin, n_energy_bins - 1)
        f_bin = min(f_bin, n_force_bins - 1)
        
        bins[(e_bin, f_bin)].append(block)
    
    print(f"    Created {len(bins)} non-empty bins")
    
    random.seed(random_seed)
    train_data = []
    valid_data = []
    
    for bin_key, bin_blocks in bins.items():
        random.shuffle(bin_blocks)
        n_train = int(len(bin_blocks) * train_ratio)
        
        for _, _, _, block in bin_blocks[:n_train]:
            train_data.extend(block)
        for _, _, _, block in bin_blocks[n_train:]:
            valid_data.extend(block)
    
    print(f"    Training:   {len(train_data)} configs")
    print(f"    Validation: {len(valid_data)} configs")
    
    return train_data, valid_data


# ============================================================================
#                           HDF5 DATASET CREATION
# ============================================================================

def _atomic_number_from_symbol(symbol: str) -> int:
    """Convert element symbol to atomic number."""
    periodic_table = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 
        'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
        'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
        'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
        'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
        'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
        'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
        'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
        'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
        'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
        'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
    }
    
    if symbol not in periodic_table:
        raise ValueError(f"Unknown element symbol: {symbol}")
    
    return periodic_table[symbol]


def _create_h5_from_data(data: List[Dict], 
                        h5_file: h5py.File, 
                        batch_size: int = 4,
                        max_force_threshold: float = 300.0,
                        conversion_factor: float = 1.0, 
                        pbc_handling: str = "auto",
                        include_stress: bool = False,
                        E0s_dict: Optional[Dict[str, float]] = None,
                        filter_config: Optional[Dict] = None) -> Dict:
    """
    Internal function to create HDF5 dataset from data.
    """
    valid_configs = 0
    skipped_force = 0
    skipped_energy = 0
    skipped_missing = 0
    skipped_element = 0
    atomization_energies = []
    extreme_outliers = []
    pbc_counts = {'True': 0, 'False': 0}
    
    valid_data = []
    
    # Calculate mean/std for sigma method
    mean_energy = None
    std_energy = None
    if filter_config and filter_config['enabled'] and filter_config['method'] == 'sigma':
        temp_energies = []
        for config in data:
            config = normalize_config_from_amlp(config)
            atom_types = config.get("atom_types", [])
            if not atom_types or "energy" not in config:
                continue
            if E0s_dict and not all(symbol in E0s_dict for symbol in atom_types):
                continue
            try:
                energy_eV = float(config["energy"]) * conversion_factor
                E_atom = calculate_atomization_energy_per_atom(energy_eV, atom_types, E0s_dict)
                temp_energies.append(E_atom)
            except (KeyError, ValueError, TypeError):
                continue
        
        if temp_energies:
            mean_energy = np.mean(temp_energies)
            std_energy = np.std(temp_energies)
    
    print("\nProcessing configurations...")
    
    for config_idx, config in enumerate(data):
        config = normalize_config_from_amlp(config)
        
        if (config_idx + 1) % 500 == 0:
            print(f"  Processed {config_idx + 1}/{len(data)}...", end='\r')
        
        if "positions" not in config or "atom_types" not in config or "energy" not in config:
            skipped_missing += 1
            continue
        
        atom_types = config["atom_types"]
        if not atom_types:
            skipped_missing += 1
            continue
        
        if E0s_dict is not None:
            if not all(symbol in E0s_dict for symbol in atom_types):
                skipped_element += 1
                continue
        
        try:
            positions = np.array(config["positions"], dtype=np.float64)
            if positions.shape != (len(atom_types), 3):
                skipped_missing += 1
                continue
        except (ValueError, TypeError):
            skipped_missing += 1
            continue
        
        try:
            energy_eV = float(config["energy"]) * conversion_factor
        except (ValueError, TypeError):
            skipped_missing += 1
            continue
        
        has_forces = "forces" in config and config["forces"] is not None
        max_force = 0.0
        if has_forces:
            try:
                forces = np.array(config["forces"], dtype=np.float64)
                if forces.shape != positions.shape:
                    has_forces = False
                else:
                    max_force = np.max(np.abs(forces))
                    if max_force >= max_force_threshold:
                        skipped_force += 1
                        continue
            except (ValueError, TypeError):
                has_forces = False
        
        if E0s_dict is not None and filter_config and filter_config['enabled']:
            atomization_energy_per_atom = calculate_atomization_energy_per_atom(
                energy_eV, atom_types, E0s_dict
            )
            atomization_energies.append(atomization_energy_per_atom)
            
            if should_remove_by_energy(atomization_energy_per_atom, filter_config, 
                                      mean_energy, std_energy):
                skipped_energy += 1
                continue
        
        has_pbc = "cell" in config and config["cell"] is not None
        if has_pbc:
            try:
                cell = np.array(config["cell"], dtype=np.float64)
                if cell.shape != (3, 3):
                    has_pbc = False
                pbc_counts['True'] += 1
            except (ValueError, TypeError):
                has_pbc = False
        else:
            pbc_counts['False'] += 1
        
        if pbc_handling == "strict" and not has_pbc:
            skipped_missing += 1
            continue
        
        has_stress = False
        if include_stress and "stress" in config and config["stress"] is not None:
            try:
                stress = np.array(config["stress"], dtype=np.float64)
                if stress.shape == (3, 3):
                    has_stress = True
            except (ValueError, TypeError):
                has_stress = False
        
        valid_data.append({
            'positions': positions,
            'atom_types': atom_types,
            'energy': energy_eV,
            'forces': forces if has_forces else None,
            'cell': cell if has_pbc else None,
            'stress': stress if has_stress else None
        })
        valid_configs += 1
    
    print(f"  Processed {len(data)} configurations" + " " * 20)
    print(f"  {valid_configs} valid configurations collected")
    
    # Write to HDF5
    print("\nWriting HDF5...")
    
    num_batches = (len(valid_data) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        if (batch_idx + 1) % 100 == 0:
            print(f"  Writing batch {batch_idx + 1}/{num_batches}...", end='\r')
        
        batch_group = h5_file.create_group(f"config_batch_{batch_idx}")
        
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(valid_data))
        batch_configs = valid_data[start_idx:end_idx]
        
        # Pad incomplete batches
        if len(batch_configs) < batch_size:
            num_padding = batch_size - len(batch_configs)
            for i in range(num_padding):
                batch_configs.append(batch_configs[i % len(batch_configs)])
        
        batch_group.attrs['num_configs'] = end_idx - start_idx
        batch_group.attrs['batch_size'] = batch_size
        
        for config_in_batch_idx, config_data in enumerate(batch_configs):
            config_group = batch_group.create_group(f"config_{config_in_batch_idx}")
            
            config_group.create_dataset("atomic_numbers", 
                                    data=np.array([_atomic_number_from_symbol(s) for s in config_data['atom_types']]))
            config_group.create_dataset("positions", data=config_data['positions'])
            
            if config_data['cell'] is not None:
                config_group.create_dataset("pbc", data=np.array([True, True, True]))
                config_group.create_dataset("cell", data=config_data['cell'])
            else:
                config_group.create_dataset("pbc", data=np.array([False, False, False]))
                config_group.create_dataset("cell", data=np.zeros((3, 3)))
            
            config_group.create_dataset("weight", data=1.0)
            config_group.create_dataset("config_type", data=np.bytes_("Default"))
            
            # Placeholder datasets
            config_group.create_dataset("charges", data=np.bytes_("None"))
            config_group.create_dataset("dipole", data=np.bytes_("None"))
            
            properties_group = config_group.create_group("properties")
            properties_group.create_dataset("energy", data=config_data['energy'])
            
            if config_data['forces'] is not None:
                properties_group.create_dataset("forces", data=config_data['forces'])
            
            if config_data['stress'] is not None:
                properties_group.create_dataset("stress", data=config_data['stress'])
            
            weights_group = config_group.create_group("property_weights")
            weights_group.create_dataset("energy_weight", data=1.0)
            weights_group.create_dataset("forces_weight", data=1.0)
            
            if config_data['stress'] is not None:
                weights_group.create_dataset("stress_weight", data=1.0)
    
    print(f"  Wrote {num_batches} batches" + " " * 20)
    
    return {
        'valid_configs': valid_configs,
        'skipped_force': skipped_force,
        'skipped_energy': skipped_energy,
        'skipped_missing': skipped_missing,
        'skipped_element': skipped_element,
        'atomization_energies': atomization_energies,
        'extreme_outliers': extreme_outliers,
        'pbc_counts': pbc_counts
    }

def create_mace_h5_dataset(input_paths: Union[str, List[str]], 
                          output_dir: Optional[str] = None,
                          dataset_name: Optional[str] = None, 
                          train_ratio: float = 0.85,
                          batch_size: int = 4,
                          max_force_threshold: float = 300.0, 
                          conversion_factor: float = 1.0,
                          pbc_handling: str = "auto",
                          include_stress: bool = False,
                          random_seed: int = 42,
                          block_size: int = 100,
                          skip_cluster_analysis: bool = False,
                          recursive: bool = False): 
    """
    Create MACE-compatible HDF5 datasets with interactive E0s-based filtering,
    cluster analysis, and intelligent hybrid splitting.
    """
    # 1. Setup Output Paths
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset_name is None:
        dataset_name = "mace_dataset"

    train_file = output_dir / f"{dataset_name}_train.h5"
    valid_file = output_dir / f"{dataset_name}_valid.h5"

    # 2. Handle Input Paths (List or Single)
    if isinstance(input_paths, str):
        input_paths = [input_paths]
        
    # 3. Load Data
    print("\n" + "="*80)
    print("LOADING DATASET".center(80))
    print("="*80)
    print(f"\nLoading data from {len(input_paths)} source(s)...")
    
    all_data = []
    
    # Unified loading logic (supports both JSON files and Directories)
    for path_str in input_paths:
        path = Path(path_str)
        
        # A. If it's a directory
        if path.is_dir():
            pattern = "**/*.json" if recursive else "*.json"
            for json_path in path.glob(pattern):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data)
                        else:
                            all_data.append(data)
                except Exception as e:
                    print(f"Skipping {json_path.name}: {e}")
                    
        # B. If it's a file
        elif path.is_file():
            if path.suffix == '.json':
                with open(path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
            elif path.suffix in ('.xyz', '.extxyz'):
                try:
                    from ase.io import read
                    frames = read(str(path), index=':')
                    if not isinstance(frames, list):
                        frames = [frames]
                    
                    loaded = 0
                    skipped = 0
                    for atoms in frames:
                        # Try to get energy from info dict
                        energy = None
                        for key in ['energy', 'Energy', 'REF_energy', 'dft_energy', 'total_energy', 'E']:
                            if key in atoms.info:
                                energy = float(atoms.info[key])
                                break
                        # Fallback to ASE calculator
                        if energy is None:
                            try:
                                energy = atoms.get_potential_energy()
                            except Exception:
                                pass
                        if energy is None:
                            skipped += 1
                            continue

                        # Try to get forces from arrays dict
                        forces = None
                        for key in ['forces', 'Forces', 'REF_forces', 'dft_forces']:
                            if key in atoms.arrays:
                                forces = atoms.arrays[key].tolist()
                                break
                        if forces is None:
                            try:
                                forces = atoms.get_forces().tolist()
                            except Exception:
                                pass

                        config = {
                            'atom_types': list(atoms.get_chemical_symbols()),
                            'positions': atoms.get_positions().tolist(),
                            'energy': energy,
                            'source_file': path.name
                        }
                        if forces is not None:
                            config['forces'] = forces
                        if atoms.get_cell() is not None and any(atoms.get_pbc()):
                            config['cell'] = atoms.get_cell().array.tolist()

                        all_data.append(config)
                        loaded += 1

                    print(f"  Loaded {loaded} frames from {path.name}" +
                        (f" ({skipped} skipped, no energy)" if skipped > 0 else ""))

                except Exception as e:
                    print(f"Warning: Could not read {path}: {e}")
                else:
                    print(f"Warning: {path} is not a valid JSON or XYZ file.")
    
    # 4. Validation
    if not all_data:
        raise ValueError("No valid data found in input paths.")
    
    print(f"  Loaded {len(all_data)} configurations")
    
    # 5. Detect species
    # NOTE: Ensure 'detect_all_species_in_json' is defined in your script!
    species_info = detect_all_species_in_json(all_data)
    display_species_statistics(species_info, len(all_data))
    
    # 6. Get E0s configuration
    # NOTE: Ensure 'get_E0s_from_user' is defined in your script!
    E0s_dict = get_E0s_from_user(species_info)
    
    # ========================================================================
    # CLUSTER ANALYSIS STEP
    # ========================================================================
    if not skip_cluster_analysis and E0s_dict is not None:
        print("\n" + "="*80)
        print("Would you like to perform cluster analysis to detect outliers?")
        print("This helps identify data contamination, different polymorphs, or DFT issues.")
        print("="*80)
        
        # Check if user wants to proceed (simple input check)
        do_cluster = input("\nRun cluster analysis? (y/n) [y]: ").strip().lower() 
        if do_cluster == '' or do_cluster == 'y':
            # NOTE: Ensure 'run_cluster_analysis' is defined!
            all_data, cluster_results = run_cluster_analysis(
                all_data, 
                E0s_dict, 
                conversion_factor,
                dataset_name,
                interactive=True
            )
            
            print(f"\n  After cluster analysis: {len(all_data)} configurations remaining")
    
    # 7. Calculate preview statistics
    if E0s_dict is not None:
        print("\n" + "="*80)
        print("Calculating preview statistics...")
        print("="*80)
        sample_size = min(500, len(all_data))
        # NOTE: Ensure 'calculate_atomization_preview' is defined!
        preview_stats = calculate_atomization_preview(
            all_data[:sample_size], 
            E0s_dict, 
            conversion_factor
        )
        
        # Get additional filter configuration
        # NOTE: Ensure 'get_energy_filter_threshold' is defined!
        filter_config = get_energy_filter_threshold(preview_stats)
    else:
        filter_config = {'enabled': False, 'method': None, 'threshold': None}
    
    # 8. Split dataset
    print("\n" + "="*80)
    print("SPLITTING DATASET".center(80))
    print("="*80)
    
    if E0s_dict is not None:
        # NOTE: Ensure 'hybrid_split' is defined!
        train_data, valid_data = hybrid_split(
            all_data, 
            E0s_dict, 
            train_ratio=train_ratio,
            block_size=block_size,
            n_strata=10,
            random_seed=random_seed
        )
    else:
        # NOTE: Ensure 'stratified_split_by_source' is defined!
        train_data, valid_data = stratified_split_by_source(
            all_data,
            train_ratio=train_ratio,
            random_seed=random_seed
        )
    
    print(f"\n  Training:   {len(train_data)} configurations")
    print(f"  Validation: {len(valid_data)} configurations")
    
    # 9. Create training set
    print("\n" + "="*80)
    print("CREATING TRAINING SET".center(80))
    print("="*80)
    
    with h5py.File(train_file, 'w') as h5f:
        # NOTE: Ensure '_create_h5_from_data' is defined!
        train_stats = _create_h5_from_data(
            train_data, h5f, batch_size, max_force_threshold,
            conversion_factor, pbc_handling, include_stress,
            E0s_dict, filter_config
        )
    
    print(f"\n  Training set saved to: {train_file}")
    
    # 10. Create validation set
    print("\n" + "="*80)
    print("CREATING VALIDATION SET".center(80))
    print("="*80)
    
    with h5py.File(valid_file, 'w') as h5f:
        valid_stats = _create_h5_from_data(
            valid_data, h5f, batch_size, max_force_threshold,
            conversion_factor, pbc_handling, include_stress,
            E0s_dict, filter_config
        )
    
    print(f"\n  Validation set saved to: {valid_file}")
    
    # 11. Final statistics
    print("\n" + "="*80)
    print("FINAL STATISTICS".center(80))
    print("="*80)
    
    print(f"\n  Training:   {train_stats['valid_configs']} valid configs")
    print(f"  Validation: {valid_stats['valid_configs']} valid configs")
    print(f"  Total:      {train_stats['valid_configs'] + valid_stats['valid_configs']} configs")
    
    total_skipped = (train_stats['skipped_force'] + valid_stats['skipped_force'] +
                    train_stats['skipped_energy'] + valid_stats['skipped_energy'] +
                    train_stats['skipped_missing'] + valid_stats['skipped_missing'])
    
    if total_skipped > 0:
        print(f"\n  Skipped (force threshold):  {train_stats['skipped_force'] + valid_stats['skipped_force']}")
        print(f"  Skipped (energy threshold): {train_stats['skipped_energy'] + valid_stats['skipped_energy']}")
        print(f"  Skipped (missing data):     {train_stats['skipped_missing'] + valid_stats['skipped_missing']}")
    
    print("\n" + "="*80)
    print("DATASET CREATION COMPLETE".center(80))
    print("="*80 + "\n")
    
    return {
    "train": str(train_file),
    "valid": str(valid_file)
    }



# ============================================================================
#                              MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert AMLP JSON trajectory data to MACE-compatible HDF5 format with cluster analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with interactive configuration
  python ml_dataset_converter.py data.json
  
  # Skip cluster analysis
  python ml_dataset_converter.py data.json --skip-cluster-analysis
  
  # Specify output directory
  python ml_dataset_converter.py data.json --output-dir ./datasets --name my_dataset
        """
    )
    
    parser.add_argument("json_file", help="Path to input JSON file")
    parser.add_argument("--output-dir", default=None, 
                       help="Output directory (default: same as input)")
    parser.add_argument("--name", default=None, 
                       help="Base name for output files")
    parser.add_argument("--train-ratio", type=float, default=0.85,
                       help="Training set ratio (default: 0.85)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Configurations per batch (default: 4)")
    parser.add_argument("--max-force", type=float, default=300.0,
                       help="Maximum force threshold in eV/A (default: 300.0)")
    parser.add_argument("--conversion-factor", type=float, default=1.0,
                       help="Energy conversion factor (default: 1.0)")
    parser.add_argument("--pbc-handling", choices=["auto", "strict", "mixed"], 
                       default="auto",
                       help="PBC handling mode (default: auto)")
    parser.add_argument("--include-stress", action="store_true",
                       help="Include stress tensor data")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for shuffling (default: 42)")
    parser.add_argument("--block-size", type=int, default=100,
                       help="Temporal block size for splitting (default: 100)")
    parser.add_argument("--skip-cluster-analysis", action="store_true",
                       help="Skip interactive cluster analysis step")
    
    args = parser.parse_args()
    
    try:
        train_file, valid_file = create_mace_h5_dataset(
            json_file=args.json_file,
            output_dir=args.output_dir,
            dataset_name=args.name,
            train_ratio=args.train_ratio,
            batch_size=args.batch_size,
            max_force_threshold=args.max_force,
            conversion_factor=args.conversion_factor,
            pbc_handling=args.pbc_handling,
            include_stress=args.include_stress,
            random_seed=args.random_seed,
            block_size=args.block_size,
            skip_cluster_analysis=args.skip_cluster_analysis
        )
        
        print("\nSuccess! Created files:")
        print(f"  Training:   {train_file}")
        print(f"  Validation: {valid_file}\n")
        
    except Exception as e:
        print(f"\nError: {e}\n")
        raise


if __name__ == "__main__":
    main()