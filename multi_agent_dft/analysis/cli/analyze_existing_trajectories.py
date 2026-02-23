#!/usr/bin/env python3
"""
Batch Free Energy Analysis from Existing Trajectories

This script allows you to analyze multiple existing trajectories using a configuration file.
Useful for batch processing multiple temperatures or systems.

Usage:
    python analyze_existing_trajectories.py config_analysis.yaml

Author: Adam Lahouari
Date: 2025
"""

import sys
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from multi_agent_dft.analysis.free_energy import MDFreeEnergy, Logger


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def analyze_trajectory(trajectory_config: Dict[str, Any],
                       global_config: Dict[str, Any],
                       logger: Logger) -> Dict[str, Any]:
    """
    Analyze a single trajectory.

    Parameters:
    -----------
    trajectory_config : dict
        Configuration for this specific trajectory
    global_config : dict
        Global configuration parameters
    logger : Logger
        Logger instance

    Returns:
    --------
    results : dict
        Analysis results
    """
    # Get trajectory-specific parameters
    traj_file = trajectory_config['trajectory_file']
    temperature = trajectory_config.get('temperature', global_config.get('temperature', 300))
    timestep = trajectory_config.get('timestep', global_config.get('timestep', 0.5))
    output_dir = Path(trajectory_config.get('output_dir',
                                            f'results_{Path(traj_file).stem}'))

    logger.log_section(f"ANALYZING: {traj_file}")
    logger.log(f"Temperature: {temperature} K")
    logger.log(f"Timestep: {timestep} fs")
    logger.log(f"Output: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if trajectory exists
    if not Path(traj_file).exists():
        logger.log(f"ERROR: Trajectory not found: {traj_file}", 'ERROR')
        return {'status': 'error', 'message': f'File not found: {traj_file}'}

    try:
        # Initialize calculator
        fe_calc = MDFreeEnergy(
            trajectory_file=traj_file,
            temperature=temperature,
            timestep=timestep,
            logger=logger
        )

        # VACF parameters
        max_lag = trajectory_config.get('max_lag', global_config.get('vacf_max_lag', None))
        window = trajectory_config.get('window', global_config.get('vacf_window', 'hann'))
        mass_weighted = trajectory_config.get('mass_weighted',
                                             global_config.get('vacf_mass_weighted', True))

        # Calculate VACF
        fe_calc.calculate_vacf(max_lag=max_lag, window=window, mass_weighted=mass_weighted)

        # VDOS parameters
        omega_max = trajectory_config.get('omega_max', global_config.get('vdos_omega_max', 4000))
        n_points = trajectory_config.get('n_points', global_config.get('vdos_n_points', 8192))

        # Calculate VDOS
        fe_calc.calculate_vdos(omega_max=omega_max, n_points=n_points)

        # Temperature array for thermodynamics
        T_min = global_config.get('T_min', 0)
        T_max = global_config.get('T_max', 500)
        n_temps = global_config.get('n_temps', 100)
        T_array = np.linspace(T_min, T_max, n_temps)

        # Calculate thermodynamic properties
        F_vib = fe_calc.calculate_free_energy(T_array)
        S_vib = fe_calc.calculate_entropy(T_array)
        C_v = fe_calc.calculate_heat_capacity(T_array)
        E_ZPE = fe_calc.calculate_zero_point_energy()

        # Convergence analysis
        convergence = None
        if global_config.get('run_convergence', True):
            fractions = global_config.get('convergence_fractions', [0.2, 0.4, 0.6, 0.8, 1.0])
            convergence = fe_calc.convergence_analysis(max_time_fractions=fractions)

            # Plot convergence
            conv_plot = output_dir / 'convergence_analysis.png'
            fe_calc.plot_convergence(convergence, output_file=str(conv_plot))

        # Phonopy comparison
        comparison = None
        phonopy_file = trajectory_config.get('phonopy_file',
                                            global_config.get('phonopy_file', None))
        if phonopy_file and Path(phonopy_file).exists():
            comparison = fe_calc.compare_with_phonopy(phonopy_file, T_array=T_array)

        # Export results
        fe_calc.export_results(str(output_dir), T_array=T_array)

        # Prepare results summary
        idx_sim = np.argmin(np.abs(T_array - temperature))
        results = {
            'status': 'success',
            'trajectory_file': traj_file,
            'temperature': temperature,
            'timestep': timestep,
            'n_frames': fe_calc.n_frames,
            'n_atoms': fe_calc.n_atoms,
            'E_ZPE': E_ZPE,
            'F_vib_at_T': F_vib[idx_sim],
            'S_vib_at_T': S_vib[idx_sim],
            'C_v_at_T': C_v[idx_sim],
            'VACF_final': fe_calc.vacf[-1],
            'VDOS_integral': np.trapz(fe_calc.vdos, fe_calc.omega),
            'output_dir': str(output_dir)
        }

        logger.log(f"✓ Analysis complete: {traj_file}")

        return results

    except Exception as e:
        logger.log(f"✗ Analysis failed: {e}", 'ERROR')
        import traceback
        logger.log(traceback.format_exc(), 'ERROR')
        return {'status': 'error', 'message': str(e), 'trajectory_file': traj_file}


def main():
    """Main execution function."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_existing_trajectories.py config_analysis.yaml")
        print("\nExample config_analysis.yaml:")
        print("""
# Global parameters
timestep: 0.5              # Default timestep (fs)
temperature: 300           # Default temperature (K)

# VACF parameters
vacf_window: 'hann'
vacf_mass_weighted: true

# VDOS parameters
vdos_omega_max: 4000
vdos_n_points: 8192

# Thermodynamics
T_min: 0
T_max: 500
n_temps: 100

# Analysis options
run_convergence: true
convergence_fractions: [0.2, 0.4, 0.6, 0.8, 1.0]

# Global phonopy file (optional)
phonopy_file: null

# Trajectories to analyze
trajectories:
  - trajectory_file: 'trajectory_300K.xyz'
    temperature: 300
    output_dir: 'results_300K'

  - trajectory_file: 'trajectory_350K.xyz'
    temperature: 350
    output_dir: 'results_350K'

  - trajectory_file: 'trajectory_400K.xyz'
    temperature: 400
    output_dir: 'results_400K'
    phonopy_file: 'phonopy_400K.yaml'  # Override global
        """)
        sys.exit(1)

    config_file = sys.argv[1]

    # Load configuration
    try:
        config = load_config(config_file)
    except Exception as e:
        print(f"ERROR: Failed to load config file: {e}")
        sys.exit(1)

    # Setup logger
    log_file = config.get('log_file', 'batch_analysis.log')
    logger = Logger(log_file=log_file)

    logger.log_section("BATCH FREE ENERGY ANALYSIS")
    logger.log(f"Configuration: {config_file}")
    logger.log(f"Log file: {log_file}")

    # Get trajectories to analyze
    trajectories = config.get('trajectories', [])

    if not trajectories:
        logger.log("ERROR: No trajectories specified in configuration", 'ERROR')
        sys.exit(1)

    logger.log(f"Number of trajectories to analyze: {len(trajectories)}")

    # Analyze each trajectory
    results_summary = []
    successful = 0
    failed = 0

    for i, traj_config in enumerate(trajectories, 1):
        logger.log("")
        logger.log(f"[{i}/{len(trajectories)}] Processing trajectory...")

        result = analyze_trajectory(traj_config, config, logger)
        results_summary.append(result)

        if result['status'] == 'success':
            successful += 1
        else:
            failed += 1

    # Final summary
    logger.log_section("BATCH ANALYSIS SUMMARY")
    logger.log(f"Total trajectories: {len(trajectories)}")
    logger.log(f"Successful: {successful}")
    logger.log(f"Failed: {failed}")
    logger.log("")

    # Print summary table
    logger.log("Results Summary:")
    logger.log("-" * 100)
    logger.log(f"{'Trajectory':<40} {'T(K)':>8} {'E_ZPE':>12} {'F_vib':>12} {'S_vib':>12} {'Status':<10}")
    logger.log("-" * 100)

    for result in results_summary:
        if result['status'] == 'success':
            traj = Path(result['trajectory_file']).name
            T = result['temperature']
            E_ZPE = result['E_ZPE']
            F = result['F_vib_at_T']
            S = result['S_vib_at_T']
            status = '✓'

            logger.log(f"{traj:<40} {T:>8.1f} {E_ZPE:>12.6f} {F:>12.6f} {S:>12.6f} {status:<10}")
        else:
            traj = Path(result.get('trajectory_file', 'unknown')).name
            logger.log(f"{traj:<40} {'---':>8} {'---':>12} {'---':>12} {'---':>12} {'✗ FAILED':<10}")

    logger.log("-" * 100)

    # Save results to file
    results_file = config.get('results_summary', 'batch_results_summary.yaml')
    with open(results_file, 'w') as f:
        yaml.dump(results_summary, f, default_flow_style=False)

    logger.log("")
    logger.log(f"Results summary saved to: {results_file}")
    logger.log_section("BATCH ANALYSIS COMPLETE")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
