#!/usr/bin/env python3
"""
Standalone Free Energy Analysis Script

This script allows you to run free energy calculations on existing MD trajectories
without needing to run MD simulations from scratch.

Usage:
    python run_free_energy_analysis.py trajectory.xyz --temperature 300 --timestep 0.5

Author: Adam Lahouari
Date: 2025
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from multi_agent_dft.analysis.free_energy import MDFreeEnergy, Logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate vibrational free energy from MD trajectory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_free_energy_analysis.py trajectory_300K.xyz --temperature 300 --timestep 0.5

  # With custom output directory
  python run_free_energy_analysis.py trajectory.xyz -T 300 -dt 0.5 -o results/

  # Extended frequency range and custom temperature range
  python run_free_energy_analysis.py trajectory.xyz -T 300 -dt 0.5 --omega-max 5000 --T-max 600

  # With phonopy comparison
  python run_free_energy_analysis.py trajectory.xyz -T 300 -dt 0.5 --phonopy thermal_properties.yaml

  # Skip convergence analysis for faster processing
  python run_free_energy_analysis.py trajectory.xyz -T 300 -dt 0.5 --no-convergence

  # Custom VACF window and FFT points
  python run_free_energy_analysis.py trajectory.xyz -T 300 -dt 0.5 --window hamming --n-points 16384
        """
    )

    # Required arguments
    parser.add_argument('trajectory', type=str,
                       help='Path to MD trajectory file (ASE-compatible format)')
    parser.add_argument('--temperature', '-T', type=float, required=True,
                       help='Simulation temperature (K)')
    parser.add_argument('--timestep', '-dt', type=float, required=True,
                       help='MD timestep (fs)')

    # Output options
    parser.add_argument('--output', '-o', type=str, default='free_energy_results',
                       help='Output directory (default: free_energy_results)')
    parser.add_argument('--log', type=str, default=None,
                       help='Log file path (default: output_dir/analysis.log)')

    # VACF options
    parser.add_argument('--max-lag', type=int, default=None,
                       help='Maximum time lag for VACF in frames (default: n_frames/2)')
    parser.add_argument('--window', type=str, default='hann',
                       choices=['hann', 'hamming', 'blackman', 'none'],
                       help='Window function for VACF (default: hann)')
    parser.add_argument('--no-mass-weight', action='store_true',
                       help='Disable mass weighting in VACF calculation')

    # VDOS options
    parser.add_argument('--omega-max', type=float, default=4000,
                       help='Maximum frequency for VDOS (cm⁻¹) (default: 4000)')
    parser.add_argument('--n-points', type=int, default=8192,
                       help='Number of FFT points for VDOS (default: 8192)')

    # Thermodynamics options
    parser.add_argument('--T-min', type=float, default=0,
                       help='Minimum temperature for thermodynamics (K) (default: 0)')
    parser.add_argument('--T-max', type=float, default=500,
                       help='Maximum temperature for thermodynamics (K) (default: 500)')
    parser.add_argument('--n-temps', type=int, default=100,
                       help='Number of temperature points (default: 100)')

    # Analysis options
    parser.add_argument('--convergence', action='store_true', default=True,
                       help='Run convergence analysis (default: True)')
    parser.add_argument('--no-convergence', action='store_false', dest='convergence',
                       help='Skip convergence analysis')
    parser.add_argument('--convergence-fractions', type=float, nargs='+',
                       default=[0.2, 0.4, 0.6, 0.8, 1.0],
                       help='Trajectory fractions for convergence (default: 0.2 0.4 0.6 0.8 1.0)')

    # Phonopy comparison
    parser.add_argument('--phonopy', type=str, default=None,
                       help='Path to phonopy thermal_properties.yaml for comparison')

    # Plotting options
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots (data only)')
    parser.add_argument('--show-plots', action='store_true',
                       help='Display plots interactively')

    # Verbosity
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress non-error output')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed progress information')

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    log_file = args.log if args.log else str(output_dir / 'analysis.log')
    logger = Logger(log_file=log_file)

    if not args.quiet:
        logger.log_section("MD-BASED FREE ENERGY ANALYSIS")
        logger.log(f"Trajectory: {args.trajectory}")
        logger.log(f"Temperature: {args.temperature} K")
        logger.log(f"Timestep: {args.timestep} fs")
        logger.log(f"Output directory: {output_dir}")

    # Check if trajectory exists
    if not Path(args.trajectory).exists():
        logger.log(f"ERROR: Trajectory file not found: {args.trajectory}", 'ERROR')
        sys.exit(1)

    try:
        # Initialize MDFreeEnergy
        if args.verbose:
            logger.log("Initializing MDFreeEnergy calculator...")

        fe_calc = MDFreeEnergy(
            trajectory_file=args.trajectory,
            temperature=args.temperature,
            timestep=args.timestep,
            logger=logger if not args.quiet else None
        )

        # Calculate VACF
        if not args.quiet:
            logger.log_section("CALCULATING VACF")

        window = None if args.window == 'none' else args.window
        mass_weighted = not args.no_mass_weight

        fe_calc.calculate_vacf(
            max_lag=args.max_lag,
            window=window,
            mass_weighted=mass_weighted
        )

        # Calculate VDOS
        if not args.quiet:
            logger.log_section("CALCULATING VDOS")

        fe_calc.calculate_vdos(
            omega_max=args.omega_max,
            n_points=args.n_points
        )

        # Calculate thermodynamic properties
        if not args.quiet:
            logger.log_section("CALCULATING THERMODYNAMIC PROPERTIES")

        T_array = np.linspace(args.T_min, args.T_max, args.n_temps)

        F_vib = fe_calc.calculate_free_energy(T_array)
        S_vib = fe_calc.calculate_entropy(T_array)
        C_v = fe_calc.calculate_heat_capacity(T_array)
        E_ZPE = fe_calc.calculate_zero_point_energy()

        # Run convergence analysis if requested
        convergence = None
        if args.convergence:
            if not args.quiet:
                logger.log_section("CONVERGENCE ANALYSIS")

            convergence = fe_calc.convergence_analysis(
                max_time_fractions=args.convergence_fractions
            )

            if not args.no_plots:
                conv_plot = output_dir / 'convergence_analysis.png'
                fe_calc.plot_convergence(
                    convergence,
                    output_file=str(conv_plot),
                    show=args.show_plots
                )

        # Compare with phonopy if provided
        comparison = None
        if args.phonopy:
            if not args.quiet:
                logger.log_section("PHONOPY COMPARISON")

            if Path(args.phonopy).exists():
                comparison = fe_calc.compare_with_phonopy(
                    args.phonopy,
                    T_array=T_array
                )
            else:
                logger.log(f"Warning: Phonopy file not found: {args.phonopy}", 'WARNING')

        # Export results
        if not args.quiet:
            logger.log_section("EXPORTING RESULTS")

        fe_calc.export_results(str(output_dir), T_array=T_array)

        # Print summary
        if not args.quiet:
            logger.log_section("ANALYSIS SUMMARY")
            logger.log(f"Trajectory length: {fe_calc.n_frames} frames ({fe_calc.n_frames * args.timestep / 1000:.2f} ps)")
            logger.log(f"Number of atoms: {fe_calc.n_atoms}")
            logger.log("")
            logger.log(f"Zero-point energy: {E_ZPE:.6f} eV/atom")
            logger.log("")

            # Find values at simulation temperature
            idx_sim = np.argmin(np.abs(T_array - args.temperature))
            logger.log(f"At T = {args.temperature} K:")
            logger.log(f"  Free energy: {F_vib[idx_sim]:.6f} eV/atom")
            logger.log(f"  Entropy: {S_vib[idx_sim]:.6f} eV/K/atom")
            logger.log(f"  Heat capacity: {C_v[idx_sim]:.6f} eV/K/atom")
            logger.log("")

            # Validation checks
            logger.log("Validation checks:")
            vacf_final = fe_calc.vacf[-1]
            vdos_integral = np.trapz(fe_calc.vdos, fe_calc.omega)
            expected_integral = 3 * fe_calc.n_atoms

            check_vacf = "✓" if abs(vacf_final) < 0.1 else "✗"
            check_vdos = "✓" if abs(vdos_integral - expected_integral) / expected_integral < 0.05 else "✗"

            logger.log(f"  {check_vacf} VACF decay: {vacf_final:.4f} (should be < 0.1)")
            logger.log(f"  {check_vdos} VDOS normalization: {vdos_integral:.2f} (expected: {expected_integral})")

            if convergence:
                logger.log("")
                logger.log(f"Convergence analysis: tested {len(convergence['n_frames'])} trajectory fractions")

            if comparison:
                logger.log("")
                logger.log(f"Phonopy comparison: loaded from {args.phonopy}")

            logger.log("")
            logger.log(f"All results saved to: {output_dir}")
            logger.log_section("ANALYSIS COMPLETE")

        return 0

    except Exception as e:
        logger.log(f"ERROR: Analysis failed: {e}", 'ERROR')
        if args.verbose:
            import traceback
            logger.log(traceback.format_exc(), 'ERROR')
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
