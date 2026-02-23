"""
Free Energy Calculation Module (CORRECTED VERSION)

This module provides tools for calculating vibrational free energy from MD trajectories
using the velocity autocorrelation function (VACF) method.

Key corrections:
- Window function preserves VACF[0] = 1.0 normalization
- All thermodynamic integrals properly integrate over cm^-1
- Improved numerical stability
- Support for trajectories with velocities (.traj format)

Date: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple, Union
from ase.io import read
import warnings


class Logger:
    """Simple logging utility for free energy calculations"""

    def __init__(self, log_file: str = None):
        self.log_file = log_file

    def log(self, message: str, level: str = 'INFO'):
        """Log message to console and optionally to file"""
        import time
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {level}: {message}"
        print(log_message)

        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_message + '\n')

    def log_section(self, title: str):
        """Log a section header"""
        separator = "=" * 60
        self.log(separator)
        self.log(f" {title} ")
        self.log(separator)


class MDFreeEnergy:
    """
    Calculate vibrational free energy from MD trajectories using
    velocity autocorrelation function (VACF) method.

    This class implements the calculation of temperature-dependent
    thermodynamic properties from MD simulations, capturing anharmonic
    effects beyond the harmonic approximation.

    Physical Constants:
        hbar = 1.054571817e-34 J·s (ℏ)
        kB = 1.380649e-23 J/K (Boltzmann constant)
        eV_to_J = 1.602176634e-19 J/eV
        cm_to_Hz = 2.99792458e10 Hz/cm⁻¹

    Formulas:
        VACF: C(t) = ⟨v(t)·v(0)⟩ / ⟨v²⟩
        VDOS: g(ω) = (2/π) ∫ C(t) cos(ωt) dt
        F_vib: F(T) = k_B T ∫ g(ω) ln[2 sinh(ℏω/2k_B T)] dω
        S_vib: S(T) = k_B ∫ g(ω) [x/(e^x-1) - ln(1-e^-x)] dω  where x=ℏω/k_B T
        C_v: C_v(T) = k_B ∫ g(ω) x² e^x/(e^x-1)² dω
        E_ZPE: E_0 = ∫ g(ω) (ℏω/2) dω

    Attributes:
        trajectory_file (str): Path to MD trajectory file
        temperature (float): Simulation temperature (K)
        timestep (float): MD timestep (fs)
        atoms_list (list): List of ASE Atoms objects from trajectory
        vacf_time (ndarray): Time array for VACF (fs)
        vacf (ndarray): Normalized velocity autocorrelation function
        omega (ndarray): Frequency array (cm⁻¹)
        vdos (ndarray): Vibrational density of states (normalized to 3N)
    """

    def __init__(self, trajectory_file: str, temperature: float, timestep: float,
                 logger: Logger = None):
        """
        Initialize MD free energy calculator.

        Parameters:
        -----------
        trajectory_file : str
            Path to trajectory file (must contain velocities!)
            Supported formats: .traj (ASE Trajectory with velocities)
                              .xyz (if velocities in extended XYZ format)
        temperature : float
            Simulation temperature (K)
        timestep : float
            MD timestep (fs)
        logger : Logger, optional
            Logger instance for output
        """
        self.trajectory_file = Path(trajectory_file)
        self.temperature = temperature
        self.timestep = timestep  # fs
        self.logger = logger if logger else Logger()

        # Physical constants (SI units)
        self.hbar = 1.054571817e-34  # J·s
        self.kB = 1.380649e-23  # J/K
        self.eV_to_J = 1.602176634e-19  # J/eV
        self.cm_to_Hz = 2.99792458e10  # Hz per cm⁻¹

        # Conversion factors
        self.fs_to_s = 1e-15  # femtoseconds to seconds
        self.amu_to_kg = 1.66053906660e-27  # atomic mass units to kg

        # Data storage
        self.atoms_list = []
        self.velocities = None  # (n_frames, n_atoms, 3) in Å/fs
        self.masses = None  # (n_atoms,) in amu
        self.n_atoms = 0
        self.n_frames = 0

        # VACF data
        self.vacf_time = None  # fs
        self.vacf = None  # dimensionless, normalized to C(0)=1

        # VDOS data
        self.omega = None  # cm⁻¹
        self.vdos = None  # modes/cm⁻¹, normalized to ∫g(ω)dω = 3N

        # Thermodynamic properties
        self.F_vib = None  # eV/atom
        self.S_vib = None  # eV/K/atom
        self.C_v = None  # eV/K/atom
        self.E_ZPE = None  # eV/atom

        self._load_trajectory()

    def _load_trajectory(self):
        """Load MD trajectory from file and extract velocities."""
        self.logger.log_section("LOADING MD TRAJECTORY")
        self.logger.log(f"Trajectory file: {self.trajectory_file}")

        if not self.trajectory_file.exists():
            raise FileNotFoundError(f"Trajectory file not found: {self.trajectory_file}")

        # Read all frames
        try:
            self.atoms_list = read(str(self.trajectory_file), index=':')
            if not isinstance(self.atoms_list, list):
                self.atoms_list = [self.atoms_list]
        except Exception as e:
            self.logger.log(f"Error reading trajectory: {e}", 'ERROR')
            raise

        self.n_frames = len(self.atoms_list)
        self.n_atoms = len(self.atoms_list[0])

        self.logger.log(f"Loaded {self.n_frames} frames")
        self.logger.log(f"Number of atoms: {self.n_atoms}")
        self.logger.log(f"Simulation temperature: {self.temperature} K")
        self.logger.log(f"Timestep: {self.timestep} fs")
        self.logger.log(f"Total simulation time: {(self.n_frames - 1) * self.timestep:.2f} fs")

        # Extract velocities and masses
        self._extract_velocities()

    def _extract_velocities(self):
        """
        Extract velocities from trajectory.
        
        Velocities must be present in the trajectory file.
        For ASE Trajectory (.traj), velocities are automatically saved.
        For XYZ, velocities must be in extended XYZ format.
        """
        self.logger.log("Extracting velocities from trajectory...")

        # Get velocities from each frame
        velocities_list = []
        for i, atoms in enumerate(self.atoms_list):
            vel = atoms.get_velocities()
            if vel is None:
                raise ValueError(
                    f"Frame {i} does not contain velocity information!\n"
                    f"MD trajectories must be saved with velocities.\n"
                    f"Use ASE Trajectory format (.traj) which automatically saves velocities."
                )
            velocities_list.append(vel)

        # Convert to array: (n_frames, n_atoms, 3)
        self.velocities = np.array(velocities_list)  # Å/fs

        # Get masses (assume constant throughout trajectory)
        self.masses = self.atoms_list[0].get_masses()  # amu

        self.logger.log(f"Velocity array shape: {self.velocities.shape}")
        self.logger.log(f"Mass array shape: {self.masses.shape}")
        
        # Sanity checks
        if np.any(np.isnan(self.velocities)):
            raise ValueError("NaN values detected in velocities!")
        if np.any(np.isinf(self.velocities)):
            raise ValueError("Infinite values detected in velocities!")
        
        # Check velocity magnitude is reasonable (not exactly zero)
        v_rms = np.sqrt(np.mean(self.velocities**2))
        self.logger.log(f"RMS velocity: {v_rms:.6f} Å/fs")
        if v_rms < 1e-10:
            raise ValueError("Velocities are essentially zero! Check MD simulation.")

    def calculate_vacf(self, max_lag: int = None, window: str = 'hann',
                       mass_weighted: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate velocity autocorrelation function using FFT.

        The VACF is defined as:
            C(t) = ⟨v(t) · v(0)⟩ / ⟨v(0) · v(0)⟩

        For mass-weighted velocities (recommended for multi-component systems):
            C(t) = ⟨√m v(t) · √m v(0)⟩ / ⟨√m v(0) · √m v(0)⟩

        CRITICAL: Window function is applied AFTER normalization to preserve C(0)=1

        Parameters:
        -----------
        max_lag : int, optional
            Maximum time lag in frames. Default: n_frames // 2
            Should be chosen such that VACF decays to near zero
        window : str, optional
            Window function for smoothing VACF before FFT
            Options: 'hann', 'hamming', 'blackman', None
            Reduces spectral leakage in VDOS calculation
        mass_weighted : bool, optional
            Use mass-weighted velocities (default: True)
            Recommended for accurate thermodynamics

        Returns:
        --------
        t : ndarray
            Time array (fs)
        C_t : ndarray
            Normalized VACF with C(0) = 1.0
        """
        self.logger.log_section("CALCULATING VELOCITY AUTOCORRELATION FUNCTION")

        if max_lag is None:
            max_lag = self.n_frames // 2

        if max_lag > self.n_frames - 1:
            max_lag = self.n_frames - 1
            self.logger.log(f"max_lag reduced to {max_lag} (n_frames - 1)", 'WARNING')

        self.logger.log(f"Maximum lag: {max_lag} frames ({max_lag * self.timestep:.2f} fs)")
        self.logger.log(f"Window function: {window}")
        self.logger.log(f"Mass-weighted: {mass_weighted}")

        # Prepare velocities: flatten to (n_frames, 3*n_atoms)
        v = self.velocities.reshape(self.n_frames, -1)  # Shape: (n_frames, 3*n_atoms)

        # Apply mass weighting if requested
        if mass_weighted:
            # Repeat masses for each dimension (x, y, z)
            mass_weights = np.sqrt(np.repeat(self.masses, 3))  # Shape: (3*n_atoms,)
            v = v * mass_weights[np.newaxis, :]  # Broadcasting

        # Calculate VACF using FFT for efficiency
        # Method: For each velocity component, compute autocorrelation and average
        vacf = np.zeros(max_lag)
        n_components = v.shape[1]

        for i in range(n_components):
            # Get velocity component
            v_i = v[:, i]

            # Zero-pad for FFT-based autocorrelation
            n_pad = 2 * len(v_i)
            v_padded = np.concatenate([v_i, np.zeros(n_pad - len(v_i))])

            # FFT-based autocorrelation: R(τ) = IFFT(|FFT(v)|²)
            fft_v = np.fft.fft(v_padded)
            acf_full = np.fft.ifft(fft_v * np.conj(fft_v)).real

            # Extract relevant part and normalize by number of samples
            acf = acf_full[:max_lag]
            normalization = np.arange(self.n_frames, self.n_frames - max_lag, -1)
            acf = acf / normalization

            vacf += acf

        # Average over all velocity components (3N components)
        vacf /= n_components

        # CRITICAL: Normalize so that C(0) = 1
        # This MUST be done BEFORE applying window function
        if vacf[0] <= 0:
            raise ValueError(f"VACF[0] = {vacf[0]} is non-positive! Something is wrong.")
        
        vacf = vacf / vacf[0]
        
        # Verify normalization
        if not np.isclose(vacf[0], 1.0, rtol=1e-6):
            self.logger.log(f"Warning: VACF[0] = {vacf[0]:.10f} (should be 1.0)", 'WARNING')

        # Apply window function if requested
        # CRITICAL: Window function is applied AFTER normalization
        # AND we preserve C(0) = 1.0 by setting window[0] = 1.0
        if window:
            if window == 'hann':
                window_func = np.hanning(len(vacf))
            elif window == 'hamming':
                window_func = np.hamming(len(vacf))
            elif window == 'blackman':
                window_func = np.blackman(len(vacf))
            else:
                raise ValueError(f"Unknown window function: {window}")

            # CRITICAL FIX: Preserve normalization by setting window[0] = 1.0
            # Standard window functions go to 0 at endpoints, which would destroy C(0)=1
            window_func[0] = 1.0
            
            vacf = vacf * window_func
            self.logger.log(f"Applied {window} window function (preserving C(0)=1.0)")

        # Time array
        self.vacf_time = np.arange(max_lag) * self.timestep  # fs
        self.vacf = vacf

        self.logger.log(f"VACF calculated: {len(self.vacf)} points")
        self.logger.log(f"VACF[0] = {self.vacf[0]:.6f} (should be 1.0)")
        self.logger.log(f"VACF decay at t={self.vacf_time[-1]:.1f} fs: {self.vacf[-1]:.6f}")
        
        # Check if VACF has decayed sufficiently
        if abs(self.vacf[-1]) > 0.1:
            self.logger.log(
                f"WARNING: VACF has not decayed to near zero (C({self.vacf_time[-1]:.1f} fs) = {self.vacf[-1]:.3f})",
                'WARNING'
            )
            self.logger.log("Consider increasing max_lag or trajectory length", 'WARNING')

        return self.vacf_time, self.vacf

    def calculate_vdos(self, omega_max: float = 4000, n_points: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate vibrational density of states from VACF using Fourier transform.

        The VDOS is obtained via cosine transform of VACF:
            g(ω) = (2/π) ∫₀^∞ C(t) cos(ωt) dt
        
        In practice, using FFT:
            g(ω) = Re[FFT(C(t))] × proper_normalization

        Parameters:
        -----------
        omega_max : float, optional
            Maximum frequency (cm⁻¹). Default: 4000 cm⁻¹
            Should cover all relevant vibrations (C-H stretch ~3000 cm⁻¹)
        n_points : int, optional
            Number of frequency points for FFT. Default: 8192
            Must be power of 2 for optimal FFT performance
            Determines frequency resolution: Δω = omega_max / (n_points/2)

        Returns:
        --------
        omega : ndarray
            Frequency array (cm⁻¹)
        g_omega : ndarray
            VDOS (modes per cm⁻¹), normalized so ∫g(ω)dω = 3N
        """
        if self.vacf is None:
            raise ValueError("VACF not calculated. Run calculate_vacf() first.")

        self.logger.log_section("CALCULATING VIBRATIONAL DENSITY OF STATES")

        # Time parameters
        dt = self.timestep * self.fs_to_s  # Convert to seconds
        
        # Zero-pad VACF for FFT
        vacf_padded = np.zeros(n_points)
        vacf_padded[:len(self.vacf)] = self.vacf

        # Perform FFT
        # For real signal, only need positive frequencies
        vdos_fft = np.fft.rfft(vacf_padded)
        
        # Get frequency array in Hz
        freq_Hz = np.fft.rfftfreq(n_points, dt)
        
        # Convert to cm⁻¹
        omega_cm = freq_Hz / self.cm_to_Hz
        
        # Extract real part (VDOS is real)
        vdos = vdos_fft.real
        
        # Apply normalization factor: (2/π) × dt
        # The factor of 2 comes from integrating only positive frequencies
        vdos = vdos * 2.0 * dt / np.pi
        
        # Limit to requested frequency range
        mask = omega_cm <= omega_max
        omega_cm = omega_cm[mask]
        vdos = vdos[mask]

        # Ensure non-negative (numerical noise can create small negative values)
        vdos = np.maximum(vdos, 0)

        # Normalize to 3N (total number of vibrational degrees of freedom)
        # This ensures ∫g(ω)dω = 3N
        integral = np.trapz(vdos, omega_cm)
        expected_integral = 3 * self.n_atoms
        
        if integral > 0:
            normalization_factor = expected_integral / integral
            vdos = vdos * normalization_factor
            self.logger.log(f"VDOS normalization factor: {normalization_factor:.6f}")
        else:
            raise ValueError("VDOS integral is zero or negative! Check VACF calculation.")

        self.omega = omega_cm
        self.vdos = vdos

        # Verify normalization
        final_integral = np.trapz(vdos, omega_cm)
        self.logger.log(f"VDOS calculated: {len(self.vdos)} points")
        self.logger.log(f"Frequency range: {omega_cm[0]:.1f} - {omega_cm[-1]:.1f} cm⁻¹")
        self.logger.log(f"Frequency resolution: {omega_cm[1] - omega_cm[0]:.2f} cm⁻¹")
        self.logger.log(f"VDOS integral: {final_integral:.2f} (should be {expected_integral})")
        
        deviation = abs(final_integral - expected_integral) / expected_integral
        if deviation > 0.05:
            self.logger.log(
                f"WARNING: VDOS integral deviates by {deviation*100:.1f}% from 3N",
                'WARNING'
            )

        return self.omega, self.vdos

    def calculate_free_energy(self, T_array: np.ndarray) -> np.ndarray:
        """
        Calculate temperature-dependent vibrational free energy.

        Formula:
            F_vib(T) = k_B T ∫ g(ω) ln[2 sinh(ℏω/2k_B T)] dω
        
        At T=0:
            F_vib(0) = E_ZPE = ∫ g(ω) (ℏω/2) dω

        CRITICAL: Integration is performed over ω in cm⁻¹, which is the
        x-axis of the VDOS. Energy conversion is handled properly.

        Parameters:
        -----------
        T_array : array-like
            Temperature array (K)

        Returns:
        --------
        F_vib : ndarray
            Vibrational free energy (eV/atom)
        """
        if self.vdos is None:
            raise ValueError("VDOS not calculated. Run calculate_vdos() first.")

        self.logger.log_section("CALCULATING FREE ENERGY")

        T_array = np.atleast_1d(T_array)
        F_vib = np.zeros(len(T_array))

        # Convert frequency from cm⁻¹ to angular frequency (rad/s)
        # ω_rad/s = 2π × ν_Hz = 2π × (ν_cm⁻¹ × c)
        omega_rad_s = self.omega * self.cm_to_Hz * 2 * np.pi  # rad/s
        
        # Energy quantum: ℏω (in Joules)
        hbar_omega_J = self.hbar * omega_rad_s  # J

        for i, T in enumerate(T_array):
            if T == 0:
                # At T=0, F_vib = E_ZPE = ∫ g(ω) (ℏω/2) dω
                # Integrate over cm⁻¹ (VDOS x-axis)
                integrand = self.vdos * 0.5 * hbar_omega_J
                F_vib[i] = np.trapz(integrand, self.omega)  # J
            else:
                # Compute x = ℏω / (2 k_B T)
                x = hbar_omega_J / (2 * self.kB * T)

                # Avoid numerical issues at small frequencies
                # sinh(x) ≈ x for small x, so ln(2 sinh(x)) ≈ ln(2x)
                mask = x > 1e-10
                integrand = np.zeros_like(self.omega)
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    integrand[mask] = self.vdos[mask] * np.log(2 * np.sinh(x[mask]))
                
                # For very small x, use limiting form: ln(2 sinh(x)) ≈ ln(2x)
                mask_small = x <= 1e-10
                if np.any(mask_small):
                    integrand[mask_small] = self.vdos[mask_small] * np.log(2 * x[mask_small])

                # Integrate over cm⁻¹
                F_vib[i] = self.kB * T * np.trapz(integrand, self.omega)  # J

        # Convert from J to eV and normalize per atom
        F_vib = F_vib / self.eV_to_J / self.n_atoms  # eV/atom

        self.F_vib = F_vib

        self.logger.log(f"Free energy calculated for {len(T_array)} temperatures")
        self.logger.log(f"F_vib at T={T_array[0]:.1f} K: {F_vib[0]:.6f} eV/atom")
        if len(T_array) > 1:
            self.logger.log(f"F_vib at T={T_array[-1]:.1f} K: {F_vib[-1]:.6f} eV/atom")

        return F_vib

    def calculate_entropy(self, T_array: np.ndarray) -> np.ndarray:
        """
        Calculate temperature-dependent vibrational entropy.

        Formula:
            S_vib(T) = k_B ∫ g(ω) [x/(e^x - 1) - ln(1 - e^(-x))] dω
            where x = ℏω / (k_B T)

        At T=0:
            S_vib(0) = 0

        Parameters:
        -----------
        T_array : array-like
            Temperature array (K)

        Returns:
        --------
        S_vib : ndarray
            Vibrational entropy (eV/K/atom)
        """
        if self.vdos is None:
            raise ValueError("VDOS not calculated. Run calculate_vdos() first.")

        self.logger.log_section("CALCULATING ENTROPY")

        T_array = np.atleast_1d(T_array)
        S_vib = np.zeros(len(T_array))

        # Convert frequency to energy
        omega_rad_s = self.omega * self.cm_to_Hz * 2 * np.pi  # rad/s
        hbar_omega_J = self.hbar * omega_rad_s  # J

        for i, T in enumerate(T_array):
            if T == 0:
                S_vib[i] = 0
            else:
                # Compute x = ℏω / (k_B T)
                x = hbar_omega_J / (self.kB * T)

                # Avoid numerical issues at small and large x
                mask = (x > 1e-10) & (x < 100)  # Avoid overflow in exp(x)
                integrand = np.zeros_like(self.omega)

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    exp_x = np.exp(x[mask])
                    # S = k_B ∫ g(ω) [x/(e^x - 1) - ln(1 - e^(-x))] dω
                    # Note: ln(1 - e^(-x)) = ln((e^x - 1)/e^x) = ln(e^x - 1) - x
                    integrand[mask] = self.vdos[mask] * (
                        x[mask] / (exp_x - 1) - np.log(1 - 1/exp_x)
                    )

                # Integrate over cm⁻¹
                S_vib[i] = self.kB * np.trapz(integrand, self.omega)  # J/K

        # Convert from J/K to eV/K and normalize per atom
        S_vib = S_vib / self.eV_to_J / self.n_atoms  # eV/K/atom

        self.S_vib = S_vib

        self.logger.log(f"Entropy calculated for {len(T_array)} temperatures")
        if len(T_array) > 0 and T_array[-1] > 0:
            self.logger.log(f"S_vib at T={T_array[-1]:.1f} K: {S_vib[-1]:.6f} eV/K/atom")

        return S_vib

    def calculate_heat_capacity(self, T_array: np.ndarray) -> np.ndarray:
        """
        Calculate temperature-dependent heat capacity.

        Formula:
            C_v(T) = k_B ∫ g(ω) x² [e^x / (e^x - 1)²] dω
            where x = ℏω / (k_B T)

        At high T, approaches Dulong-Petit limit: C_v → 3k_B per atom

        Parameters:
        -----------
        T_array : array-like
            Temperature array (K)

        Returns:
        --------
        C_v : ndarray
            Heat capacity (eV/K/atom)
        """
        if self.vdos is None:
            raise ValueError("VDOS not calculated. Run calculate_vdos() first.")

        self.logger.log_section("CALCULATING HEAT CAPACITY")

        T_array = np.atleast_1d(T_array)
        C_v = np.zeros(len(T_array))

        # Convert frequency to energy
        omega_rad_s = self.omega * self.cm_to_Hz * 2 * np.pi  # rad/s
        hbar_omega_J = self.hbar * omega_rad_s  # J

        for i, T in enumerate(T_array):
            if T == 0:
                C_v[i] = 0
            else:
                # Compute x = ℏω / (k_B T)
                x = hbar_omega_J / (self.kB * T)

                # Avoid numerical issues
                mask = (x > 1e-10) & (x < 100)
                integrand = np.zeros_like(self.omega)

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    exp_x = np.exp(x[mask])
                    # C_v = k_B ∫ g(ω) x² e^x / (e^x - 1)² dω
                    integrand[mask] = self.vdos[mask] * x[mask]**2 * exp_x / (exp_x - 1)**2

                # Integrate over cm⁻¹
                C_v[i] = self.kB * np.trapz(integrand, self.omega)  # J/K

        # Convert from J/K to eV/K and normalize per atom
        C_v = C_v / self.eV_to_J / self.n_atoms  # eV/K/atom

        self.C_v = C_v

        self.logger.log(f"Heat capacity calculated for {len(T_array)} temperatures")
        if len(T_array) > 0 and T_array[-1] > 0:
            self.logger.log(f"C_v at T={T_array[-1]:.1f} K: {C_v[-1]:.6f} eV/K/atom")
            # Dulong-Petit limit: 3k_B per atom
            dulong_petit = 3 * self.kB / self.eV_to_J
            self.logger.log(f"Dulong-Petit limit (3k_B): {dulong_petit:.6f} eV/K/atom")
            
            # Check if approaching classical limit at high T
            if T_array[-1] > 300:
                ratio = C_v[-1] / dulong_petit
                self.logger.log(f"C_v / Dulong-Petit ratio: {ratio:.3f} (→1.0 at high T)")

        return C_v

    def calculate_zero_point_energy(self) -> float:
        """
        Calculate zero-point energy (quantum ground state energy).

        Formula:
            E_ZPE = ∫ g(ω) (ℏω/2) dω

        This is the vibrational energy at T=0 K due to quantum zero-point motion.

        Returns:
        --------
        E_ZPE : float
            Zero-point energy (eV/atom)
        """
        if self.vdos is None:
            raise ValueError("VDOS not calculated. Run calculate_vdos() first.")

        # Convert frequency to energy
        omega_rad_s = self.omega * self.cm_to_Hz * 2 * np.pi  # rad/s
        hbar_omega_J = self.hbar * omega_rad_s  # J

        # Calculate ZPE: ∫ g(ω) (ℏω/2) dω
        # Integrate over cm⁻¹
        integrand = self.vdos * 0.5 * hbar_omega_J
        E_ZPE = np.trapz(integrand, self.omega)  # J

        # Convert from J to eV and normalize per atom
        E_ZPE = E_ZPE / self.eV_to_J / self.n_atoms  # eV/atom

        self.E_ZPE = E_ZPE

        self.logger.log(f"Zero-point energy: {E_ZPE:.6f} eV/atom")

        return E_ZPE

    def compare_with_phonopy(self, phonopy_yaml: str, T_array: np.ndarray = None) -> Dict[str, Any]:
        """
        Compare MD-based VDOS and thermodynamics with phonopy harmonic calculation.

        This allows quantifying anharmonic effects: Difference = MD - Phonopy

        Parameters:
        -----------
        phonopy_yaml : str
            Path to phonopy output file (e.g., 'phonopy.yaml' or 'thermal_properties.yaml')
        T_array : array-like, optional
            Temperature array for comparison. If None, uses same as phonopy.

        Returns:
        --------
        comparison : dict
            Dictionary containing comparison data and statistics
        """
        self.logger.log_section("COMPARING WITH PHONOPY")

        # Try to load phonopy data
        try:
            import yaml as pyyaml
            with open(phonopy_yaml, 'r') as f:
                phonopy_data = pyyaml.safe_load(f)
        except Exception as e:
            self.logger.log(f"Error loading phonopy file: {e}", 'ERROR')
            return None

        comparison = {
            'phonopy_file': phonopy_yaml,
            'md_method': 'VACF',
            'md_temperature': self.temperature,
        }

        # Extract phonopy thermal properties if available
        if 'thermal_properties' in phonopy_data:
            thermal = phonopy_data['thermal_properties']

            # Extract data
            T_phonopy = np.array([entry['temperature'] for entry in thermal])
            F_phonopy = np.array([entry['free_energy'] for entry in thermal])  # kJ/mol
            S_phonopy = np.array([entry['entropy'] for entry in thermal])  # J/K/mol
            Cv_phonopy = np.array([entry['heat_capacity'] for entry in thermal])  # J/K/mol

            # Note: Unit conversion depends on unit cell size
            # This requires knowing the number of atoms in the unit cell

            comparison['phonopy_temperatures'] = T_phonopy.tolist()
            comparison['phonopy_free_energy'] = F_phonopy.tolist()
            comparison['phonopy_entropy'] = S_phonopy.tolist()
            comparison['phonopy_heat_capacity'] = Cv_phonopy.tolist()

            self.logger.log(f"Loaded phonopy thermal properties for {len(T_phonopy)} temperatures")
            self.logger.log("Note: Unit conversion may be needed based on unit cell size")

        self.logger.log("Phonopy comparison data loaded")
        self.logger.log("Use plot methods with phonopy_data to visualize differences")

        return comparison

    def convergence_analysis(self, max_time_fractions: list = None) -> Dict[str, Any]:
        """
        Analyze convergence of thermodynamic properties with trajectory length.

        Tests whether the trajectory is long enough by recalculating properties
        using different fractions of the total trajectory.

        Parameters:
        -----------
        max_time_fractions : list, optional
            Fractions of trajectory to use (e.g., [0.2, 0.4, 0.6, 0.8, 1.0])
            Default: [0.2, 0.4, 0.6, 0.8, 1.0]

        Returns:
        --------
        convergence : dict
            Dictionary containing:
            - fractions: List of trajectory fractions tested
            - n_frames: Number of frames for each fraction
            - F_vib, S_vib, C_v, E_ZPE: Properties at each fraction
        """
        if max_time_fractions is None:
            max_time_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

        self.logger.log_section("CONVERGENCE ANALYSIS")
        self.logger.log(f"Testing trajectory fractions: {max_time_fractions}")

        convergence = {
            'fractions': max_time_fractions,
            'n_frames': [],
            'F_vib': [],
            'S_vib': [],
            'C_v': [],
            'E_ZPE': []
        }

        # Save original state
        original_atoms_list = self.atoms_list.copy()
        original_velocities = self.velocities.copy()
        original_n_frames = self.n_frames

        # Test each fraction
        for frac in max_time_fractions:
            n_frames_test = int(original_n_frames * frac)
            if n_frames_test < 100:
                self.logger.log(f"Skipping fraction {frac}: too few frames ({n_frames_test})", 'WARNING')
                continue

            self.logger.log(f"Testing with {n_frames_test} frames ({frac*100:.0f}%)")

            # Temporarily truncate trajectory
            self.atoms_list = original_atoms_list[:n_frames_test]
            self.velocities = original_velocities[:n_frames_test]
            self.n_frames = n_frames_test

            # Recalculate VACF and VDOS with appropriate max_lag
            max_lag = min(n_frames_test // 2, int(original_n_frames * 0.5))
            self.calculate_vacf(max_lag=max_lag)
            self.calculate_vdos()

            # Calculate thermodynamic properties at simulation temperature
            T = self.temperature
            F = self.calculate_free_energy([T])[0]
            S = self.calculate_entropy([T])[0]
            Cv = self.calculate_heat_capacity([T])[0]
            E_zpe = self.calculate_zero_point_energy()

            convergence['n_frames'].append(n_frames_test)
            convergence['F_vib'].append(F)
            convergence['S_vib'].append(S)
            convergence['C_v'].append(Cv)
            convergence['E_ZPE'].append(E_zpe)

        # Restore original state and recalculate with full trajectory
        self.atoms_list = original_atoms_list
        self.velocities = original_velocities
        self.n_frames = original_n_frames

        self.calculate_vacf()
        self.calculate_vdos()

        self.logger.log("Convergence analysis complete")
        
        # Check convergence
        if len(convergence['F_vib']) >= 2:
            last_two = convergence['F_vib'][-2:]
            rel_change = abs(last_two[-1] - last_two[-2]) / abs(last_two[-1])
            if rel_change < 0.01:
                self.logger.log(f"✓ F_vib converged (relative change < 1%)")
            else:
                self.logger.log(f"⚠ F_vib may not be converged (relative change = {rel_change*100:.2f}%)", 'WARNING')

        return convergence

    def plot_vacf(self, output_file: str = None, show: bool = False):
        """
        Plot velocity autocorrelation function.
        
        A good VACF should:
        - Start at C(0) = 1.0
        - Decay smoothly to near zero
        - Show oscillations (vibrational modes)
        """
        if self.vacf is None:
            raise ValueError("VACF not calculated. Run calculate_vacf() first.")

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.vacf_time, self.vacf, 'b-', linewidth=2, label='VACF')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axhline(y=1.0, color='r', linestyle=':', alpha=0.3, label='C(0) = 1.0')

        ax.set_xlabel('Time (fs)', fontsize=14)
        ax.set_ylabel('C(t)', fontsize=14)
        ax.set_title(f'Velocity Autocorrelation Function (T = {self.temperature} K)', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.2, 1.2])

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.log(f"VACF plot saved: {output_file}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_vdos(self, output_file: str = None, show: bool = False, phonopy_data: dict = None):
        """
        Plot vibrational density of states.
        
        Parameters:
        -----------
        output_file : str, optional
            Output filename for plot
        show : bool, optional
            Display plot
        phonopy_data : dict, optional
            Phonopy data for comparison
        """
        if self.vdos is None:
            raise ValueError("VDOS not calculated. Run calculate_vdos() first.")

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.omega, self.vdos, 'b-', linewidth=2, label=f'MD VACF (T={self.temperature}K)')
        ax.fill_between(self.omega, 0, self.vdos, alpha=0.2)

        if phonopy_data and 'frequencies' in phonopy_data and 'dos' in phonopy_data:
            ax.plot(phonopy_data['frequencies'], phonopy_data['dos'],
                   'r--', linewidth=2, label='Phonopy (harmonic)', alpha=0.7)

        ax.set_xlabel('Frequency (cm⁻¹)', fontsize=14)
        ax.set_ylabel('VDOS g(ω) (modes/cm⁻¹)', fontsize=14)
        ax.set_title('Vibrational Density of States', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, self.omega[-1])
        ax.set_ylim(bottom=0)

        # Add integral as text
        integral = np.trapz(self.vdos, self.omega)
        ax.text(0.98, 0.98, f'∫g(ω)dω = {integral:.1f}\n(should be {3*self.n_atoms})',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.log(f"VDOS plot saved: {output_file}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_thermodynamics(self, T_array: np.ndarray, output_file: str = None,
                           show: bool = False, phonopy_data: dict = None):
        """
        Plot temperature-dependent thermodynamic properties.

        Creates a 2×2 subplot showing:
        - Free energy F(T)
        - Entropy S(T)
        - Heat capacity C_v(T)
        - Zero-point energy

        Parameters:
        -----------
        T_array : array-like
            Temperature array (K)
        output_file : str, optional
            Output filename for plot
        show : bool, optional
            Display plot
        phonopy_data : dict, optional
            Phonopy data for comparison
        """
        # Calculate properties if not already done
        if self.F_vib is None or len(self.F_vib) != len(T_array):
            self.calculate_free_energy(T_array)
        if self.S_vib is None or len(self.S_vib) != len(T_array):
            self.calculate_entropy(T_array)
        if self.C_v is None or len(self.C_v) != len(T_array):
            self.calculate_heat_capacity(T_array)
        if self.E_ZPE is None:
            self.calculate_zero_point_energy()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Free energy
        axes[0, 0].plot(T_array, self.F_vib, 'b-', linewidth=2, label='MD (VACF)')
        if phonopy_data and 'F_vib' in phonopy_data:
            axes[0, 0].plot(phonopy_data['T'], phonopy_data['F_vib'],
                           'r--', linewidth=2, label='Phonopy (harmonic)')
        axes[0, 0].set_xlabel('Temperature (K)', fontsize=12)
        axes[0, 0].set_ylabel('F_vib (eV/atom)', fontsize=12)
        axes[0, 0].set_title('Vibrational Free Energy', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Entropy
        axes[0, 1].plot(T_array, self.S_vib, 'b-', linewidth=2, label='MD (VACF)')
        if phonopy_data and 'S_vib' in phonopy_data:
            axes[0, 1].plot(phonopy_data['T'], phonopy_data['S_vib'],
                           'r--', linewidth=2, label='Phonopy (harmonic)')
        axes[0, 1].set_xlabel('Temperature (K)', fontsize=12)
        axes[0, 1].set_ylabel('S_vib (eV/K/atom)', fontsize=12)
        axes[0, 1].set_title('Vibrational Entropy', fontsize=14)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Heat capacity
        axes[1, 0].plot(T_array, self.C_v, 'b-', linewidth=2, label='MD (VACF)')
        if phonopy_data and 'C_v' in phonopy_data:
            axes[1, 0].plot(phonopy_data['T'], phonopy_data['C_v'],
                           'r--', linewidth=2, label='Phonopy (harmonic)')
        # Add Dulong-Petit limit
        dulong_petit = 3 * self.kB / self.eV_to_J
        axes[1, 0].axhline(y=dulong_petit, color='k', linestyle=':',
                          alpha=0.5, label=f'Dulong-Petit (3k_B={dulong_petit:.6f})')
        axes[1, 0].set_xlabel('Temperature (K)', fontsize=12)
        axes[1, 0].set_ylabel('C_v (eV/K/atom)', fontsize=12)
        axes[1, 0].set_title('Heat Capacity', fontsize=14)
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)

        # Summary info
        info_text = f'Zero-point energy: {self.E_ZPE:.6f} eV/atom\n'
        info_text += f'Simulation T: {self.temperature} K\n'
        info_text += f'Number of atoms: {self.n_atoms}\n'
        info_text += f'Trajectory length: {self.n_frames * self.timestep / 1000:.1f} ps'
        
        axes[1, 1].text(0.5, 0.5, info_text,
                       ha='center', va='center', fontsize=12,
                       transform=axes[1, 1].transAxes,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.log(f"Thermodynamics plot saved: {output_file}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_convergence(self, convergence: dict, output_file: str = None, show: bool = False):
        """
        Plot convergence analysis results.
        
        Shows how thermodynamic properties converge as trajectory length increases.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        n_frames = convergence['n_frames']
        times = [n * self.timestep / 1000 for n in n_frames]  # Convert to ps

        # Free energy convergence
        axes[0, 0].plot(times, convergence['F_vib'], 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Trajectory length (ps)', fontsize=12)
        axes[0, 0].set_ylabel('F_vib (eV/atom)', fontsize=12)
        axes[0, 0].set_title('Free Energy Convergence', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)

        # Entropy convergence
        axes[0, 1].plot(times, convergence['S_vib'], 'go-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Trajectory length (ps)', fontsize=12)
        axes[0, 1].set_ylabel('S_vib (eV/K/atom)', fontsize=12)
        axes[0, 1].set_title('Entropy Convergence', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)

        # Heat capacity convergence
        axes[1, 0].plot(times, convergence['C_v'], 'ro-', linewidth=2, markersize=8)
        dulong_petit = 3 * self.kB / self.eV_to_J
        axes[1, 0].axhline(y=dulong_petit, color='k', linestyle='--', alpha=0.5,
                          label='Dulong-Petit')
        axes[1, 0].set_xlabel('Trajectory length (ps)', fontsize=12)
        axes[1, 0].set_ylabel('C_v (eV/K/atom)', fontsize=12)
        axes[1, 0].set_title('Heat Capacity Convergence', fontsize=14)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # ZPE convergence
        axes[1, 1].plot(times, convergence['E_ZPE'], 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Trajectory length (ps)', fontsize=12)
        axes[1, 1].set_ylabel('E_ZPE (eV/atom)', fontsize=12)
        axes[1, 1].set_title('Zero-Point Energy Convergence', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            self.logger.log(f"Convergence plot saved: {output_file}")

        if show:
            plt.show()
        else:
            plt.close()

    def export_results(self, output_dir: str, T_array: np.ndarray = None):
        """
        Export all results in multiple formats (NPY, CSV, plots).

        Parameters:
        -----------
        output_dir : str
            Output directory path
        T_array : array-like, optional
            Temperature array for thermodynamic calculations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        self.logger.log_section("EXPORTING RESULTS")
        self.logger.log(f"Output directory: {output_dir}")

        # Export VACF
        if self.vacf is not None:
            # NPY format
            vacf_file = output_dir / 'vacf.npy'
            np.save(vacf_file, np.column_stack([self.vacf_time, self.vacf]))
            self.logger.log(f"VACF data (NPY): {vacf_file}")

            # CSV format
            vacf_csv = output_dir / 'vacf.csv'
            np.savetxt(vacf_csv, np.column_stack([self.vacf_time, self.vacf]),
                      header='time_fs,vacf', delimiter=',', comments='')
            self.logger.log(f"VACF data (CSV): {vacf_csv}")

            # Plot
            self.plot_vacf(output_file=output_dir / 'vacf.png')

        # Export VDOS
        if self.vdos is not None:
            # NPY format
            vdos_file = output_dir / 'vdos.npy'
            np.save(vdos_file, np.column_stack([self.omega, self.vdos]))
            self.logger.log(f"VDOS data (NPY): {vdos_file}")

            # CSV format
            vdos_csv = output_dir / 'vdos.csv'
            np.savetxt(vdos_csv, np.column_stack([self.omega, self.vdos]),
                      header='frequency_cm-1,vdos', delimiter=',', comments='')
            self.logger.log(f"VDOS data (CSV): {vdos_csv}")

            # Plot
            self.plot_vdos(output_file=output_dir / 'vdos.png')

        # Export thermodynamic properties
        if T_array is not None:
            T_array = np.atleast_1d(T_array)

            # Calculate if not already done
            if self.F_vib is None or len(self.F_vib) != len(T_array):
                self.calculate_free_energy(T_array)
            if self.S_vib is None or len(self.S_vib) != len(T_array):
                self.calculate_entropy(T_array)
            if self.C_v is None or len(self.C_v) != len(T_array):
                self.calculate_heat_capacity(T_array)
            if self.E_ZPE is None:
                self.calculate_zero_point_energy()

            # NPY format
            thermo_file = output_dir / 'thermodynamics.npy'
            thermo_data = np.column_stack([T_array, self.F_vib, self.S_vib, self.C_v])
            np.save(thermo_file, thermo_data)
            self.logger.log(f"Thermodynamics data (NPY): {thermo_file}")

            # CSV format
            thermo_csv = output_dir / 'thermodynamics.csv'
            header = 'temperature_K,F_vib_eV_per_atom,S_vib_eV_per_K_per_atom,C_v_eV_per_K_per_atom'
            np.savetxt(thermo_csv, thermo_data, header=header, delimiter=',', comments='')
            self.logger.log(f"Thermodynamics data (CSV): {thermo_csv}")

            # Plot
            self.plot_thermodynamics(T_array, output_file=output_dir / 'thermodynamics.png')

        # Export summary
        summary_file = output_dir / 'summary.txt'
        with open(summary_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("MD-BASED VIBRATIONAL FREE ENERGY CALCULATION - SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("TRAJECTORY INFORMATION:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Trajectory file: {self.trajectory_file}\n")
            f.write(f"Simulation temperature: {self.temperature} K\n")
            f.write(f"MD timestep: {self.timestep} fs\n")
            f.write(f"Number of atoms: {self.n_atoms}\n")
            f.write(f"Number of frames: {self.n_frames}\n")
            f.write(f"Total time: {(self.n_frames - 1) * self.timestep / 1000:.2f} ps\n\n")

            if self.E_ZPE is not None:
                f.write("ZERO-POINT ENERGY:\n")
                f.write("-" * 70 + "\n")
                f.write(f"E_ZPE = {self.E_ZPE:.6f} eV/atom\n\n")

            if self.vacf is not None:
                f.write("VELOCITY AUTOCORRELATION FUNCTION:\n")
                f.write("-" * 70 + "\n")
                f.write(f"Number of points: {len(self.vacf)}\n")
                f.write(f"Maximum lag: {self.vacf_time[-1]:.2f} fs\n")
                f.write(f"VACF[0]: {self.vacf[0]:.6f} (should be 1.0)\n")
                f.write(f"VACF decay at end: {self.vacf[-1]:.6f}\n\n")

            if self.vdos is not None:
                f.write("VIBRATIONAL DENSITY OF STATES:\n")
                f.write("-" * 70 + "\n")
                f.write(f"Number of points: {len(self.vdos)}\n")
                f.write(f"Frequency range: {self.omega[0]:.1f} - {self.omega[-1]:.1f} cm⁻¹\n")
                integral = np.trapz(self.vdos, self.omega)
                f.write(f"VDOS integral: {integral:.2f}\n")
                f.write(f"Expected integral (3N): {3 * self.n_atoms}\n")
                deviation = abs(integral - 3*self.n_atoms) / (3*self.n_atoms) * 100
                f.write(f"Deviation: {deviation:.2f}%\n\n")

            if T_array is not None and self.F_vib is not None:
                f.write("THERMODYNAMIC PROPERTIES:\n")
                f.write("-" * 70 + "\n")
                f.write(f"{'T (K)':>10} {'F_vib':>15} {'S_vib':>15} {'C_v':>15}\n")
                f.write(f"{'':>10} {'(eV/atom)':>15} {'(eV/K/atom)':>15} {'(eV/K/atom)':>15}\n")
                f.write("-" * 70 + "\n")
                for i, T in enumerate(T_array):
                    f.write(f"{T:>10.1f} {self.F_vib[i]:>15.6f} {self.S_vib[i]:>15.6f} {self.C_v[i]:>15.6f}\n")
                
                f.write("\n")
                dulong_petit = 3 * self.kB / self.eV_to_J
                f.write(f"Dulong-Petit limit (3k_B): {dulong_petit:.6f} eV/K/atom\n")

        self.logger.log(f"Summary: {summary_file}")
        self.logger.log("Export complete")


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python free_energy.py <trajectory_file> <temperature>")
        print("Example: python free_energy.py md_300K.traj 300")
        sys.exit(1)
    
    traj_file = sys.argv[1]
    temperature = float(sys.argv[2])
    
    # Create calculator
    fe_calc = MDFreeEnergy(
        trajectory_file=traj_file,
        temperature=temperature,
        timestep=0.5  # fs
    )
    
    # Calculate VACF and VDOS
    fe_calc.calculate_vacf(max_lag=2500, window='hann')
    fe_calc.calculate_vdos(omega_max=4000, n_points=8192)
    
    # Calculate thermodynamic properties
    T_range = np.linspace(0, 500, 100)
    fe_calc.calculate_free_energy(T_range)
    fe_calc.calculate_entropy(T_range)
    fe_calc.calculate_heat_capacity(T_range)
    fe_calc.calculate_zero_point_energy()
    
    # Export results
    output_dir = f"free_energy_{int(temperature)}K"
    fe_calc.export_results(output_dir, T_range)
    
    print(f"\nResults exported to: {output_dir}/")
    print(f"Zero-point energy: {fe_calc.E_ZPE:.6f} eV/atom")