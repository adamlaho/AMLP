"""
Script: md-an.py (Partial replication for layered 2D systems + only original-atom coordination)
Description:
    This script performs one or more operations on an input XYZ file:
      1. Geometry Optimization
      2. Phonon/Vibrational Calculations
      3. Molecular Dynamics (MD) Simulation
      4. Radial Distribution Function (RDF) Analysis
      5. Coordination Number Analysis (Histogram)
      6. Plots and Comparisons
      7. Single-Point Calculation (new option)

    UPDATED FOR PARTIAL REPLICATION:
    - If you have a 2D or layered system, you can replicate only in-plane
      (or as desired) so you do NOT get hundreds of neighbors in the out-of-plane
      direction.
    - The function replicate_cell_for_rdf takes a "replicate_dims" argument
      that can be booleans or integers.
    - Coordination analysis only counts neighbors for the original atoms.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys, re, time, random
import numpy as np
import matplotlib.pyplot as plt
import yaml, torch
from tqdm import tqdm
import itertools, threading
from math import ceil
from scipy.ndimage import gaussian_filter1d

from ase.io import read, write
from ase import units
from ase.geometry.cell import Cell
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import BFGS, LBFGS, FIRE
from ase.build import make_supercell
from ase.geometry import find_mic

np.random.seed(701)

# ----------------------------------------------------------------------
# 1. Helper Functions
# ----------------------------------------------------------------------

def run_with_spinner(fn, *args, **kwargs):
    """Runs a blocking function in a separate thread while displaying a spinner."""
    spinner = itertools.cycle(['-', '/', '|', '\\'])
    result = [None]
    exception = [None]
    def target():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    thread = threading.Thread(target=target)
    thread.start()
    while thread.is_alive():
        sys.stdout.write(next(spinner))
        sys.stdout.flush()
        time.sleep(0.1)
        sys.stdout.write('\b')
    thread.join()
    if exception[0]:
        raise exception[0]
    return result[0]

def load_config(config_filename):
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)
    return config

def parse_xyz_header(filename):
    with open(filename, 'r') as f:
        f.readline()  # Skip number of atoms.
        comment_line = f.readline().strip()
    energy_match = re.search(r'Energy:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', comment_line)
    cell_match = re.search(r'Cell:\s*([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)', comment_line)
    if not energy_match or not cell_match:
        raise ValueError("Could not parse header for energy or cell parameters in file: " + filename)
    energy_hartree = float(energy_match.group(1))
    a = float(cell_match.group(1))
    b = float(cell_match.group(2))
    c = float(cell_match.group(3))
    alpha = float(cell_match.group(4))
    beta = float(cell_match.group(5))
    gamma = float(cell_match.group(6))
    return energy_hartree, [a, b, c, alpha, beta, gamma]

def select_band_path(lattice_description):
    """(Optional) Returns a typical band path for various lattice types."""
    desc = lattice_description.lower()
    if "mcl(" in desc or "variant name: mcl" in desc:
        return "GYHCEM1AXH1,MDZ,YD"
    if "face-centered cubic" in desc or "fcc" in desc:
        return "GXWKGLUWLK,UX"
    elif "body-centered cubic" in desc or "bcc" in desc:
        return "GHNGPH,PN"
    elif "primitive cubic" in desc or "cubic" in desc:
        return "GXMGRX,MR"
    elif "tetragonal" in desc:
        return "GXMGZRAZ,XR,MA"
    elif "body-centered tetragonal" in desc:
        return "GXMGZPNZ1M,XP"
    elif "orthorhombic" in desc:
        if "primitive" in desc:
            return "GXSYGZURTZ,YT,UX,SR"
        elif "face-centered" in desc:
            return "GYTZGXA1Y,TX1,XAZ,LG"
        elif "body-centered" in desc:
            return "GXLTWRX1ZGYSW,L1Y,Y1Z"
        elif "base-centered" in desc:
            return "GXSRAZGYX1A1TY,ZT"
    elif "hexagonal" in desc:
        return "GMKGALHA,LM,KH"
    elif "rhombohedral" in desc:
        return "GLB1,BZGX,QFP1Z,LP"
    elif "monoclinic" in desc:
        if "primitive" in desc:
            return "GYHCEM1AXH1,MDZ,YD"
        elif "base-centered" in desc:
            return "GYFLI,I1ZF1,YX1,XGN,MG"
    elif "triclinic" in desc:
        return "XGY,LGZ,NGM,RG"
    elif "oblique" in desc:
        return "GYHCH1XG"
    elif "rectangular" in desc:
        if "primitive" in desc:
            return "GXSYGS"
        else:
            return "GXA1YG"
    elif "square" in desc:
        return "MGXM"
    elif "line" in desc:
        return "GX"
    else:
        raise ValueError("Unknown lattice type: " + lattice_description)

# ----------------------------------------------------------------------
# 2. Partial Replication for 2D or 3D
# ----------------------------------------------------------------------

def replicate_cell_for_rdf(atoms, rmax, margin=2.0, replicate_dims=None):
    """
    Replicate each lattice vector so that its length*N_i >= (rmax + margin),
    BUT only for directions the user wants to replicate.

    replicate_dims can be:
      - None (default): replicate all 3 directions if needed
      - A list/tuple of booleans, e.g. [True, True, False], meaning replicate
        the first two directions but not the third (2D system).
      - A list/tuple of integers, e.g. [2, 2, 1], meaning replicate exactly
        Nx=2, Ny=2, Nz=1 times.

    If your system is 2D and you do not want to replicate out-of-plane,
    set replicate_dims = [True, True, False] or something similar.
    """
    cell = atoms.cell
    a1, a2, a3 = cell[0], cell[1], cell[2]
    L1 = np.linalg.norm(a1)
    L2 = np.linalg.norm(a2)
    L3 = np.linalg.norm(a3)

    if replicate_dims is None:
        replicate_dims = [True, True, True]

    Nx = 1
    Ny = 1
    Nz = 1

    if all(isinstance(x, bool) for x in replicate_dims):
        if replicate_dims[0]:
            Nx = max(1, int(np.ceil((rmax + margin)/ L1)))
        if replicate_dims[1]:
            Ny = max(1, int(np.ceil((rmax + margin)/ L2)))
        if replicate_dims[2]:
            Nz = max(1, int(np.ceil((rmax + margin)/ L3)))
    else:
        Nx, Ny, Nz = replicate_dims

    if Nx == 1 and Ny == 1 and Nz == 1:
        return atoms.copy()

    transform_matrix = np.diag([Nx, Ny, Nz])
    super_atoms = make_supercell(atoms, transform_matrix)

    print(f"Replicating cell with Nx={Nx}, Ny={Ny}, Nz={Nz}")
    print("New supercell vectors:\n", super_atoms.cell)
    return super_atoms

# ----------------------------------------------------------------------
# 3. RDF Routines
# ----------------------------------------------------------------------

def compute_rdf_custom(atoms, rmin=0.0, rmax=10.0, nbins=100, atom_types="all", use_pbc=True):
    n_atoms = len(atoms)
    volume = atoms.get_volume()
    density = n_atoms / volume
    distances = []

    if atom_types != "all" and not isinstance(atom_types, list):
        atom_types = [atom_types]

    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            if atom_types == "all" or (atoms[i].symbol in atom_types and atoms[j].symbol in atom_types):
                d = atoms.get_distance(i, j, mic=use_pbc)
                distances.append(d)

    distances = np.array(distances)
    bins = np.linspace(rmin, rmax, nbins+1)
    hist, edges = np.histogram(distances, bins=bins)
    r = 0.5 * (edges[1:] + edges[:-1])
    dr = edges[1] - edges[0]
    shell_volumes = 4 * np.pi * r**2 * dr
    norm = n_atoms * density * shell_volumes
    g_r = hist / norm
    return r, g_r

def compute_and_plot_rdf(atoms, label, output_dir, base_name, suffix, config):
    rmin = config.get("rdf_min", 0.0)
    rmax = config.get("rdf_rmax", 10.0)
    nbins = config.get("rdf_nbins", 100)
    atom_types = config.get("rdf_atom_types", "all")
    use_pbc = config.get("pbc", True)

    replicate_dims = config.get("replicate_dims", None)
    replicate_atoms = replicate_cell_for_rdf(atoms, rmax, margin=2.0, replicate_dims=replicate_dims)

    print(f"\nCalculating RDF for '{label}'")
    print("Replicated cell for RDF, new cell:\n", replicate_atoms.get_cell())
    print("Using periodic boundary conditions (mic):", use_pbc)

    r, g_r = compute_rdf_custom(replicate_atoms, rmin=rmin, rmax=rmax, nbins=nbins,
                                atom_types=atom_types, use_pbc=use_pbc)
    rdf_filename = os.path.join(output_dir, f"{base_name}_{suffix}_rdf.png")

    plt.figure()
    plt.plot(r, g_r, label=label)
    plt.xlabel("r (Å)")
    plt.ylabel("g(r)")
    plt.title(f"RDF: {label}")
    plt.legend()
    plt.xlim(rmin, rmax)
    plt.savefig(rdf_filename, dpi=config.get("plot_dpi", 300))
    plt.close()
    print(f"RDF plot saved as {rdf_filename}")
    return r, g_r

def compute_average_rdf_from_trajectory(trajectory_file, output_dir, base_name, suffix, config):
    rmin = config.get("rdf_min", 0.0)
    rmax = config.get("rdf_rmax", 10.0)
    nbins = config.get("rdf_nbins", 100)
    atom_types = config.get("rdf_atom_types", "all")
    use_pbc = config.get("pbc", True)
    replicate_dims = config.get("replicate_dims", None)

    frames = read(trajectory_file, index=':')
    total_frames = len(frames)
    nframes_limit = config.get("rdf_nframes", None)
    if nframes_limit is not None and isinstance(nframes_limit, int) and nframes_limit < total_frames:
        frames = random.sample(frames, nframes_limit)
        print(f"Randomly selected {nframes_limit} frames out of {total_frames} in {trajectory_file}")
    else:
        print(f"Processing {total_frames} frames in {trajectory_file}")

    rdf_list = []
    for frame_atoms in tqdm(frames, desc="Processing configurations", unit="conf"):
        replicate_atoms = replicate_cell_for_rdf(frame_atoms, rmax, margin=2.0, replicate_dims=replicate_dims)
        r_f, g_f = compute_rdf_custom(replicate_atoms, rmin=rmin, rmax=rmax, nbins=nbins,
                                      atom_types=atom_types, use_pbc=use_pbc)
        rdf_list.append(g_f)

    rdf_array = np.array(rdf_list)
    avg_g_r = np.mean(rdf_array, axis=0)

    replicate_test = replicate_cell_for_rdf(frames[0], rmax, margin=2.0, replicate_dims=replicate_dims)
    r_bins, _ = compute_rdf_custom(replicate_test, rmin=rmin, rmax=rmax, nbins=nbins,
                                   atom_types=atom_types, use_pbc=use_pbc)

    rdf_filename = os.path.join(output_dir, f"{base_name}_{suffix}_avg_rdf.png")
    plt.figure()
    plt.plot(r_bins, avg_g_r, label="Average RDF")
    plt.xlabel("r (Å)")
    plt.ylabel("g(r)")
    plt.title("Average RDF from MD Trajectory")
    plt.legend()
    plt.xlim(rmin, rmax)
    plt.savefig(rdf_filename, dpi=config.get("plot_dpi", 300))
    plt.close()
    print(f"Average RDF plot saved as {rdf_filename}")
    return r_bins, avg_g_r

def process_existing_md_trajectories(md_traj_dir, output_dir, base_name, config):
    md_files = [f for f in os.listdir(md_traj_dir) if f.endswith("_md.xyz")]
    if not md_files:
        print("No MD trajectory files found in", md_traj_dir)
        return {}
    md_rdf_results = {}
    for idx, f in enumerate(md_files, start=1):
        print(f"Processing trajectory {idx}/{len(md_files)}: {f}")
        trajectory_file = os.path.join(md_traj_dir, f)
        try:
            temp_str = f.split("_T")[1].split("_md")[0]
            temp = float(temp_str)
        except Exception as e:
            print(f"Could not extract temperature from filename {f}: {e}")
            continue

        if config.get("rdf_average", False):
            r, g = compute_average_rdf_from_trajectory(trajectory_file, os.path.dirname(trajectory_file),
                                                        f.replace(".xyz",""), "rdf", config)
        else:
            last_atoms = read(trajectory_file, index=-1)
            r, g = compute_and_plot_rdf(last_atoms, label=f"MD T={temp}K",
                                        output_dir=os.path.dirname(trajectory_file),
                                        base_name=f.replace(".xyz",""), suffix="rdf", config=config)
        md_rdf_results[temp] = (r, g)

    if md_rdf_results:
        plt.figure()
        for temp, (r, g) in md_rdf_results.items():
            plt.plot(r, g, label=f"MD {temp}K")
        plt.xlabel("r (Å)")
        plt.ylabel("g(r)")
        plt.title("RDF from MD Trajectories")
        plt.legend()
        combined_rdf_filename = os.path.join(output_dir, f"{base_name}_combined_rdf.png")
        plt.savefig(combined_rdf_filename, dpi=config.get("plot_dpi", 300))
        plt.close()
        print(f"Combined RDF plot saved as {combined_rdf_filename}")
    return md_rdf_results

# ----------------------------------------------------------------------
# 4. Optimization, Phonons, MD
# ----------------------------------------------------------------------

def run_optimization(atoms, config, base_name, opt_log_filename, results_dir):
    optimizer_choice = config.get("optimizer", "BFGS").upper()
    fmax = float(config.get("fmax", 0.001))
    traj_file = os.path.join(results_dir, config.get("optimizer_trajectory", base_name + "_opt.traj"))
    restart_file = config.get("optimizer_restart", None)

    print(f"Using optimizer: {optimizer_choice}")
    if optimizer_choice in ("BFGS", "BFGSLINESEARCH", "QUASINEWTON"):
        opt = BFGS(atoms, logfile=opt_log_filename, trajectory=traj_file, restart=restart_file)
    elif optimizer_choice == "LBFGS":
        opt = LBFGS(atoms, logfile=opt_log_filename, trajectory=traj_file, restart=restart_file)
    elif optimizer_choice == "FIRE":
        opt = FIRE(atoms, logfile=opt_log_filename, trajectory=traj_file)
    else:
        print(f"Optimizer '{optimizer_choice}' not recognized; defaulting to BFGS.")
        opt = BFGS(atoms, logfile=opt_log_filename, trajectory=traj_file, restart=restart_file)

    progress_bar = tqdm(desc="Optimization Iteration", unit="iter")
    def optimization_callback():
        progress_bar.update(1)
    opt.attach(optimization_callback, interval=1)

    print("Starting geometry optimization using", optimizer_choice, "optimizer...")
    opt.run(fmax=fmax)
    progress_bar.close()

    optimized_energy = atoms.get_potential_energy()
    print(f"Optimized energy: {optimized_energy:.3f} eV")
    torch.cuda.empty_cache()

    r_opt, g_opt = compute_and_plot_rdf(atoms, label="Optimized",
                                        output_dir=results_dir,
                                        base_name=base_name,
                                        suffix="opt",
                                        config=config)
    return r_opt, g_opt

def calculate_phonons(atoms, base_name, calc, displacement=0.01, output_dir="."):
    from ase.vibrations import Vibrations
    original_cwd = os.getcwd()
    try:
        os.chdir(output_dir)
        vib_name = base_name + "_vib"
        print("Changing working directory to", output_dir, "for phonon calculation.")
        vib = Vibrations(atoms, name=vib_name, delta=displacement)
        print("Running vibrational (phonon) calculation using Vibrations...")
        run_with_spinner(vib.run)
        vib.summary()
        frequencies = vib.get_frequencies()
        frequencies = np.real_if_close(frequencies, tol=1e-5)
        vib_filename = os.path.join(output_dir, base_name + "_vib.txt")
        with open(vib_filename, "w") as f:
            f.write("Vibration frequencies (cm^-1):\n")
            for freq in frequencies:
                f.write(f"{freq}\n")
        print(f"Vibration frequencies saved to {vib_filename}")
        if config.get("phonon_save_data", True):
            raw_filename = os.path.join(output_dir, base_name + "_vib_data.npy")
            np.save(raw_filename, frequencies)
            print(f"Raw vibration data saved to {raw_filename}")
        torch.cuda.empty_cache()
    finally:
        os.chdir(original_cwd)
    return frequencies

def plot_phonon_bandstructure(atoms, base_name, calc, band_path_str, supercell=(3,3,3), delta=0.05, output_dir="."):
    from ase.phonons import Phonons
    ph = Phonons(atoms, calc, supercell=supercell, delta=delta)
    ph.run()
    ph.read(acoustic=True)
    ph.clean()

    grid = tuple(config.get("phonon_grid", [20,20,20]))
    npts = config.get("phonon_npts", 200)
    width = config.get("phonon_width", 1e-3)

    dos = ph.get_dos(kpts=grid).sample_grid(npts=npts, width=width)
    emax = dos.get_energies().max() * 1.1

    fig = plt.figure(figsize=(7,4))
    ax = fig.add_axes([0.12,0.07,0.75,0.85])
    ax.plot(dos.get_weights(), dos.get_energies(), label="DOS")
    ax.set_xlabel("DOS (arb. units)")
    ax.set_ylabel("Frequency (cm^-1)")
    ax.set_title("Phonon DOS")
    plt.savefig(os.path.join(output_dir, base_name + "_vib_DOS.png"), dpi=config.get("plot_dpi", 300))
    plt.close()
    print(f"Phonon DOS plot saved as {os.path.join(output_dir, base_name + '_vib_DOS.png')}")
    torch.cuda.empty_cache()

def plot_full_phonon_dispersion(ph, base_name, output_dir=".", grid_points=50, fixed_kz=0.0, branch=0):
    import numpy as np
    import matplotlib.pyplot as plt
    grid_points = config.get("phonon_dispersion_grid_points", grid_points)
    fixed_kz = config.get("phonon_fixed_kz", fixed_kz)
    branch = config.get("phonon_branch", branch)
    contour_levels = config.get("phonon_contour_levels", 100)

    kx = np.linspace(0,1,grid_points)
    ky = np.linspace(0,1,grid_points)
    KX, KY = np.meshgrid(kx,ky)
    frequencies = np.zeros_like(KX)

    for i in range(grid_points):
        for j in range(grid_points):
            kpt = [KX[j,i], KY[j,i], fixed_kz]
            try:
                freq = ph.get_frequencies(kpt)
                frequencies[j,i] = freq[branch]
            except Exception as e:
                frequencies[j,i] = np.nan
                print(f"Warning: could not compute frequencies at kpt {kpt}: {e}")

    plt.figure(figsize=(8,6))
    cp = plt.contourf(KX, KY, frequencies, levels=contour_levels, cmap='viridis')
    plt.colorbar(cp, label="Frequency (arb. units)")
    plt.xlabel("$k_x$ (fractional)")
    plt.ylabel("$k_y$ (fractional)")
    plt.title(f"Vibration Dispersion (branch {branch}) at fixed $k_z={fixed_kz}$")
    dispersion_filename = os.path.join(output_dir, base_name + f"_full_vib_dispersion_branch{branch}_kz{fixed_kz}.png")
    plt.savefig(dispersion_filename, dpi=config.get("plot_dpi", 300))
    plt.close()
    print(f"Full vibration dispersion plot (2D slice) saved as {dispersion_filename}")
    torch.cuda.empty_cache()

# ----------------------------------------------------------------------
# 5. Coordination Analysis
# ----------------------------------------------------------------------

def compute_neighbors_for_original_atoms(original_atoms, cutoff=12.0, atom_types=["N"], margin=2.0, replicate_dims=None):
    """
    For each 'center' atom in the original cell, count how many neighbors
    are within 'cutoff' in a partially replicated environment. We do not
    count neighbors for the replicated "center" atoms, only for the original
    atoms.
    
    replicate_dims can be booleans or integers. If you have a 2D system,
    you might do replicate_dims=[True, True, False], etc.
    """
    env_atoms = replicate_cell_for_rdf(original_atoms, rmax=cutoff, margin=margin,
                                       replicate_dims=replicate_dims)

    center_positions = []
    if atom_types == "all":
        center_positions = [atom.position for atom in original_atoms]
    else:
        if isinstance(atom_types, str):
            atom_types = [atom_types]
        for atom in original_atoms:
            if atom.symbol in atom_types:
                center_positions.append(atom.position)

    neighbor_counts = []
    env_positions = env_atoms.get_positions()
    cell = env_atoms.cell
    pbc = env_atoms.pbc

    for cpos in center_positions:
        delta = env_positions - cpos
        mic_vecs, dist = find_mic(delta, cell=cell, pbc=pbc)
        same_mask = dist < 1e-6
        is_neighbor = (dist < cutoff) & (~same_mask)
        neighbor_counts.append(np.sum(is_neighbor))

    return neighbor_counts

def plot_coordination_numbers(atoms, output_dir, base_name, config, label=""):
    """
    Plot histogram of neighbor counts for original atoms, referencing
    a partially replicated environment (if needed).
    """
    cutoff = config.get("coordination_cutoff", 3.0)
    atom_types = config.get("rdf_atom_types", "all")
    replicate_dims = config.get("replicate_dims", None)

    neighbor_counts = compute_neighbors_for_original_atoms(
        original_atoms=atoms,
        cutoff=cutoff,
        atom_types=atom_types,
        margin=2.0,
        replicate_dims=replicate_dims
    )

    if len(neighbor_counts) == 0:
        print("No center atoms found for the specified types, skipping histogram.")
        return

    plt.figure()
    bins = range(min(neighbor_counts), max(neighbor_counts)+2)
    plt.hist(neighbor_counts, bins=bins, align='left', edgecolor='black')
    plt.xlabel("Number of neighbors")
    plt.ylabel("Frequency")
    plt.title(f"Coordination histogram for {label} atoms: {atom_types}\n(Cutoff = {cutoff} Å)")
    plot_filename = os.path.join(output_dir, f"{base_name}_{label}_coordination_hist.png")
    plt.savefig(plot_filename, dpi=config.get("plot_dpi", 300))
    plt.close()
    print(f"Coordination number histogram saved as {plot_filename}")

# ----------------------------------------------------------------------
# 6. MD Routines
# ----------------------------------------------------------------------

def simpleMD(atoms, temp, calc, trajectory, interval, steps, timestep=0.5, init_vel=True):
    """
    Run an MD simulation and compute the RDF on the final geometry (or average if rdf_average=True).
    """
    atoms.set_calculator(calc)
    if init_vel:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
        Stationary(atoms)
        ZeroRotation(atoms)
    dyn = Langevin(atoms, timestep * units.fs, temperature_K=temp,
                   friction=config.get("md_friction", 0.01))
    times, energies, temperatures, speeds = [], [], [], []
    if os.path.exists(trajectory):
        os.remove(trajectory)
    log_filename = os.path.splitext(trajectory)[0] + '.log'
    f_log = open(log_filename, 'w')
    f_log.write("Time(fs), Energy(eV), Temperature(K), Speed(ns/day)\n")
    start_time = time.time()
    md_bar = tqdm(total=steps, desc=f"MD simulation at {temp}K", unit="step")
    counter = 0

    def log_and_save():
        nonlocal counter
        current_time = dyn.get_time() / units.fs
        current_energy = atoms.get_potential_energy()
        current_temp = atoms.get_temperature()
        elapsed_wall_time = time.time() - start_time
        simulation_time_ns = current_time / 1e6
        speed_ns_per_day = (simulation_time_ns * 86400 / elapsed_wall_time) if elapsed_wall_time > 0 else 0.0
        times.append(current_time)
        energies.append(current_energy)
        temperatures.append(current_temp)
        speeds.append(speed_ns_per_day)
        write(trajectory, atoms, format='extxyz', append=True)
        log_message = (f"Time: {current_time:8.2f} fs | Energy: {current_energy:12.3f} eV | "
                       f"Temperature: {current_temp:8.2f} K | Speed: {speed_ns_per_day:8.2f} ns/day")
        print(log_message)
        f_log.write(f"{current_time:.2f}, {current_energy:.3f}, {current_temp:.2f}, {speed_ns_per_day:.2f}\n")
        f_log.flush()
        counter += interval
        md_bar.update(interval)

    dyn.attach(log_and_save, interval=interval)
    t_start = time.time()
    dyn.run(steps)
    t_end = time.time()
    f_log.close()
    md_bar.close()
    print(f"\nMD simulation at {temp} K completed in {(t_end-t_start)/60:.2f} minutes.")
    base_md_name = os.path.splitext(os.path.basename(trajectory))[0]

    if config.get("rdf_average", False):
        r_md, g_md = compute_average_rdf_from_trajectory(trajectory, os.path.dirname(trajectory),
                                                         base_md_name, "rdf", config)
    else:
        r_md, g_md = compute_and_plot_rdf(atoms, label=f"MD T={temp}K",
                                          output_dir=os.path.dirname(trajectory),
                                          base_name=base_md_name,
                                          suffix="rdf",
                                          config=config)
    return r_md, g_md

def run_md(atoms, config, base_name, calc, md_traj_dir):
    """
    Runs multiple MD simulations at different temperatures (Temp_initial -> Temp_final in steps).
    """
    temp_initial = config.get("Temp_initial", 50)
    temp_final = config.get("Temp_final", 350)
    temp_step = config.get("Temp_step", 25)
    md_steps = config.get("Step", 2000000)
    timestep = config.get("timestep", 0.5)
    save_interval = config.get("MD_save_interval", 5000)
    first_simulation = True
    md_rdf_results = {}

    temp_range = np.arange(temp_initial, temp_final+temp_step, temp_step)
    for temp in temp_range:
        trajectory_file = os.path.join(md_traj_dir, f"{base_name}_T{int(temp)}_md.xyz")
        print(f"\nStarting MD simulation at {temp} K...")
        r_md, g_md = simpleMD(atoms, temp=temp, calc=calc,
                              trajectory=trajectory_file, interval=save_interval, steps=md_steps,
                              timestep=timestep, init_vel=first_simulation)
        md_rdf_results[temp] = (r_md, g_md)
        first_simulation = False
        torch.cuda.empty_cache()
    return md_rdf_results

# ----------------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Usage: python3 md-an.py <input_file.xyz> config_file=\"config.yaml\"")
    input_file = os.path.abspath(sys.argv[1])
    config_filename = sys.argv[2].split("=")[-1].strip('"')
    config = load_config(config_filename)

    plot_dpi = config.get("plot_dpi", 300)

    results_dir = os.path.abspath(config.get("output_dir", "."))
    os.makedirs(results_dir, exist_ok=True)
    md_traj_dir = os.path.join(results_dir, config.get("md_trajectory_dir", "md"))
    os.makedirs(md_traj_dir, exist_ok=True)
    plot_dir = os.path.join(results_dir, config.get("plot_dir", "plots"))
    os.makedirs(plot_dir, exist_ok=True)
    phonon_dir = os.path.join(results_dir, config.get("phonon_dir", "phonon"))
    os.makedirs(phonon_dir, exist_ok=True)

    print("Results will be stored in:", results_dir)

    # --- Determine cell parameters ---
    if config.get("readcell_info", True):
        ref_energy, cell_params = parse_xyz_header(input_file)
        print(f"Cell parameters from file: a={cell_params[0]:.3f}, b={cell_params[1]:.3f}, c={cell_params[2]:.3f}, "
              f"alpha={cell_params[3]:.1f}, beta={cell_params[4]:.1f}, gamma={cell_params[5]:.1f}")
    else:
        cell_params = config.get("cell_params")
        if cell_params is None:
            raise ValueError("readcell_info is set to False, but cell_params is not provided in the config file.")
        print(f"Cell parameters from config: a={cell_params[0]:.3f}, b={cell_params[1]:.3f}, c={cell_params[2]:.3f}, "
              f"alpha={cell_params[3]:.1f}, beta={cell_params[4]:.1f}, gamma={cell_params[5]:.1f}")

    atoms = read(input_file, index=0)
    atoms.set_cell(Cell.fromcellpar(cell_params))

    pbc_flag = config.get("pbc", True)
    atoms.set_pbc(pbc_flag)
    print("Using periodic boundary conditions (PBC) for RDF calculations:", pbc_flag)
    print("Cell used for RDF calculations:", atoms.get_cell())

    # --- Initialize MACE Calculator (example) ---
    from mace.calculators import MACECalculator
    device_option = config.get("device", "gpu").lower()
    if device_option == "cpu":
        device_str = "cpu"
    else:
        gpu_devices = config.get("gpus", ["cuda:0"])
        device_str = ",".join(gpu_devices)
    model_paths = config.get("model_paths", ["/path/to/your/model"])
    mace_calc = MACECalculator(model_paths=model_paths, device=device_str)
    atoms.calc = mace_calc

    # Save a copy as the DFT_reference
    refDFT_atoms = atoms.copy()
    refDFT_atoms.calc = mace_calc
    refDFT_atoms.set_pbc(pbc_flag)
    print("DFT_reference structure:")
    print("Cell parameters:", refDFT_atoms.get_cell())
    print("Using PBC for RDF calculation:", pbc_flag)

    # Coordination analysis for DFT_reference (only original atoms as centers)
    plot_coordination_numbers(refDFT_atoms, plot_dir,
                              os.path.splitext(os.path.basename(input_file))[0] + "_DFT",
                              config, label="DFT_reference")

    # Disable symmetry reduction
    atoms.info['symprec'] = 0.0
    refDFT_atoms.info['symprec'] = 0.0

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    opt_log_filename = os.path.join(results_dir, f"{base_name}_opt.log")

    # --- Single-Point Calculation ---
    if config.get("single_point", False):
        print("\nPerforming single-point calculation on the input structure...")
        sp_energy = atoms.get_potential_energy()
        sp_forces = atoms.get_forces()
        sp_file = os.path.join(results_dir, f"{base_name}_sp.xyz")
        write(sp_file, atoms)
        print(f"Single-point calculation results:")
        print(f"  Energy: {sp_energy:.3f} eV")
        print(f"  Forces:\n{sp_forces}")
        print(f"Structure saved as {sp_file}")

    # --- Geometry Optimization ---
    rdf_opt = None
    if config.get("geo_opt", False):
        print("\nRunning geometry optimization...")
        r_opt, g_opt = run_optimization(atoms, config, base_name, opt_log_filename, results_dir)
        rdf_opt = (r_opt, g_opt)
        opt_file = os.path.join(results_dir, f"{base_name}_opt.xyz")
        write(opt_file, atoms)
        print(f"Optimized structure saved as {opt_file}")

        plot_coordination_numbers(atoms, plot_dir, base_name + "_opt", config, label="Optimized")

    # --- Vibrational Calculations for DFT_reference ---
    if config.get("vib_input", True):
        displacement = config.get("phonon_delta", 0.01)
        print("\nCalculating vibrations for the input (DFT_reference) structure...")
        ref_frequencies = calculate_phonons(refDFT_atoms, base_name + "_ref", mace_calc,
                                             displacement=displacement, output_dir=phonon_dir)
        print("DFT_reference vibration frequencies (cm^-1):")
        print(ref_frequencies)

        try:
            band_path_str_ref = select_band_path(refDFT_atoms.cell.get_bravais_lattice().description())
        except Exception as e:
            print(f"Error selecting band path for reference: {e}")
            band_path_str_ref = "GXWKGLUWLK,UX"

        from ase.phonons import Phonons
        plot_phonon_bandstructure(refDFT_atoms, base_name + "_ref", mace_calc, band_path_str_ref,
                                  supercell=tuple(config.get("phonon_supercell", [3,3,3])),
                                  delta=config.get("phonon_band_delta", 0.05),
                                  output_dir=plot_dir)

    # --- Vibrational Calculations for Optimized Structure ---
    if config.get("phonon", False):
        displacement = config.get("phonon_delta", 0.01)
        supercell = tuple(config.get("phonon_supercell", [3,3,3]))
        band_delta = config.get("phonon_band_delta", 0.05)
        print("\nCalculating vibrations for the optimized structure...")
        frequencies = calculate_phonons(atoms, base_name, mace_calc,
                                        displacement=displacement, output_dir=phonon_dir)
        print("Optimized vibration frequencies (cm^-1):")
        print(frequencies)

        if config.get("phonon_plot", True):
            try:
                band_path_str = select_band_path(atoms.cell.get_bravais_lattice().description())
            except Exception as e:
                print(f"Error selecting band path for optimized: {e}")
                band_path_str = "GXWKGLUWLK,UX"

            plot_phonon_bandstructure(atoms, base_name, mace_calc, band_path_str,
                                      supercell=supercell, delta=band_delta, output_dir=plot_dir)
            from ase.phonons import Phonons
            ph_full = Phonons(atoms, mace_calc, supercell=supercell, delta=band_delta)
            ph_full.run()
            try:
                if ph_full.cache and any(ph_full.cache.values()):
                    ph_full.read(acoustic=True)
                else:
                    print("Phonon full dispersion cache is empty, skipping read.")
            except Exception as e:
                print(f"Error reading phonon data for full dispersion: {e}")
            ph_full.clean()
            plot_full_phonon_dispersion(ph_full, base_name,
                                        grid_points=config.get("phonon_dispersion_grid_points", 50),
                                        fixed_kz=config.get("phonon_fixed_kz", 0.0),
                                        branch=config.get("phonon_branch", 0),
                                        output_dir=plot_dir)

    # --- MD RDF Processing (if available) ---
    md_rdf_results = {}
    if config.get("MD_run", False):
        md_rdf_results = run_md(atoms, config, base_name, mace_calc, md_traj_dir)
    elif config.get("rdf_run", False):
        print("\nProcessing existing MD trajectory files for RDF...")
        md_rdf_results = process_existing_md_trajectories(md_traj_dir, results_dir, base_name, config)

    # --- RDF Comparison: DFT_reference vs. Optimized vs. Individual MD Curves ---
    if config.get("rdf_compare", False):
        print("\nComputing RDF comparison among DFT_reference, Optimized, and MD curves...")
        r_dft, g_dft = compute_and_plot_rdf(refDFT_atoms, label="DFT_reference",
                                            output_dir=results_dir,
                                            base_name=base_name+"_DFT",
                                            suffix="rdf", config=config)
        sigma = config.get("rdf_smoothing_sigma", 2)
        g_dft_smoothed = gaussian_filter1d(g_dft, sigma=sigma) if sigma > 0 else g_dft

        r_opt, g_opt = compute_and_plot_rdf(atoms, label="Optimized",
                                            output_dir=results_dir,
                                            base_name=base_name+"_opt",
                                            suffix="rdf", config=config)
        g_opt_smoothed = gaussian_filter1d(g_opt, sigma=sigma) if sigma > 0 else g_opt

        plt.figure()
        plt.plot(r_dft, g_dft_smoothed, label="DFT_reference")
        plt.plot(r_opt, g_opt_smoothed, label="MACE-Optimized")
        for temp, (r_md, g_md) in md_rdf_results.items():
            plt.plot(r_md, g_md, label=f"MD {temp}K")
        plt.xlabel("r (Å)")
        plt.ylabel("g(r)")
        plt.title("RDF Comparison")
        plt.legend()
        rdf_compare_filename = os.path.join(results_dir, f"{base_name}_rdf_comparison.png")
        plt.savefig(rdf_compare_filename, dpi=plot_dpi)
        plt.close()
        print(f"RDF comparison plot saved as {rdf_compare_filename}")

    # --- Combined RDF Plot from MD (if desired) ---
    if config.get("rdf_run", False) and md_rdf_results:
        plt.figure()
        for temp, (r_md, g_md) in md_rdf_results.items():
            plt.plot(r_md, g_md, label=f"MD {temp}K")
        plt.xlabel("r (Å)")
        plt.ylabel("g(r)")
        plt.title("Combined RDF from MD Trajectories")
        plt.legend()
        combined_rdf_filename = os.path.join(results_dir, f"{base_name}_combined_rdf.png")
        plt.savefig(combined_rdf_filename, dpi=plot_dpi)
        plt.close()
        print(f"Combined RDF plot saved as {combined_rdf_filename}")
