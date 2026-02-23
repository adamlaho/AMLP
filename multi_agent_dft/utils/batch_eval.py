import argparse
import glob
import os
from typing import Dict, List, Optional
import ase.data
import ase.io
import numpy as np
import time

# Import torch first
import torch

# Patch torch.load BEFORE importing e3nn
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    # Force weights_only=False if not specified
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# Now safe to import e3nn and mace
from e3nn import o3
from mace import data
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.tools import torch_geometric, torch_tools, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", help="path to model", required=True)
    parser.add_argument("--output", help="output path", required=True)
    parser.add_argument(
        "--configs",
        help="input file (xyz/cif) OR input directory containing xyz/cif files",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument(
        "--enable_cueq",
        help="enable cuequivariance acceleration",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--batch_size",
        help="batch size for evaluation",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--head",
        help="Model head used for evaluation",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--info_prefix",
        help="prefix for info/array tags in output atoms",
        type=str,
        default="mace_",
    )
    parser.add_argument(
        "--compute",
        help="what to compute: 'e' (energy only), 'ef' (energy+forces, default), or 'efs' (energy+forces+stress)",
        type=str,
        choices=["e", "ef", "efs"],
        default="ef",
    )
    parser.add_argument(
        "--compute_density",
        help="Compute and store density (g/cm³)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--verbose",
        help="Print detailed batch processing information",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--compute_stress",
        help="(Deprecated: use --compute efs) Compute stress tensor",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def compute_density(atoms):
    """
    Compute density in g/cm³
    
    Args:
        atoms: ASE Atoms object
        
    Returns:
        density in g/cm³
    """
    # Get atomic masses in amu (atomic mass units)
    masses = atoms.get_masses()
    total_mass_amu = np.sum(masses)
    
    # Convert to grams (1 amu = 1.66053906660e-24 g)
    total_mass_g = total_mass_amu * 1.66053906660e-24
    
    # Get cell volume in Å³
    volume_ang3 = atoms.get_volume()
    
    # Convert to cm³ (1 Å³ = 1e-24 cm³)
    volume_cm3 = volume_ang3 * 1e-24
    
    # Calculate density in g/cm³
    density = total_mass_g / volume_cm3
    
    return density


def run(args: argparse.Namespace) -> None:
    # Handle backward compatibility
    if args.compute_stress:
        args.compute = "efs"
    
    # Parse what to compute
    compute_forces = args.compute in ["ef", "efs"]
    compute_stress = args.compute == "efs"
    
    # 1. Setup Device and Dtype
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # 2. Load model
    print(f"Loading model: {args.model}")
    model = torch.load(f=args.model, map_location=device)
    
    if args.enable_cueq:
        print("Enabling cuequivariance...")
        model = run_e3nn_to_cueq(model)
        
    model.to(device)
    model.eval()

    # 3. Load data
    atoms_list = []

    if os.path.isdir(args.configs):
        files = glob.glob(os.path.join(args.configs, "*.xyz")) + \
                glob.glob(os.path.join(args.configs, "*.cif"))
        files = sorted(files)
        
        if not files:
            raise ValueError(f"No .xyz or .cif files found in directory: {args.configs}")
        
        print(f"Found {len(files)} file(s)")
        
        for f in files:
            try:
                content = ase.io.read(f, index=":")
                if isinstance(content, list):
                    atoms_list.extend(content)
                else:
                    atoms_list.append(content)
            except Exception as e:
                print(f"Warning: Skipping {os.path.basename(f)} - {str(e)}")
                continue
                
    elif os.path.isfile(args.configs):
        try:
            content = ase.io.read(args.configs, index=":")
            if isinstance(content, list):
                atoms_list.extend(content)
            else:
                atoms_list.append(content)
        except Exception as e:
            raise ValueError(f"Error reading {args.configs}: {str(e)}")
    else:
        raise FileNotFoundError(f"Path not found: {args.configs}")

    if not atoms_list:
        raise ValueError("No valid structures could be loaded!")

    density_flag = " | Density: ON" if args.compute_density else ""
    print(f"Structures: {len(atoms_list)} | Batch size: {args.batch_size} | Computing: {args.compute}{density_flag}")
    # Apply head tag if requested
    if args.head is not None:
        for atoms in atoms_list:
            atoms.info["head"] = args.head

    # Convert to MACE configs
    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    try:
        heads = model.heads
    except AttributeError:
        heads = None

    # Create Data Loader
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max), heads=heads
            )
            for config in configs
        ],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # 4. Evaluation Loop
    energies_list = []
    stresses_list = []
    forces_list_flat = []
    
    total_batches = len(data_loader)
    start_time = time.time()

    for batch_idx, batch in enumerate(data_loader, 1):
        batch_start_time = time.time()
        
        batch = batch.to(device)
        input_data = batch.to_dict()
        
        # Get batch statistics
        num_structures = batch.num_graphs
        num_atoms = batch.num_nodes
        
        # Print progress
        if args.verbose:
            print(f"Batch {batch_idx}/{total_batches}: {num_structures} structures, {num_atoms} atoms")
        else:
            print(f"[{batch_idx}/{total_batches}] {num_structures} structs, {num_atoms} atoms", end="", flush=True)
        
        # Compute
        if compute_forces or compute_stress:
            input_data["positions"].requires_grad_(True)
            output = model(input_data, compute_force=True, compute_stress=compute_stress)
        else:
            with torch.no_grad():
                output = model(input_data, compute_force=False, compute_stress=False)

        # Collect results
        e = torch_tools.to_numpy(output["energy"])
        if e.ndim == 2:
            e = e.flatten()
        energies_list.append(e)

        if compute_stress:
            stresses_list.append(torch_tools.to_numpy(output["stress"]))

        if compute_forces:
            batch_forces = torch_tools.to_numpy(output["forces"])
            split_indices = batch.ptr[1:].cpu().numpy()
            forces_split = np.split(batch_forces, indices_or_sections=split_indices, axis=0)
            forces_list_flat.extend(forces_split[:-1])
        
        # Timing
        batch_time = time.time() - batch_start_time
        if not args.verbose:
            elapsed = time.time() - start_time
            eta = (elapsed / batch_idx) * (total_batches - batch_idx)
            print(f" | {batch_time:.2f}s | ETA: {eta:.0f}s")

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.2f}s ({len(atoms_list)/total_time:.1f} structs/s)")

    # 5. Aggregate Results
    total_energies = np.concatenate(energies_list, axis=0)
    
    if compute_stress:
        total_stresses = np.concatenate(stresses_list, axis=0)
        assert len(atoms_list) == total_stresses.shape[0]

    if compute_forces:
        assert len(atoms_list) == len(forces_list_flat)
    
    assert len(atoms_list) == len(total_energies)

    # 6. Store data
    for i, atoms in enumerate(atoms_list):
        atoms.calc = None
        
        # Clean metadata
        for key in ['occupancy', 'spacegroup', 'unit_cell', 'spacegroup_kinds']:
            atoms.info.pop(key, None)
        for key in ['spacegroup_kinds']:
            atoms.arrays.pop(key, None)
        
        # Add predictions
        atoms.info[f"{args.info_prefix}energy"] = total_energies[i]
        
        if compute_forces:
            atoms.arrays[f"{args.info_prefix}forces"] = forces_list_flat[i]
        
        if compute_stress:
            # Voigt notation: [xx, yy, zz, yz, xz, xy]
            s = total_stresses[i]
            atoms.info[f"{args.info_prefix}stress"] = np.array([
                s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]
            ])
        
        # Compute density if requested
        if args.compute_density:
            density = compute_density(atoms)
            atoms.info[f"{args.info_prefix}density"] = density

    # 7. Write output
    print(f"Writing: {args.output}")
    ase.io.write(args.output, images=atoms_list, format="extxyz")
    print("Done.")


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
