#!/usr/bin/env python3
"""
Script to patch MACE's atomic_data.py to fix the PBC tensor warning.
This patches the MACE library code to remove the unnecessary list wrapper.
"""

import sys
import os
from pathlib import Path

def find_mace_path():
    """Find the MACE installation path."""
    try:
        import mace
        mace_path = Path(mace.__file__).parent
        return mace_path
    except ImportError:
        print("ERROR: MACE is not installed!")
        return None

def patch_atomic_data(mace_path):
    """Patch the atomic_data.py file."""
    atomic_data_file = mace_path / "data" / "atomic_data.py"

    if not atomic_data_file.exists():
        print(f"ERROR: Could not find {atomic_data_file}")
        return False

    print(f"Found MACE atomic_data.py at: {atomic_data_file}")

    # Read the file
    with open(atomic_data_file, 'r') as f:
        content = f.read()

    # Check if already patched
    if 'torch.tensor(config.pbc, dtype=torch.bool)' in content:
        print("✓ File is already patched!")
        return True

    # Create backup
    backup_file = atomic_data_file.with_suffix('.py.backup')
    if not backup_file.exists():
        with open(backup_file, 'w') as f:
            f.write(content)
        print(f"✓ Created backup at: {backup_file}")

    # Apply patch
    original_line = 'torch.tensor(config.pbc, dtype=torch.bool)'
    patched_line = 'torch.tensor([config.pbc], dtype=torch.bool)'
    if original_line in content:
        print("processing")
        content = content.replace(original_line, patched_line)

        # Write patched content
        with open(atomic_data_file, 'w') as f:
            f.write(content)

        print(f"Successfully patched {atomic_data_file}")
        print(f"  Changed: {original_line}")
        print(f"  To:      {patched_line}")
        return True
    else:
        print(f"WARNING: Could not find the line to patch!")
        print(f"Looking for: {original_line}")
        return False

def main():
    print("="*70)
    print("MACE Atomic Data Patcher")
    print("="*70)
    print()

    mace_path = find_mace_path()
    if not mace_path:
        return 1

    print(f"MACE installation: {mace_path}")
    print()

    success = patch_atomic_data(mace_path)

    print()
    print("="*70)
    if success:
        print("Patch completed successfully!")
        print("The PBC tensor warning should now be resolved.")
        print()
        print("To verify, run your MACE training again.")
    else:
        print("Patch failed!")
        print("You may need to manually edit the file.")
    print("="*70)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
