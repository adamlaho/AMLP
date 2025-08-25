#!/usr/bin/env bash
#
# generate_all_POTCARs.sh
#
# Finds every POSCAR under a root directory and, in each POSCAR-containing
# directory, builds the corresponding POTCAR by concatenating the right
# element potentials in order.
#
# Usage:
#   ./generate_all_POTCARs.sh [--root_dir=PATH] [--potcar_dir=PATH]
#
# Options:
#   --root_dir=PATH    Directory to scan for POSCAR files (default: .)
#   --potcar_dir=PATH  Directory where all element POTCARs live
#                      (default: /vast/al9500/vasp_POTCAR)
#   --help             Show this help message and exit

# Default values
root_dir="."
pots_dir="/vast/al9500/vasp_POTCAR"

# Parse long options
for arg in "$@"; do
  case $arg in
    --root_dir=*)
      root_dir="${arg#*=}"
      ;;
    --potcar_dir=*)
      pots_dir="${arg#*=}"
      ;;
    --help)
      echo "Usage: $0 [--root_dir=PATH] [--potcar_dir=PATH]"
      echo
      echo "  --root_dir=PATH    Directory to scan for POSCAR files (default: .)"
      echo "  --potcar_dir=PATH  Directory where all element POTCARs live"
      echo "                     (default: /vast/al9500/vasp_POTCAR)"
      echo "  --help             Show this help message and exit"
      exit 0
      ;;
    *)
      echo "Error: Unknown option '$arg'" >&2
      echo "Run '$0 --help' for usage." >&2
      exit 1
      ;;
  esac
done

# Sanity checks
if [ ! -d "$root_dir" ]; then
  echo "Error: root directory '$root_dir' does not exist." >&2
  exit 1
fi

if [ ! -d "$pots_dir" ]; then
  echo "Error: POTCAR repository '$pots_dir' does not exist." >&2
  exit 1
fi

# Main loop: find every POSCAR and process it
find "$root_dir" -type f -name POSCAR | while IFS= read -r poscar_file; do
  work_dir=$(dirname "$poscar_file")
  out_potcar="$work_dir/POTCAR"

  echo
  echo "Processing POSCAR in: $work_dir"

  # 1) read the 6th line (element symbols)
  elements_line=$(sed -n '6p' "$poscar_file")
  if [ -z "$elements_line" ]; then
    echo "Warning: POSCAR '$poscar_file' has no 6th line; skipping." >&2
    continue
  fi

  # split into array
  read -r -a elements <<< "$elements_line"

  # 2) truncate/create the output POTCAR
  : > "$out_potcar" 2>/dev/null
  if [ $? -ne 0 ]; then
    echo "Error: Cannot write to '$out_potcar'." >&2
    continue
  fi

  # 3) for each element, in order, but only once
  declare -A seen=()
  for elem in "${elements[@]}"; do
    if [ "${seen[$elem]+_}" ]; then
      continue
    fi
    seen[$elem]=1

    # locate the source POTCAR
    if [ -f "$pots_dir/$elem/POTCAR" ]; then
      src="$pots_dir/$elem/POTCAR"
    elif [ -f "$pots_dir/$elem" ]; then
      src="$pots_dir/$elem"
    else
      echo "Error: No POTCAR found for element '$elem' under '$pots_dir'." >&2
      src=""
    fi

    if [ -n "$src" ]; then
      echo "Appending $elem from $src"
      cat "$src" >> "$out_potcar"
      if [ $? -ne 0 ]; then
        echo "Error: Failed to append '$src' to '$out_potcar'." >&2
        break
      fi
    fi
  done

  echo "Finished creating '$out_potcar'"
done
