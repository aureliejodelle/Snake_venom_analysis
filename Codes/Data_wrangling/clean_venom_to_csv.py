from __future__ import annotations
"""
Step 2:  clean_venom_to_csv

Reads every PDB from organized_pdb/<type>/ (output of step 1),
extracts ATOM/HETATM x,y,z coordinates, and saves a matching CSV:

    clean_data/
        serine/
        kallikrein/
        SNMP/
        PLA/
        ...

Each CSV: one row per atom, columns x, y, z.

Usage:
    python clean_venom_to_csv.py                    # uses config defaults
    python clean_venom_to_csv.py: input organized_pdb/ ; output clean_data/
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))  
try:
    from config import config as _cfg
    _DEFAULT_INPUT  = str(_cfg.ORGANIZED_DIR)
    _DEFAULT_OUTPUT = str(_cfg.CLEAN_DIR)
except ImportError:
    _cfg            = None
    _DEFAULT_INPUT  = "organized_pdb/"
    _DEFAULT_OUTPUT = "clean_data/"

# Core

def extract_coordinates(pdb_path: Path) -> Optional[pd.DataFrame]:
    """Parse ATOM/HETATM records and return a DataFrame with x, y, z columns."""
    x, y, z = [], [], []
    with pdb_path.open() as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x.append(float(line[30:38]))
                    y.append(float(line[38:46]))
                    z.append(float(line[46:54]))
                except ValueError:
                    pass

    if not x:
        return None
    return pd.DataFrame({"x": x, "y": y, "z": z})


def clean(input_root: Path, output_root: Path) -> None:
    """
    Walk input_root recursively, convert every .pdb to a .csv in the
    mirrored location under output_root.
    """
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    pdb_files = sorted(input_root.rglob("*.pdb"))
    if not pdb_files:
        print(f"[WARN] No .pdb files found under '{input_root}'")
        return

    print(f"[SCAN] Found {len(pdb_files)} PDB file(s) under '{input_root}'")

    stats = {"saved": 0, "skipped": 0, "no_atoms": 0}

    for pdb_path in pdb_files:
        # Mirror the subfolder structure under output_root
        relative = pdb_path.relative_to(input_root)
        csv_path = output_root / relative.with_suffix(".csv")

        #Skip if already done 
        if csv_path.exists():
            print(f"  [SKIP] {pdb_path.name}  (CSV already exists)")
            stats["skipped"] += 1
            continue

        print(f"  Processing: {pdb_path}")

        df = extract_coordinates(pdb_path)
        if df is None:
            print("No atoms found — skipping")
            stats["no_atoms"] += 1
            continue

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f" Saved: {csv_path}  ({len(df)} atoms)")
        stats["saved"] += 1

    print("\n Summary ")
    print(f"  Converted : {stats['saved']}")
    print(f"  Skipped   : {stats['skipped']}  (CSV already existed)")
    print(f"  No atoms  : {stats['no_atoms']}")
    print(f"  Output dir: {output_root.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 2 — convert organised PDB files to x,y,z coordinate CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  "-i", default=_DEFAULT_INPUT,
                   help="Root folder produced by step 1 (organize_pdb_by_type)")
    p.add_argument("--output", "-o", default=_DEFAULT_OUTPUT,
                   help="Destination root for CSV files")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clean(input_root=Path(args.input), output_root=Path(args.output))
    print("\nDONE ✔")
