from __future__ import annotations
#!/usr/bin/env python3
"""
Step 1: organize_pdb_by_type
------------------------------
Look at the predicted structures and search for rank_001 PDB files and use protein's type 
from SnakeVenom.csv file based on protein_names and Type columns to classify them.

    organized_pdb/
        serine/
        kallikrein/
        SNMP/
        PLA/
        ...

Usage:
    python organize_pdb_by_type.py                  # uses config defaults
    python organize_pdb_by_type.py --dry-run        # preview only
    python organize_pdb_by_type.py --move           # move instead of copy
    python organize_pdb_by_type.py \\
        --input  colabfold_output_results/ \\
        --output organized_pdb/ \\
        --csv    SnakeVenom.csv
"""

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
try:
    from config import config as _cfg
    _DEFAULT_INPUT    = str(_cfg.COLABFOLD_DIR)
    _DEFAULT_OUTPUT   = str(_cfg.ORGANIZED_DIR)
    _DEFAULT_CSV      = str(_cfg.METADATA_CSV)
    _DEFAULT_REPORT   = str(_cfg.UNMATCHED_REPORT)
except ImportError:
    _cfg              = None
    _DEFAULT_INPUT    = "colabfold_output_results/"
    _DEFAULT_OUTPUT   = "organized_pdb/"
    _DEFAULT_CSV      = "SnakeVenom.csv"
    _DEFAULT_REPORT   = "unmatched_report.csv"



def load_protein_type_map(csv_path: Path) -> Dict[str, str]:
    """Return {protein_name_raw: type} from the CSV."""
    mapping: Dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name  = row.get("Protein_names", "").strip()
            ptype = row.get("Type", "").strip()
            if name and ptype:
                mapping[name] = ptype
    print(f"[CSV] Loaded {len(mapping)} protein → type entries from '{csv_path}'")
    return mapping


def normalize(name: str) -> str:
    """
    Normalize a protein name for fuzzy matching between filenames and CSV.

    Patterns handled (CSV to filename):
      - Parentheses / brackets : (EC 3.4.24.46) : __EC_3.4.24.46__  → space
      - Comma, colon, semicolon, slash, hyphen  : space
      - Apostrophe                              : dropped
      - Underscores                             : space
      - Possessive _s_ (Russel_s to Russels)     : collapsed
      - Multiple spaces                         : single space
    """
    s = name.lower()
    s = s.replace("'", "")                        # drop apostrophes
    s = re.sub(r"[(){}\[\]]", " ", s)             # brackets to space
    s = re.sub(r"[,;:/\\-]",  " ", s)             # punctuation to space
    s = s.replace("_", " ")                       # underscores to space
    s = re.sub(r"\b(\w+) s\b", r"\1s", s)         # Russel s to Russels
    s = re.sub(r" +", " ", s)                     # collapse spaces
    return s.strip()


def extract_protein_name(pdb_path: Path) -> str:
    """
    Extract protein name from a ColabFold rank_001 filename.

    Steps:
      1. Strip everything from 'rank_001' onward.
      2. Strip trailing ColabFold run-state tokens (_unrelaxed, _relaxed, …).
      3. Strip trailing underscores/spaces.
    """
    stem  = pdb_path.stem
    idx   = stem.lower().find("rank_001")
    protein_part = stem[:idx] if idx != -1 else stem

    colabfold_suffixes = re.compile(
        r"[_ ]*(unrelaxed|relaxed|model_\d+|seed_\d+|alphafold\w*|ptm)[_ ]*$",
        re.IGNORECASE,
    )
    while True:
        cleaned = colabfold_suffixes.sub("", protein_part)
        if cleaned == protein_part:
            break
        protein_part = cleaned

    return protein_part.rstrip("_").strip()


def find_type(protein_name: str, mapping: Dict[str, str]) -> Optional[str]:
    """
    Match protein_name against the CSV mapping (normalized, three strategies):
      1. Exact match after normalization.
      2. Normalized CSV key contains the normalized protein name.
      3. Normalized protein name contains a normalized CSV key.
    """
    key = normalize(protein_name)
    norm_map = {normalize(k): v for k, v in mapping.items()}

    if key in norm_map:
        return norm_map[key]
    for csv_key, ptype in norm_map.items():
        if key in csv_key:
            return ptype
    for csv_key, ptype in norm_map.items():
        if csv_key in key:
            return ptype
    return None


def write_unmatched_report(rows: List[Dict], report_path: Path) -> None:
    """Write unmatched entries to a CSV for manual inspection."""
    if not rows:
        print("[REPORT] No unmatched files.")
        return
    fieldnames = ["pdb_file", "extracted_protein_name", "normalized_name", "full_path"]
    with report_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[REPORT] Unmatched report → '{report_path}' ({len(rows)} entries)")



# ---------------------------------------------------------------------------

def organize(
    input_dir:        Path,
    output_dir:       Path,
    csv_path:         Path,
    copy:             bool = True,
    dry_run:          bool = False,
    unmatched_report: Path = Path("unmatched_report.csv"),
) -> None:

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    mapping   = load_protein_type_map(csv_path)
    pdb_files = sorted(input_dir.rglob("*rank_001*.pdb"))
    print(f"[SCAN] Found {len(pdb_files)} rank_001 PDB file(s) under '{input_dir}'")

    stats: Dict[str, int] = {"processed": 0, "skipped": 0, "unmatched": 0}
    unmatched_rows: List[Dict] = []

    for pdb in pdb_files:
        protein_name = extract_protein_name(pdb)
        ptype        = find_type(protein_name, mapping)

        if ptype is None:
            print(f"  [WARN] No type found for '{protein_name}' ({pdb.name})")
            stats["unmatched"] += 1
            unmatched_rows.append({
                "pdb_file":               pdb.name,
                "extracted_protein_name": protein_name,
                "normalized_name":        normalize(protein_name),
                "full_path":              str(pdb),
            })
            continue

        dest_dir  = output_dir / ptype
        dest_file = dest_dir / pdb.name

        #Skip if already done
        if not dry_run and dest_file.exists():
            print(f"  [SKIP] {pdb.name}  (already in {ptype}/)")
            stats["skipped"] += 1
            continue

        print(f"  [{ptype}] {pdb.name}")

        if dry_run:
            action_word = "copy" if copy else "move"
            exists_note = " (already exists)" if dest_file.exists() else ""
            print(f"         → would {action_word} to {dest_file}{exists_note}")
            stats["processed"] += 1
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        if copy:
            shutil.copy2(pdb, dest_file)
        else:
            shutil.move(str(pdb), dest_file)
        stats["processed"] += 1

    write_unmatched_report(unmatched_rows, unmatched_report)

    action = "Dry-run" if dry_run else ("Copied" if copy else "Moved")
    print("\n--- Summary ---")
    print(f"  {action}    : {stats['processed']}")
    print(f"  Skipped   : {stats['skipped']}  (already existed)")
    print(f"  Unmatched : {stats['unmatched']}")
    if not dry_run:
        print(f"  Output dir: {output_dir.resolve()}")
    if unmatched_rows:
        print(f"  Report    : {unmatched_report.resolve()}")



# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 1: copy rank_001 PDB files into typed sub-folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  "-i", default=_DEFAULT_INPUT,  help="ColabFold output root")
    p.add_argument("--output", "-o", default=_DEFAULT_OUTPUT, help="Destination root")
    p.add_argument("--csv",    "-c", default=_DEFAULT_CSV,    help="SnakeVenom metadata CSV")
    p.add_argument("--unmatched-report", default=_DEFAULT_REPORT,
                   help="Path for the unmatched proteins report CSV")
    p.add_argument("--move",    action="store_true", help="Move files instead of copying")
    p.add_argument("--dry-run", action="store_true", help="Preview without touching files")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    organize(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        csv_path=Path(args.csv),
        copy=not args.move,
        dry_run=args.dry_run,
        unmatched_report=Path(args.unmatched_report),
    )
