from __future__ import annotations
""""
End-to-end pipeline for snake venom PDB processing.

  Step 1  --organize_pdb        Sort rank_001 PDBs into typed folders
  Step 2  --clean_data          Extract x,y,z coordinates to CSV
  Step 3  --persistent_homology Compute persistent homology (Julia / Ripserer)
  Step 4  --hypergraphs         Build hypergraph incidence matrices from PH cycles
  Step 5  --curvature           Compute Forman-Ricci curvature (4 variants)

Usage

  # Full pipeline (all 5 steps)
  python run_snake_venom_pipeline.py

  # Individual steps
  python run_snake_venom_pipeline.py --organize_pdb
  python run_snake_venom_pipeline.py --clean_data
  python run_snake_venom_pipeline.py --persistent_homology
  python run_snake_venom_pipeline.py --hypergraphs
  python run_snake_venom_pipeline.py --curvature

  # Combine steps
  python run_snake_venom_pipeline.py --hypergraphs --curvature

  # Dry-run step 1 (preview matches, writes unmatched_report.csv, no files copied)
  python run_snake_venom_pipeline.py --organize_pdb --dry-run

  # Override paths (any step)
  python run_snake_venom_pipeline.py --organize_pdb \\
      --colabfold-dir  /data/colabfold/ \\
      --organized-dir  /data/organized/ \\
      --csv            /data/SnakeVenom.csv

  # Override Julia settings
  python run_snake_venom_pipeline.py --persistent_homology \\
      --julia-bin /usr/local/bin/julia --julia-threads 8

  # Show current configuration without running anything
  python run_snake_venom_pipeline.py --show-config
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
try:
    from config import config
except ImportError:
    print("[ERROR] config.py not found. Make sure it is in the same directory.")
    sys.exit(1)

from organize_pdb_by_type import organize
from clean_venom_to_csv   import clean


def banner(title: str, width: int = 65) -> None:
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def step_header(num: int, total: int, name: str, width: int = 65) -> None:
    print(f"\n{'─' * width}")
    print(f"  STEP {num}/{total} — {name}")
    print("─" * width)


def elapsed(start: float) -> str:
    s = time.time() - start
    return f"{s:.1f}s" if s < 60 else f"{s / 60:.1f}min"


def run_organize_pdb(args: argparse.Namespace) -> bool:
    """Step 1: copy/move rank_001 PDBs into organized_pdb."""
    try:
        organize(
            input_dir        = Path(args.colabfold_dir),
            output_dir       = Path(args.organized_dir),
            csv_path         = Path(args.csv),
            copy             = not args.move,
            dry_run          = args.dry_run,
            unmatched_report = Path(args.unmatched_report),
        )
        return True
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return False


def run_clean_data(args: argparse.Namespace) -> bool:
    """Step 2: extract x,y,z coordinates from PDBs into clean_data."""
    try:
        clean(
            input_root  = Path(args.organized_dir),
            output_root = Path(args.clean_dir),
        )
        return True
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return False


def run_persistent_homology(args: argparse.Namespace) -> bool:
    """
    Step 3: run persistent_homology.jl via Julia.

    """
    jl_script = Path(__file__).parent.parent / "persistent_homology.jl"
    if not jl_script.exists():
        print(f"[ERROR] Julia script not found: {jl_script}")
        return False

    julia_bin     = args.julia_bin
    julia_threads = str(args.julia_threads)

    env_vars = {
        "PH_INPUT_ROOT":  str(Path(args.clean_dir)),
        "PH_OUTPUT_ROOT": str(Path(args.ph_dir)),
        "JULIA_THREADS":  julia_threads,
    }

    cmd = [julia_bin, f"--threads={julia_threads}", str(jl_script)]

    print(f"  Julia binary  : {julia_bin}")
    print(f"  Julia threads : {julia_threads}")
    print(f"  Input root    : {env_vars['PH_INPUT_ROOT']}")
    print(f"  Output root   : {env_vars['PH_OUTPUT_ROOT']}")
    print(f"  Command       : {' '.join(cmd)}\n")

    import os
    full_env = {**os.environ, **env_vars}

    try:
        result = subprocess.run(cmd, env=full_env, check=True)
        return result.returncode == 0
    except FileNotFoundError:
        print(f"[ERROR] Julia binary not found: '{julia_bin}'")
        print("        Set --julia-bin or export JULIA_BIN=/path/to/julia")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Julia exited with code {e.returncode}")
        return False


def run_hypergraphs(args: argparse.Namespace) -> bool:
    """Step 4 — build hypergraph incidence matrices from PH representatives."""
    import os


    env_patch: dict[str, str] = {}
    if args.types:
        env_patch["PIPELINE_CLASSES"] = ",".join(args.types)

    try:
        import importlib
        import compute_hypergraphs as chg
        # Override config if path is customized
        chg.config.PH_DIR = Path(args.ph_dir)
        chg.config.HG_DIR = Path(args.hg_dir)
        chg.main()
        return True
    except Exception as e:
        print(f"[ERROR] Hypergraph computation failed: {e}")
        return False


def run_curvature(args: argparse.Namespace) -> bool:
    """Step 5: compute Forman-Ricci curvature F(e) = 2|e| - D."""
    try:
        import compute_curvature as cc
        cc.config.HG_DIR = Path(args.hg_dir)
        cc.config.RC_DIR = Path(args.rc_dir)
        cc.main()
        return True
    except Exception as e:
        print(f"[ERROR] Curvature computation failed: {e}")
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Snake venom PDB pipeline — run one, several, or all steps.\n"
            "With no step flag, all 5 steps run in sequence."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Step selection
    steps = p.add_argument_group(
        "step selection",
        "Pass one or more flags to run specific steps. "
        "With no flag all steps run in sequence.",
    )
    steps.add_argument("--organize_pdb",        action="store_true",
                       help="Step 1: copy rank_001 PDBs into organized_pdb/<type>/")
    steps.add_argument("--clean_data",          action="store_true",
                       help="Step 2: extract x,y,z coordinates to clean_data/<type>/")
    steps.add_argument("--persistent_homology", action="store_true",
                       help="Step 3: compute persistent homology (Julia / Ripserer)")
    steps.add_argument("--hypergraphs",         action="store_true",
                       help="Step 4: build hypergraph incidence matrices from PH cycles")
    steps.add_argument("--curvature",           action="store_true",
                       help="Step 5: compute Forman-Ricci curvature  F(e) = 2|e| - D")

    paths = p.add_argument_group("paths (override config defaults)")
    paths.add_argument("--colabfold-dir",   default=str(config.COLABFOLD_DIR),
                       help="ColabFold raw output root")
    paths.add_argument("--csv",             default=str(config.METADATA_CSV),
                       help="SnakeVenom metadata CSV")
    paths.add_argument("--unmatched-report",default=str(config.UNMATCHED_REPORT),
                       help="Unmatched proteins report CSV")
    paths.add_argument("--organized-dir",   default=str(config.ORGANIZED_DIR),
                       help="Step 1 output - typed PDB folders")
    paths.add_argument("--clean-dir",       default=str(config.CLEAN_DIR),
                       help="Step 2 output - coordinate CSVs")
    paths.add_argument("--ph-dir",          default=str(config.PH_DIR),
                       help="Step 3 output - persistent homology JSON")
    paths.add_argument("--hg-dir",          default=str(config.HG_DIR),
                       help="Step 4 output - hypergraph JSON")
    paths.add_argument("--rc-dir", default=str(config.RC_DIR),
                       help="Step 5 output - Ricci curvature (default: %(default)s)")

    #Julia settings
    julia = p.add_argument_group("Julia settings (step 3)")
    julia.add_argument("--julia-bin",     default=config.JULIA_BIN,
                       help="Path to the Julia executable")
    julia.add_argument("--julia-threads", default=config.JULIA_THREADS, type=int,
                       help="Number of Julia threads for step 3")


    behaviour = p.add_argument_group("behaviour")
    behaviour.add_argument("--dry-run", action="store_true",
                           help="Step 1 only: preview without copying files")
    behaviour.add_argument("--move",    action="store_true",
                           help="Step 1: move files instead of copying")
    behaviour.add_argument("--types",   nargs="+", metavar="TYPE",
                           help="Restrict steps 4-5 to specific protein types "
                                "(e.g. --types serine kallikrein)")
    behaviour.add_argument("--show-config", action="store_true",
                           help="Print current configuration and exit")

    return p.parse_args()



# Main


def main() -> None:
    args           = parse_args()
    pipeline_start = time.time()

    #show-config
    if args.show_config:
        print(config.summary())
        sys.exit(0)

    #Determine which steps to run
    any_step_selected = any([
        args.organize_pdb, args.clean_data, args.persistent_homology,
        args.hypergraphs,  args.curvature,
    ])
    # Default: run everything
    run_all = not any_step_selected

    steps_to_run = {
        "organize_pdb":        run_all or args.organize_pdb,
        "clean_data":          run_all or args.clean_data,
        "persistent_homology": run_all or args.persistent_homology,
        "hypergraphs":         run_all or args.hypergraphs,
        "curvature":           run_all or args.curvature,
    }

    # --dry-run implies organize_pdb only
    if args.dry_run:
        steps_to_run = {k: (k == "organize_pdb") for k in steps_to_run}

    active_steps = [k for k, v in steps_to_run.items() if v]
    total_steps  = len(active_steps)

    # Header 
    print("Snake Venom PDB Processing Pipeline")
    print(f"  Steps to run : {' to '.join(active_steps)}")
    if args.dry_run:
        print("  Mode         : DRY-RUN (step 1 preview only, no files written)")
    elif args.move:
        print("  Mode         : MOVE (files will be moved, not copied in step 1)")
    print()

    results: dict[str, bool] = {}
    step_num = 0

    # Step 1: organize_pdb 
    if steps_to_run["organize_pdb"]:
        step_num += 1
        step_header(step_num, total_steps, "organize_pdb - sort PDBs by protein type")
        if not config.validate_step("organize_pdb"):
            print("[ERROR] Prerequisite check failed for organize_pdb.")
            results["organize_pdb"] = False
        else:
            t0 = time.time()
            ok = run_organize_pdb(args)
            results["organize_pdb"] = ok
            print(f"\n  organize_pdb {'✔' if ok else '✗'} completed in {elapsed(t0)}")

        if args.dry_run:
            banner("DRY-RUN COMPLETE")
            print(f"  Total time : {elapsed(pipeline_start)}")
            print(f"  Review '{args.unmatched_report}' for unmatched proteins.")
            print("  Re-run without --dry-run to copy files and continue.\n")
            return

    #Step 2: clean_data
    if steps_to_run["clean_data"]:
        step_num += 1
        step_header(step_num, total_steps, "clean_data - extract coordinates to CSV")
        if not config.validate_step("clean_data"):
            print("[ERROR] Prerequisite check failed. Run --organize_pdb first.")
            results["clean_data"] = False
        else:
            t0 = time.time()
            ok = run_clean_data(args)
            results["clean_data"] = ok
            print(f"\n  clean_data {'✔' if ok else '✗'} completed in {elapsed(t0)}")

    #Step 3: persistent_homology 
    if steps_to_run["persistent_homology"]:
        step_num += 1
        step_header(step_num, total_steps,
                    "persistent_homology - Ripserer H0/H1 (Julia)")
        if not config.validate_step("persistent_homology"):
            print("[ERROR] Prerequisite check failed. Run --clean_data first.")
            results["persistent_homology"] = False
        else:
            t0 = time.time()
            ok = run_persistent_homology(args)
            results["persistent_homology"] = ok
            print(f"\n  persistent_homology {'✔' if ok else '✗'} completed in {elapsed(t0)}")

    #Step 4: hypergraphs
    if steps_to_run["hypergraphs"]:
        step_num += 1
        step_header(step_num, total_steps,
                    "hypergraphs build incidence matrices from PH cycles")
        if not config.validate_step("hypergraphs"):
            print("[ERROR] Prerequisite check failed. Run --persistent_homology first.")
            results["hypergraphs"] = False
        else:
            t0 = time.time()
            ok = run_hypergraphs(args)
            results["hypergraphs"] = ok
            print(f"\n  hypergraphs {'✔' if ok else '✗'} completed in {elapsed(t0)}")

    # Step 5 : curvature 
    if steps_to_run["curvature"]:
        step_num += 1
        step_header(step_num, total_steps,
                    "curvature - Forman-Ricci  F(e) = 2|e| - D")
        if not config.validate_step("curvature"):
            print("[ERROR] Prerequisite check failed. Run --hypergraphs first.")
            results["curvature"] = False
        else:
            t0 = time.time()
            ok = run_curvature(args)
            results["curvature"] = ok
            print(f"\n  curvature {'✔' if ok else '✗'} completed in {elapsed(t0)}")

    #Final summary 
    banner("PIPELINE SUMMARY")
    all_ok = True
    step_labels = {
        "organize_pdb":        "Step 1  organize_pdb",
        "clean_data":          "Step 2  clean_data",
        "persistent_homology": "Step 3  persistent_homology",
        "hypergraphs":         "Step 4  hypergraphs",
        "curvature":           "Step 5  curvature",
    }
    for key, label in step_labels.items():
        if key in results:
            icon = "✔" if results[key] else "✗"
            print(f"  {icon}  {label}")
            if not results[key]:
                all_ok = False

    print(f"\n  Total time : {elapsed(pipeline_start)}")
    if Path(args.unmatched_report).exists() and steps_to_run.get("organize_pdb"):
        print(f"  Unmatched  : {args.unmatched_report}  review if count > 0")
    if all_ok:
        print("\n All steps completed successfully.\n")
    else:
        print("\n One or more steps failed — check output above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
