"""
run_visualizations.py ; Visualization / analysis orchestrator

Runs the visualization and analysis scripts in dependency order, skipping any
step whose outputs already exist (unless --force). Each step runs as a
subprocess; its live output is echoed and written to logs/viz_<step>_<ts>.log.

Datasets

  dataset alphafold      AlphaFold models           (SNAKE_DATASET=alphafold)
  dataset experimental   experimental PDB entries   (SNAKE_DATASET=pdb)
  dataset both           alphafold, then experimental
With no --dataset, an existing SNAKE_DATASET is used, otherwise alphafold.

Steps

  FEATURES
    v1  extract_features.py         per-protein feature table -> protein_features.csv
  ANALYSIS
    v2  analysis_snake_venom.py     composition (incl. amino acids), distributions, statistics,
                                    significance, correlations, cycle geometry,
                                    size control + classification             [needs v1]
    v3  analysis_snake_venom.py     the same analyses on the standard features  -> disulfide_data/results/standard
    v4  analysis_snake_venom.py     the same analyses on the augmented features -> disulfide_data/results/augmented
    v8  analysis_disulfide.py       augmented PH, lasso threading, size/count controls,
                                    bootstrap, standard-vs-augmented, family curves  [needs v1]
    v10 function_prediction.py      function-prediction experiment             [needs v8]
    v13 analysis_controls.py        random-staple null on full barcodes (needs ripser) [needs v1]
  PER-PROTEIN FIGURES
    v12 per_protein_figures.py      standard vs augmented barcodes + persistence diagram
                                    (featured proteins; run it with --all for every protein)

The full-dataset protein x protein W2 (full_protein_wasserstein.py) is a
multi-hour/day HPC job with its own launcher (codes/run_hpc.sh); it is NOT here.

Usage
-----
  python run_visualizations.py                           # everything not yet done
  python run_visualizations.py --dataset both            # both datasets
  python run_visualizations.py --dataset experimental --status
  python run_visualizations.py --list
  python run_visualizations.py --step v2 --dataset alphafold
  python run_visualizations.py --step v2 --force
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

_ALIAS = {"alphafold": "alphafold", "af": "alphafold", "predicted": "alphafold",
          "colabfold": "alphafold", "experimental": "pdb", "exp": "pdb", "pdb": "pdb"}


def _peek_dataset():
    a = sys.argv
    for i, x in enumerate(a):
        if x == "--dataset" and i + 1 < len(a):
            return a[i + 1].strip().lower()
        if x.startswith("--dataset="):
            return x.split("=", 1)[1].strip().lower()
    return None


_REQ = _peek_dataset()
if _REQ == "both":
    os.environ["SNAKE_DATASET"] = "alphafold"
elif _REQ in _ALIAS:
    os.environ["SNAKE_DATASET"] = _ALIAS[_REQ]
else:
    if not os.environ.get("SNAKE_DATASET", "").strip():
        os.environ["SNAKE_DATASET"] = "alphafold"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config
import ssbond_features as ssf


def _is_experimental():
    return config.dataset in ("pdb", "experimental", "exp")


def _script_key():
    return "experimental" if _is_experimental() else "alphafold"

_FEATS = lambda: config.VIZ_FEATURES_DIR / "protein_features.csv"
_need_feats = {"type": "file", "path": _FEATS, "hint": "run step v1 first"}

ONLY = []

STEPS = {
    "v1":  {"script": "extract_features.py", "positional_dataset": True,
            "description": "Per-protein feature table -> protein_features.csv",
            "depends_on": [],
            "input_check": {"type": "dir_json", "path": lambda: config.PH_DIR,
                            "hint": "compute persistent homology first"},
            "done_check": {"type": "file", "path": _FEATS},
            "output_dir": lambda: config.VIZ_FEATURES_DIR, "estimated": "1-5 min"},

    "v5":  {"script": "extract_features.py", "positional_dataset": True,
            "args": ["--variant", "standard"],
            "description": "Disulfide-track STANDARD feature table -> disulfide_data/features/standard",
            "depends_on": [],
            "input_check": {"type": "dir_any", "path": lambda: ssf.DD / "processed_data" / "persistent_homology" / "standard",
                            "hint": "disulfide_data/processed_data/persistent_homology/standard is missing"},
            "done_check": {"type": "file", "path": lambda: ssf.feature_table("standard")},
            "output_dir": lambda: ssf.DD / "features" / "standard", "estimated": "1-5 min"},

    "v6":  {"script": "extract_features.py", "positional_dataset": True,
            "args": ["--variant", "augmented"],
            "description": "Disulfide-track AUGMENTED feature table -> disulfide_data/features/augmented",
            "depends_on": [],
            "input_check": {"type": "dir_any", "path": lambda: ssf.DD / "processed_data" / "persistent_homology" / "augmented",
                            "hint": "disulfide_data/processed_data/persistent_homology/augmented is missing"},
            "done_check": {"type": "file", "path": lambda: ssf.feature_table("augmented")},
            "output_dir": lambda: ssf.DD / "features" / "augmented", "estimated": "1-5 min"},

    "v2":  {"script": "analysis_snake_venom.py", "positional_dataset": True,
            "description": "Composition, distributions, statistics, significance, correlations, "
                           "cycle geometry, robustness (writes pi_per_protein.csv)",
            "depends_on": ["v1"], "input_check": _need_feats,
            "done_check": {"type": "file",
                           "path": lambda: config.RESULTS_DIR / "statistics" / "summary_report.txt"},
            "output_dir": lambda: config.RESULTS_DIR, "estimated": "5-15 min"},

    "v3":  {"script": "analysis_snake_venom.py", "positional_dataset": True,
            "args": ["--variant", "standard"],
            "description": "Main-data analyses on the disulfide-track STANDARD features "
                           "-> disulfide_data/results/standard",
            "depends_on": ["v1", "v5"],
            "input_check": {"type": "file", "path": lambda: ssf.feature_table("standard"),
                            "hint": "disulfide-track standard feature table missing"},
            "done_check": {"type": "file",
                           "path": lambda: config.DS_RESULTS_DIR / "standard" / "statistics" / "summary_report.txt"},
            "output_dir": lambda: config.DS_RESULTS_DIR / "standard", "estimated": "5-15 min"},

    "v4":  {"script": "analysis_snake_venom.py", "positional_dataset": True,
            "args": ["--variant", "augmented"],
            "description": "Main-data analyses on the disulfide-track AUGMENTED features "
                           "-> disulfide_data/results/augmented",
            "depends_on": ["v1", "v6"],
            "input_check": {"type": "file", "path": lambda: ssf.feature_table("augmented"),
                            "hint": "disulfide-track augmented feature table missing"},
            "done_check": {"type": "file",
                           "path": lambda: config.DS_RESULTS_DIR / "augmented" / "statistics" / "summary_report.txt"},
            "output_dir": lambda: config.DS_RESULTS_DIR / "augmented", "estimated": "5-15 min"},

    "v8":  {"script": "analysis_disulfide.py", "positional_dataset": True,
            "description": "Disulfide-augmented PH, lasso threading, size/count controls, bootstrap",
            "depends_on": ["v1"], "input_check": _need_feats,
            "done_check": {"type": "file",
                           "path": lambda: config.DS_RESULTS_DIR / "controls_report.txt"},
            "output_dir": lambda: config.DS_RESULTS_DIR, "estimated": "10-40 min"},

    "v10": {"script": "function_prediction.py", "positional_dataset": True,
            "description": "Function-prediction experiment (toxin activity, coagulation)",
            "depends_on": ["v8"],
            "input_check": {"type": "file",
                            "path": lambda: config.DS_RESULTS_DIR / "disulfide_features.csv",
                            "hint": "run step v8 first"},
            "done_check": {"type": "file",
                           "path": lambda: config.DS_RESULTS_DIR / "function_prediction" / "function_report.txt"},
            "output_dir": lambda: config.DS_RESULTS_DIR / "function_prediction", "estimated": "2-5 min"},

    "v13": {"script": "analysis_controls.py", "positional_dataset": True,
            "description": "Random-staple null (full barcodes)",
            "depends_on": ["v1"], "input_check": _need_feats,
            "done_check": {"type": "file",
                           "path": lambda: config.DS_RESULTS_DIR / "controls" / "random_staple_summary.csv"},
            "output_dir": lambda: config.DS_RESULTS_DIR / "controls", "estimated": "10-30 min"},

    "v12": {"script": "per_protein_figures.py", "positional_dataset": True,
            "description": "Per-protein standard vs augmented barcodes and persistence diagrams",
            "depends_on": ["v1"], "input_check": _need_feats,
            "done_check": {"type": "file",
                           "path": lambda: config.RESULTS_DIR / "per_protein" / "summary.csv"},
            "output_dir": lambda: config.RESULTS_DIR / "per_protein", "estimated": "1-2 min"},
}

RUN_ORDER = ["v1", "v5", "v6", "v2", "v3", "v4", "v8", "v10", "v12", "v13"]


# HELPERS


def _is_done(sid):
    spec = STEPS[sid]["done_check"]
    path = spec["path"]()
    if spec["type"] == "file":
        return path.exists() and path.stat().st_size > 0
    pat = {"dir_pdf": "**/*.pdf", "dir_png": "**/*.png", "dir_json": "**/*.json"}[spec["type"]]
    return path.exists() and any(path.rglob(pat))


def _input_ready(sid):
    spec = STEPS[sid]["input_check"]
    path = spec["path"]()
    hint = spec.get("hint", "")
    if spec["type"] == "file":
        if not path.exists():
            return False, f"missing file: {path}  ({hint})"
        if path.stat().st_size == 0:
            return False, f"empty file: {path}"
        return True, ""
    if not path.exists():
        return False, f"missing directory: {path}  ({hint})"
    if spec["type"] == "dir_json":
        if not any(path.rglob("*.json")):
            return False, f"no JSON files in: {path}  ({hint})"
    else:
        if not any(p.is_file() for p in path.rglob("*")):
            return False, f"no files in: {path}  ({hint})"
    return True, ""


def _missing_deps(sid):
    return [d for d in STEPS[sid]["depends_on"] if not _is_done(d)]


def _count_outputs(sid):
    d = STEPS[sid]["output_dir"]()
    if not d.exists():
        return {"pdf": 0, "png": 0, "tiff": 0, "csv": 0}
    return {"pdf": len(list(d.rglob("*.pdf"))), "png": len(list(d.rglob("*.png"))),
            "tiff": len(list(d.rglob("*.tiff"))), "csv": len(list(d.rglob("*.csv")))}


def _fmt_counts(c):
    return f"{c['pdf']}pdf {c['png']}png {c['tiff']}tiff {c['csv']}csv"


def _run_step(sid, force):
    spec = STEPS[sid]
    print("\n" + "=" * 65)
    print(f"STEP {sid}:  {spec['description']}  [{_script_key()}]")
    print("=" * 65)

    ready, reason = _input_ready(sid)
    if not ready:
        print(f"  BLOCKED — {reason}")
        return False

    if not force and _is_done(sid):
        print(f"  already done ({_fmt_counts(_count_outputs(sid))}) -> {spec['output_dir']()}")
        print("  (use --force to re-run)")
        return True

    print(f"  script    : {spec['script']}")
    print(f"  estimated : {spec['estimated']}")
    print(f"  output    : {spec['output_dir']()}")

    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = config.LOGS_DIR / f"viz_{sid}_{ts}.log"
    _codes = Path(__file__).resolve().parent.parent
    _script = next((p for p in [_codes / spec["script"], *_codes.glob("*/" + spec["script"])]
                    if p.exists()), _codes / spec["script"])
    cmd = [sys.executable, str(_script)]
    if spec.get("positional_dataset"):
        cmd.append(_script_key())
    elif config.dataset:
        cmd += ["--dataset", config.dataset]
    cmd += spec.get("args", [])
    if ONLY and spec["script"] == "analysis_snake_venom.py":
        cmd += ["--only", *ONLY]
    print(f"  log       : {log}\n")
    try:
        with open(log, "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    encoding="utf-8", errors="replace", bufsize=1,
                                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                                    cwd=str(Path(__file__).parent.parent))
            for line in proc.stdout:
                print(line, end="", flush=True)
                lf.write(line)
            proc.wait()
        if proc.returncode != 0:
            print(f"\n  FAILED (exit {proc.returncode}); see {log}")
            return False
    except Exception as e:
        print(f"\n  ERROR running {spec['script']}: {e}")
        return False
    print(f"\n  done -> {spec['output_dir']()}")
    return True


def _run_with_deps(sid, force, done, failed):
    for dep in STEPS[sid]["depends_on"]:
        if dep in failed:
            print(f"\nSTEP {sid}: skipped, it needs {dep}, which did not complete")
            failed.add(sid)
            return False
        if dep not in done and not _is_done(dep):
            if not _run_with_deps(dep, force, done, failed):
                failed.add(sid)
                return False
    ok = _run_step(sid, force)
    (done if ok else failed).add(sid)
    return ok


def print_status():
    print("\n" + "=" * 78)
    print(f"VISUALIZATION PIPELINE STATUS  [{_script_key()}]")
    print("=" * 78)
    print(f"  {'step':<5}{'state':<11}{'outputs':<30}{'script'}")
    print("  " + "-" * 72)
    for sid in RUN_ORDER:
        if _is_done(sid):
            state, out = "done", _fmt_counts(_count_outputs(sid))
        else:
            ready, _ = _input_ready(sid)
            state = "pending" if ready and not _missing_deps(sid) else "blocked"
            out = ""
        print(f"  {sid:<5}{state:<11}{out:<30}{STEPS[sid]['script']}")
    print()


def print_list():
    print("\nSteps (run order):")
    for sid in RUN_ORDER:
        print(f"  {sid:<5}{STEPS[sid]['script']:<30}{STEPS[sid]['description']}")
    print()


def _without_dataset(argv):
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--dataset":
            skip = True
            continue
        if a.startswith("--dataset="):
            continue
        out.append(a)
    return out


def main():
    ap = argparse.ArgumentParser(description="Visualization / analysis orchestrator",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--dataset", choices=["alphafold", "experimental", "pdb", "both"],
                    help="Dataset to process (default: SNAKE_DATASET, else alphafold).")
    ap.add_argument("--step", choices=list(STEPS.keys()),
                    help="Run a single step (its dependencies run first).")
    ap.add_argument("--force", action="store_true", help="Re-run even if outputs exist.")
    ap.add_argument("--only", nargs="+", default=None,
                    help="v2/v3/v4 only: run just these plot types (composition hist kde ecdf "
                         "violinbox statistics significance correlations cycles controls wasserstein)")
    ap.add_argument("--status", action="store_true", help="Show step status and exit.")
    ap.add_argument("--list", action="store_true", help="List the steps and exit.")
    args = ap.parse_args()
    ONLY[:] = args.only or []

    if args.list:
        print_list(); return

    if args.dataset == "both":
        rest = _without_dataset(sys.argv[1:])
        rc = 0
        for k in ("alphafold", "experimental"):
            print(f"\n{'#' * 65}\n# DATASET: {k}\n{'#' * 65}", flush=True)
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), *rest,
                                "--dataset", k])
            rc = rc or r.returncode
        sys.exit(rc)

    if args.status:
        print_status(); return

    done, failed = set(), set()
    order = [args.step] if args.step else RUN_ORDER
    results = {sid: _run_with_deps(sid, args.force, done, failed) for sid in order}

    print("\n" + "=" * 65)
    print(f"SUMMARY  [{_script_key()}]")
    print("=" * 65)
    for sid in order:
        print(f"  {sid:<5}{'OK' if results.get(sid) else 'skipped/blocked/failed'}")
    print()


if __name__ == "__main__":
    main()
