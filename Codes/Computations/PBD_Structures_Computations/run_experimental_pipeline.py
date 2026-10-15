#!/usr/bin/env python3
"""
run_experimental_pipeline.py

Drive the existing codes/ topology pipeline over the experimental-structure
dataset produced by prepare_experimental_pdbs.py.

Steps

  2  clean          organized_pdb/<cls>/*.pdb   -> clean_data/<cls>/*.csv   (all-atom)
  3  PH             clean_data                  -> processed_data/persistent_homology
                    (Julia / Ripserer, H0+H1)
  4  hypergraphs    persistent_homology         -> processed_data/hypergraphs
  5  curvature      hypergraphs                 -> processed_data/ricci_curvature
  6  features       PH+HG+RC                     -> features/protein_features.csv
  7  ss-augmented   ca_clouds + ss_bonds_ca     -> h1_bars_ca/{standard,augmented}
                    (Julia / Ripserer on the Ca distance matrix, staple = 1e-4)
  8  ss-features    h1_bars_ca + ss_bonds_ca     -> results/ssbond_features.csv

Usage
-----
    python run_experimental_pipeline.py                     # all steps
    python run_experimental_pipeline.py --steps clean ph    # subset
    python run_experimental_pipeline.py --julia-threads 8
    python run_experimental_pipeline.py --out <experimental_data>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CODES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODES_DIR.parent
PIPE_CODES = CODES_DIR                               # the topology pipeline 
COMP = PIPE_CODES / "computation"
DEFAULT_OUT = PROJECT_ROOT / "PDB_computation" / "experimental_data"

ALL_STEPS = ["clean", "ph", "hypergraphs", "curvature", "features",
             "ss_augmented", "ss_features", "ss_curvature", "ss_figures",
             "viz", "function"]
VIZ = PIPE_CODES / "visualization"


def banner(msg):
    print("\n" + "=" * 68 + f"\n  {msg}\n" + "=" * 68, flush=True)


def run(cmd, env_extra=None, cwd=None):
    env = {**os.environ, **(env_extra or {})}
    env.setdefault("PYTHONIOENCODING", "utf-8")
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run(cmd, env=env, cwd=str(cwd) if cwd else None)
    if r.returncode != 0:
        raise SystemExit(f"[error] step failed (exit {r.returncode}): {cmd}")



def _h1_stats(bars):
    pers = [float(b[1]) - float(b[0]) for b in bars
            if b and b[0] is not None and b[1] is not None]
    if not pers:
        return dict(count=0, median=0.0, sum=0.0, max=0.0)
    pers.sort()
    m = pers[len(pers) // 2] if len(pers) % 2 else \
        (pers[len(pers) // 2 - 1] + pers[len(pers) // 2]) / 2
    return dict(count=len(pers), median=float(m), sum=float(sum(pers)),
                max=float(max(pers)))


def build_ssbond_features(out_root: Path):
    import csv
    h1 = out_root / "h1_bars_ca"
    ss = out_root / "ss_bonds_ca"
    std_root, aug_root = h1 / "standard", h1 / "augmented"
    if not std_root.exists():
        print("[warn] no h1_bars_ca/standard -- run the ss_augmented step first")
        return
    rows = []
    for std in sorted(std_root.glob("*/*.json")):
        cls, pid = std.parent.name, std.stem
        aug = aug_root / cls / f"{pid}.json"
        if not aug.exists():
            continue
        with open(std) as fh:
            s = _h1_stats(json.load(fh).get("dim_1", []))
        with open(aug) as fh:
            a = _h1_stats(json.load(fh).get("dim_1", []))
        n_ss, matched = 0, False
        ssf = ss / cls / f"{pid}.json"
        if ssf.exists():
            with open(ssf) as fh:
                d = json.load(fh)
            n_ss, matched = int(d.get("n_ssbond", 0)), bool(d.get("ss_matched"))
        rows.append(dict(
            Protein_ID=pid, Protein_Class=cls, n_ssbond=n_ss, ss_matched=matched,
            h1c_std=s["count"], h1med_std=s["median"], h1sum_std=s["sum"],
            h1max_std=s["max"], h1c_aug=a["count"], h1med_aug=a["median"],
            h1sum_aug=a["sum"], h1max_aug=a["max"],
            h1c_ss=a["count"] - s["count"], h1sum_ss=a["sum"] - s["sum"],
            has_ss=(matched and n_ss > 0)))
    # chain_length per protein (from the cysteine/disulfide table) -> logsize
    import math
    clen = {}
    cyc = out_root / "results" / "cysteine_disulfide_per_protein.csv"
    if cyc.exists():
        with open(cyc, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    clen[r["Protein_ID"]] = int(r["chain_length"])
                except (ValueError, KeyError):
                    pass
    for r in rows:
        n = clen.get(r["Protein_ID"])
        r["chain_length"] = n if n else ""
        r["logsize"] = round(math.log(n), 6) if n else ""

    res = out_root / "results"
    res.mkdir(parents=True, exist_ok=True)
    cols = ["Protein_ID", "Protein_Class", "chain_length", "n_ssbond", "ss_matched",
            "h1c_std", "h1med_std", "h1sum_std", "h1max_std",
            "h1c_aug", "h1med_aug", "h1sum_aug", "h1max_aug",
            "logsize", "h1c_ss", "h1sum_ss", "has_ss"]
    # ssbond_features.csv AND revision_ssbond_features.csv
    for name in ("ssbond_features.csv", "revision_ssbond_features.csv"):
        with (res / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    n_has = sum(1 for r in rows if r["has_ss"])
    print(f"[ok] ssbond features -> {res / 'ssbond_features.csv'} + "
          f"revision_ssbond_features.csv  ({len(rows)} proteins, "
          f"{n_has} carry a disulfide staple)")


# --------------------------------------------------------------------------- 
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=ALL_STEPS)
    ap.add_argument("--dataset", default="pdb",
                    help="config dataset profile (default: pdb). Every child "
                         "script resolves its paths + B-factor label from this.")
    ap.add_argument("--julia-bin", default=os.environ.get("JULIA_BIN", "julia"))
    ap.add_argument("--julia-threads", type=int,
                    default=int(os.environ.get("JULIA_THREADS", "4")))
    args = ap.parse_args()

    # Single switch: every child process (which imports config) inherits this and
    # resolves all paths + the B-factor label from the dataset profile.
    os.environ["SNAKE_DATASET"] = args.dataset

    out = Path(args.out)
    org = out / "organized_pdb"
    clean = out / "clean_data"
    ph = out / "processed_data" / "persistent_homology"
    hg = out / "processed_data" / "hypergraphs"
    rc = out / "processed_data" / "ricci_curvature"
    feat = out / "features"
    ca = out / "ca_clouds"
    ssb = out / "ss_bonds_ca"
    h1 = out / "h1_bars_ca"

    if not org.exists():
        print(f"[error] {org} not found -- run prepare_experimental_pdbs.py first")
        return 1

    jt = str(args.julia_threads)
    t0 = time.time()

    if "clean" in args.steps:
        banner("Step 2 - clean (all-atom coordinates -> CSV)")
        run([sys.executable, str(COMP / "clean_venom_to_csv.py"),
             "-i", str(org), "-o", str(clean)])

    if "ph" in args.steps:
        banner("Step 3 - persistent homology (Ripserer, Julia)")
        run([args.julia_bin, f"--threads={jt}", str(COMP / "persistent_homology.jl")],
            env_extra={"PH_INPUT_ROOT": str(clean), "PH_OUTPUT_ROOT": str(ph),
                       "JULIA_THREADS": jt})

    if "hypergraphs" in args.steps:
        banner("Step 4 - hypergraphs")
        run([sys.executable, str(COMP / "compute_hypergraphs.py")],
            env_extra={"SNAKE_PH_DIR": str(ph), "SNAKE_HG_DIR": str(hg)})

    if "curvature" in args.steps:
        banner("Step 5 - Forman-Ricci curvature")
        run([sys.executable, str(COMP / "compute_curvature.py")],
            env_extra={"SNAKE_HG_DIR": str(hg), "SNAKE_RC_DIR": str(rc)})

    if "features" in args.steps:
        banner("Step 6 - feature table")
        feat.mkdir(parents=True, exist_ok=True)
        run([sys.executable, str(COMP / "extract_features.py")],
            env_extra={"SNAKE_PH_DIR": str(ph), "SNAKE_HG_DIR": str(hg),
                       "SNAKE_RC_DIR": str(rc), "SNAKE_FEATURES_DIR": str(feat)})

    if "ss_augmented" in args.steps:
        banner("Step 7 - Ca disulfide-augmented PH (Ripserer, Julia)")
        run([args.julia_bin, f"--threads={jt}",
             str(COMP / "augmented_persistent_homology.jl")],
            env_extra={"CA_INPUT_ROOT": str(ca), "SS_ROOT": str(ssb),
                       "H1_OUTPUT_ROOT": str(h1), "JULIA_NUM_THREADS": jt})

    if "ss_features" in args.steps:
        banner("Step 8 - disulfide standard-vs-augmented feature table")
        build_ssbond_features(out)

    if "ss_curvature" in args.steps:
        banner("Step 8a - disulfide standard-vs-augmented CURVATURE "
               "(full PH from Cα distance matrices -> hypergraphs -> FRC)")
        dis = out / "disulfide"
        # 1) build standard + augmented C-alpha distance matrices from the clouds
        run([sys.executable, str(COMP / "dump_distance_matrices_from_clouds.py")],
            env_extra={"CA_CLOUDS_ROOT": str(ca), "SS_ROOT": str(ssb),
                       "DM_OUT_ROOT": str(dis / "dm")})
        # 2) full PH (with representatives) on each variant, then 3) HG + 4) FRC
        for variant in ("standard", "augmented"):
            run([args.julia_bin, f"--threads={jt}", str(CODES_DIR / "full_ph_from_dm.jl")],
                env_extra={"AUG_DM_ROOT": str(dis / "dm" / variant),
                           "PH_OUT_ROOT": str(dis / "ph" / variant),
                           "JULIA_NUM_THREADS": jt})
            utf8 = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
            run([sys.executable, str(COMP / "compute_hypergraphs.py")],
                env_extra={"SNAKE_PH_DIR": str(dis / "ph" / variant),
                           "SNAKE_HG_DIR": str(dis / "hg" / variant), **utf8})
            run([sys.executable, str(COMP / "compute_curvature.py")],
                env_extra={"SNAKE_HG_DIR": str(dis / "hg" / variant),
                           "SNAKE_RC_DIR": str(dis / "rc" / variant), **utf8})

    if "ss_figures" in args.steps:
        banner("Step 8b - disulfide comparison figures -> h1_bars_ca/results")
        run([sys.executable, str(CODES_DIR / "plot_experimental_ssbond.py"),
             "--out", str(out)])

    if "viz" in args.steps:
        banner("Step 9 - figures (main_data/results layout)")
        results = out / "results"
        viz_env = {
            "SNAKE_PH_DIR": str(ph), "SNAKE_HG_DIR": str(hg), "SNAKE_RC_DIR": str(rc),
            "SNAKE_FEATURES_DIR": str(feat), "SNAKE_RESULTS_DIR": str(results),
            "SNAKE_HIGH_IMPACT_DIR": str(results), "SNAKE_DATA_DIR": str(org),
            # experimental structures store a crystallographic B-factor in the
            # B-factor column (not AlphaFold pLDDT) 
            "SNAKE_BFACTOR_LABEL": "B-factor",
        }
        
        for script in ("analyze_feature_correlations.py",
                       "analyze_aa_composition.py",
                       "analysis_snake_venom.py",
                       "analyze_robustness.py",
                       # per-family H1 loop size (fig_cycle_size_*) + curvature-vs-size
                       "analyze_cycle_size_curvature.py",
                       # plot_feature_bars supersedes plot_cycle_count_bar +
                       # plot_ssbond_count_bar (identical fig_*_count output) -> once.
                       "plot_feature_bars.py"):
            run([sys.executable, str(VIZ / script)], env_extra=viz_env)

    if "function" in args.steps:
        banner("Step 10 - function prediction (GO labels from RCSB)")
        results = out / "results"
        labels = results / "function_labels.csv"
        run([sys.executable, str(CODES_DIR / "fetch_go_labels.py"), "--out", str(out)])
        run([sys.executable, str(VIZ / "predict_function.py")], env_extra={
            "SNAKE_FEATURES_DIR": str(feat), "SNAKE_RESULTS_DIR": str(results),
            "SNAKE_HIGH_IMPACT_DIR": str(results), "SNAKE_DATA_DIR": str(org),
            "SNAKE_FUNCTION_LABELS": str(labels), "SNAKE_BFACTOR_LABEL": "B-factor"})

    banner(f"PIPELINE DONE in {(time.time() - t0) / 60:.1f} min  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
