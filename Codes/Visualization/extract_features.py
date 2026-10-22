"""
extract_features.py

Extract per-protein features -> CSV

Reads JSON outputs from pipeline steps 4, 5, 6 and builds a  feature
table for every protein in every class (original + homolog groups).

Features extracted (21 total):
  Persistence (H1 barcodes) : count, mean, median, max, sum, std
  Curvature (Forman-Ricci)  : mean, median, max, min, std, sum,
                               n_positive, n_negative, n_zero
  Hypergraph structure      : n_nodes, n_hyperedges, avg_edge_size,
                               max_edge_size, connectivity, density

Outputs:
  results/features/protein_features.csv             all hyperedges included
  results/features/protein_features_no_outliers.csv outliers removed per protein
                                                     before computing aggregated
                                                     features (IQR method on
                                                     curvature and persistence)

Usage:
  python extract_features.py alphafold
  python extract_features.py experimental
  python extract_features.py both
"""

import sys
import json
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

warnings.filterwarnings("ignore")
import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import viz_common as vc
import ssbond_features as ssf
from viz_common import config, long_path


def ds_path(kind, cls, *tail):
    """First existing disulfide-track path for hypergraphs / ricci_curvature."""
    full = {"hg": "hypergraphs", "rc": "ricci_curvature"}[kind]
    roots = [ssf.DD / "processed_data" / full, ssf.DS / "processed_data" / full, ssf.DS / kind]
    cands = [r / v / cls / Path(*tail) for r in roots for v in (VARIANT, "")]
    return next((p for p in cands if long_path(p).exists()), cands[0])


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# CONSTANTS

VARIANT = vc.VARIANT
FEATURES_CSV = (ssf.DD / "features" / VARIANT / "protein_features.csv") if VARIANT else vc.FEATURES_CSV
FEATURES_NO_OL_CSV  = FEATURES_CSV.with_name("protein_features_no_outliers.csv")
NUM_WORKERS  = 8

FEATURE_COLS = [
    "H1_count", "H1_mean", "H1_median", "H1_max", "H1_sum", "H1_std",
    "Curv_mean", "Curv_median", "Curv_max", "Curv_min", "Curv_std", "Curv_sum",
    "Curv_n_positive", "Curv_n_negative", "Curv_n_zero",
    "HG_n_nodes", "HG_n_hyperedges",
    "HG_avg_edge_size", "HG_max_edge_size",
    "HG_connectivity", "HG_density",
]

# FEATURE LOADERS

def load_hypergraph(class_name: str, protein_id: str) -> dict:
    path = (ds_path("hg", class_name, "hyperedge_map", f"{protein_id}.json") if VARIANT
            else config.HG_DIR / class_name / "hyperedge_map" / f"{protein_id}.json")
    keys = ["HG_n_nodes","HG_n_hyperedges","HG_avg_edge_size",
            "HG_max_edge_size","HG_connectivity","HG_density"]
    out  = {k: np.nan for k in keys}
    if not long_path(path).exists():
        s = (ds_path("hg", class_name, "summary", f"{protein_id}.json") if VARIANT
             else config.HG_DIR / class_name / "summary" / f"{protein_id}.json")
        if long_path(s).exists():
            try:
                with open(long_path(s)) as f:
                    d = json.load(f)
                out["HG_n_nodes"]      = float(d.get("n_nodes",      np.nan))
                out["HG_n_hyperedges"] = float(d.get("n_hyperedges", np.nan))
            except Exception:
                pass
        return out
    try:
        with open(long_path(path)) as f:
            data = json.load(f)
        edges = list(data.values()) if isinstance(data, dict) else list(data)
        nodes, sizes = set(), []
        for e in edges:
            if isinstance(e, list):
                nodes.update(e); sizes.append(len(e))
        n, m = len(nodes), len(sizes)
        if n and m:
            possible = n*(n-1)/2 if n > 1 else 1
            out.update({
                "HG_n_nodes":       float(n),
                "HG_n_hyperedges":  float(m),
                "HG_avg_edge_size": float(np.mean(sizes)),
                "HG_max_edge_size": float(np.max(sizes)),
                "HG_connectivity":  float(m/n),
                "HG_density":       float(m/possible),
            })
    except Exception:
        pass
    return out


# OUTLIER REMOVAL

def iqr_filter(values: np.ndarray) -> np.ndarray:
    """
    Remove outliers using the IQR method.
    Returns filtered array keeping values within
    [Q1 - 1.5*IQR,  Q3 + 1.5*IQR].
    Returns original array unchanged if fewer than 4 values.
    """
    if len(values) < 4:
        return values
    q1, q3 = np.percentile(values, [25, 75])
    iqr     = q3 - q1
    lo, hi  = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return values[(values >= lo) & (values <= hi)]


def load_curvature_raw(class_name: str, protein_id: str) -> np.ndarray:
    """Return raw curvature array (all hyperedges) for one protein."""
    path = (ds_path("rc", class_name, f"{protein_id}.json") if VARIANT
            else config.RC_DIR / class_name / f"{protein_id}.json")
    if not long_path(path).exists():
        return np.array([])
    try:
        with open(long_path(path)) as f:
            data = json.load(f)
        vals = (np.array(list(data.values()), dtype=float)
                if isinstance(data, dict)
                else np.array(data, dtype=float))
        return vals[np.isfinite(vals)]
    except Exception:
        return np.array([])


def load_persistence_raw(class_name: str, protein_id: str) -> np.ndarray:
    """Return raw persistence array (finite bars only) for one protein."""
    if VARIANT:
        bars = vc.load_bars(class_name, protein_id, VARIANT)
        if bars is None or not len(bars):
            return np.array([])
        arr = bars[:, 1] - bars[:, 0]
        return arr[np.isfinite(arr)]
    path = config.PH_DIR / class_name / "barcodes" / f"{protein_id}.json"
    if not long_path(path).exists():
        return np.array([])
    try:
        with open(long_path(path)) as f:
            data = json.load(f)
        pers = [
            float(b[1]) - float(b[0])
            for b in data.get("dim_1", [])
            if b[0] is not None and b[1] is not None
        ]
        arr = np.array(pers, dtype=float)
        return arr[np.isfinite(arr)]
    except Exception:
        return np.array([])


def curvature_features_from_array(vals: np.ndarray) -> dict:
    """Compute curvature features."""
    keys = ["Curv_mean","Curv_median","Curv_max","Curv_min","Curv_std","Curv_sum",
            "Curv_n_positive","Curv_n_negative","Curv_n_zero"]
    out = {k: np.nan for k in keys}
    if len(vals):
        out.update({
            "Curv_mean":       float(np.mean(vals)),
            "Curv_median":     float(np.median(vals)),
            "Curv_max":        float(np.max(vals)),
            "Curv_min":        float(np.min(vals)),
            "Curv_std":        float(np.std(vals)),
            "Curv_sum":        float(np.sum(vals)),
            "Curv_n_positive": float(np.sum(vals > 0)),
            "Curv_n_negative": float(np.sum(vals < 0)),
            "Curv_n_zero":     float(np.sum(vals == 0)),
        })
    return out


def persistence_features_from_array(arr: np.ndarray) -> dict:
    """Compute persistence features."""
    out = {k: np.nan for k in
           ["H1_count","H1_mean","H1_median","H1_max","H1_sum","H1_std"]}
    if len(arr):
        out.update({
            "H1_count":  float(len(arr)),
            "H1_mean":   float(np.mean(arr)),
            "H1_median": float(np.median(arr)),
            "H1_max":    float(np.max(arr)),
            "H1_sum":    float(np.sum(arr)),
            "H1_std":    float(np.std(arr)),
        })
    return out


# PROCESS ONE PROTEIN  both raw and outlier-removed

def process_protein(class_name: str, protein_id: str) -> dict:
    """
    Returns a dict with two sub-dicts:
      'raw': features computed on all values (original behaviour)
      'no_outliers': features computed after IQR filtering curvature and persistence
    Both include Protein_Class, Protein_ID, and hypergraph features (unchanged).
    """
    hg_feats = load_hypergraph(class_name, protein_id)

    # Raw arrays
    curv_arr = load_curvature_raw(class_name, protein_id)
    pers_arr = load_persistence_raw(class_name, protein_id)

    # Filtered arrays
    curv_filt = iqr_filter(curv_arr)
    pers_filt = iqr_filter(pers_arr)

    base = {"Protein_Class": class_name, "Protein_ID": protein_id}
    base.update(hg_feats)

    row_raw = {**base,
               **curvature_features_from_array(curv_arr),
               **persistence_features_from_array(pers_arr)}

    row_no_ol = {**base,
                 **curvature_features_from_array(curv_filt),
                 **persistence_features_from_array(pers_filt),
                 "Curv_n_outliers_removed": int(len(curv_arr) - len(curv_filt)),
                 "Pers_n_outliers_removed": int(len(pers_arr) - len(pers_filt))}

    return {"raw": row_raw, "no_outliers": row_no_ol}


def collect_proteins() -> list:
    pairs = []
    if VARIANT:
        seen = set()
        for root in vc.bar_roots(VARIANT):
            if not long_path(root).is_dir():
                continue
            for cls_dir in sorted(long_path(root).iterdir()):
                for sub in ("barcodes", "PH_1", ""):
                    folder = cls_dir / sub if sub else cls_dir
                    if not folder.is_dir():
                        continue
                    for f in sorted(folder.iterdir()):
                        if f.is_file() and f.suffix.lower() in vc.BAR_EXTS \
                                and (cls_dir.name, f.stem) not in seen:
                            seen.add((cls_dir.name, f.stem))
                            pairs.append((cls_dir.name, f.stem))
            if pairs:
                break
        return pairs
    if not config.PH_DIR.exists():
        return pairs
    for cls_dir in sorted(config.PH_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        bc = cls_dir / "barcodes"
        if bc.exists():
            for jf in sorted(bc.glob("*.json")):
                pairs.append((cls_dir.name, jf.stem))
    return pairs


# MAIN

def main():
    ap = argparse.ArgumentParser(description="Per-protein feature extraction")
    ap.add_argument("dataset", nargs="?", default="alphafold", choices=vc.KEYS,
                    help="alphafold | experimental | both")
    ap.add_argument("--variant", choices=("standard", "augmented"), default=None,
                    help="Build the disulfide-track feature table for this variant "
                         "(disulfide_data/features/<variant>/protein_features.csv)")
    args = ap.parse_args()
    if args.dataset == "both":
        sys.exit(vc.run_both(__file__))
    print("=" * 60)
    print(f"FEATURE EXTRACTION  [{vc.KEY}{' / ' + VARIANT if VARIANT else ''}]")
    print("=" * 60)
    print(f"Start   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output 1: {FEATURES_CSV}")
    print(f"Output 2: {FEATURES_NO_OL_CSV}  (IQR outliers removed)\n")

    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    proteins = collect_proteins()

    if not proteins:
        print("No proteins found. Run pipeline steps 4-6 first.")
        sys.exit(1)

    # Detect which classes are originals vs homologs
    counts = Counter(c for c,_ in proteins)
    all_cls = list(counts.keys())
    originals = [c for c in all_cls
                 if not any(c.startswith(o+"_") for o in all_cls if o != c)]

    print(f"Found {len(proteins)} proteins across {len(counts)} class(es):")
    for cls in sorted(counts.keys()):
        role = "original" if cls in originals else "homolog"
        print(f"  [{role:8s}]  {cls:<55} n={counts[cls]}")
    print()

    rows_raw   = []
    rows_no_ol = []

    with tqdm(total=len(proteins), desc="Extracting features", unit="prot") as pbar:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futs = {ex.submit(process_protein, c, p): (c,p) for c,p in proteins}
            for fut in as_completed(futs):
                result = fut.result()
                rows_raw.append(result["raw"])
                rows_no_ol.append(result["no_outliers"])
                pbar.update(1)

    # Save raw CSV (original behaviour) 
    ordered_cols = ["Protein_Class", "Protein_ID"] + [c for c in FEATURE_COLS if c in rows_raw[0]]
    df_raw = pd.DataFrame(rows_raw)
    df_raw = df_raw[[c for c in ordered_cols if c in df_raw.columns]]
    df_raw.to_csv(FEATURES_CSV, index=False)
    print(f"\nSaved {len(df_raw)} rows → {FEATURES_CSV}")

    #Save outlier-removed CSV 
    extra_cols = ["Curv_n_outliers_removed", "Pers_n_outliers_removed"]
    ordered_no_ol = ordered_cols + [c for c in extra_cols if c in rows_no_ol[0]]
    df_no_ol = pd.DataFrame(rows_no_ol)
    df_no_ol = df_no_ol[[c for c in ordered_no_ol if c in df_no_ol.columns]]
    df_no_ol.to_csv(FEATURES_NO_OL_CSV, index=False)
    print(f"Saved {len(df_no_ol)} rows → {FEATURES_NO_OL_CSV}")

    #Feature coverage table 
    print("\nFeature coverage  [raw | no_outliers] (non-NaN %):")
    for col in FEATURE_COLS:
        if col in df_raw.columns:
            n_raw   = df_raw[col].notna().sum()
            n_no_ol = df_no_ol[col].notna().sum() if col in df_no_ol.columns else 0
            pct_r   = 100 * n_raw   / len(df_raw)
            pct_n   = 100 * n_no_ol / len(df_no_ol)
            bar_r   = "#" * int(pct_r/5) + "-" * (20-int(pct_r/5))
            print(f"  {col:<25} raw |{bar_r}| {pct_r:.0f}%   no_ol {pct_n:.0f}%")

    #Outlier removal summary 
    if "Curv_n_outliers_removed" in df_no_ol.columns:
        mean_co = df_no_ol["Curv_n_outliers_removed"].mean()
        mean_po = df_no_ol["Pers_n_outliers_removed"].mean()
        print(f"\nMean curvature outliers removed per protein : {mean_co:.1f}")
        print(f"Mean persistence outliers removed per protein: {mean_po:.1f}")

    print(f"\nDone. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
