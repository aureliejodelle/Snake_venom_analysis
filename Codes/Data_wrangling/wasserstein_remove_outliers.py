"""
wasserstein_remove_outliers.py
Here we removed outliers based on chain length as we did on the previous analysis
Usage:
    python codes/computation/wasserstein_remove_outliers.py
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
for _p in ("codes", "codes/visualization", "codes/computation"):
    sys.path.insert(0, str(ROOT / _p))

import full_protein_wasserstein as W                       
from plot_style import use_computer_modern, disp, disp_list  
from ph_io import VALID_CLASSES, _savefig, _stars  
from dist_stats import (permanova, permdisp, mantel, partial_mantel,  
                        variance_partition, nn_class_accuracy)

use_computer_modern()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WDIR   = ROOT / "AlphaFold_computation" / "main_data" / "results" / "wasserstein"
MATRIX = WDIR / "wasserstein_full_matrix.csv"
IDMAP  = ROOT / "AlphaFold_computation" / "disulfide_data" / "id_map.csv"
CLEAN  = ROOT / "AlphaFold_computation" / "main_data" / "clean_data"
OUT    = WDIR / "no_outliers"


# outliers + size covariate 

def collect_outlier_labels(uid2label: dict) -> tuple[list, list]:
    """uids in every clean_data/*_outliers/ folder -> matrix labels."""
    uids, labels = [], []
    for d in sorted(CLEAN.glob("*_outliers")):
        for f in sorted(d.glob("*.csv")):
            uids.append(f.stem)
            lab = uid2label.get(f.stem)
            if lab is not None:
                labels.append(lab)
    return uids, labels


PH = ROOT / "AlphaFold_computation" / "main_data" / "processed_data" / "persistent_homology"


def cloud_size(label: str, label2uid: dict) -> float:
    """H0-barcode count == number of Calpha points in the cloud.

    Read from the clean_data cloud CSV (row count) where present; fall back to
    len(dim_0_barcode) in the PH JSON for the handful of proteins whose cloud CSV
    is not staged. Verified identical on proteins that carry both.
    """
    fam = label.split("::", 1)[0]
    uid = label2uid.get(label)
    if uid is None:
        return np.nan
    csv = CLEAN / fam / f"{uid}.csv"
    if csv.exists():
        with open(csv) as fh:
            return float(sum(1 for _ in fh) - 1)       #
    js = PH / fam / "PH_1" / f"{uid}.json"
    if js.exists():
        return float(len(json.load(open(js)).get("dim_0_barcode", [])))
    return np.nan


# reconstructed heatmaps (their original generator is gone) 

def plot_full_heatmap(D, classes_of, out_path):
    """Full protein x protein W2 heatmap, blocked and labelled by family."""
    present = [c for c in VALID_CLASSES if c in classes_of]
    arr = np.asarray(classes_of)
    bounds, centers, pos = [], [], 0
    for c in present:
        n = int((arr == c).sum())
        centers.append(pos + n / 2.0)
        pos += n
        bounds.append(pos)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(D, cmap="magma", interpolation="nearest")
    for b in bounds[:-1]:
        ax.axhline(b - 0.5, color="white", lw=0.6, alpha=0.7)
        ax.axvline(b - 0.5, color="white", lw=0.6, alpha=0.7)
    ax.set_xticks(centers); ax.set_xticklabels(disp_list(present), rotation=45, ha="right")
    ax.set_yticks(centers); ax.set_yticklabels(disp_list(present))
    fig.colorbar(im, ax=ax, label="W2 (H1)")
    ax.set_title(f"Full protein-level W2 ({D.shape[0]} proteins)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_family_heatmap(M, present, out_path):
    """6x6 between-family W2 heatmap (annotated)."""
    K = len(present)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(M, cmap="magma")
    ax.set_xticks(range(K)); ax.set_xticklabels(disp_list(present), rotation=45, ha="right")
    ax.set_yticks(range(K)); ax.set_yticklabels(disp_list(present))
    for a in range(K):
        for b in range(K):
            ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if M[a, b] < M.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, label="mean W2 (H1)")
    ax.set_title("Between-family W2", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, out_path)


# significance layer (same tests as full_protein_wasserstein) 

def significance(D, classes_of, sizes, n_perm, seed):
    arr = np.asarray(classes_of)
    X = (arr[:, None] != arr[None, :]).astype(float)
    Y = np.abs(sizes[:, None] - sizes[None, :]).astype(float)
    pn = permanova(D, classes_of, n_perm=n_perm, seed=seed)
    pd_ = permdisp(D, classes_of, n_perm=n_perm, seed=seed)
    mt = mantel(D, X, n_perm=n_perm, seed=seed)
    pm = partial_mantel(D, X, Y, n_perm=n_perm, seed=seed)
    vp = variance_partition(D, X, Y)
    nn = nn_class_accuracy(D, classes_of, sizes)
    table = pd.DataFrame([
        dict(test="PERMANOVA", statistic=pn["F"], effect=pn["R2"], p_value=pn["p"]),
        dict(test="PERMDISP", statistic=pd_["F"], effect=np.nan, p_value=pd_["p"]),
        dict(test="Mantel_class", statistic=mt["r"], effect=mt["r"], p_value=mt["p"]),
        dict(test="PartialMantel_class_given_size", statistic=pm["r_partial"],
             effect=pm["r_partial"], p_value=pm["p"]),
        dict(test="Mantel_size", statistic=pm["r_dy"], effect=pm["r_dy"], p_value=np.nan),
        dict(test="VarPart_class_unique", statistic=vp["class_unique"],
             effect=vp["class_unique"], p_value=np.nan),
        dict(test="VarPart_size_unique", statistic=vp["size_unique"],
             effect=vp["size_unique"], p_value=np.nan),
        dict(test="NN_topology", statistic=nn["topology"], effect=nn["topology"], p_value=np.nan),
        dict(test="NN_size_only", statistic=nn["size_only"], effect=nn["size_only"], p_value=np.nan),
        dict(test="NN_chance", statistic=nn["chance"], effect=nn["chance"], p_value=np.nan),
    ])
    return table, dict(pn=pn, pd=pd_, mt=mt, pm=pm, vp=vp, nn=nn)


def write_report(path, n, present, M, s):
    L = []
    L.append("=" * 68)
    L.append("FULL-BARCODE PROTEIN-LEVEL W2 -> BETWEEN-FAMILY 6x6  (OUTLIERS REMOVED)")
    L.append("=" * 68)
    L.append(f"Proteins            : {n}  (per-family outliers removed; distances "
             f"subset from the saved full matrix, not recomputed)")
    L.append("Distance            : W2 (order 2), Euclidean (L2) ground metric")
    L.append("Solver              : scipy exact assignment (subset of saved matrix)")
    L.append("")
    L.append("6 x 6 between-family W2 (diagonal = within-family spread):")
    L.append("  " + " " * 12 + "".join(f"{disp(c):>12}" for c in present))
    for a, ca in enumerate(present):
        L.append(f"  {disp(ca):<12}" + "".join(f"{M[a, b]:>12.4f}" for b in range(len(present))))
    L.append("")
    L.append(f"Significance on the full {n}x{n} matrix (label permutation):")
    L.append(f"  PERMANOVA        pseudo-F={s['pn']['F']:.2f}  R2={s['pn']['R2']:.4f}  "
             f"p={s['pn']['p']:.4g} ({_stars(s['pn']['p'])})")
    L.append(f"  PERMDISP         F={s['pd']['F']:.2f}  p={s['pd']['p']:.4g}  "
             f"({'heterogeneous' if s['pd']['p'] < 0.05 else 'homogeneous'} spread)")
    L.append(f"  Mantel (class)   r={s['mt']['r']:.4f}  p={s['mt']['p']:.4g}")
    L.append(f"  Partial Mantel   r(class|size)={s['pm']['r_partial']:.4f}  p={s['pm']['p']:.4g}")
    L.append(f"  Mantel (size)    r={s['pm']['r_dy']:.4f}")
    L.append(f"  Var. partition   class-unique R2={s['vp']['class_unique']:.4f}  "
             f"size-unique R2={s['vp']['size_unique']:.4f}")
    L.append(f"  NN same-class    topology={s['nn']['topology']*100:.1f}%  "
             f"size-only={s['nn']['size_only']*100:.1f}%  chance={s['nn']['chance']*100:.1f}%")
    L.append("")
    Path(path).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


# main 

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("REMOVE OUTLIERS FROM FULL W2 MATRIX + REPRODUCE FIGURES")
    print("=" * 65)

    idmap = pd.read_csv(IDMAP)
    uid2label = {r.uid: f"{r.family}::{r.stem}" for r in idmap.itertuples()}
    label2uid = {v: k for k, v in uid2label.items()}

    out_uids, out_labels = collect_outlier_labels(uid2label)

    df = pd.read_csv(MATRIX, index_col=0)
    labelset = set(df.index)
    present_out = [l for l in out_labels if l in labelset]
    absent_out  = [u for u in out_uids if uid2label.get(u) not in labelset]
    keep = [l for l in df.index if l not in set(present_out)]

    print(f"outlier proteins flagged : {len(out_uids)}")
    print(f"  present in matrix, removed : {len(present_out)}")
    print(f"  not in matrix (already absent): {len(absent_out)} -> {absent_out}")
    print(f"matrix {df.shape[0]} -> {len(keep)} proteins\n")

    dfc = df.loc[keep, keep]
    dfc.to_csv(OUT / "wasserstein_full_matrix.csv")
    D = dfc.to_numpy(dtype=float)
    classes_of = [s.split("::", 1)[0] for s in dfc.index.astype(str)]

    sizes = np.array([cloud_size(l, label2uid) for l in dfc.index], dtype=float)
    n_missing = int(np.isnan(sizes).sum())
    if n_missing:
        print(f"WARNING: {n_missing} kept proteins have no cloud CSV for the size covariate")

    # 6x6 summary (both filenames, identical content)
    present, M = W.summarise_6x6(D, classes_of)
    m6 = pd.DataFrame(M, index=present, columns=present)
    m6.to_csv(OUT / "family_w2_6x6.csv")
    m6.to_csv(OUT / "wasserstein_between_family_6x6.csv")

    # figures identical in style to full_protein_wasserstein.py
    W.plot_6x6(M, present, OUT / "fig_family_w2_6x6.pdf")
    W.plot_dendrogram(M, present, OUT / "fig_family_w2_dendrogram.pdf")
    W.plot_distance_histogram(D, classes_of, OUT / "fig_w2_histogram.pdf")
    W.plot_distance_ecdf(D, classes_of, OUT / "fig_w2_ecdf.pdf")

    # reconstructed heatmaps
    plot_full_heatmap(D, classes_of, OUT / "wasserstein_heatmap_full.pdf")
    plot_family_heatmap(M, present, OUT / "wasserstein_heatmap_family6x6.pdf")

    # significance + report
    print("Running PERMANOVA / Mantel / size control ...")
    table, s = significance(D, classes_of, sizes, args.n_perm, args.seed)
    table.to_csv(OUT / "wasserstein_full_significance.csv", index=False)

    write_report(OUT / "wasserstein_full_report.txt", len(keep), present, M, s)
    print("\nDone. Outputs in:", OUT)


if __name__ == "__main__":
    main()
