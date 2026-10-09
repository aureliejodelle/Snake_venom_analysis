"""
2-Wasserstein analysis of snake venom H1
persistence diagrams, summarised to a between-family 6 x 6 matrix.

Usage

    python full_protein_wasserstein.py                  # full run
    python full_protein_wasserstein.py --delta 0.001    # tighter (slower) W2
    python full_protein_wasserstein.py --workers 6
    python full_protein_wasserstein.py --finalize-only   # skip compute

Outputs

    wasserstein_full_matrix.csv        803 x 803 protein W2 matrix
    family_w2_6x6.csv                  6 x 6 between/within-family mean W2
    wasserstein_full_significance.csv  PERMANOVA / Mantel / size control
    fig_family_w2_6x6.pdf/png          6 x 6 heatmap
    fig_family_w2_dendrogram.pdf/png   family dendrogram from the 6 x 6
    wasserstein_full_report.txt
"""

from __future__ import annotations

import sys
import time
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).resolve().parent.parent))
import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1])) 
from plot_style import use_computer_modern, disp, disp_list
from ph_io import (ROOT, PH_DIR, PH_SUBDIR, VALID_CLASSES, CLASS_COLORS,
                   _read_json, _diagram_from_dict, _savefig, _stars, wasserstein2)
from dist_stats import (permanova, permdisp, mantel, partial_mantel,
                        variance_partition, nn_class_accuracy)

use_computer_modern()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ORDER = 2.0
INTERNAL_P = 2.0

RESULTS = ROOT / "results" / "wasserstein_full"
CACHE   = RESULTS / "_cache"


#data loading 

def load_all(dim=1):
    """Load every protein's

    Returns labels, classes_of, diagrams, and sizes. Proteins with an empty
    H1 diagram are skipped.
    """
    labels, classes_of, diagrams, sizes = [], [], [], []
    for cls in VALID_CLASSES:
        cdir = PH_DIR / cls / PH_SUBDIR
        if not cdir.exists():
            continue
        n = 0
        for f in sorted(cdir.glob("*.json")):
            d = _read_json(f)
            if d is None:
                continue
            dgm = _diagram_from_dict(d, dim, 0)
            if len(dgm) == 0:
                continue
            labels.append(f.stem)
            classes_of.append(cls)
            diagrams.append(np.ascontiguousarray(dgm, dtype=float))
            sizes.append(len(d.get("dim_0_barcode", [])))
            n += 1
        print(f"  {cls:<12}: {n} diagrams")
    return labels, classes_of, diagrams, np.asarray(sizes, dtype=float)



_DIAGS = _I = _J = None
_DELTA = 0.01


def _init(diags, I, J, delta):
    global _DIAGS, _I, _J, _DELTA
    _DIAGS, _I, _J, _DELTA = diags, I, J, delta


def _work(rng):
    """Compute W2 for the pair-index range."""
    start, end = rng
    out = np.empty(end - start, dtype=float)
    for t, k in enumerate(range(start, end)):
        a = _DIAGS[_I[k]]; b = _DIAGS[_J[k]]
        out[t] = wasserstein2(a, b, order=ORDER, internal_p=INTERNAL_P)
    return start, out



def compute_matrix(diagrams, delta, workers, chunk=1000, flush_every=2,
                   cache_dir=None):

    cache_dir = cache_dir or CACHE
    n = len(diagrams)
    I, J = np.triu_indices(n, k=1)
    P = len(I)
    nchunks = (P + chunk - 1) // chunk

    cache_dir.mkdir(parents=True, exist_ok=True)
    dpath, kpath = cache_dir / "dists.dat", cache_dir / "done.dat"
    resume = dpath.exists() and kpath.exists()
    dists = np.memmap(dpath, dtype="float64", mode="r+" if resume else "w+", shape=(P,))
    done  = np.memmap(kpath, dtype="uint8",   mode="r+" if resume else "w+", shape=(nchunks,))
    if not resume:
        dists[:] = np.nan; done[:] = 0; dists.flush(); done.flush()

    todo = [(c * chunk, min((c + 1) * chunk, P)) for c in range(nchunks) if done[c] == 0]
    n_done0 = nchunks - len(todo)
    print(f"  pairs={P:,}  chunks={nchunks}  already done={n_done0}  remaining={len(todo)}")
    if not todo:
        return _assemble(dists, I, J, n)

    t0 = time.time()
    processed = 0
    with Pool(processes=workers, initializer=_init,
              initargs=(diagrams, I, J, delta)) as pool:
        for start, out in pool.imap_unordered(_work, todo):
            dists[start:start + len(out)] = out
            done[start // chunk] = 1
            processed += 1
            if processed % flush_every == 0 or processed == len(todo):
                dists.flush(); done.flush()
                frac = (n_done0 + processed) / nchunks
                el = time.time() - t0
                eta = el / processed * (len(todo) - processed)
                print(f"    {n_done0 + processed}/{nchunks} chunks "
                      f"({frac*100:5.1f}%)  elapsed {el/3600:5.2f} h  "
                      f"ETA {eta/3600:5.2f} h", flush=True)
    dists.flush(); done.flush()
    return _assemble(dists, I, J, n)


def _assemble(dists, I, J, n):
    D = np.zeros((n, n), dtype=float)
    D[I, J] = dists
    D[J, I] = dists
    return D


#6 x 6 summary + plots

def summarise_6x6(D, classes_of):
    """Mean within-/between-family W2 to K x K matrix (where diagonal = within-family)."""
    present = [c for c in VALID_CLASSES if c in classes_of]
    idx = {c: np.where(np.asarray(classes_of) == c)[0] for c in present}
    K = len(present)
    M = np.zeros((K, K))
    for a, ca in enumerate(present):
        for b, cb in enumerate(present):
            ia, ib = idx[ca], idx[cb]
            sub = D[np.ix_(ia, ib)]
            if ca == cb:
                m = sub[np.triu_indices(len(ia), k=1)]
                M[a, b] = float(m.mean()) if m.size else 0.0
            else:
                M[a, b] = float(sub.mean())
    return present, M


def plot_6x6(M, present, out_path):
    K = len(present)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(M, cmap="magma")
    ax.set_xticks(range(K)); ax.set_xticklabels(disp_list(present), rotation=45, ha="right")
    ax.set_yticks(range(K)); ax.set_yticklabels(disp_list(present))
    for a in range(K):
        for b in range(K):
            ax.text(b, a, f"{M[a, b]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if M[a, b] < M.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, label="mean W2 (H1)")
    ax.set_title("Between-family W2\n"
                 "diagonal = within-family spread", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_dendrogram(M, present, out_path, method="average"):
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    B = M.copy(); np.fill_diagonal(B, 0.0)          # between-family distances only
    Z = linkage(squareform(B, checks=False), method=method)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    dn = dendrogram(Z, labels=disp_list(present), ax=ax, color_threshold=0,
                    above_threshold_color="#555555", leaf_font_size=12)
    name_by_disp = {disp(c): c for c in present}
    for lbl in ax.get_xticklabels():
        cls = name_by_disp.get(lbl.get_text())
        if cls:
            lbl.set_color(CLASS_COLORS[cls]); lbl.set_fontweight("bold")
    ax.set_ylabel("mean W2 (H1) between families", fontsize=12)
    ax.set_title(f"Family clustering by full-barcode W2 ({method} linkage)",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _savefig(fig, out_path)


def _within_between(D, classes_of):
    """Vectorise the strict lower triangle of D, split into within- and
    between-family pairwise distances."""
    arr = np.asarray(classes_of)
    iu = np.tril_indices(D.shape[0], k=-1)
    d = D[iu]
    same = (arr[:, None] == arr[None, :])[iu]
    return d[same], d[~same]


def plot_distance_histogram(D, classes_of, out_path, bins=60):
    """Histogram of the pairwise W2 distances, within- vs between-family."""
    within, between = _within_between(D, classes_of)
    lo = float(min(within.min(), between.min()))
    hi = float(max(within.max(), between.max()))
    edges = np.linspace(lo, hi, bins + 1)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(between, bins=edges, density=True, color="#e41a1c", alpha=0.55,
            label=f"between family (n={between.size:,})")
    ax.hist(within, bins=edges, density=True, color="#4daf4a", alpha=0.60,
            label=f"within family (n={within.size:,})")
    ax.axvline(between.mean(), color="#e41a1c", ls="--", lw=1.5)
    ax.axvline(within.mean(),  color="#4daf4a", ls="--", lw=1.5)
    ax.set_xlabel("W2 (H1) between proteins", fontsize=12)
    ax.set_ylabel("density", fontsize=12)
    ax.set_title("Distribution of pairwise W2 distances\nwithin vs between toxin families",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_distance_ecdf(D, classes_of, out_path):
    """ECDF of the pairwise W2 distances: within vs between and each
    family's distances to all other families (right)."""
    def ecdf(a):
        x = np.sort(a); return x, np.arange(1, x.size + 1) / x.size

    within, between = _within_between(D, classes_of)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    for data, col, lab in [(within, "#4daf4a", "within family"),
                           (between, "#e41a1c", "between family")]:
        x, y = ecdf(data)
        ax.plot(x, y, color=col, lw=2, label=f"{lab} (n={data.size:,})")
    ax.set_xlabel("W2 (H1) between proteins", fontsize=12)
    ax.set_ylabel("ECDF", fontsize=12)
    ax.set_title("Within vs between family", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    arr = np.asarray(classes_of)
    present = [c for c in VALID_CLASSES if c in classes_of]
    for c in present:
        idx = np.where(arr == c)[0]; other = np.where(arr != c)[0]
        x, y = ecdf(D[np.ix_(idx, other)].ravel())
        ax.plot(x, y, color=CLASS_COLORS[c], lw=1.8, label=disp(c))
    ax.set_xlabel("W2 (H1) to other families", fontsize=12)
    ax.set_ylabel("ECDF", fontsize=12)
    ax.set_title("Per-family distance to the rest", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _savefig(fig, out_path)


def make_all_figures(D, classes_of, present, M):
    """Draw every wasserstein_full figure (heatmap, dendrogram, histogram, ECDF)."""
    plot_6x6(M, present, RESULTS / "fig_family_w2_6x6.pdf")
    plot_dendrogram(M, present, RESULTS / "fig_family_w2_dendrogram.pdf")
    plot_distance_histogram(D, classes_of, RESULTS / "fig_w2_histogram.pdf")
    plot_distance_ecdf(D, classes_of, RESULTS / "fig_w2_ecdf.pdf")


#main

def main():
    ap = argparse.ArgumentParser(description="Full-barcode protein-level W2 "
                                             "-> between-family 6x6")
    ap.add_argument("--dim", type=int, default=1, choices=[0, 1])
    ap.add_argument("--delta", type=float, default=0.01,
                    help="Hera relative accuracy (default 0.01 = 1%%; 0.001 = tighter).")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--finalize-only", action="store_true",
                    help="Skip compute; build all outputs (matrix, 6x6, stats, figures) "
                         )
    ap.add_argument("--plots-only", action="store_true",
                    help="Redraw every figure (heatmap, dendrogram, histogram, ECDF) "
                         )
    ap.add_argument("--force", action="store_true",
                    help="Recompute the full W2 matrix even if it already exists "
                         )
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("FULL-BARCODE PROTEIN-LEVEL W2  to  BETWEEN-FAMILY 6x6")
    print("=" * 65)

    # plots only: rebuild figures from the saved 803x803 matrix CSV
    
    mat = RESULTS / "wasserstein_full_matrix.csv"
    if mat.exists() and not args.plots_only and not args.force:
        print(f"Existing matrix found ({mat.name}); skipping compute and "
              f"replotting only.\nPass --force to recompute the full W2.")
        args.plots_only = True

    if args.plots_only:
        if not mat.exists():
            raise SystemExit(f"--plots-only needs {mat}. Run the full job "
                             )
        print(f"Redrawing all figures from {mat} ...")
        df = pd.read_csv(mat, index_col=0)
        D = df.to_numpy(dtype=float)
        classes_of = [s.split("::", 1)[0] for s in df.index.astype(str)]
        present, M = summarise_6x6(D, classes_of)
        pd.DataFrame(M, index=present, columns=present).to_csv(RESULTS / "family_w2_6x6.csv")
        make_all_figures(D, classes_of, present, M)
        print("Done (plots-only). Figures in:", RESULTS)
        return

    print(f"order p={ORDER:g}  internal_p={INTERNAL_P:g}  Hera delta={args.delta:g}  "
          f"workers={args.workers}\n")

    print("Loading full (non-truncated) H1 diagrams ...")
    labels, classes_of, diagrams, sizes = load_all(args.dim)
    n = len(diagrams)
    if n < 3:
        raise SystemExit(f"Only {n} diagrams.")
    total_bars = int(sum(len(d) for d in diagrams))
    print(f"  Total: {n} proteins, {total_bars:,} H1 bars "
          f"(mean {total_bars/n:.0f}/protein)\n")

    meta_path = CACHE / "meta.json"
    meta = dict(n=n, delta=args.delta, chunk=args.chunk, labels=labels)
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old.get("n") != n or old.get("labels") != labels or old.get("chunk") != args.chunk:
            raise SystemExit("Cached run does not match current dataset/chunk. "
                             )
    else:
        CACHE.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta))

    if args.finalize_only:
        I, J = np.triu_indices(n, k=1)
        dists = np.memmap(CACHE / "dists.dat", dtype="float64", mode="r", shape=(len(I),))
        if np.isnan(dists).any():
            raise SystemExit("Cache is incomplete; run without --finalize-only.")
        D = _assemble(np.asarray(dists), I, J, n)
    else:
        print("Computing full pairwise W2 ...")
        D = compute_matrix(diagrams, args.delta, args.workers, args.chunk)

    #save full matrix
    idx_labels = [f"{c}::{l}" for c, l in zip(classes_of, labels)]
    pd.DataFrame(D, index=idx_labels, columns=idx_labels).to_csv(
        RESULTS / "wasserstein_full_matrix.csv")
    print(f"\n  Saved -> {RESULTS / 'wasserstein_full_matrix.csv'}")

    #6 x 6 summary + figures
    present, M = summarise_6x6(D, classes_of)
    pd.DataFrame(M, index=present, columns=present).to_csv(RESULTS / "family_w2_6x6.csv")
    make_all_figures(D, classes_of, present, M)

    #significance on the full matrix
    print("\nRunning PERMANOVA / Mantel / size control ...")
    arr = np.asarray(classes_of)
    X = (arr[:, None] != arr[None, :]).astype(float)
    Y = np.abs(sizes[:, None] - sizes[None, :]).astype(float)
    pn = permanova(D, classes_of, n_perm=args.n_perm, seed=args.seed)
    pd_ = permdisp(D, classes_of, n_perm=args.n_perm, seed=args.seed)
    mt = mantel(D, X, n_perm=args.n_perm, seed=args.seed)
    pm = partial_mantel(D, X, Y, n_perm=args.n_perm, seed=args.seed)
    vp = variance_partition(D, X, Y)
    nn = nn_class_accuracy(D, classes_of, sizes)
    pd.DataFrame([
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
    ]).to_csv(RESULTS / "wasserstein_full_significance.csv", index=False)

    #report
    lines = []
    lines.append("=" * 68)
    lines.append("FULL-BARCODE PROTEIN-LEVEL W2 to BETWEEN-FAMILY 6x6")
    lines.append("=" * 68)
    lines.append(f"Proteins            : {n}   H1 bars total: {total_bars:,} "
                 f"(mean {total_bars/n:.0f}/protein")
    lines.append(f"Distance            : W2 (order 2), Euclidean (L2)")
    lines.append(f"Solver              : scipy")
    lines.append("")
    lines.append("6 x 6 between-family W2 (diagonal = within-family spread):")
    lines.append("  " + " " * 12 + "".join(f"{disp(c):>12}" for c in present))
    for a, ca in enumerate(present):
        lines.append(f"  {disp(ca):<12}" +
                     "".join(f"{M[a, b]:>12.4f}" for b in range(len(present))))
    lines.append("")
    lines.append("Significance on the full 803x803 matrix (label permutation):")
    lines.append(f"  PERMANOVA        pseudo-F={pn['F']:.2f}  R2={pn['R2']:.4f}  "
                 f"p={pn['p']:.4g} ({_stars(pn['p'])})")
    lines.append(f"  PERMDISP         F={pd_['F']:.2f}  p={pd_['p']:.4g}  "
                 f"({'heterogeneous' if pd_['p'] < 0.05 else 'homogeneous'} spread)")
    lines.append(f"  Mantel (class)   r={mt['r']:.4f}  p={mt['p']:.4g}")
    lines.append(f"  Partial Mantel   r(class|size)={pm['r_partial']:.4f}  p={pm['p']:.4g}")
    lines.append(f"  Mantel (size)    r={pm['r_dy']:.4f}")
    lines.append(f"  Var. partition   class-unique R2={vp['class_unique']:.4f}  "
                 f"size-unique R2={vp['size_unique']:.4f}")
    lines.append(f"  NN same-class    topology={nn['topology']*100:.1f}%  "
                 f"size-only={nn['size_only']*100:.1f}%  chance={nn['chance']*100:.1f}%")
    lines.append("")
    (RESULTS / "wasserstein_full_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("Done. Outputs in:", RESULTS)


if __name__ == "__main__":
    main()
