"""
analysis_controls.py

Control analyses for the disulfide-augmented persistent homology, for one dataset
(alphafold and experimental) or both. H1 is recomputed here from the C-alpha clouds with
Vietoris-Rips (ripser); a disulfide "staple" is a pair of C-alpha atoms joined at
distance 10^{-4}.

Usage:
  python analysis_controls.py alphafold
  python analysis_controls.py experimental [--seeds 20] [--workers 4] [--stats-only]
  python analysis_controls.py both

results/controls/   (under disulfide_data/results)
  disulfide_random_staple_h1med.csv   uid, Protein_Class, seed, h1med
  random_staple_summary.csv           observed vs random augmented H1 median per family
  random_staple_null                  figure
"""
from __future__ import annotations

import sys
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import viz_common as vc
import ssbond_features as ssf
from viz_common import disp, palette, savefig, long_path

warnings.filterwarnings("ignore")

OUT = vc.DSRES / "controls"
MIN_LOOP = 3


def _h1(D):
    """H1 bars of the full Vietoris-Rips filtration (no cutoff)."""
    from ripser import ripser
    dg = ripser(D, maxdim=1, distance_matrix=True)["dgms"][1]
    return dg[np.isfinite(dg[:, 1])] if len(dg) else np.empty((0, 2))


def _median(dg):
    p = dg[:, 1] - dg[:, 0]
    p = p[p > 0]
    return float(np.median(p)) if len(p) else np.nan


def _staple(D, pairs):
    D = D.copy()
    for i, j in pairs:
        D[i, j] = D[j, i] = 0.0
    return D


def _work(job):
    """One protein with real bonds: observed augmented median."""
    from scipy.spatial.distance import pdist, squareform
    uid, cls, pdb, ss, seeds = job
    _, X = ssf.read_ca(pdb)
    n = len(X)
    D = squareform(pdist(X))
    obs = dict(uid=uid, Protein_Class=cls, h1med_obs=_median(_h1(_staple(D, ss))))
    rand = []
    for s in range(1, seeds + 1):
        rng = np.random.default_rng(s * 100003 + n)
        pairs = []
        while len(pairs) < len(ss):
            i, j = sorted(rng.integers(0, n, 2))
            if j - i >= MIN_LOOP and (int(i), int(j)) not in pairs:
                pairs.append((int(i), int(j)))
        rand.append(dict(uid=uid, Protein_Class=cls, seed=s, h1med=_median(_h1(_staple(D, pairs)))))
    return obs, rand


def compute(feat, ann, seeds, workers):
    jobs = []
    for _, r in feat[feat["Protein_Class"].isin(vc.VALID)].iterrows():
        pdb = vc.find_pdb(r["Protein_Class"], r["Protein_ID"])
        if not long_path(pdb).exists():
            continue
        seq, X = ssf.read_ca(pdb)
        if len(X) < 4:
            continue
        ss, matched = ann(r["Protein_Class"], r["Protein_ID"], seq)
        if matched and ss:
            jobs.append((ssf.track_id(r["Protein_ID"]), r["Protein_Class"], pdb, ss, seeds))
    print(f"  {len(jobs)} proteins carry a mapped disulfide")
    obs, rand = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (o, rr) in enumerate(ex.map(_work, jobs, chunksize=4), 1):
            obs.append(o)
            rand += rr
            if k % 100 == 0:
                print(f"  {k}/{len(jobs)}")
    return pd.DataFrame(obs), pd.DataFrame(rand)


def plot_null(obs, per_seed, classes):
    cls = [c for c in classes if c in per_seed["Protein_Class"].values]
    colors = dict(zip(classes, palette(classes)))
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [per_seed.loc[per_seed["Protein_Class"] == c, "h1med"].values for c in cls]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False,
                    medianprops=dict(color="black", lw=1.5))
    for b, c in zip(bp["boxes"], cls):
        b.set_facecolor(colors[c])
        b.set_alpha(0.55)
    ax.scatter(range(1, len(cls) + 1), [obs.get(c, np.nan) for c in cls], marker="D", s=60,
               color="#e41a1c", zorder=5, label="real disulfide bonds")
    ax.set_xticks(range(1, len(cls) + 1))
    ax.set_xticklabels([disp(c) for c in cls], rotation=25, ha="right")
    ax.set_ylabel("Mean augmented H1 median persistence")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    savefig(fig, OUT / "random_staple_null")


def main():
    import os
    ap = argparse.ArgumentParser(description="Random-staple control",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", default="alphafold", choices=vc.KEYS,
                    help="alphafold | experimental | both")
    ap.add_argument("--seeds", type=int, default=20, help="random-staple replicates per protein")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--stats-only", action="store_true", help="tables only; skip the figure")
    args = ap.parse_args()
    if args.dataset == "both":
        sys.exit(vc.run_both(__file__))
    vc.STATE["stats_only"] = args.stats_only
    try:
        import ripser  # noqa: F401
    except ImportError:
        sys.exit("ripser is required: pip install ripser")

    print("=" * 65)
    print(f"RANDOM-STAPLE CONTROL  [{vc.KEY}]")
    print("=" * 65)
    ann = ssf.make_annotator(vc.SS_JSON, vc.SS_CSV)
    if ann is None:
        sys.exit(f"No disulfide annotation found (looked in {vc.SS_JSON} and SnakeVenom.csv).")
    obs, rand = compute(vc.load_features(), ann, args.seeds, args.workers)
    if rand.empty:
        sys.exit("No protein with a mapped disulfide and a PDB file was found.")

    OUT.mkdir(parents=True, exist_ok=True)
    rand.to_csv(OUT / "disulfide_random_staple_h1med.csv", index=False)
    classes = [c for c in vc.VALID if c in set(obs["Protein_Class"])]
    observed = obs.groupby("Protein_Class")["h1med_obs"].mean()
    per_seed = rand.groupby(["Protein_Class", "seed"])["h1med"].mean().reset_index()
    rows = []
    for c, g in per_seed.groupby("Protein_Class"):
        o = observed.get(c, np.nan)
        rows.append(dict(Protein_Class=c, observed=o, random_mean=g["h1med"].mean(),
                         random_sd=g["h1med"].std(),
                         p_random_ge_observed=float((np.sum(g["h1med"] >= o) + 1) / (len(g) + 1))))
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "random_staple_summary.csv", index=False)
    print(summ.to_string(index=False))
    if not args.stats_only:
        plot_null(observed, per_seed, classes)
    print(f"\nDone. Results in: {OUT}")


if __name__ == "__main__":
    main()
