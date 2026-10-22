"""
per_protein_figures.py

One figure per protein comparing the standard and the disulfide-augmented H1
barcodes: A standard barcode, B augmented barcode (same axes, one row per loop),
C persistence diagram with an arrow from each standard loop to its augmented
position. Loops that gain more than EPS Angstrom of persistence under
augmentation are drawn in red.

Standard and augmented loops are matched greedily by nearest neighbour in
(birth, death); augmentation only lowers a few Cys-Cys distances, so it is a
small perturbation of the diagram and the matching is reliable.

Usage:
  python per_protein_figures.py alphafold                 # the two featured proteins
  python per_protein_figures.py experimental --all        # every protein
  python per_protein_figures.py both --all
  python per_protein_figures.py alphafold --protein Cytotoxin_10 --protein BaspPLA2-II
  python per_protein_figures.py alphafold --all --class 3ft --force

results/per_protein/
  examples/<id>      featured proteins, every journal format
  <class>/<id>       --all: PDF and PNG for each protein
  summary.csv        loops, persistence totals and strengthened-loop counts per protein
"""
from __future__ import annotations

import sys
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist

import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import viz_common as vc
import ssbond_features as ssf
from viz_common import long_path

OUT = vc.DSRES / "per_protein"
EPS = 0.10
EXAMPLES = ("Cytotoxin_10__Toxin_CM-4a", "Acidic_phospholipase_A2_2__BaspPLA2-II")
BULK_FORMATS, BULK_DPI = ("pdf", "png"), 300
SUMMARY = ("Protein_ID", "Protein_Class", "N", "n_raw", "n_aug", "n_strengthened",
           "sumpers_raw", "sumpers_aug")


def match(raw, aug):
    """Greedy nearest-neighbour matching in (birth, death) space."""
    n, m = len(raw), len(aug)
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    D = cdist(raw, aug)
    ur, ua, matches = set(), set(), []
    for flat in np.argsort(D, axis=None):
        i, j = divmod(int(flat), m)
        if i in ur or j in ua:
            continue
        matches.append((i, j))
        ur.add(i)
        ua.add(j)
        if len(matches) == min(n, m):
            break
    return (matches, [i for i in range(n) if i not in ur], [j for j in range(m) if j not in ua])


def analyse(cls, pid):
    raw, aug = vc.load_bars(cls, pid, "standard"), vc.load_bars(cls, pid, "augmented")
    if raw is None or aug is None:
        return None
    pr = raw[:, 1] - raw[:, 0] if len(raw) else np.zeros(0)
    pa = aug[:, 1] - aug[:, 0] if len(aug) else np.zeros(0)
    matches, ru, au = match(raw, aug)
    rows = [(i, j, bool((pa[j] - pr[i]) > EPS))
            for i, j in sorted(matches, key=lambda ij: float(-pr[ij[0]]))]
    rows += [(i, None, False) for i in sorted(ru, key=lambda i: float(-pr[i]))]
    rows += [(None, j, False) for j in sorted(au, key=lambda j: float(-pa[j]))]
    pdb = vc.find_pdb(cls, pid)
    n_res = len(ssf.read_ca(pdb)[0]) if long_path(pdb).exists() else np.nan
    return dict(Protein_ID=pid, Protein_Class=cls, N=n_res, n_raw=len(raw), n_aug=len(aug),
                n_strengthened=sum(r[2] for r in rows),
                sumpers_raw=float(pr.sum()), sumpers_aug=float(pa.sum()),
                raw=raw, aug=aug, pr=pr, pa=pa, rows=rows, matches=matches)


def draw(info, base, formats=None, dpi=None):
    raw, aug, rows = info["raw"], info["aug"], info["rows"]
    R = len(rows)
    fig = plt.figure(figsize=(19, 5.6))
    axL = fig.add_subplot(1, 3, 1)
    axM = fig.add_subplot(1, 3, 2, sharex=axL, sharey=axL)
    axR = fig.add_subplot(1, 3, 3)
    top = max([float(b[:, 1].max()) for b in (raw, aug) if len(b)] or [1.0])
    for ax, bars, side, n in ((axL, raw, 0, info["n_raw"]), (axM, aug, 1, info["n_aug"])):
        for y, row in enumerate(rows):
            idx = row[side]
            if idx is None:
                continue
            b, d = bars[idx]
            hi = row[2]
            ax.plot([b, d], [R - y, R - y], color=vc.COL_AUG if hi else vc.COL_STD,
                    lw=1.9 if hi else 1.4, solid_capstyle="butt", zorder=3 if hi else 2)
        ax.set_xlim(-0.2, top + 0.2)
        ax.set_ylim(0.5, R + 1.5)
        ax.set_yticks([])
        ax.set_xlabel("Filtration scale (Angstrom)")
        ax.spines["left"].set_visible(False)
        ax.grid(False)
        ax.text(0.98, 0.97, f"{n} loops", transform=ax.transAxes, ha="right", va="top", fontsize=10)

    lim = top + 0.4
    axR.plot([0, lim], [0, lim], color="#bbb", lw=1, zorder=1)
    if len(raw):
        axR.scatter(raw[:, 0], raw[:, 1], s=22, facecolors="none", edgecolors=vc.COL_STD,
                    linewidths=1.0, alpha=.8, zorder=2)
    for i, j in info["matches"]:
        if (info["pa"][j] - info["pr"][i]) > EPS:
            axR.annotate("", xy=(aug[j, 0], aug[j, 1]), xytext=(raw[i, 0], raw[i, 1]),
                         arrowprops=dict(arrowstyle="->", color=vc.COL_AUG, lw=1.6), zorder=4)
            axR.scatter(aug[j, 0], aug[j, 1], s=34, color=vc.COL_AUG, zorder=5)
    axR.set_xlim(-0.3, lim)
    axR.set_ylim(-0.3, lim)
    axR.set_aspect("equal")
    axR.set_xlabel("Birth (Angstrom)")
    axR.set_ylabel("Death (Angstrom)")
    axR.grid(False)
    axR.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markeredgecolor=vc.COL_STD,
               markerfacecolor="none", label="standard loop"),
        Line2D([0], [0], marker="o", color=vc.COL_AUG, lw=0, label="strengthened by augmentation"),
        Line2D([0], [0], color=vc.COL_AUG, lw=1.6, label=r"standard $\rightarrow$ augmented")],
        loc="lower right", frameon=False, fontsize=10)
    vc.add_panel_letters([axL, axM, axR])
    fig.tight_layout()
    vc.savefig(fig, base, formats, dpi)


def main():
    ap = argparse.ArgumentParser(
        description="Per-protein standard vs augmented barcodes and persistence diagrams",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", default="alphafold", choices=vc.KEYS,
                    help="alphafold | experimental | both")
    ap.add_argument("--all", action="store_true", help="draw every protein (PDF + 300 dpi PNG)")
    ap.add_argument("--protein", action="append", default=[],
                    help="Protein_ID substring of a featured protein (repeatable)")
    ap.add_argument("--class", dest="cls", default=None, help="restrict to one class")
    ap.add_argument("--force", action="store_true", help="redraw figures that already exist")
    args = ap.parse_args()
    if args.dataset == "both":
        sys.exit(vc.run_both(__file__))

    feat = vc.load_features()
    feat = feat[feat["Protein_Class"].isin(vc.VALID)]
    if args.cls:
        feat = feat[feat["Protein_Class"] == args.cls]
    print(f"Per-protein figures [{vc.KEY}]: {len(feat)} proteins, "
          f"{'all' if args.all else 'featured only'}")

    infos, t0 = {}, time.time()
    for n, (_, r) in enumerate(feat.iterrows(), 1):
        cls, pid = r["Protein_Class"], r["Protein_ID"]
        try:
            info = analyse(cls, pid)
        except Exception as e:
            print(f"  !! {cls}/{pid}: {e}")
            continue
        if info is None:
            continue
        infos[pid] = info
        if args.all:
            base = OUT / cls / ssf.track_id(pid)
            if args.force or not (base.parent / f"{base.name}.pdf").exists():
                try:
                    draw(info, base, BULK_FORMATS, BULK_DPI)
                except Exception as e:
                    print(f"  !! {cls}/{pid}: {e}")
        if n % 50 == 0:
            print(f"  {n}/{len(feat)}  ({time.time() - t0:.0f}s)")

    if not infos:
        print("No protein has both standard and augmented H1 bars. Exiting.")
        sys.exit(1)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{k: v for k, v in i.items() if k in SUMMARY} for i in infos.values()])
    summary.insert(1, "file_id", [ssf.track_id(p) for p in summary["Protein_ID"]])
    summary.to_csv(OUT / "summary.csv", index=False)

    chosen = []
    for p in (args.protein or list(EXAMPLES)):
        hit = [i for pid, i in infos.items() if p in pid]
        if hit:
            chosen.append(hit[0])
        else:
            print(f"  no protein matches '{p}'")
    if not chosen and not args.protein:
        for cls in ("3ft", "PLA"):
            cand = [i for i in infos.values() if i["Protein_Class"] == cls]
            if cand:
                chosen.append(max(cand, key=lambda i: i["sumpers_aug"] - i["sumpers_raw"]))
    for info in chosen:
        draw(info, OUT / "examples" / ssf.track_id(info["Protein_ID"]))

    print(f"\nDone. {len(infos)} proteins analysed, {len(chosen)} featured -> {OUT}")


if __name__ == "__main__":
    main()
