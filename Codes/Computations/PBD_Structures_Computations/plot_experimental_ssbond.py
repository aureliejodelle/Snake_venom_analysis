
"""
plot_experimental_ssbond.py

Disulfide (C-alpha standard-vs-augmented) distribution comparison
Reads the per-protein table built by run_experimental_pipeline.py
(results/ssbond_features.csv), whose schema matches the disulfide_data one:
  Protein_Class, n_ssbond, has_ss,
  h1c_std / h1med_std / h1sum_std / h1max_std   (plain C-alpha Ripserer H1)
  h1c_aug / h1med_aug / h1sum_aug / h1max_aug   (staple-augmented H1)
  h1c_ss = h1c_aug - h1c_std                     (loops added by the staples)

Figures (PDF+PNG), 
  cycle_count_bar        per-family median H1 count, standard vs augmented
  delta_h1_bar           per-family mean H1 loops gained by the staples (aug-std)
  ssbond_count_bar       per-family median disulfide-bond count
  delta_h1_violin        per-family distribution of h1c_ss


Usage:  python plot_experimental_ssbond.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PIPE_CODES = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE_CODES))
try:
    from plot_style import use_computer_modern, disp, CLASS_COLORS
    use_computer_modern()
except Exception:                                          # noqa: BLE001
    def disp(c): return {"3ft": "3FTx", "PLA": "PLA$_2$", "SNMP": "SVMP"}.get(c, c)
    CLASS_COLORS = {"SNMP": "#e41a1c", "PLA": "#377eb8", "serine": "#4daf4a",
                    "3ft": "#984ea3", "kallikrein": "#ff7f00", "kunitz": "#a65628"}

C_STD, C_AUG = "#52636E", "#D1495B"      # standard / augmented (repo convention)
ORDER = ["3ft", "PLA", "serine", "SNMP", "kallikrein", "kunitz"]
DPI = 200
N_MIN = 15      # 
HATCH = "///"   # hatch used to de-weight small-n families show


def _weak(n):
    return n < N_MIN


def _label(classes, ns):
    return [f"{disp(c)}\n($n={n}$)" for c, n in zip(classes, ns)]


def _order(classes):
    present = [c for c in ORDER if c in classes]
    return present + [c for c in classes if c not in present]


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {name}")


def paired_bar(df, out_dir, col_std, col_aug, ylabel, title, name, agg="median"):
    classes = _order(sorted(df["Protein_Class"].unique()))
    fn = np.median if agg == "median" else np.mean
    ns = [int((df.Protein_Class == c).sum()) for c in classes]
    s = [fn(df[df.Protein_Class == c][col_std]) for c in classes]
    a = [fn(df[df.Protein_Class == c][col_aug]) for c in classes]
    x = np.arange(len(classes)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bs = ax.bar(x - w / 2, s, w, color=C_STD, edgecolor="black", linewidth=0.8,
                label="standard", zorder=3)
    ba = ax.bar(x + w / 2, a, w, color=C_AUG, edgecolor="black", linewidth=0.8,
                label="augmented", zorder=3)
    for i, n in enumerate(ns):
        if _weak(n):
            bs[i].set_hatch(HATCH); bs[i].set_alpha(0.55)
            ba[i].set_hatch(HATCH); ba[i].set_alpha(0.55)
    ax.set_xticks(x); ax.set_xticklabels(_label(classes, ns), fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12); ax.set_title(title)
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); _save(fig, out_dir, name)


def collect_curvature(out: Path):
    """Per-protein median Forman-Ricci curvature, standard vs augmented (built by full_ph_from_dm.jl)
    compute_hypergraphs.py -> compute_curvature.py. Returns None if not computed."""
    base = out / "disulfide" / "rc"
    std, aug = base / "standard", base / "augmented"
    if not (std.exists() and aug.exists()):
        return None
    rows = []
    for cls_dir in sorted(p for p in std.iterdir() if p.is_dir() and not p.name.startswith(".")):
        cls = cls_dir.name
        for f in cls_dir.glob("*.json"):
            af = aug / cls / f.name
            if not af.exists():
                continue
            s = list(json.load(open(f)).values())
            a = list(json.load(open(af)).values())
            if not s or not a:
                continue
            rows.append(dict(Protein_Class=cls, Protein_ID=f.stem,
                             curv_std=float(np.median(s)), curv_aug=float(np.median(a))))
    return pd.DataFrame(rows) if rows else None


def paired_box(df, out_dir, col_std, col_aug, ylabel, title, name,
               order=None, show_n=True):
    """Per-family grouped boxplot, standard vs augmented (2 boxes per family),
    mirroring the predicted-set persistence_boxplot. Small-n families hatched.
    order overrides the shared family ordering; show_n toggles the per-family
    (n=..) annotation in the x-tick labels."""
    from matplotlib.patches import Patch
    if order is not None:
        present = set(df["Protein_Class"].unique())
        classes = [c for c in order if c in present]
        classes += [c for c in sorted(present) if c not in classes]
    else:
        classes = _order(sorted(df["Protein_Class"].unique()))
    ns = [int((df.Protein_Class == c).sum()) for c in classes]
    std_data = [df.loc[df.Protein_Class == c, col_std].dropna().values for c in classes]
    aug_data = [df.loc[df.Protein_Class == c, col_aug].dropna().values for c in classes]
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(9, 5.2))

    def _box(data, positions, color):
        bp = ax.boxplot(data, positions=positions, widths=0.34, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.4),
                        flierprops=dict(marker="o", markersize=2.2, alpha=0.3, markeredgewidth=0))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(color); patch.set_alpha(0.65)
            if _weak(ns[i]):
                patch.set_hatch(HATCH)
        return bp

    _box(std_data, x - 0.19, C_STD)
    _box(aug_data, x + 0.19, C_AUG)
    labels = _label(classes, ns) if show_n else [disp(c) for c in classes]
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12); ax.set_title(title)
    ax.legend(handles=[Patch(facecolor=C_STD, alpha=0.65, label="standard"),
                       Patch(facecolor=C_AUG, alpha=0.65, label="augmented")],
              frameon=False, loc="best")
    ax.set_xlim(-0.6, len(classes) - 0.4)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); _save(fig, out_dir, name)


def single_bar(df, out_dir, col, ylabel, title, name, agg="mean"):
    classes = _order(sorted(df["Protein_Class"].unique()))
    fn = {"mean": np.mean, "median": np.median}[agg]
    ns = [int((df.Protein_Class == c).sum()) for c in classes]
    vals = [fn(df[df.Protein_Class == c][col]) for c in classes]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(range(len(classes)), vals, 0.68,
                  color=[CLASS_COLORS.get(c, "#888") for c in classes],
                  edgecolor="black", linewidth=0.8, zorder=3)
    for i, n in enumerate(ns):
        if _weak(n):
            bars[i].set_hatch(HATCH); bars[i].set_alpha(0.55)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(_label(classes, ns), fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12); ax.set_title(title)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); _save(fig, out_dir, name)


def delta_violin(df, out_dir, col, ylabel, title, name):
    classes = _order(sorted(df["Protein_Class"].unique()))
    ns = [int((df.Protein_Class == c).sum()) for c in classes]
    data = [df[df.Protein_Class == c][col].values for c in classes]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    parts = ax.violinplot(data, positions=range(len(classes)), widths=0.7,
                          showmedians=False, showextrema=False)
    for b, c, n in zip(parts["bodies"], classes, ns):
        b.set_facecolor(CLASS_COLORS.get(c, "#888")); b.set_alpha(0.55)
        if _weak(n):
            b.set_hatch(HATCH); b.set_edgecolor("black"); b.set_linewidth(0.8)
        else:
            b.set_edgecolor("white")
    ax.boxplot(data, positions=range(len(classes)), widths=0.16, patch_artist=True,
               boxprops=dict(facecolor="white", linewidth=1.2),
               medianprops=dict(color="black", linewidth=1.8),
               flierprops=dict(marker="o", markersize=2, alpha=0.4, markeredgewidth=0))
    ax.axhline(0, color="grey", lw=0.9, ls="--", zorder=1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(_label(classes, ns), fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12); ax.set_title(title)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); _save(fig, out_dir, name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_out = Path(__file__).resolve().parent.parent / "PDB_computation" / "experimental_data"
    ap.add_argument("--out", default=str(default_out))
    args = ap.parse_args()

    out = Path(args.out)
    src = out / "results" / "ssbond_features.csv"
    if not src.exists():
        print(f"[error] {src} not found -- run the ss_features step first")
        return 1
    out_dir = out / "h1_bars_ca" / "results"

    df = pd.read_csv(src)
    print(f"Disulfide figures from {src.name} ({len(df)} proteins) -> {out_dir}")

    sub = df[df["has_ss"] == True] if "has_ss" in df.columns else df  
    if sub.empty:
        sub = df

    paired_bar(sub, out_dir, "h1c_std", "h1c_aug",
               "Median H1 cycle count",
               "H1 cycle count per family: standard vs augmented",
               "cycle_count_bar", agg="median")
    paired_box(sub, out_dir, "h1med_std", "h1med_aug",
               "H1 median persistence",
               "H1 persistence by family: standard vs augmented",
               "persistence_boxplot",
               order=["SNMP", "3ft", "serine", "PLA", "kunitz"], show_n=False)

    # curvature std-vs-aug (only if the disulfide/rc tree has been computed)
    cdf = collect_curvature(out)
    if cdf is not None and not cdf.empty:
        cdf.to_csv(out / "results" / "curvature_ssbond_features.csv", index=False)
        paired_box(cdf, out_dir, "curv_std", "curv_aug",
                   "median Forman--Ricci curvature",
                   "Median curvature by family: standard vs augmented",
                   "curvature_boxplot",
                   order=["SNMP", "3ft", "serine", "PLA", "kunitz"], show_n=False)
    else:
        print("  [skip] curvature_boxplot (no disulfide/rc tree; run the "
              "augmented-curvature steps first)")
    single_bar(sub, out_dir, "h1c_ss",
               "Mean H1 loops gained (augmented - standard)",
               "Topological effect of disulfide staples per family",
               "delta_h1_bar", agg="mean")
    single_bar(df, out_dir, "n_ssbond",
               "Median disulfide-bond count",
               "Disulfide bonds per family",
               "ssbond_count_bar", agg="median")
    delta_violin(sub, out_dir, "h1c_ss",
                 "H1 loops gained (augmented - standard)",
                 "Distribution of disulfide-induced H1 loops per family",
                 "delta_h1_violin")

    # small per-family summary CSV alongside the figures
    g = sub.groupby("Protein_Class")
    summ = pd.DataFrame({
        "n": g.size(),
        "median_h1_std": g["h1c_std"].median(),
        "median_h1_aug": g["h1c_aug"].median(),
        "mean_delta_h1": g["h1c_ss"].mean(),
        "median_n_ssbond": df.groupby("Protein_Class")["n_ssbond"].median(),
    }).reindex(_order(list(g.groups)))
    out_dir.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out_dir / "ssbond_summary_per_family.csv")
    print(f"  [ok] ssbond_summary_per_family.csv")
    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
