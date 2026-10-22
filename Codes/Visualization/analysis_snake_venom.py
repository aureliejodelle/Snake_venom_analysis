"""
analysis_snake_venom.py

Snake-venom visualisation and statistics.
Disulfide analyses live in analysis_disulfide.py.

Usage:
  python analysis_snake_venom.py alphafold
  python analysis_snake_venom.py experimental
  python analysis_snake_venom.py both
  python analysis_snake_venom.py alphafold --only hist kde statistics
  python analysis_snake_venom.py experimental --stats-only

results/
  composition/    pi_per_protein.csv, composition_per_class.csv, aa_enrichment.csv,
                  aa_biochemical_per_class.csv, aa_presence_per_class.csv,
                  aa_shap_per_class.csv, aa_composition, aa_shap_per_class
  violinbox/ hist/ kde/ ecdf/  <feature>: one figure per feature in each folder
  statistics/     mann_whitney_pairwise.csv, kruskal_wallis.csv, levene_results.csv,
                  bartlett_results.csv, fligner_killeen_results.csv,
                  combined_variance_summary.csv, correlations_spearman.csv, curvature_cyclesize_spearman.csv,
                  size_control.csv, classification.csv, summary_report.txt
  significance/   pairwise_significance, effect_size_heatmap, kruskal_wallis_summary,
                  variance_tests
  correlations/   <y>_vs_<x>_all (pooled, with Spearman rho), <y>_vs_<x>_families (one panel per family);
                  H1_median_vs_chain_length_spearman: within-family rho with bootstrap CI
  cycles/         cycle_count, cycle_loop_size, curvature_vs_loop_size (separate figures)
  controls/       size_control_variance, classification_accuracy (more in analysis_controls.py)
  wasserstein/    family_w2_dendrogram, drawn from family_w2_6x6.csv (written by the W2 job)
"""
from __future__ import annotations

import sys
import json
import argparse
import warnings
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, spearmanr, mannwhitneyu, kruskal

import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import viz_common as vc
from viz_common import disp, disp_list, present, palette, col, stars, fdr_bh, savefig
from statistical_tests import (run_mann_whitney, run_kruskal_wallis, run_variance_tests,
                               variance_consensus)

warnings.filterwarnings("ignore")

SECTIONS = ("composition", "hist", "kde", "ecdf", "violinbox", "statistics", "significance",
            "correlations", "cycles", "controls", "wasserstein")
KINDS = ("hist", "kde", "ecdf", "violinbox")
OUT = {k: vc.RES / k for k in ("composition", "violinbox", "hist", "kde", "ecdf", "statistics",
                               "significance", "correlations", "cycles", "controls")}

PAIRS = [
    ("chain_length", "Curv_median"),
    ("H1_median",    "Curv_median"),
    ("cys_fraction", "Curv_median"),
    ("chain_length", "H1_median"),
    ("cys_fraction", "H1_median"),
    ("chain_length", "pI"),
    ("cys_fraction", "pI"),
    (vc.BFAC_MEAN,   "Curv_median"),
    (vc.BFAC_MEAN,   "H1_median"),
    ("H1_count",     "Curv_median"),
    ("cys_fraction", "curv_norm"),
]

AA_ROW_ORDER = ["3ft", "PLA", "serine", "SNMP", "kallikrein", "kunitz"]
AA_NAMES = {"A": "Ala", "C": "Cys", "D": "Asp", "E": "Glu", "F": "Phe", "G": "Gly",
            "H": "His", "I": "Ile", "K": "Lys", "L": "Leu", "M": "Met", "N": "Asn",
            "P": "Pro", "Q": "Gln", "R": "Arg", "S": "Ser", "T": "Thr", "V": "Val",
            "W": "Trp", "Y": "Tyr"}
BIOCATS = {
    "Nonpolar aliphatic": list("GAVLIMP"),
    "Aromatic":           list("FWY"),
    "Polar uncharged":    list("STNQ"),
    "Positively charged": list("RKH"),
    "Negatively charged": list("DE"),
    "Cysteine":           list("C"),
}
BIOCAT_COLORS = ["#8DA0CB", "#FC8D62", "#66C2A5", "#E78AC3", "#A6D854", "#FFD92F"]


# COMPOSITION

def _aa_tables(df):
    cols = [f"aa_{a}" for a in vc.AA_LETTERS]
    if not set(cols) <= set(df.columns):
        return None
    d = df.dropna(subset=cols)
    if d["Protein_Class"].nunique() < 2:
        return None
    gm, gs = d[cols].mean(), d[cols].std().replace(0, 1e-9)
    z = (d.groupby("Protein_Class")[cols].mean() - gm) / gs
    z.columns = vc.AA_LETTERS
    order = [c for c in AA_ROW_ORDER if c in z.index] + [c for c in z.index if c not in AA_ROW_ORDER]
    z = z.loc[order]
    star = pd.DataFrame("", index=z.index, columns=z.columns)
    for c in z.index:
        m = d["Protein_Class"] == c
        pv = []
        for a in vc.AA_LETTERS:
            x, y = d.loc[m, f"aa_{a}"], d.loc[~m, f"aa_{a}"]
            pv.append(mannwhitneyu(x, y, alternative="two-sided")[1]
                      if len(x) >= 3 and len(y) >= 3 else 1.0)
        _, pc = fdr_bh(pv)
        for a, p in zip(vc.AA_LETTERS, pc):
            star.loc[c, a] = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    bio = pd.DataFrame({cat: d[[f"aa_{a}" for a in mem]].sum(axis=1) for cat, mem in BIOCATS.items()})
    bio = bio.groupby(d["Protein_Class"].values).mean().loc[order]
    presence = (d[cols] > 0).groupby(d["Protein_Class"].values).mean().loc[order]
    presence.columns = vc.AA_LETTERS
    return d, z, star, bio, presence


def plot_aa_composition(z, star, bio):
    fig = plt.figure(figsize=(19, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.2])
    ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    vmax = float(np.abs(z.values).max())
    sns.heatmap(z.astype(float), ax=ax_a, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                linewidths=0.4, linecolor="#cccccc", annot=star.values, fmt="",
                annot_kws={"size": 7, "color": "black", "fontweight": "bold"},
                cbar_kws={"label": "Enrichment z-score", "shrink": 0.8})
    ax_a.set_xticklabels([f"{a}\n({AA_NAMES[a]})" for a in z.columns], rotation=0, fontsize=8)
    ax_a.set_yticklabels(disp_list(list(z.index)), rotation=0, fontsize=10)
    ax_a.set_xlabel("Amino acid", fontsize=11)
    ax_a.set_ylabel("")
    ax_a.grid(False)
    x = np.arange(len(bio))
    bottom = np.zeros(len(bio))
    for cat, color in zip(BIOCATS, BIOCAT_COLORS):
        vals = bio[cat].values
        ax_b.bar(x, vals, bottom=bottom, color=color, label=cat, edgecolor="white", linewidth=0.5)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.04:
                ax_b.text(i, b + v / 2, f"{v * 100:.0f}%", ha="center", va="center",
                          fontsize=7, fontweight="bold")
        bottom += vals
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(disp_list(list(bio.index)), rotation=30, ha="right", fontsize=9)
    ax_b.set_ylabel("Fraction of residues", fontsize=11)
    ax_b.set_ylim(0, 1)
    ax_b.legend(fontsize=7, framealpha=0.9, loc="upper left", bbox_to_anchor=(1.01, 1))
    vc.add_panel_letters([ax_a, ax_b])
    fig.tight_layout()
    savefig(fig, OUT["composition"] / "aa_composition")


def run_aa_shap(df):
    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        print("  shap / scikit-learn not installed; per-class SHAP skipped")
        return
    aa_cols = [f"aa_{a}" for a in vc.AA_LETTERS]
    if not set(aa_cols) <= set(df.columns):
        return
    num = df.select_dtypes(include=[np.number])
    feat_cols = [c for c in num.columns if num[c].notna().any() and num[c].std() > 1e-9]
    if not set(aa_cols) <= set(feat_cols):
        return
    sub = df[df["Protein_Class"].isin(AA_ROW_ORDER)]
    le = LabelEncoder()
    y = le.fit_transform(sub["Protein_Class"].values)
    X = SimpleImputer(strategy="median").fit_transform(sub[feat_cols].values)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
    rf.fit(X, y)
    raw = shap.TreeExplainer(rf).shap_values(X)
    sv = np.array(raw)
    k = len(le.classes_)
    if sv.ndim == 3 and sv.shape[0] == k:
        sv3 = sv
    elif sv.ndim == 3 and sv.shape[2] == k:
        sv3 = sv.transpose(2, 0, 1)
    elif isinstance(raw, list) and len(raw) == k:
        sv3 = np.stack(raw, axis=0)
    else:
        return
    idx = [feat_cols.index(c) for c in aa_cols]
    tab = pd.DataFrame({c: sv3[i][:, idx].mean(axis=0) for i, c in enumerate(le.classes_)},
                       index=vc.AA_LETTERS)
    order = [c for c in AA_ROW_ORDER if c in tab.columns]
    tab = tab[order]
    tab.to_csv(OUT["composition"] / "aa_shap_per_class.csv")
    if vc.STATE["stats_only"]:
        return
    fig, axes = plt.subplots(1, len(order), figsize=(3.2 * len(order), 5.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, order):
        vals = tab[c].values
        ax.barh(vc.AA_LETTERS, vals, color=["#d73027" if v > 0 else "#4575b4" for v in vals],
                edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.text(0.97, 0.02, disp(c), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, fontweight="bold", color=col(c))
        ax.set_xlabel("Mean SHAP", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Amino acid", fontsize=11)
    axes[0].invert_yaxis()
    vc.add_panel_letters(axes)
    fig.tight_layout()
    savefig(fig, OUT["composition"] / "aa_shap_per_class")


def run_composition(df, comp):
    out = OUT["composition"]
    out.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out / "pi_per_protein.csv", index=False)
    vc.HIGH.mkdir(parents=True, exist_ok=True)
    comp.to_csv(vc.HIGH / "pi_per_protein.csv", index=False)
    cols = [c for c in ("chain_length", "cys_fraction", "pI", vc.BFAC_MEAN,
                        "H1_median", "Curv_median") if c in df.columns]
    rows = []
    for c in present(df):
        g = df[df["Protein_Class"] == c]
        r = {"Protein_Class": c, "n": len(g)}
        for k in cols:
            r[f"{k}_mean"] = round(float(g[k].mean()), 4)
            r[f"{k}_median"] = round(float(g[k].median()), 4)
        if vc.BFAC_STEM == "plddt" and vc.BFAC_MEAN in g.columns:
            r["frac_plddt_ge70"] = round(float((g[vc.BFAC_MEAN].dropna() >= 70).mean()), 4)
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "composition_per_class.csv", index=False)
    res = _aa_tables(df)
    if res is not None:
        d, z, star, bio, presence = res
        z.to_csv(out / "aa_enrichment.csv")
        bio.to_csv(out / "aa_biochemical_per_class.csv")
        presence.round(4).to_csv(out / "aa_presence_per_class.csv")
        if not vc.STATE["stats_only"]:
            plot_aa_composition(z, star, bio)
        run_aa_shap(df)
    print(f"  composition outputs -> {out}")


# DISTRIBUTIONS


def _violin_box_ax(ax, groups, colors, labels):
    allv = np.concatenate([g for g in groups if len(g)])
    q1, q3 = np.percentile(allv, [5, 95])
    pad = (q3 - q1) * 0.5 if q3 > q1 else max(abs(q3) * 0.5, 0.5)
    y_min, y_max = q1 - pad, q3 + pad * 1.2
    pos = range(len(groups))
    parts = ax.violinplot(groups, positions=pos, widths=0.6, showmeans=False, showmedians=False)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(colors[i])
        b.set_alpha(0.22)
    for k in ("cmins", "cmaxes", "cbars"):
        if k in parts:
            parts[k].set_visible(False)
    bp = ax.boxplot(groups, positions=pos, widths=0.30, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=3, alpha=0.35,
                                    markerfacecolor="gray", markeredgewidth=0),
                    medianprops=dict(color="black", lw=2.0))
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i])
        box.set_alpha(0.8)
    for i, g in enumerate(groups):
        ax.text(i, y_min - pad * 0.1, f"n={len(g)}", ha="center", va="top",
                fontsize=8, color="#666666")
    ax.set_ylim(y_min - pad * 0.15, y_max)
    ax.set_xticks(list(pos))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.grid(axis="y", alpha=0.2, lw=0.5)


def _draw_groups(ax, kind, groups, classes, colors):
    plotted = False
    for g, cls, c in zip(groups, classes, colors):
        label = f"{disp(cls)} (n={len(g)})"
        if len(g) < (4 if kind == "kde" else 3):
            continue
        if kind == "hist":
            ax.hist(g, bins=min(30, max(5, int(np.sqrt(len(g))) * 2)), density=True,
                    alpha=0.45, color=c, edgecolor="white", linewidth=0.4, label=label)
            plotted = True
        elif kind == "kde":
            try:
                kde = gaussian_kde(g, bw_method="scott")
            except Exception:
                continue
            xs = np.linspace(g.min(), g.max(), 300)
            ax.plot(xs, kde(xs), color=c, lw=2.0, label=label)
            ax.fill_between(xs, kde(xs), alpha=0.07, color=c)
            ax.axvline(np.median(g), color=c, lw=1.0, ls="--", alpha=0.65)
            plotted = True
        else:
            v = np.sort(g)
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, lw=2.0, label=label)
            plotted = True
    return plotted


DIST_KINDS = {"hist": "Density", "kde": "Density", "ecdf": "F(x)"}


def plot_distribution(df, classes, colors, feat, kinds):
    groups = [df.loc[df["Protein_Class"] == c, feat].dropna().values for c in classes]
    groups = [g[np.isfinite(g)] for g in groups]
    if not any(len(g) >= 3 for g in groups):
        return
    label = vc.feat_label(feat)
    if "violinbox" in kinds:
        fig, ax = plt.subplots(figsize=(7, 5))
        _violin_box_ax(ax, groups, colors, disp_list(classes))
        ax.set_ylabel(label, fontsize=11)
        fig.tight_layout()
        savefig(fig, OUT["violinbox"] / feat)
    for kind, ylabel in DIST_KINDS.items():
        if kind not in kinds:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        if _draw_groups(ax, kind, groups, classes, colors):
            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            if kind == "ecdf":
                ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8, framealpha=0.9)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            savefig(fig, OUT[kind] / feat)
        else:
            plt.close(fig)


def run_distributions(df, classes, feats, kinds):
    if vc.STATE["stats_only"]:
        return
    sub = df[df["Protein_Class"].isin(classes)]
    colors = palette(classes)
    for feat in feats:
        plot_distribution(sub, classes, colors, feat, kinds)


# STATISTICS


def _write_summary(pw, kw, var, df, classes, feats, out):
    lines = ["=" * 65, "STATISTICAL ANALYSIS - SNAKE VENOM", "=" * 65,
             f"Dataset   : {vc.KEY}",
             f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             f"Classes   : {len(classes)}"]
    for c in classes:
        lines.append(f"  {c:<45} n={(df['Protein_Class'] == c).sum()}")
    lines += [f"Features  : {len(feats)}", "", "─" * 65,
              "KRUSKAL-WALLIS TEST  (are any classes different?)", "─" * 65]
    if kw.empty:
        lines.append("No Kruskal-Wallis results available.")
    else:
        lines += [f"Features tested : {len(kw)}  (FDR-BH corrected)",
                  f"Significant     : {int(kw['significant'].sum())}", "",
                  f"{'Feature':<30} {'H':>8} {'p-value':>12} {'p-FDR':>12} {'sig':>5}", "-" * 70]
        for _, r in kw.iterrows():
            mark = ("  ***" if r["p_fdr"] < 0.001 else "   **" if r["p_fdr"] < 0.01
                    else "    *" if r["p_fdr"] < 0.05 else "   ns")
            lines.append(f"{r['display']:<30} {r['H_stat']:>8.2f} "
                         f"{r['p_value']:>12.2e} {r['p_fdr']:>12.2e}{mark}")
    lines += ["", "─" * 65, "MANN-WHITNEY U  (top 20 pairwise comparisons by p_fdr)", "─" * 65]
    if pw.empty:
        lines.append("No Mann-Whitney results available.")
    else:
        lines += [f"Total tests     : {len(pw)}  (FDR-BH corrected)",
                  f"Significant     : {int(pw['significant'].sum())}", ""]
        top = pw[pw["significant"]].nsmallest(20, "p_fdr")
        if top.empty:
            top = pw.nsmallest(20, "p_fdr")
        lines += [f"{'Feature':<22} {'Group1':<20} {'Group2':<20} {'delta':>7} {'p_fdr':>10}",
                  "-" * 82]
        for _, r in top.iterrows():
            lines.append(f"{vc.feat_label(r['feature']):<22} {r['group1']:<20} {r['group2']:<20} "
                         f"{r['cliffs_delta']:>+7.3f} {r['p_fdr']:>10.2e}")
    comb = var.get("combined", pd.DataFrame())
    lines += ["", "─" * 65, "VARIANCE TESTS  (Levene, Bartlett, Fligner-Killeen; FDR-BH)", "─" * 65]
    if comb.empty:
        lines.append("No variance-test results available.")
    else:
        lines += [f"{'Feature':<30} {'Levene':>8} {'Bartlett':>9} {'Fligner':>8} {'tests sig':>10}",
                  "-" * 70]
        for _, r in comb.iterrows():
            lines.append(f"{r['display']:<30} {'yes' if r['levene_sig'] else 'no':>8} "
                         f"{'yes' if r['bartlett_sig'] else 'no':>9} "
                         f"{'yes' if r['fligner_sig'] else 'no':>8} {int(r['n_tests_significant']):>10}")
        cons = variance_consensus(var)
        lines += ["", "Significant in all three: "
                  + (", ".join(vc.feat_label(f) for f in cons) or "none")]
    (out / "summary_report.txt").write_text("\n".join(lines), encoding="utf-8")


def run_statistics(df, classes, feats, pw, kw, var):
    out = OUT["statistics"]
    out.mkdir(parents=True, exist_ok=True)
    if not pw.empty:
        pw.to_csv(out / "mann_whitney_pairwise.csv", index=False)
        print(f"  Mann-Whitney U: {len(pw)} tests, {int(pw['significant'].sum())} significant")
    if not kw.empty:
        kw.to_csv(out / "kruskal_wallis.csv", index=False)
        print(f"  Kruskal-Wallis: {len(kw)} features, {int(kw['significant'].sum())} significant")
    for name, fname in (("levene", "levene_results.csv"), ("bartlett", "bartlett_results.csv"),
                        ("fligner", "fligner_killeen_results.csv"),
                        ("combined", "combined_variance_summary.csv")):
        d = var.get(name)
        if d is not None and not d.empty:
            d.to_csv(out / fname, index=False)
    comb = var.get("combined")
    if comb is not None and not comb.empty:
        print(f"  Variance tests: {int((comb['n_tests_significant'] == 3).sum())} "
              "features significant in all three")
    _write_summary(pw, kw, var, df, classes, feats, out)


# SIGNIFICANCE


def plot_kruskal_summary(kw):
    if kw.empty:
        return
    k = kw.sort_values("p_fdr")
    fig, ax = plt.subplots(figsize=(7, max(4, len(k) * 0.38)))
    ax.barh(k["display"], k["neg_log10_p"],
            color=["#E74C3C" if s else "#BDC3C7" for s in k["significant"]],
            edgecolor="white", height=0.7)
    ax.axvline(-np.log10(0.05), color="gray", lw=1.0, ls="--", label="p=0.05")
    ax.set_xlabel("-log₁₀(p-value)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    savefig(fig, OUT["significance"] / "kruskal_wallis_summary")


def plot_effect_size_heatmap(pw, order):
    if pw.empty:
        return
    feats = [f for f in vc.FEATURE_COLS if f in pw["feature"].values]
    pairs = [f"{a} vs {b}" for a, b in combinations(order, 2)]
    rows = [vc.feat_label(f) for f in feats]
    delta = pd.DataFrame(np.nan, index=rows, columns=pairs)
    star = pd.DataFrame("", index=rows, columns=pairs)
    for _, r in pw.iterrows():
        fd = vc.feat_label(r["feature"])
        if fd not in delta.index:
            continue
        ps, pr = f"{r['group1']} vs {r['group2']}", f"{r['group2']} vs {r['group1']}"
        if ps in pairs:
            delta.loc[fd, ps] = r["cliffs_delta"]
            star.loc[fd, ps] = stars(r["p_fdr"])
        elif pr in pairs:
            delta.loc[fd, pr] = -r["cliffs_delta"]
            star.loc[fd, pr] = stars(r["p_fdr"])
    fig, ax = plt.subplots(figsize=(max(5, len(pairs) * 1.6), max(5, len(feats) * 0.45)))
    sns.heatmap(delta.astype(float), ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.35, linecolor="white", annot=star.values, fmt="",
                cbar_kws={"label": "Cliff's delta", "shrink": 0.65})
    ax.grid(False)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    savefig(fig, OUT["significance"] / "effect_size_heatmap")


def _pairwise_matrix(d, column):
    dd = d.dropna(subset=[column])
    cls_o = present(dd)
    n = len(cls_o)
    if n < 2:
        return None
    idx = {c: i for i, c in enumerate(cls_o)}
    mat = np.full((n, n), np.nan)
    star = [["" for _ in range(n)] for _ in range(n)]
    pairs, pvals = [], []
    for c1, c2 in combinations(cls_o, 2):
        a = dd[dd["Protein_Class"] == c1][column].values
        b = dd[dd["Protein_Class"] == c2][column].values
        if len(a) < 1 or len(b) < 1:
            continue
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        pairs.append((idx[c1], idx[c2]))
        pvals.append(p)
    if not pvals:
        return None
    _, pfdr = fdr_bh(np.array(pvals))
    for (i, j), pf in zip(pairs, pfdr):
        lo, hi = max(i, j), min(i, j)
        mat[lo, hi] = -np.log10(np.clip(pf, 1e-60, 1))
        star[lo][hi] = stars(pf)
    return cls_o, mat, star


def plot_pairwise_significance(df):
    panels = []
    for column in ("H1_median", "Curv_median"):
        if column in df.columns:
            res = _pairwise_matrix(df, column)
            if res is not None:
                panels.append(res)
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6.8 * len(panels), 5.6))
    axes = np.atleast_1d(axes)
    for ax, (cls_o, mat, star) in zip(axes, panels):
        n = len(cls_o)
        mask = np.triu(np.ones((n, n), bool))
        vmax = np.nanmax(mat) if np.isfinite(np.nanmax(mat)) else 1.0
        im = ax.imshow(np.ma.masked_array(mat, mask), cmap="YlOrRd", vmin=0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
            r"$-\log_{10}(p_{\rm FDR})$", fontsize=11)
        ax.set_xticks(range(n))
        ax.set_xticklabels(disp_list(cls_o), rotation=30, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(disp_list(cls_o))
        ax.grid(False)
        for i in range(n):
            for j in range(n):
                if mask[i, j] or not star[i][j]:
                    continue
                ax.text(j, i, star[i][j], ha="center", va="center", fontsize=10,
                        fontweight="bold", color="white" if mat[i, j] > 0.6 * vmax else "black")
    vc.add_panel_letters(axes)
    fig.tight_layout()
    savefig(fig, OUT["significance"] / "pairwise_significance")


def plot_variance_heatmap(var):
    tests = [(n, var.get(k)) for n, k in (("Levene", "levene"), ("Bartlett", "bartlett"),
                                          ("Fligner-Killeen", "fligner"))]
    tests = [(n, d) for n, d in tests if d is not None and not d.empty]
    if not tests:
        return
    feats = [f for f in vc.FEATURE_COLS if any(f in d["feature"].values for _, d in tests)]
    labels = [vc.feat_label(f) for f in feats]
    data = pd.DataFrame({n: d.set_index("feature")["neg_log10_p_fdr"].reindex(feats).values
                         for n, d in tests}, index=labels)
    annot = pd.DataFrame({n: d.set_index("feature")["sig_stars"].reindex(feats).fillna("").values
                          for n, d in tests}, index=labels)
    fig, ax = plt.subplots(figsize=(6.5, max(4.5, len(labels) * 0.42)))
    sns.heatmap(data.astype(float), ax=ax, cmap="YlOrRd", linewidths=0.4, linecolor="white",
                annot=annot.values, fmt="",
                cbar_kws={"label": r"$-\log_{10}(p_{\rm FDR})$", "shrink": 0.7})
    ax.grid(False)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.yticks(fontsize=9)
    fig.tight_layout()
    savefig(fig, OUT["significance"] / "variance_tests")


def run_significance(df, classes, pw, kw, var):
    if vc.STATE["stats_only"]:
        return
    plot_kruskal_summary(kw)
    plot_effect_size_heatmap(pw, classes)
    plot_pairwise_significance(df)
    plot_variance_heatmap(var)


# CORRELATIONS


def _boot_ci(x, y, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.column_stack([x, y])
    vals = []
    for _ in range(n_boot):
        s = arr[rng.integers(0, len(arr), len(arr))]
        r = spearmanr(s[:, 0], s[:, 1]).correlation
        if np.isfinite(r):
            vals.append(r)
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def spearman_rows(d, x, y):
    sub = d.dropna(subset=[x, y])
    rows = []
    for name, g in [("all", sub)] + [(c, sub[sub["Protein_Class"] == c]) for c in present(sub)]:
        if len(g) < 5:
            continue
        rho, p = spearmanr(g[x], g[y])
        lo, hi = _boot_ci(g[x].values, g[y].values)
        rows.append(dict(x=x, y=y, group=name, n=len(g), rho=round(float(rho), 4),
                         p_value=float(p), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
    return rows


def pair_figure(d, x, y):
    """Two figures per pair."""
    sub = d.dropna(subset=[x, y])
    cls_o = present(sub)
    if not cls_o:
        return
    fig, ax = plt.subplots(figsize=(7, 5.4))
    for c in cls_o:
        s = sub[sub["Protein_Class"] == c]
        ax.scatter(s[x], s[y], color=col(c), s=22, alpha=0.6, linewidths=0, label=disp(c))
    if len(sub) >= 5:
        rho, p = spearmanr(sub[x], sub[y])
        if sub[x].nunique() > 1:
            slope, icpt = np.polyfit(sub[x], sub[y], 1)
            xs = np.linspace(sub[x].min(), sub[x].max(), 100)
            ax.plot(xs, slope * xs + icpt, color="black", lw=1.4, ls="--", zorder=5)
        ax.text(0.03, 0.96, f"all data: n = {len(sub)}\nSpearman ρ = {rho:.2f}, p = {p:.1e}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
    ax.legend(fontsize=8, framealpha=0.8, loc="lower right")
    ax.set_xlabel(vc.feat_label(x), fontsize=11)
    ax.set_ylabel(vc.feat_label(y), fontsize=11)
    ax.grid(alpha=0.18, lw=0.4)
    fig.tight_layout()
    savefig(fig, OUT["correlations"] / f"{y}_vs_{x}_all")

    fig, axes = vc.grid(len(cls_o))
    for ax, c in zip(axes, cls_o):
        s = sub[sub["Protein_Class"] == c]
        ax.scatter(s[x], s[y], color=col(c), s=26, alpha=0.7, linewidths=0)
        ax.text(0.97, 0.96, f"{disp(c)} (n={len(s)})", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=col(c))
        if len(s) >= 5:
            rho = spearmanr(s[x], s[y]).correlation
            ax.text(0.97, 0.05, f"ρ = {rho:.2f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=9)
        ax.set_xlabel(vc.feat_label(x), fontsize=10)
        ax.set_ylabel(vc.feat_label(y), fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.18, lw=0.4)
    for ax in axes[len(cls_o):]:
        ax.set_visible(False)
    vc.add_panel_letters(axes)
    fig.tight_layout()
    savefig(fig, OUT["correlations"] / f"{y}_vs_{x}_families")


def plot_spearman_bars(res, x, y):
    d = res[(res["x"] == x) & (res["y"] == y) & (res["group"] != "all")]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    pos = np.arange(len(d))
    ax.bar(pos, d["rho"], width=0.6, color=[col(g) for g in d["group"]], alpha=0.8,
           edgecolor="white")
    ax.errorbar(pos, d["rho"], yerr=[d["rho"] - d["ci_lo"], d["ci_hi"] - d["rho"]],
                fmt="none", ecolor="black", capsize=5)
    for i, (_, r) in enumerate(d.iterrows()):
        up = r["rho"] >= 0
        ax.text(i, r["ci_hi"] + 0.03 if up else r["ci_lo"] - 0.05, r["sig_stars"],
                ha="center", va="bottom" if up else "top")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(pos)
    ax.set_xticklabels(disp_list(list(d["group"])))
    ax.set_ylabel(rf"Spearman $\rho$ ({vc.feat_label(x)} vs {vc.feat_label(y)})")
    fig.tight_layout()
    savefig(fig, OUT["correlations"] / f"{y}_vs_{x}_spearman")


def run_correlations(df):
    rows = []
    for x, y in PAIRS:
        if x not in df.columns or y not in df.columns:
            continue
        print(f"  {y} vs {x}")
        rows.extend(spearman_rows(df, x, y))
        if not vc.STATE["stats_only"]:
            pair_figure(df, x, y)
    if rows:
        res = pd.DataFrame(rows)
        _, res["p_fdr"] = fdr_bh(res["p_value"].values)
        res["sig_stars"] = res["p_fdr"].apply(stars)
        OUT["statistics"].mkdir(parents=True, exist_ok=True)
        res.to_csv(OUT["statistics"] / "correlations_spearman.csv", index=False)
        if not vc.STATE["stats_only"]:
            plot_spearman_bars(res, "chain_length", "H1_median")


# CYCLES


def collect_cycles():
    recs = []
    for fam in vc.VALID:
        fam_dir = vc.HG / fam / "hyperedge_map"
        if not fam_dir.is_dir():
            continue
        for hf in fam_dir.glob("*.json"):
            with open(hf) as fh:
                hm = json.load(fh)
            sizes = [len(v) for v in hm.values()]
            rf = vc.RC / fam / hf.name
            if not sizes or not rf.exists():
                continue
            with open(rf) as fh:
                curv = list(json.load(fh).values())
            if curv:
                recs.append(dict(Protein_Class=fam, mean_size=float(np.mean(sizes)),
                                 med_curv=float(np.median(curv))))
    return pd.DataFrame(recs)


def _curv_size_ax(ax, cyc):
    fams = [f for f in vc.VALID if (cyc["Protein_Class"] == f).any()]
    for f in fams:
        s = cyc[cyc["Protein_Class"] == f]
        ax.scatter(s["mean_size"], s["med_curv"], s=16, alpha=0.5, color=col(f),
                   edgecolors="none", label=disp(f))
    for f in fams:
        s = cyc[cyc["Protein_Class"] == f]
        ax.scatter(s["mean_size"].mean(), s["med_curv"].median(), s=160, marker="D",
                   color=col(f), edgecolors="black", linewidths=1.2, zorder=6)
    ax.set_xlabel("Mean H1 loop size per protein (residues)")
    ax.set_ylabel("Median Forman-Ricci curvature per protein")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")


def run_cycles(df):
    cyc = collect_cycles()
    plots = []
    if "H1_count" in df.columns:
        plots.append(("cycle_count", (6.5, 4.8), lambda ax: vc.bar_by_class(
            ax, df, "H1_count", np.median, "Median H1 cycle count", "{:.0f}", True)))
    if not cyc.empty:
        plots.append(("cycle_loop_size", (6.5, 4.8), lambda ax: vc.bar_by_class(
            ax, cyc, "mean_size", np.mean, "Mean H1 loop size (residues)", "{:.1f}", True)))
        plots.append(("curvature_vs_loop_size", (7, 5.2), lambda ax: _curv_size_ax(ax, cyc)))
        rho, p = spearmanr(cyc["mean_size"], cyc["med_curv"])
        OUT["statistics"].mkdir(parents=True, exist_ok=True)
        pd.DataFrame([dict(rho=rho, p_value=p, n=len(cyc))]).to_csv(
            OUT["statistics"] / "curvature_cyclesize_spearman.csv", index=False)
        print(f"  proteins with cycle data: {len(cyc)}; Spearman(loop size, curvature) "
              f"rho={rho:.3f}, p={p:.2e}")
    else:
        print(f"  no hyperedge/curvature files under {vc.HG} / {vc.RC}; loop-size plots skipped")
    if vc.STATE["stats_only"]:
        return
    for name, size, draw in plots:
        fig, ax = plt.subplots(figsize=size)
        draw(ax)
        fig.tight_layout()
        savefig(fig, OUT["cycles"] / name)


# ROBUSTNESS: size control and family classification


def run_robustness(df, classes):
    import statsmodels.formula.api as smf
    d0 = df[df["Protein_Class"].isin(classes)].copy()
    if "chain_length" not in d0.columns:
        return
    d0["logsize"] = np.log(d0["chain_length"])
    rows = []
    for column, label in (("H1_count", "H1 cycle count"),
                          ("Curv_median", "Median curvature"),
                          ("H1_median", "Median persistence")):
        if column not in d0.columns:
            continue
        d = d0[[column, "logsize", "Protein_Class"]].dropna().rename(columns={column: "y"})
        if d["Protein_Class"].nunique() < 2 or len(d) < 10:
            continue
        r2_size = smf.ols("y ~ logsize", d).fit().rsquared
        r2_fam = smf.ols("y ~ C(Protein_Class)", d).fit().rsquared
        r2_full = smf.ols("y ~ logsize + C(Protein_Class)", d).fit().rsquared
        d["resid"] = smf.ols("y ~ logsize", data=d).fit().resid
        k = d["Protein_Class"].nunique()
        H, p = kruskal(*[g["resid"].values for _, g in d.groupby("Protein_Class")])
        Hr, _ = kruskal(*[g["y"].values for _, g in d.groupby("Protein_Class")])
        rows.append(dict(descriptor=column, label=label, r2_size=r2_size, r2_family=r2_fam,
                         r2_family_beyond_size=r2_full - r2_size,
                         eps2_raw=(Hr - k + 1) / (len(d) - k),
                         eps2_resid=(H - k + 1) / (len(d) - k), p_resid=p))
    sc = pd.DataFrame(rows)
    OUT["statistics"].mkdir(parents=True, exist_ok=True)
    if not sc.empty:
        sc.to_csv(OUT["statistics"] / "size_control.csv", index=False)
        for _, r in sc.iterrows():
            print(f"  {r['label']}: R2 size {r['r2_size']:.3f}, family beyond size "
                  f"{r['r2_family_beyond_size']:.3f}, KW p={r['p_resid']:.1e}")

    size_f = [c for c in ("chain_length", "HG_n_nodes") if c in d0.columns]
    pers_f = [c for c in ("H1_mean", "H1_median", "H1_max", "H1_std") if c in d0.columns]
    curv_f = [c for c in ("Curv_mean", "Curv_median", "Curv_std", "Curv_min", "Curv_max")
              if c in d0.columns]
    sets = {"size only": size_f, "persistence only": pers_f, "curvature only": curv_f,
            "persistence + curvature": pers_f + curv_f,
            "size + persistence + curvature": size_f + pers_f + curv_f}
    clf_rows = []
    for name, cols in sets.items():
        if not cols:
            continue
        try:
            m, s = vc.balanced_accuracy_cv(d0, cols)
        except Exception as e:
            print(f"  classification '{name}' skipped ({str(e)[:50]})")
            continue
        clf_rows.append(dict(feature_set=name, balanced_acc=m, sd=s))
        print(f"  {name:<32} balanced accuracy = {m:.3f} ± {s:.3f}")
    clf = pd.DataFrame(clf_rows)
    if not clf.empty:
        clf.to_csv(OUT["statistics"] / "classification.csv", index=False)
    if vc.STATE["stats_only"] or (sc.empty and clf.empty):
        return
    if not sc.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(sc))
        w = 0.27
        ax.bar(x - w, sc["r2_size"], w, label="Explained by size", color="#888888")
        ax.bar(x, sc["r2_family"], w, label="Explained by family", color="#377eb8")
        ax.bar(x + w, sc["r2_family_beyond_size"], w, label="Family beyond size", color="#e41a1c")
        for xi, v in zip(x + w, sc["r2_family_beyond_size"]):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(sc["label"], fontsize=10)
        ax.set_ylabel("Variance explained ($R^2$)")
        ax.set_ylim(0, 1)
        ax.legend(framealpha=0.8, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        savefig(fig, OUT["controls"] / "size_control_variance")
    if not clf.empty:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        xc = np.arange(len(clf))
        ax.bar(xc, clf["balanced_acc"], yerr=clf["sd"], capsize=4,
               color=["#888888", "#377eb8", "#4daf4a", "#984ea3", "#e41a1c"][:len(clf)])
        for xi, v, e in zip(xc, clf["balanced_acc"], clf["sd"]):
            ax.text(xi, v + e + 0.01, f"{v:.2f}", ha="center", fontsize=9)
        ax.axhline(1.0 / len(classes), ls="--", color="black", lw=1,
                   label=f"Chance = {1.0 / len(classes):.2f}")
        ax.set_xticks(xc)
        ax.set_xticklabels(clf["feature_set"], rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Balanced accuracy (5-fold CV)")
        ax.set_ylim(0, 1)
        ax.legend(framealpha=0.8, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        savefig(fig, OUT["controls"] / "classification_accuracy")


# WASSERSTEIN


def run_wasserstein():
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    src = vc.RES / "wasserstein" / "family_w2_6x6.csv"
    if not src.exists():
        print(f"  {src.name} not found; run the W2 job first")
        return
    m = pd.read_csv(src, index_col=0)
    fams = list(m.index)
    M = m.values.astype(float)
    np.fill_diagonal(M, 0.0)
    Z = linkage(squareform(M, checks=False), method="average")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    dendrogram(Z, labels=disp_list(fams), ax=ax, color_threshold=0,
               above_threshold_color="#555555", leaf_font_size=12)
    by_label = {disp(c): c for c in fams}
    for lbl in ax.get_xticklabels():
        c = by_label.get(lbl.get_text())
        if c:
            lbl.set_color(col(c))
    ax.set_ylabel("Mean W2 (H1) between families")
    ax.grid(False)
    fig.tight_layout()
    savefig(fig, src.parent / "family_w2_dendrogram")


# MAIN


def main():
    ap = argparse.ArgumentParser(
        description="Snake-venom analysis without disulfide bonds",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", default="alphafold", choices=vc.KEYS,
                    help="alphafold | experimental | both")
    ap.add_argument("--only", nargs="+", choices=SECTIONS + ("distributions",), default=None,
                    help="Run only the listed sections: " + ", ".join(SECTIONS))
    ap.add_argument("--variant", choices=("standard", "augmented"), default=None,
                    help="Use the disulfide-track standard / augmented H1 features; "
                         "results go to disulfide_data/results/<variant>")
    ap.add_argument("--stats-only", action="store_true",
                    help="Tables and statistics only; skip all figures")
    args = ap.parse_args()
    if args.dataset == "both":
        sys.exit(vc.run_both(__file__))
    vc.STATE["stats_only"] = args.stats_only
    run = set(args.only or SECTIONS)
    if "distributions" in run:
        run |= set(KINDS)
    if vc.VARIANT:
        run -= {"composition", "wasserstein"}

    print("=" * 65)
    print(f"SNAKE VENOM ANALYSIS  [{vc.KEY}{' / ' + vc.VARIANT if vc.VARIANT else ''}]")
    print("=" * 65)
    print(f"Sections : {', '.join(s for s in SECTIONS if s in run)}")
    print(f"Mode     : {'tables only' if args.stats_only else 'tables + figures'}")
    print(f"Start    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input    : {vc.FEATURES_CSV}")
    print(f"Output   : {vc.RES}\n")

    df, comp = vc.build_dataset()
    feats = vc.avail_features(df)
    sizes = df["Protein_Class"].value_counts()
    classes = [c for c in present(df) if sizes.get(c, 0) > vc.MIN_CLASS_SIZE]
    skipped = [c for c in present(df) if sizes.get(c, 0) <= vc.MIN_CLASS_SIZE]
    if skipped:
        print(f"Excluded from distributions/statistics (n <= {vc.MIN_CLASS_SIZE}): "
              + ", ".join(f"{c} (n={sizes.get(c, 0)})" for c in skipped))
    if not classes:
        print("No classes with n > MIN_CLASS_SIZE remain. Exiting.")
        sys.exit(1)

    pw = kw = pd.DataFrame()
    var = {}
    if run & {"statistics", "significance"}:
        print("Computing all-class statistics ...")
        sub = df[df["Protein_Class"].isin(classes)]
        pw = run_mann_whitney(sub, feats)
        kw = run_kruskal_wallis(sub, classes, feats)
        var = run_variance_tests(sub, feats, classes)

    if "composition" in run:
        print("\n[composition]")
        run_composition(df, comp)
    kinds = [k for k in KINDS if k in run]
    if kinds:
        print(f"\n[{', '.join(kinds)}]")
        run_distributions(df, classes, feats, kinds)
    if "statistics" in run:
        print("\n[statistics]")
        run_statistics(df, classes, feats, pw, kw, var)
    if "significance" in run:
        print("\n[significance]")
        run_significance(df, classes, pw, kw, var)
    if "correlations" in run:
        print("\n[correlations]")
        run_correlations(df)
    if "cycles" in run:
        print("\n[cycles]")
        run_cycles(df)
    if "controls" in run:
        print("\n[controls]")
        run_robustness(df, classes)
    if "wasserstein" in run:
        print("\n[wasserstein]")
        if vc.STATE["stats_only"]:
            print("  figures only; skipped")
        else:
            run_wasserstein()

    print(f"\n{'=' * 65}")
    print(f"Done.  Results in: {vc.RES}")
    print(f"End    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
