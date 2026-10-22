"""
statistical_tests.py

Statistical library shared by the analysis scripts.

  Location  Mann-Whitney U (pairwise, Cliff's delta) and Kruskal-Wallis
  Spread    Levene (median-centred), Bartlett, Fligner-Killeen
  Repeated-CV comparisons  repeat_ci (Nadeau-Bengio t-interval) and tost (equivalence)

Every family of p-values is corrected with FDR-BH at alpha = 0.05.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal, levene, bartlett, fligner, t as t_dist
from statsmodels.stats.multitest import multipletests

FEATURE_DISPLAY = {
    "H1_count":         "H1 Bar Count",
    "H1_mean":          "Mean Persistence",
    "H1_median":        "Median Persistence",
    "H1_max":           "Max Persistence",
    "H1_sum":           "Sum Persistence",
    "H1_std":           "Std Persistence",
    "Curv_mean":        "Mean Curvature",
    "Curv_median":      "Median Curvature",
    "Curv_max":         "Max Curvature",
    "Curv_min":         "Min Curvature",
    "Curv_std":         "Std Curvature",
    "Curv_sum":         "Sum Curvature",
    "Curv_n_positive":  "N Positive Curvature",
    "Curv_n_negative":  "N Negative Curvature",
    "HG_n_nodes":       "HG Nodes",
    "HG_n_hyperedges":  "HG Hyperedges",
    "HG_avg_edge_size": "HG Avg Edge Size",
    "HG_max_edge_size": "HG Max Edge Size",
    "HG_connectivity":  "HG Connectivity",
    "HG_density":       "HG Density",
}


def stars(p: float) -> str:
    if p < 0.0001: return "****"
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "ns"


def fdr_bh(pvals, alpha: float = 0.05):
    """Benjamini-Hochberg: returns (reject, adjusted p)."""
    rej, padj, _, _ = multipletests(np.asarray(pvals, float), alpha=alpha, method="fdr_bh")
    return rej, padj


def _finish(df: pd.DataFrame, stat_col: str | None = None) -> pd.DataFrame:
    if stat_col:
        df = df.rename(columns={"stat": stat_col})
    df["p_fdr"] = fdr_bh(df["p_value"].values)[1]
    df["significant"] = df["p_fdr"] < 0.05
    df["neg_log10_p"] = -np.log10(df["p_value"].clip(1e-300))
    df["neg_log10_p_fdr"] = -np.log10(df["p_fdr"].clip(1e-300))
    df["sig_stars"] = df["p_fdr"].apply(stars)
    return df



# LOCATION TESTS


def run_mann_whitney(df_sub: pd.DataFrame, feats: list) -> pd.DataFrame:
    """All pairwise Mann-Whitney U tests; effect size Cliff's delta."""
    groups = df_sub["Protein_Class"].unique().tolist()
    records = []
    for feat in feats:
        for g1, g2 in combinations(groups, 2):
            d1 = df_sub[df_sub["Protein_Class"] == g1][feat].dropna().values
            d2 = df_sub[df_sub["Protein_Class"] == g2][feat].dropna().values
            if len(d1) < 3 or len(d2) < 3:
                continue
            try:
                u, p = mannwhitneyu(d1, d2, alternative="two-sided")
            except Exception:
                continue
            n1, n2 = len(d1), len(d2)
            med2 = float(np.median(d2))
            fc = float(np.median(d1) / med2) if med2 != 0 else np.nan
            records.append(dict(
                feature=feat, display=FEATURE_DISPLAY.get(feat, feat),
                group1=g1, group2=g2, n1=n1, n2=n2,
                median1=float(np.median(d1)), median2=med2,
                mean1=float(d1.mean()), mean2=float(d2.mean()),
                std1=float(d1.std()), std2=float(d2.std()),
                cliffs_delta=float(1 - 2 * u / (n1 * n2)),
                fold_change=fc, log2fc=float(np.log2(fc)) if (fc and fc > 0) else np.nan,
                u_stat=float(u), p_value=float(p)))
    return _finish(pd.DataFrame(records)) if records else pd.DataFrame()


def run_kruskal_wallis(df_sub: pd.DataFrame, classes: list, feats: list) -> pd.DataFrame:
    """Kruskal-Wallis H-test across all classes, one row per feature."""
    records = []
    for feat in feats:
        groups = [df_sub[df_sub["Protein_Class"] == c][feat].dropna().values for c in classes]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 2:
            continue
        try:
            h, p = kruskal(*groups)
        except Exception:
            continue
        records.append(dict(feature=feat, display=FEATURE_DISPLAY.get(feat, feat),
                            H_stat=float(h), p_value=float(p), n_classes=len(groups)))
    if not records:
        return pd.DataFrame()
    return _finish(pd.DataFrame(records)).sort_values("p_value")



# SPREAD TESTS


_VARIANCE = {
    "levene":   (lambda *g: levene(*g, center="median"), "levene_stat"),
    "bartlett": (bartlett, "bartlett_stat"),
    "fligner":  (fligner, "fligner_stat"),
}


def _variance_test(df_sub, feats, groups, name):
    fn, stat_col = _VARIANCE[name]
    records = []
    for feat in feats:
        data = [df_sub[df_sub["Protein_Class"] == g][feat].dropna().values for g in groups]
        data = [d for d in data if len(d) >= 3]
        if len(data) < 2:
            continue
        try:
            stat, p = fn(*data)
        except Exception:
            continue
        records.append(dict(feature=feat, display=FEATURE_DISPLAY.get(feat, feat),
                            stat=float(stat), p_value=float(p), n_groups=len(data),
                            groups=", ".join(groups)))
    if not records:
        return pd.DataFrame()
    return _finish(pd.DataFrame(records), stat_col).sort_values("p_value")


def run_variance_tests(df_sub: pd.DataFrame, feats: list, groups: list) -> dict:
    """Levene, Bartlett and Fligner-Killeen per feature, plus a merged summary.

    Returns {"levene": df, "bartlett": df, "fligner": df, "combined": df}.
    """
    out = {name: _variance_test(df_sub, feats, groups, name) for name in _VARIANCE}
    rows = []
    for feat in feats:
        row = {"feature": feat, "display": FEATURE_DISPLAY.get(feat, feat)}
        for name, (_, stat_col) in _VARIANCE.items():
            d = out[name]
            if d.empty or feat not in d["feature"].values:
                row.update({f"{name}_stat": np.nan, f"{name}_p": np.nan,
                            f"{name}_p_fdr": np.nan, f"{name}_sig": False})
            else:
                r = d[d["feature"] == feat].iloc[0]
                row.update({f"{name}_stat": r[stat_col], f"{name}_p": r["p_value"],
                            f"{name}_p_fdr": r["p_fdr"], f"{name}_sig": bool(r["significant"])})
        row["n_tests_significant"] = sum(row[f"{n}_sig"] for n in _VARIANCE)
        rows.append(row)
    comb = pd.DataFrame(rows)
    out["combined"] = comb.sort_values("n_tests_significant", ascending=False) if not comb.empty else comb
    return out


def variance_consensus(var: dict) -> list:
    """Features flagged by all three variance tests."""
    sets = [set(var[n][var[n]["significant"]]["feature"]) for n in _VARIANCE
            if n in var and not var[n].empty]
    return sorted(set.intersection(*sets)) if sets else []


def _repeat_means(fold_scores, n_splits, n_repeats):
    return np.asarray(fold_scores).reshape(n_repeats, n_splits).mean(axis=1)


def repeat_ci(fold_scores, n_splits=5, n_repeats=20):
    """(mean, lo, hi): 95% t-interval over the repeat-level means of a repeated-CV score
    or paired difference. Folds within a repeat share training data, so they are averaged
    first."""
    reps = _repeat_means(fold_scores, n_splits, n_repeats)
    m = reps.mean()
    h = t_dist.ppf(0.975, len(reps) - 1) * reps.std(ddof=1) / np.sqrt(len(reps))
    return m, m - h, m + h


def tost(fold_scores, margin, alpha=0.05, n_splits=5, n_repeats=20):
    """Two one-sided tests that a repeated-CV paired difference lies within . Returns (p, equivalent)."""
    reps = _repeat_means(fold_scores, n_splits, n_repeats)
    n, m = len(reps), reps.mean()
    se = reps.std(ddof=1) / np.sqrt(n)
    p = max(t_dist.sf((m + margin) / se, n - 1), t_dist.cdf((m - margin) / se, n - 1))
    return p, p < alpha
