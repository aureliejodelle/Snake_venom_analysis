"""
analysis_disulfide.py

Disulfide-bond analyses: augmented persistent homology,
size/count controls, bootstrap significance, standard-vs-augmented comparison
and family-level curves.

Usage:
  python analysis_disulfide.py alphafold
  python analysis_disulfide.py experimental
  python analysis_disulfide.py both
  python analysis_disulfide.py alphafold --stats-only

Inputs (paths from config.py for the chosen dataset; see ssbond_features.py):
  disulfide pairs   <SS_BONDS_DIR>/<class>/<id>.json, then SnakeVenom.csv (alphafold)
  H1 bars           Ripserer barcodes (ph_io) or <H1BARS_DIR>/{standard,augmented}/<class>/<id>.json
  paired features   <disulfide features>/{standard,augmented}/protein_features.csv

Every comparison is made on proteins carrying at least one mapped disulfide
('has_ss'); proteins without one have augmented == standard by construction.

results/disulfide/
  disulfide_features.csv     per-protein augmented-PH and lasso descriptors
  controls_report.txt        variance partition, classification, bootstrap, count baseline,
                             paired standard-vs-augmented tests
  bond_counts                A median, B mean, C total S-S bonds per family
  standard_vs_augmented      A median H1 cycle count, B median persistence, C median curvature
  disulfide_topology         A disulfide-loop count, B lasso threading, C threaded loops vs bonds
  family_curves              rows = families; birth, death and persistence curves
  family_signal              A family-unique R2 beyond size / size + count, B classification ladder
"""
from __future__ import annotations

import sys
import time
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr, wilcoxon

import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import viz_common as vc
import ssbond_features as ssf
from viz_common import disp, disp_list, present, col, savefig, long_path

warnings.filterwarnings("ignore")

OUT = vc.DSRES
GRIDN = 260
MIN_LOOP = 3
GLN_THRESH = 0.5
_LOG: list = []
PLOT = {"fam": [], "ladder": [], "chance": None}

PH_STD = ["h1med_std", "h1sum_std", "h1max_std", "h1c_std"]
PH_AUG = ["h1med_aug", "h1sum_aug", "h1max_aug", "h1c_aug", "h1c_ss", "h1sum_ss"]
LASSO = ["lasso_gln_max", "lasso_gln_sum", "lasso_gln_mean", "n_lasso", "n_loops"]


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    _LOG.append(msg)

# GEOMETRY AND TOPOLOGY HELPERS

def bar_stats(bars):
    if bars is None or len(bars) == 0:
        return 0, np.nan, 0.0, np.nan
    p = bars[:, 1] - bars[:, 0]
    p = p[p > 0]
    if not len(p):
        return 0, np.nan, 0.0, np.nan
    return len(p), float(np.median(p)), float(p.sum()), float(p.max())


def gln(A, B):
    if len(A) < 2 or len(B) < 2:
        return 0.0
    R1, R2 = A[:-1][:, None, :], A[1:][:, None, :]
    R3, R4 = B[:-1][None, :, :], B[1:][None, :, :]
    r13, r14, r23, r24 = R3 - R1, R4 - R1, R3 - R2, R4 - R2

    def unit(v):
        nn = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.where(nn < 1e-12, 1.0, nn)

    n1, n2 = unit(np.cross(r13, r14)), unit(np.cross(r14, r24))
    n3, n4 = unit(np.cross(r24, r23)), unit(np.cross(r23, r13))

    def asin_dot(a, b):
        return np.arcsin(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0))

    star = asin_dot(n1, n2) + asin_dot(n2, n3) + asin_dot(n3, n4) + asin_dot(n4, n1)
    sign = np.sign(np.sum(np.cross(R4 - R3, R2 - R1) * r13, axis=-1))
    return float(np.sum(star * sign) / (4.0 * np.pi))


def protein_lassos(X, ss):
    N, out = len(X), []
    for i, j in ss:
        a, b = (i, j) if i < j else (j, i)
        if b - a < MIN_LOOP:
            continue
        loop = np.vstack([X[a:b + 1], X[a]])
        out.append(max(abs(gln(loop, X[0:a + 1])), abs(gln(loop, X[b:N]))))
    return out


# PER-PROTEIN TABLE


def build_table(feat, ann):
    sub = feat[feat["Protein_Class"].isin(vc.VALID)].reset_index(drop=True)
    say(f"Standard / augmented H1 + lasso GLN for {len(sub)} proteins (full barcodes, no threshold)")
    rows, t0 = [], time.time()
    for n, (_, r) in enumerate(sub.iterrows(), 1):
        cls, pid = r["Protein_Class"], r["Protein_ID"]
        pdb = vc.find_pdb(cls, pid)
        if not long_path(pdb).exists():
            continue
        pseq, X = ssf.read_ca(pdb)
        if len(X) < 4:
            continue
        ss, matched = ann(cls, pid, pseq)
        sb, ab = vc.load_bars(cls, pid, "standard"), vc.load_bars(cls, pid, "augmented")
        if sb is not None and ab is not None:
            c_s, m_s, s_s, x_s = bar_stats(sb)
            c_a, m_a, s_a, x_a = bar_stats(ab)
        else:
            c_s = m_s = s_s = x_s = c_a = m_a = s_a = x_a = np.nan
        if matched:
            g = np.array(protein_lassos(X, ss))
            lasso = dict(n_loops=int(g.size), n_lasso=int((g >= GLN_THRESH).sum()) if g.size else 0,
                         lasso_gln_max=float(g.max()) if g.size else 0.0,
                         lasso_gln_sum=float(g.sum()) if g.size else 0.0,
                         lasso_gln_mean=float(g.mean()) if g.size else 0.0)
        else:
            lasso = dict(n_loops=np.nan, n_lasso=np.nan, lasso_gln_max=np.nan,
                         lasso_gln_sum=np.nan, lasso_gln_mean=np.nan)
        rows.append(dict(Protein_ID=pid, Protein_Class=cls, chain_length=len(X),
                         n_ssbond=len(ss), ss_matched=matched,
                         h1c_std=c_s, h1med_std=m_s, h1sum_std=s_s, h1max_std=x_s,
                         h1c_aug=c_a, h1med_aug=m_a, h1sum_aug=s_a, h1max_aug=x_a, **lasso))
        if n % 100 == 0:
            print(f"  {n}/{len(sub)}  ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    try:
        cols = ["h1c_std", "h1med_std", "h1sum_std", "h1max_std",
                "h1c_aug", "h1med_aug", "h1sum_aug", "h1max_aug"]
        rip = ssf.build(write=False)[["Protein_ID"] + cols]
        match = df["Protein_ID"].isin(set(rip["Protein_ID"])).mean()
        if match >= 0.9:
            df = df.drop(columns=[c for c in cols if c in df.columns]).merge(
                rip, on="Protein_ID", how="left")
            say(f"[ripserer] H1 std/aug features from the feature tables "
                f"({df['h1med_aug'].notna().sum()}/{len(df)} matched)")
    except Exception as e:
        say(f"[warn] standard/augmented feature tables not usable: {type(e).__name__}: {e}")
    if df["h1med_aug"].notna().sum() == 0:
        say("[warn] no standard/augmented H1 bars or feature tables found; "
            "augmented-PH descriptors are empty")
        say(vc.bars_report(sub.iloc[0]["Protein_Class"], sub.iloc[0]["Protein_ID"]))
    df["logsize"] = np.log(df["chain_length"])
    df["h1c_ss"] = df["h1c_aug"] - df["h1c_std"]
    df["h1sum_ss"] = df["h1sum_aug"] - df["h1sum_std"]
    df["has_ss"] = df["ss_matched"].astype(bool) & (df["n_ssbond"] > 0)
    return df


# STATISTICAL SECTIONS

def _variance_control(d, descriptors):
    say(f"  {'descriptor':<38}{'R2 size':>9}{'R2 fam|size':>13}{'KW p':>11}")
    for column, lab in descriptors:
        try:
            rs, rf, p = vc.variance_partition(d, column)
            say(f"  {lab:<38}{rs:>9.3f}{rf:>13.3f}{p:>11.1e}")
        except Exception as e:
            say(f"  {lab:<38} skipped ({str(e)[:40]})")


def _classify(d, sets):
    out = {}
    for name, cols in sets.items():
        try:
            m, s = vc.balanced_accuracy_cv(d, cols)
            out[name] = (m, s)
            say(f"  {name:<34}  {m:.3f} ± {s:.3f}")
        except Exception as e:
            say(f"  {name:<34}  skipped ({str(e)[:40]})")
    return out


def section_augmented(df):
    say("\n" + "=" * 68)
    say("1. AUGMENTED PH -- family-unique R2 (variance beyond size)")
    say("=" * 68)
    _variance_control(df, [
        ("h1med_std", "std median persistence"),
        ("h1med_aug", "augmented median persistence"),
        ("h1sum_std", "std total persistence"),
        ("h1sum_aug", "augmented total persistence"),
        ("h1sum_ss",  "disulfide-loop persistence (aug-std)"),
        ("h1c_ss",    "disulfide-loop count (aug-std)"),
        ("n_ssbond",  "raw disulfide-bond count")])
    try:
        say(f"\n  sanity: Spearman(new loops from augmentation, n_ssbond) = "
            f"{spearmanr(df['h1c_ss'], df['n_ssbond'], nan_policy='omit').statistic:+.3f}")
    except Exception:
        pass
    say("\n  Classification (5-fold balanced accuracy):")
    _classify(df, {
        "size only":                 ["chain_length"],
        "size + std topology":       ["chain_length", *PH_STD],
        "size + disulfide topology": ["chain_length", *PH_AUG, "n_ssbond"]})


def section_lasso(df):
    say("\n" + "=" * 68)
    say("2. LASSO THREADING -- family-unique R2 (variance beyond size)")
    say("=" * 68)
    lassed = df["n_lasso"]
    say(f"  proteins with >=1 threaded loop (|GLN|>={GLN_THRESH}): {(lassed > 0).mean() * 100:.0f}%  "
        f"(median n_lasso where present = {lassed[lassed > 0].median() if (lassed > 0).any() else 0:.0f})")
    _variance_control(df, [
        ("lasso_gln_max",  "max loop-tail threading (|GLN|)"),
        ("lasso_gln_sum",  "total threading (sum |GLN|)"),
        ("lasso_gln_mean", "mean threading (|GLN|)"),
        ("n_lasso",        "number of threaded loops"),
        ("n_loops",        "number of disulfide loops"),
        ("n_ssbond",       "raw disulfide-bond count")])
    d2 = df.dropna(subset=["n_lasso", "n_ssbond"])
    if len(d2) > 3:
        say(f"\n  Spearman(n_lasso, n_ssbond) = {spearmanr(d2['n_lasso'], d2['n_ssbond']).statistic:+.3f}")
    say("\n  Classification (5-fold balanced accuracy):")
    _classify(df, {
        "size only":              ["chain_length"],
        "size + disulfide count": ["chain_length", "n_ssbond", "n_loops"],
        "size + lasso threading": ["chain_length", "n_ssbond", *LASSO]})


def section_bootstrap(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
    bdf = df.dropna(subset=["h1med_std", "h1med_aug", "chain_length"]).reset_index(drop=True)
    nb = len(bdf)
    fams = [f for f in vc.VALID if (bdf["Protein_Class"] == f).any()]
    if nb < 10 or len(fams) < 2:
        say("\n3. BOOTSTRAP skipped (not enough proteins with standard and augmented H1)")
        return
    fam = bdf["Protein_Class"].values
    X_size = np.column_stack([np.ones(nb), np.log(bdf["chain_length"].values)])
    X_full = np.column_stack([X_size] + [(fam == f).astype(float) for f in fams[:-1]])

    def r2(y, X, idx):
        yy, XX = y[idx], X[idx]
        beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        res = yy - XX @ beta
        return 1 - (res @ res) / ((yy - yy.mean()) ** 2).sum()

    def fbs(y, idx):
        return r2(y, X_full, idx) - r2(y, X_size, idx)

    y_std, y_aug = bdf["h1med_std"].values, bdf["h1med_aug"].values
    by_fam = {f: np.where(fam == f)[0] for f in fams}
    rng = np.random.default_rng(0)
    B = 2000
    bs_std, bs_aug, bs_diff = np.empty(B), np.empty(B), np.empty(B)
    for b in range(B):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in by_fam.values()])
        bs_std[b], bs_aug[b] = fbs(y_std, idx), fbs(y_aug, idx)
        bs_diff[b] = bs_aug[b] - bs_std[b]
    ci = lambda x: np.percentile(x, [2.5, 97.5])
    allidx = np.arange(nb)
    say("\n" + "=" * 68)
    say(f"3. BOOTSTRAP (n={nb}): family signal beyond size (R^2), B={B}, stratified")
    say("=" * 68)
    say(f"  plain Ca median persistence   : {fbs(y_std, allidx):.3f}  "
        f"95% CI [{ci(bs_std)[0]:.3f}, {ci(bs_std)[1]:.3f}]")
    say(f"  disulfide-augmented           : {fbs(y_aug, allidx):.3f}  "
        f"95% CI [{ci(bs_aug)[0]:.3f}, {ci(bs_aug)[1]:.3f}]")
    d0 = fbs(y_aug, allidx) - fbs(y_std, allidx)
    lo, hi = ci(bs_diff)
    p_dir = 2 * min((bs_diff <= 0).mean(), (bs_diff >= 0).mean())
    say(f"  DIFFERENCE (aug - plain)      : {d0:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
        f"   (2-sided p ~ {p_dir:.4f})")
    say(f"  --> improvement {'IS' if lo > 0 else 'is NOT'} significant "
        f"(CI {'excludes' if lo > 0 else 'includes'} 0)")

    say("\n  Repeated CV (5x20), paired plain vs disulfide topology:")
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
    ycls = bdf["Protein_Class"].values

    def scores(cols):
        X = bdf[cols].fillna(bdf[cols].median()).values
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=0, n_jobs=-1)
        return cross_val_score(clf, X, ycls, cv=rcv, scoring="balanced_accuracy")

    sp = scores(["chain_length", *PH_STD])
    sd = scores(["chain_length", *PH_AUG, "n_ssbond"])
    m, lo, hi = vc.st.repeat_ci(sd - sp)
    say(f"    size+plain topology     : {sp.mean():.3f} ± {sp.std():.3f}")
    say(f"    size+disulfide topology : {sd.mean():.3f} ± {sd.std():.3f}")
    say(f"    paired diff (diss-plain): {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    say(f"    --> classification improvement {'IS' if lo > 0 else 'is NOT'} reliably > 0")


def section_count_baseline(df):
    say(f"\n{'=' * 74}\nCOUNT BASELINE -- {len(df)} proteins with a mapped disulfide\n{'=' * 74}")
    say(f"  {'descriptor':<38}{'fam|size':>10}{'fam|size+cnt':>14}{'KW p|s+c':>11}")
    PLOT["fam"].clear()
    for column, lab in [("h1med_std", "plain median persistence"),
                        ("h1med_aug", "augmented median persistence"),
                        ("h1sum_aug", "augmented total persistence"),
                        ("h1c_aug", "augmented loop count"),
                        ("h1sum_ss", "disulfide-loop persistence"),
                        ("lasso_gln_sum", "lasso total threading"),
                        ("n_lasso", "lasso threaded-loop count")]:
        try:
            _, r_s, _ = vc.variance_partition(df, column)
            _, r_sc, p = vc.variance_partition(df, column, extra=("n_ssbond",))
            say(f"  {lab:<38}{r_s:>10.3f}{r_sc:>14.3f}{p:>11.1e}")
            PLOT["fam"].append((lab, r_s, r_sc))
        except Exception as e:
            say(f"  {lab:<38} skipped ({str(e)[:40]})")
    say("\n  Nested CV ladder (5-fold balanced accuracy)")
    res = _classify(df, {
        "size only":                      ["chain_length"],
        "size + count":                   ["chain_length", "n_ssbond"],
        "size + plain-PH":                ["chain_length", *PH_STD],
        "size + count + augmented-PH":    ["chain_length", "n_ssbond", *PH_AUG],
        "size + count + lasso":           ["chain_length", "n_ssbond", *LASSO],
        "size + count + PH + lasso":      ["chain_length", "n_ssbond", *PH_AUG, *LASSO],
        "size + augmented-PH (no count)": ["chain_length", *PH_AUG]})
    PLOT["ladder"] = [(k, v[0], v[1]) for k, v in res.items()]
    PLOT["chance"] = 1.0 / max(df["Protein_Class"].nunique(), 1)
    if "size + count" in res:
        base = res["size + count"][0]
        say("\n  Decisive gaps")
        for k, lab in (("size + count + augmented-PH", "augmented-PH beyond count"),
                       ("size + count + lasso", "lasso beyond count")):
            if k in res:
                say(f"  {lab:<26}: {res[k][0] - base:+.3f}")


def paired_features(df):
    s_f, a_f = ssf.feature_table("standard"), ssf.feature_table("augmented")
    if not (s_f.exists() and a_f.exists()):
        say(f"\n[standard vs augmented] feature tables not found ({s_f}, {a_f}); skipped")
        return None
    m = pd.read_csv(s_f).merge(pd.read_csv(a_f), on="Protein_ID", suffixes=("_std", "_aug"))
    m["Protein_Class"] = m["Protein_Class_std"]
    has = {ssf.track_id(p) for p in df.loc[df["has_ss"], "Protein_ID"]}
    m = m[m["Protein_Class"].isin(vc.VALID) & m["Protein_ID"].isin(has)].reset_index(drop=True)
    if m.empty:
        say("\n[standard vs augmented] no proteins with a mapped disulfide; skipped")
        return None
    say("\n" + "=" * 68)
    say(f"STANDARD vs AUGMENTED (paired, {len(m)} proteins with >=1 disulfide)")
    say("=" * 68)
    for column, lab in COMPARE:
        if f"{column}_std" not in m.columns:
            continue
        d = m[[f"{column}_std", f"{column}_aug"]].dropna()
        delta = d[f"{column}_aug"].values - d[f"{column}_std"].values
        try:
            p = wilcoxon(d[f"{column}_aug"], d[f"{column}_std"]).pvalue if np.any(delta != 0) else 1.0
        except ValueError:
            p = 1.0
        say(f"  {lab:<24} mean delta {delta.mean():+.4g}   Wilcoxon p = {p:.2e}   (n={len(d)})")
    return m


# FIGURES

def _box_by_class(ax, d, column, ylabel):
    d = d.dropna(subset=[column])
    cls_o = present(d) if len(d) else []
    if not cls_o:
        ax.set_visible(False)
        return
    data = [d[d["Protein_Class"] == c][column].values for c in cls_o]
    bp = ax.boxplot(data, positions=range(len(cls_o)), widths=0.55, patch_artist=True,
                    showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.4,
                                                     markerfacecolor="gray", markeredgewidth=0),
                    medianprops=dict(color="black", lw=1.8))
    for b, c in zip(bp["boxes"], cls_o):
        b.set_facecolor(col(c))
        b.set_alpha(0.8)
    ax.set_xticks(range(len(cls_o)))
    ax.set_xticklabels(disp_list(cls_o), rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", alpha=0.2)


def plot_bond_counts(df):
    d = df[df["ss_matched"].astype(bool)]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax, agg, lab, fmt in ((axes[0], np.median, "Median S-S bond count", "{:.0f}"),
                              (axes[1], np.mean, "Mean S-S bond count", "{:.1f}"),
                              (axes[2], np.sum, "Total S-S bonds", "{:.0f}")):
        vc.bar_by_class(ax, d, "n_ssbond", agg, lab, fmt, True)
    vc.add_panel_letters(axes)
    fig.tight_layout()
    savefig(fig, OUT / "bond_counts")


def plot_topology(df):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    _box_by_class(axes[0], df, "h1c_ss", "Disulfide-loop count (augmented − standard)")
    _box_by_class(axes[1], df, "lasso_gln_max", "Maximum loop-tail threading (|GLN|)")
    m = df.dropna(subset=["n_ssbond", "n_lasso"])
    if len(m):
        rng = np.random.default_rng(0)
        for c in present(m):
            s = m[m["Protein_Class"] == c]
            axes[2].scatter(s["n_ssbond"] + rng.uniform(-0.12, 0.12, len(s)),
                            s["n_lasso"] + rng.uniform(-0.12, 0.12, len(s)),
                            color=col(c), s=22, alpha=0.6, linewidths=0, label=disp(c))
        axes[2].set_xlabel("Disulfide bonds", fontsize=10)
        axes[2].set_ylabel("Threaded loops", fontsize=10)
        axes[2].legend(fontsize=8, framealpha=0.9)
        axes[2].grid(alpha=0.18)
    else:
        axes[2].set_visible(False)
    vc.add_panel_letters(axes)
    fig.tight_layout()
    savefig(fig, OUT / "disulfide_topology")


COMPARE = (("H1_count", "H1 cycle count"),
           ("H1_median", "H1 median persistence"),
           ("H1_mean", "H1 mean persistence"),
           ("Curv_median", "Median Forman-Ricci curvature"),
           ("Curv_mean", "Mean Forman-Ricci curvature"))


def plot_standard_vs_augmented(m):
    """One standard-vs-augmented boxplot figure per descriptor, a box pair per family."""
    if m is None:
        return
    fams = present(m)
    x = np.arange(len(fams))
    for column, ylabel in COMPARE:
        if f"{column}_std" not in m.columns or f"{column}_aug" not in m.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5.2))
        for off, suffix, color in ((-0.19, "std", vc.COL_STD), (0.19, "aug", vc.COL_AUG)):
            data = [m.loc[m["Protein_Class"] == f, f"{column}_{suffix}"].dropna().values for f in fams]
            bp = ax.boxplot(data, positions=x + off, widths=0.34, patch_artist=True,
                            medianprops=dict(color="black", linewidth=1.4),
                            flierprops=dict(marker="o", markersize=2.2, alpha=0.3, markeredgewidth=0))
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.65)
        ns = [int((m["Protein_Class"] == f).sum()) for f in fams]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(disp_list(fams), ns)], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(-0.6, len(fams) - 0.4)
        ax.legend(handles=[Patch(facecolor=vc.COL_STD, alpha=0.65, label="Standard"),
                           Patch(facecolor=vc.COL_AUG, alpha=0.65, label="Augmented")],
                  frameon=False)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        savefig(fig, OUT / "standard_vs_augmented" / column)


def _frac_curve(vals, grid):
    if len(vals) == 0:
        return np.full(len(grid), np.nan)
    v = np.sort(vals)
    return np.searchsorted(v, grid, side="right") / len(v)


def _surv_curve(vals, grid):
    if len(vals) == 0:
        return np.full(len(grid), np.nan)
    v = np.sort(vals)
    return 1.0 - np.searchsorted(v, grid, side="left") / len(v)


def _band(ax, grid, curves, color, label):
    if not curves:
        return
    M = np.vstack(curves)
    ax.fill_between(grid, np.nanpercentile(M, 25, axis=0), np.nanpercentile(M, 75, axis=0),
                    color=color, alpha=0.18, lw=0)
    ax.plot(grid, np.nanmedian(M, axis=0), color=color, lw=2.0, label=label)


def plot_family_curves(feat, df):
    has = set(df.loc[df["has_ss"], "Protein_ID"])
    sub = feat[feat["Protein_Class"].isin(vc.VALID) & feat["Protein_ID"].isin(has)]
    fams = present(sub)
    if not fams:
        return
    top = 0.0
    for cls in fams:
        for pid in sub.loc[sub["Protein_Class"] == cls, "Protein_ID"]:
            for var in ("standard", "augmented"):
                b = vc.load_bars(cls, pid, var)
                if b is not None and len(b):
                    top = max(top, float(b[:, 1].max()))
    xs = np.linspace(0, top if top > 0 else 1.0, GRIDN)
    fig, axes = plt.subplots(len(fams), 3, figsize=(15, 4.2 * len(fams)))
    axes = np.atleast_2d(axes)
    for r, cls in enumerate(fams):
        B, D, Praw = ({"std": [], "aug": []} for _ in range(3))
        pmax = 0.0
        for pid in sub.loc[sub["Protein_Class"] == cls, "Protein_ID"]:
            for var, key in (("standard", "std"), ("augmented", "aug")):
                bars = vc.load_bars(cls, pid, var)
                if bars is None or len(bars) == 0:
                    continue
                B[key].append(_frac_curve(bars[:, 0], xs))
                D[key].append(_frac_curve(bars[:, 1], xs))
                p = bars[:, 1] - bars[:, 0]
                Praw[key].append(p)
                pmax = max(pmax, float(p.max()))
        if not (B["std"] or B["aug"]):
            for ax in axes[r]:
                ax.set_visible(False)
            continue
        xp = np.linspace(0, pmax * 1.02 if pmax > 0 else 1.0, GRIDN)
        P = {k: [_surv_curve(p, xp) for p in Praw[k]] for k in Praw}
        for ax, (grid, curves_s, curves_a, xl) in zip(axes[r], (
                (xs, B["std"], B["aug"], "Filtration scale (Angstrom)"),
                (xs, D["std"], D["aug"], "Filtration scale (Angstrom)"),
                (xp, P["std"], P["aug"], "Persistence (Angstrom)"))):
            _band(ax, grid, curves_s, vc.COL_STD, "Standard (median)")
            _band(ax, grid, curves_a, vc.COL_AUG, "Augmented (median)")
            ax.set_xlim(grid[0], grid[-1])
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel(xl, fontsize=10)
            ax.grid(alpha=0.18)
        axes[r][0].set_ylabel(f"{disp(cls)}\nFraction of loops (cumulative)", fontsize=10)
        axes[r][2].set_ylabel("Fraction of loops (survival)", fontsize=10)
        axes[r][0].text(0.97, 0.05, f"n={len(B['std'])}", transform=axes[r][0].transAxes,
                        ha="right", va="bottom", fontsize=9)
        if r == 0:
            axes[0][0].legend(frameon=False, loc="lower right", fontsize=8)
    vc.add_panel_letters(axes)
    fig.tight_layout()
    savefig(fig, OUT / "family_curves")


def plot_family_signal():
    fam, lad = PLOT["fam"], PLOT["ladder"]
    if not fam and not lad:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    if fam:
        y = np.arange(len(fam))
        axes[0].barh(y - 0.2, [f[1] for f in fam], height=0.38, color="#2980B9", label="Beyond size")
        axes[0].barh(y + 0.2, [f[2] for f in fam], height=0.38, color="#E67E22",
                     label="Beyond size + bond count")
        axes[0].set_yticks(y)
        axes[0].set_yticklabels([f[0] for f in fam], fontsize=9)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Family-unique $R^2$", fontsize=10)
        axes[0].legend(fontsize=9, framealpha=0.9)
        axes[0].grid(axis="x", alpha=0.2)
    else:
        axes[0].set_visible(False)
    if lad:
        y = np.arange(len(lad))
        axes[1].barh(y, [l[1] for l in lad], xerr=[l[2] for l in lad], height=0.6,
                     color="#7F8C8D", ecolor="black", capsize=3)
        if PLOT["chance"]:
            axes[1].axvline(PLOT["chance"], ls="--", color="#333", lw=1.2,
                            label=f"chance = {PLOT['chance']:.2f}")
            axes[1].legend(fontsize=9, framealpha=0.9, loc="lower right")
        axes[1].set_yticks(y)
        axes[1].set_yticklabels([l[0] for l in lad], fontsize=9)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Balanced accuracy (5-fold)", fontsize=10)
        axes[1].grid(axis="x", alpha=0.2)
    else:
        axes[1].set_visible(False)
    vc.add_panel_letters(axes)
    fig.tight_layout()
    savefig(fig, OUT / "family_signal")


# MAIN

def main():
    ap = argparse.ArgumentParser(
        description="Disulfide-bond analyses and figures",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", default="alphafold", choices=vc.KEYS,
                    help="alphafold | experimental | both")
    ap.add_argument("--stats-only", action="store_true",
                    help="Tables and report only; skip figures")
    args = ap.parse_args()
    if args.dataset == "both":
        sys.exit(vc.run_both(__file__))
    vc.STATE["stats_only"] = args.stats_only

    print("=" * 65)
    print(f"DISULFIDE ANALYSIS  [{vc.KEY}]")
    print("=" * 65)
    print(f"Start    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Features : {vc.FEATURES_CSV}")
    print(f"Output   : {OUT}\n")

    ann = ssf.make_annotator(vc.SS_JSON, vc.SS_CSV)
    if ann is None:
        print(f"No disulfide annotation found (looked in {vc.SS_JSON} and SnakeVenom.csv). Exiting.")
        sys.exit(1)

    feat = vc.load_features()
    df = build_table(feat, ann)
    if df.empty:
        print("No proteins with PDB files and annotation. Exiting.")
        sys.exit(1)
    sub = df[df["has_ss"]].reset_index(drop=True)
    if sub.empty:
        print("No protein carries a mapped disulfide. Exiting.")
        sys.exit(1)

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "disulfide_features.csv", index=False)
    say(f"Saved {len(df)} rows; {len(sub)} carry >=1 mapped disulfide")

    section_augmented(sub)
    section_lasso(sub)
    section_bootstrap(sub)
    section_count_baseline(sub)
    paired = paired_features(df)
    (OUT / "controls_report.txt").write_text("\n".join(_LOG), encoding="utf-8")
    print(f"\n  report: {OUT / 'controls_report.txt'}")

    if not args.stats_only:
        plot_bond_counts(df)
        plot_topology(sub)
        plot_standard_vs_augmented(paired)
        plot_family_curves(feat, df)
        plot_family_signal()

    print(f"\n{'=' * 65}")
    print(f"Done.  Results in: {OUT}")
    print(f"End    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
