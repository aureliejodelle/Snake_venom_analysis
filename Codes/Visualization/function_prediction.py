"""
function_prediction.py

Target: UniProt GO "toxin activity" (binary), which varies WITHIN the mixed families
(PLA2, serine) and is therefore partly orthogonal to family and size. Secondary target:
"blood coagulation". Baselines are the competitors a reviewer will ask for: size,
sequence (amino-acid composition), size + sequence, then size + sequence + plain or
disulfide topology. Repeated stratified 5x20 CV, ROC-AUC, paired differences against
size + sequence with Nadeau-Bengio t-intervals and TOST equivalence, overall and within
the two label-balanced families.

Usage:
  python function_prediction.py alphafold
  python function_prediction.py experimental --labels labels.csv
  python function_prediction.py both

Inputs:
  results/disulfide/disulfide_features.csv   written by analysis_disulfide.py
  labels  --labels CSV (Protein_ID, toxin, coag) or SNAKE_FUNCTION_LABELS; otherwise the
          GO columns of SnakeVenom.csv are matched by sequence

results/function_prediction/
  function_prediction        toxin-activity ROC-AUC per feature set, all families
  function_features.csv      per-protein size, topology, amino-acid fractions and labels
  function_auc.csv           every target / subset / feature-set result
  function_report.txt        the same results as text
"""
from __future__ import annotations

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import viz_common as vc
import ssbond_features as ssf
from statistical_tests import repeat_ci, tost
from viz_common import long_path, savefig

warnings.filterwarnings("ignore")

OUT = vc.DSRES / "function_prediction"
AAS = "ACDEFGHIKLMNPQRSTVWY"
MIN_POSITIVES = 15
GREY, GREEN, BLUE, PURPLE, RED = "#7f7f7f", "#2ca02c", "#1f77b4", "#9467bd", "#d62728"

AA = [f"aa_{a}" for a in AAS]
SIZE = ["chain_length"]
PLAIN = ["h1med_std", "h1sum_std", "h1max_std", "h1c_std"]
DISULFIDE = ["h1med_aug", "h1sum_aug", "h1c_aug", "h1c_ss", "h1sum_ss", "n_ssbond"]
SETS = {
    "size only":                     (SIZE, GREY),
    "sequence (AA composition)":     (AA, GREEN),
    "size + sequence":               (SIZE + AA, BLUE),
    "size+seq + plain topology":     (SIZE + AA + PLAIN, PURPLE),
    "size+seq + disulfide topology": (SIZE + AA + DISULFIDE, RED),
}
BASELINE = "size + sequence"
TICKS = {"size only": "size\nonly", "sequence (AA composition)": "sequence\n(AA comp.)",
         "size + sequence": "size +\nsequence", "size+seq + plain topology": "size+seq +\nplain topology",
         "size+seq + disulfide topology": "size+seq +\ndisulfide topology"}

REPORT, RECORDS = [], []


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    REPORT.append(line)


def make_labeller(labels_csv):
    """(Protein_ID, sequence) -> (toxin, coag) or None."""
    if labels_csv:
        lab = pd.read_csv(labels_csv)
        known = {str(r.Protein_ID): (int(r.toxin), int(r.coag)) for r in lab.itertuples()}
        return lambda pid, seq: known.get(str(pid))
    cat = ssf.catalogue()
    if cat is None:
        return None
    d = pd.read_csv(cat)
    d = d[d["Type"].isin(vc.VALID)]
    seqs = [s if isinstance(s, str) else "" for s in d["Sequence"]]
    tox = d["GO_molecular_function"].str.contains("toxin activity", case=False, na=False).astype(int)
    coag = d["GO_biological_process"].str.contains("coagulation", case=False, na=False).astype(int)
    rows = list(zip(seqs, tox, coag))

    def label(pid, seq):
        for s, t, c in rows:
            if s and seq in s:
                return int(t), int(c)
        return None

    return label


def build_table(top, label_of):
    rows = []
    for _, r in top.iterrows():
        pdb = vc.find_pdb(r["Protein_Class"], r["Protein_ID"])
        if not long_path(pdb).exists():
            continue
        seq, _ = ssf.read_ca(pdb)
        lbl = label_of(r["Protein_ID"], seq)
        if lbl is None:
            continue
        comp = {f"aa_{a}": seq.count(a) / max(len(seq), 1) for a in AAS}
        rows.append(dict(r, toxin=lbl[0], coag=lbl[1], **comp))
    return pd.DataFrame(rows)


def evaluate(data, target, title, subset):
    """Repeated-CV ROC-AUC per feature set; returns {name: fold scores} or None."""
    y = data[target].values
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < MIN_POSITIVES:
        say(f"  [{title}] skipped (too few positives: {int(y.sum())}/{len(y)})")
        return None
    say(f"\n  {title}  (n={len(y)}, positives={int(y.sum())})")
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
    scores, base = {}, None
    for name, (cols, _) in SETS.items():
        cols = [c for c in cols if data[c].notna().any()]
        X = data[cols].fillna(data[cols].median()).values
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=0, n_jobs=-1)
        auc = cross_val_score(clf, X, y, cv=rcv, scoring="roc_auc")
        scores[name] = auc
        rec = dict(target=target, subset=subset, n=len(y), positives=int(y.sum()),
                   feature_set=name, auc_mean=auc.mean(), auc_sd=auc.std())
        tag = ""
        if base is not None:
            d = auc - base
            m, lo, hi = repeat_ci(d)
            eqv = [f"±{mg:.2f}" for mg in (0.01, 0.02, 0.03) if tost(d, mg)[1]]
            rec.update(delta_vs_baseline=m, delta_lo=lo, delta_hi=hi,
                       tost_equivalent=",".join(eqv))
            tag = (f"   Δvs({BASELINE})={m:+.3f} [{lo:+.3f},{hi:+.3f}]{'  *' if lo > 0 else ''}"
                   f"   TOST-equiv@{{{','.join(eqv) if eqv else 'none'}}}")
        say(f"    {name:<30} AUC={auc.mean():.3f} ± {auc.std():.3f}{tag}")
        RECORDS.append(rec)
        if name == BASELINE:
            base = auc
    return scores


def plot_auc(scores):
    names = list(SETS)
    means = [scores[n].mean() for n in names]
    sds = [scores[n].std() for n in names]
    pos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(pos, means, yerr=sds, width=0.68, color=[SETS[n][1] for n in names],
           edgecolor="white", linewidth=0.8, zorder=3, capsize=5,
           error_kw=dict(lw=1.2, ecolor="#333"))
    for x, m, s in zip(pos, means, sds):
        ax.text(x, m + s + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0.5, ls="--", color="#333", lw=1.2)
    ax.set_xticks(pos)
    ax.set_xticklabels([TICKS[n] for n in names], fontsize=9)
    ax.set_ylabel(r"Toxin-activity ROC-AUC (5$\times$20 CV)")
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()
    savefig(fig, OUT / "function_prediction")


def main():
    ap = argparse.ArgumentParser(description="Function-prediction experiment",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", default="alphafold", choices=vc.KEYS,
                    help="alphafold | experimental | both")
    ap.add_argument("--labels", default=os.environ.get("SNAKE_FUNCTION_LABELS"),
                    help="CSV with Protein_ID, toxin, coag (default: GO terms of SnakeVenom.csv)")
    ap.add_argument("--stats-only", action="store_true", help="tables only; skip the figure")
    args = ap.parse_args()
    if args.dataset == "both":
        sys.exit(vc.run_both(__file__))

    src = next((p for p in (vc.DSRES / "disulfide_features.csv", ssf.OUT_CSV)
                if p.exists()), None)
    if src is None:
        sys.exit("No disulfide feature table found. Run analysis_disulfide.py first.")
    label_of = make_labeller(args.labels)
    if label_of is None:
        sys.exit("No function labels: pass --labels or provide SnakeVenom.csv.")

    top = pd.read_csv(src)
    top = top[top["Protein_Class"].isin(vc.VALID)]
    df = build_table(top, label_of)
    if df.empty:
        sys.exit("No protein could be matched to a function label.")
    say(f"Function prediction [{vc.KEY}]: matched {len(df)} proteins with function labels")
    say(f"  toxin-activity positives: {df['toxin'].mean() * 100:.0f}%   "
        f"coagulation positives: {df['coag'].mean() * 100:.0f}%")

    say("\n" + "=" * 70)
    say("TARGET: toxin activity  (does topology add over size + sequence?)")
    say("=" * 70)
    overall = evaluate(df, "toxin", "ALL families", "all")
    for fam in ("PLA", "serine"):
        evaluate(df[df["Protein_Class"] == fam].reset_index(drop=True), "toxin",
                 f"within {fam}", fam)
    say("\n" + "=" * 70)
    say("TARGET: blood coagulation involvement")
    say("=" * 70)
    evaluate(df, "coag", "ALL families", "all")

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "function_features.csv", index=False)
    pd.DataFrame(RECORDS).to_csv(OUT / "function_auc.csv", index=False)
    with open(OUT / "function_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT))
    if overall is not None and not args.stats_only:
        plot_auc(overall)
    say(f"\nDone. Results in: {OUT}")


if __name__ == "__main__":
    main()
