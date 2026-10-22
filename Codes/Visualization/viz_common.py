"""
viz_common.py

Shared setup for the snake-venom visualisation scripts.

Dataset keys (first positional argument of every script):
  alphafold     AlphaFold models           -> SNAKE_DATASET=alphafold
  experimental  experimental PDB entries   -> SNAKE_DATASET=pdb
  both          run alphafold, then experimental

"""
from __future__ import annotations

import io
import os
import sys
import logging
import subprocess
from pathlib import Path

KEYS = ("alphafold", "experimental", "both")
_ALIAS = {"alphafold": "alphafold", "af": "alphafold", "predicted": "alphafold",
          "colabfold": "alphafold", "experimental": "experimental", "exp": "experimental",
          "pdb": "experimental"}
_CONFIG_NAME = {"alphafold": "alphafold", "experimental": "pdb"}


def _key():
    for a in sys.argv[1:]:
        if a in KEYS:
            return a
    return _ALIAS.get(os.environ.get("SNAKE_DATASET", "").strip().lower(), "alphafold")


def _variant():
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a.startswith("--variant="):
            return a.split("=", 1)[1]
        if a == "--variant" and i + 1 < len(argv):
            return argv[i + 1]
    return os.environ.get("SNAKE_VARIANT", "").strip().lower()


VARIANT = _variant() if _variant() in ("standard", "augmented") else ""
KEY = _key()
os.environ["SNAKE_DATASET"] = _CONFIG_NAME.get(KEY, "alphafold")

HERE = Path(__file__).resolve().parent
CODES = HERE.parent
ROOT = CODES.parent
for _p in (str(CODES), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

from config import config, long_path
from plot_style import use_computer_modern, disp, disp_list, class_color
import ssbond_features as ssf
import statistical_tests as st

Image.MAX_IMAGE_PIXELS = None
logging.getLogger("fontTools").setLevel(logging.ERROR)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

use_computer_modern()
mpl.rcParams.update({
    "axes.grid":      True,
    "axes.grid.axis": "y",
    "grid.alpha":     0.18,
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.8,
})

DPI = 600
FORMATS = ("pdf", "tiff", "png")
MIN_CLASS_SIZE = 10
N_MIN = 15
HATCH = "///"
THRESH = None     
STATE = {"stats_only": False}

VALID = ssf.VALID
COL_STD, COL_AUG = "#52636E", "#D1495B"

DSRES = Path(config.DS_RESULTS_DIR)
RES = DSRES / VARIANT if VARIANT else Path(config.RESULTS_DIR)
HIGH = RES if VARIANT else Path(config.HIGH_IMPACT_DIR)
FEATURES_CSV = Path(config.VIZ_FEATURES_DIR) / "protein_features.csv"
PDB_DIR = Path(config.PDB_DATA_DIR)
HG = Path(config.HG_DIR)
RC = Path(config.RC_DIR)
SS_JSON = Path(config.SS_BONDS_DIR)
BARS_CACHE = Path(config.H1BARS_DIR)


SS_CSV = None if KEY == "experimental" else ssf.catalogue()

BFAC_LABEL = config.BFACTOR_LABEL
BFAC_STEM = "".join(c for c in BFAC_LABEL.lower() if c.isalnum()) or "plddt"
BFAC_MEAN, BFAC_MED = f"{BFAC_STEM}_mean", f"{BFAC_STEM}_median"

FEATURE_COLS = [
    "H1_count", "H1_mean", "H1_median",
    "Curv_median", "Curv_mean",
    "chain_length", "cys_fraction", "pI", BFAC_MEAN, BFAC_MED,
]

FEATURE_DISPLAY = st.FEATURE_DISPLAY
FEATURE_DISPLAY.update({
    "curv_norm":    "Curvature / Chain Length",
    "chain_length": "Chain Length",
    "cys_fraction": "Cysteine Fraction",
    "pI":           "Isoelectric Point (pI)",
    BFAC_MEAN:      f"Mean {BFAC_LABEL}",
    BFAC_MED:       f"Median {BFAC_LABEL}",
})

AA3TO1 = ssf.THREE2ONE
AA_LETTERS = list("ACDEFGHIKLMNPQRSTVWY")
PKA_NTERM, PKA_CTERM = 8.6, 3.65
PKA_SIDE = {"D": 3.90, "E": 4.07, "C": 8.18, "Y": 10.46, "H": 6.04, "K": 10.54, "R": 12.48}
BASE_RES = {"H", "K", "R"}

stars = st.stars
fdr_bh = st.fdr_bh


def feat_label(f):
    return FEATURE_DISPLAY.get(f, f)


def run_both(script):
    rest = [a for a in sys.argv[1:] if a not in KEYS]
    rc = 0
    for k in ("alphafold", "experimental"):
        print(f"\n{'#' * 65}\n# DATASET: {k}\n{'#' * 65}", flush=True)
        r = subprocess.run([sys.executable, str(script), k, *rest])
        rc = rc or r.returncode
    return rc


# CLASSES, COLOURS, FIGURE HELPERS


def present(d):
    have = set(d["Protein_Class"].unique())
    return [c for c in VALID if c in have] + sorted(c for c in have if c not in VALID)


def palette(classes):
    return [class_color(c, i) for i, c in enumerate(classes)]


def col(c):
    return class_color(c, VALID.index(c) if c in VALID else len(VALID))


def grid(n, ncols=4, w=4.4, h=4.0):
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * w, nrows * h))
    return fig, np.atleast_1d(axes).ravel()


def add_panel_letters(axes, fontsize=14):
    visible = [a for a in np.array(axes).flatten() if a.get_visible()]
    for i, ax in enumerate(visible):
        ax.text(-0.16, 1.08, chr(65 + i), transform=ax.transAxes, fontsize=fontsize,
                fontweight="bold", ha="left", va="bottom")


def savefig(fig, path, formats=None, dpi=None):
    
    if STATE["stats_only"]:
        plt.close(fig)
        return
    formats = formats or FORMATS
    dpi = dpi or DPI
    path = Path(path)
    base = path.with_suffix("") if path.suffix.lower() in {".pdf", ".png", ".tif", ".tiff"} else path
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        if fmt in ("pdf", "eps", "svg"):
            fig.savefig(long_path(base.parent / f"{base.name}.{fmt}"), format=fmt,
                        bbox_inches="tight", facecolor="white")
        elif fmt not in ("tif", "tiff", "png"):
            raise ValueError(f"Unsupported format: {fmt}")
    raster = [f for f in formats if f in ("tif", "tiff", "png")]
    if raster:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        for fmt in raster:
            target = long_path(base.parent / f"{base.name}.{fmt}")
            if fmt == "png":
                img.save(target, format="PNG", dpi=(dpi, dpi))
            else:
                img.save(target, format="TIFF", dpi=(dpi, dpi), compression="tiff_lzw")
        img.close()
    plt.close(fig)
    try:
        shown = base.relative_to(RES)
    except ValueError:
        shown = base.name
    print(f"  saved {shown}")


def bar_by_class(ax, d, column, agg, ylabel, fmt, mark_small):
    """One bar per class (value label, n inside); classes below N_MIN are hatched."""
    cls_o = present(d.dropna(subset=[column]))
    if not cls_o:
        ax.set_visible(False)
        return
    centers, ns = [], []
    for c in cls_o:
        v = d.loc[d["Protein_Class"] == c, column].dropna().values
        centers.append(float(agg(v)))
        ns.append(len(v))
    x = np.arange(len(cls_o))
    ymax = max(centers) or 1.0
    bars = ax.bar(x, centers, width=0.68, color=[col(c) for c in cls_o],
                  edgecolor="white", linewidth=0.8, zorder=3)
    for b, n in zip(bars, ns):
        if mark_small and n < N_MIN:
            b.set_hatch(HATCH)
            b.set_alpha(0.55)
            b.set_edgecolor("black")
    for xi, cv, n in zip(x, centers, ns):
        ax.text(float(xi), cv + ymax * 0.015, fmt.format(cv), ha="center", va="bottom", fontsize=9)
        ax.text(float(xi), ymax * 0.02, f"n={n}", ha="center", va="bottom", fontsize=8,
                color="black" if (mark_small and n < N_MIN) else "white",
                fontweight="bold", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(disp_list(cls_o), rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax * 1.12)
    ax.grid(axis="y", alpha=0.25, zorder=0)


# DATA

def load_features():
    if not FEATURES_CSV.exists():
        print(f"Features CSV not found: {FEATURES_CSV}")
        print("Run extract_features.py first.")
        sys.exit(1)
    feat = pd.read_csv(FEATURES_CSV)
    other = feat.loc[~feat["Protein_Class"].isin(VALID), "Protein_Class"].value_counts()
    if len(other):
        print("  not analysed (outside the six venom families): "
              + ", ".join(f"{c} (n={n})" for c, n in other.items()))
        feat = feat[feat["Protein_Class"].isin(VALID)].reset_index(drop=True)
    if not VARIANT:
        return feat
    table = ssf.feature_table(VARIANT)
    if not table.exists():
        print(f"No {VARIANT} feature table found: {table}")
        sys.exit(1)
    tab = pd.read_csv(table).rename(columns={"Protein_ID": "uid"})
    tab = tab.drop(columns=[c for c in ("Protein_Class",) if c in tab.columns])
    feat = feat.assign(uid=feat["Protein_ID"].map(ssf.track_id))
    feat = feat.drop(columns=[c for c in tab.columns if c != "uid" and c in feat.columns])
    feat = feat.merge(tab, on="uid", how="inner").drop(columns="uid")
    print(f"  {VARIANT} features: {len(feat)} proteins matched ({table})")
    return feat


PH_NOTE = {}


BAR_SUBDIRS = ("barcodes", "PH_1")
BAR_EXTS = (".json", ".csv", ".txt", ".tsv", ".npy")


def bar_roots(variant):
    """Folders holding <class>/{barcodes,PH_1} for one variant, in order of preference."""
    roots = [ssf.DD / "processed_data" / "persistent_homology" / variant,
             ssf.DS / "ph" / variant,
             ssf.DS / "processed_data" / "persistent_homology" / variant,
             BARS_CACHE / variant]
    if variant == "standard":
        roots.append(Path(config.PH_DIR))
    if KEY != "experimental":
        roots.append(ssf.DD.parent.parent / "h1_bars_ca" / variant)
    return roots


def bar_dirs(cls, variant):
    """Folders that may hold the H1 barcodes of one family, in order of preference."""
    dirs = []
    for root in bar_roots(variant):
        dirs.append(root / cls)
        dirs += [root / cls / sub for sub in BAR_SUBDIRS]
    return dirs


def _bars_from_file(f, thresh):
    """H1 bars (N * 2) from one barcode file; None if the file cannot be read."""
    import json
    f = long_path(f)
    try:
        if f.suffix == ".json":
            with open(f) as fh:
                d = json.load(fh)
            raw = d.get("dim_1", d.get("H1", d.get("h1", []))) if isinstance(d, dict) else d
            rows = []
            for b in raw:
                if isinstance(b, dict):
                    rows.append((b.get("birth"), b.get("death")))
                else:
                    rows.append((b[0], b[1]))
        elif f.suffix == ".npy":
            rows = np.load(f, allow_pickle=True).reshape(-1, 2).tolist()
        else:
            tab = pd.read_csv(f, sep=None, engine="python", header=None)
            try:
                tab.iloc[0].astype(float)
            except (TypeError, ValueError):
                head = [str(c).strip() for c in tab.iloc[0]]
                tab = tab.iloc[1:]
                tab.columns = head
                tab = tab.apply(pd.to_numeric, errors="coerce")
            if tab.shape[1] < 2:
                return None
            cols = {str(c).lower(): c for c in tab.columns}
            if "dim" in cols:
                tab = tab[tab[cols["dim"]] == 1]
            bc, dc = cols.get("birth", tab.columns[-2]), cols.get("death", tab.columns[-1])
            rows = list(zip(tab[bc], tab[dc]))
    except Exception:
        return None
    out = []
    for b, d in rows:
        try:
            b = float(b)
            d = float(d) if d is not None else np.inf
        except (TypeError, ValueError):
            continue
        if not np.isfinite(d):
            if thresh is None:
                continue
            d = thresh
        out.append([b, d])
    return np.asarray(out, dtype=float).reshape(-1, 2)


def load_bars(cls, pid, variant, thresh=THRESH):
    """H1 bars (N x 2) for 'standard' or 'augmented'; None when not available.
    Reads the Ripserer barcodes through ph_io when it can, else the barcode files
    (<variant>/<class>/barcodes or PH_1, json / csv / npy)."""
    tid = ssf.track_id(pid)
    try:
        from ph_io import load_h1_bars
        b = load_h1_bars(cls, tid, variant, thresh)
        if b is not None:
            return np.asarray(b, dtype=float).reshape(-1, 2)
    except Exception as e:
        PH_NOTE.setdefault("ph_io", f"{type(e).__name__}: {e}")
    for folder in bar_dirs(cls, variant):
        if not long_path(folder).is_dir():
            continue
        for name in dict.fromkeys((tid, pid)):
            found = [folder / f"{name}{ext}" for ext in BAR_EXTS]
            for pattern in (f"{name}.*", f"{name}_*", f"*_{name}.*"):
                found += sorted(long_path(folder).glob(pattern))
            for f in found:
                if f.suffix.lower() in BAR_EXTS and long_path(f).exists():
                    bars = _bars_from_file(f, thresh)
                    if bars is not None:
                        return bars
    return None


def bars_report(cls, pid):
    """Where H1 bars were looked for; printed when none can be found."""
    lines = [f"[diagnose] example protein: {cls}/{pid}  (track id {ssf.track_id(pid)})",
             f"[diagnose] id_map.csv: {ssf.ID_MAP}  ({'found' if ssf.ID_MAP.exists() else 'MISSING'})",
             "[diagnose] ph_io: " + PH_NOTE.get("ph_io", "importable, returned nothing for this protein")]
    for variant in ("standard", "augmented"):
        for d in bar_dirs(cls, variant):
            if d.is_dir():
                ex = ", ".join(sorted(x.name for x in d.iterdir())[:3]) or "empty"
                lines.append(f"[diagnose]   {variant:<9} {d}  (found; e.g. {ex})")
            else:
                lines.append(f"[diagnose]   {variant:<9} {d}  (folder MISSING)")
        t = ssf.feature_table(variant)
        lines.append(f"[diagnose]   {variant:<9} {t}  ({'found' if t.exists() else 'MISSING'})")
    return "\n".join(lines)


def _net_charge(seq, ph):
    c = 1.0 / (1.0 + 10 ** (ph - PKA_NTERM)) - 1.0 / (1.0 + 10 ** (PKA_CTERM - ph))
    for aa, pka in PKA_SIDE.items():
        n = seq.count(aa)
        if n:
            c += (n / (1.0 + 10 ** (ph - pka)) if aa in BASE_RES
                  else -n / (1.0 + 10 ** (pka - ph)))
    return c


def compute_pi(seq, tol=1e-4):
    if not seq:
        return float("nan")
    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        c = _net_charge(seq, mid)
        if abs(c) < tol:
            return round(mid, 3)
        lo, hi = (mid, hi) if c > 0 else (lo, mid)
    return round((lo + hi) / 2.0, 3)


def extract_pdb(pdb_path):
    """Composition and B-factor / pLDDT summary of the Ca trace of one structure."""
    seq, seen, vals = [], set(), []
    try:
        with open(long_path(pdb_path)) as fh:
            for line in fh:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    key = (line[21], line[22:26].strip())
                    if key in seen:
                        continue
                    seen.add(key)
                    seq.append(AA3TO1.get(line[17:20].strip(), "X"))
                    try:
                        vals.append(float(line[60:66]))
                    except ValueError:
                        pass
    except Exception:
        return None
    if not seq:
        return None
    s = "".join(seq)
    n = len(s)
    v = np.array(vals) if vals else np.array([np.nan])
    out = {"chain_length": n, "n_cys": s.count("C"),
           "cys_fraction": s.count("C") / n if n else np.nan,
           "pI": compute_pi(s), BFAC_MEAN: float(np.nanmean(v)),
           BFAC_MED: float(np.nanmedian(v))}
    out.update({f"aa_{a}": s.count(a) / n for a in AA_LETTERS})
    return out


def find_pdb(cls, pid):
    pdb = PDB_DIR / cls / f"{pid}.pdb"
    if not long_path(pdb).exists() and PDB_DIR.is_dir():
        for sub in PDB_DIR.iterdir():
            cand = sub / f"{pid}.pdb"
            if sub.is_dir() and long_path(cand).exists():
                return cand
    return pdb


def build_dataset():
    feat = load_features()
    print(f"Loaded {len(feat)} proteins, {feat['Protein_Class'].nunique()} classes")
    print("Extracting composition from PDB files ...")
    recs = []
    for _, r in feat.iterrows():
        pid, cls = r["Protein_ID"], r["Protein_Class"]
        pdb = find_pdb(cls, pid)
        e = extract_pdb(pdb) if long_path(pdb).exists() else None
        rec = {"Protein_ID": pid, "Protein_Class": cls}
        if e:
            rec.update(e)
        recs.append(rec)
    comp = pd.DataFrame(recs)
    for c in ("chain_length", "n_cys", "cys_fraction", "pI", BFAC_MEAN, BFAC_MED,
              *[f"aa_{a}" for a in AA_LETTERS]):
        if c not in comp.columns:
            comp[c] = np.nan
    print(f"  extracted for {comp['pI'].notna().sum()} / {len(feat)} proteins")
    keys = ("Protein_ID", "Protein_Class")
    feat = feat.drop(columns=[c for c in comp.columns if c not in keys and c in feat.columns])
    df = feat.merge(comp.drop(columns=["Protein_Class"]), on="Protein_ID", how="left")
    if "Curv_median" in df.columns:
        df["curv_norm"] = df["Curv_median"] / df["chain_length"]
    return df, comp


def avail_features(d):
    return [f for f in FEATURE_COLS
            if f in d.columns and d[f].notna().sum() > 3 and d[f].nunique() > 1]


# MODELLING HELPERS

def variance_partition(d0, column, extra=()):
    """(R2 of log-size, R2 added by family, Kruskal-Wallis p of the size-residuals)."""
    import statsmodels.formula.api as smf
    from scipy.stats import kruskal
    d = d0[[column, "logsize", "Protein_Class", *extra]].dropna().rename(columns={column: "y"})
    base = "logsize" + "".join(f" + {e}" for e in extra)
    r2_base = smf.ols(f"y ~ {base}", d).fit().rsquared
    r2_full = smf.ols(f"y ~ {base} + C(Protein_Class)", d).fit().rsquared
    d = d.assign(resid=smf.ols(f"y ~ {base}", data=d).fit().resid)
    _, p = kruskal(*[g["resid"].values for _, g in d.groupby("Protein_Class")])
    return r2_base, r2_full - r2_base, p


def balanced_accuracy_cv(d, cols, n_estimators=400):
    """5-fold stratified balanced accuracy (mean, sd) of a random forest on `cols`."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    X = d[cols].fillna(d[cols].median()).values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced",
                                 random_state=0, n_jobs=-1)
    ba = cross_val_score(clf, X, d["Protein_Class"].values, cv=cv, scoring="balanced_accuracy")
    return float(ba.mean()), float(ba.std())
