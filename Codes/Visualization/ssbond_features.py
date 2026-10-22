"""
ssbond_features.py

Disulfide input layer shared by every disulfide script. Paths come from
config.py, so the active SNAKE_DATASET profile decides the dataset.

  Protein IDs: the AlphaFold disulfide track names files by a short uid
    (id_map.csv: uid <-> long stem). 'track_id' maps a stem to its track name;
    experimental structures need no mapping.
  'read_ca': C-alpha reader (residue order, duplicate residues skipped), the
    same indexing the Ca clouds and staple pairs use.
  * 'make_annotator': disulfide pairs for a protein, from the staple JSON files
    ('ca_pairs', 0-based) and, as a fallback, SnakeVenom.csv sequence matching.
  * 'build': the per-protein standard-vs-augmented table written to
    revision_ssbond_features.csv, rebuilt from the Ripserer feature tables:

      <disulfide features>/standard/protein_features.csv    H1_* (plain)
      <disulfide features>/augmented/protein_features.csv   H1_* (staple)
      <PH>/standard/<cls>/barcodes/<uid>.json                dim_0 -> chain_length
      <SS_BONDS_DIR>/<cls>/<uid>.json                        n_ssbond, ss_matched

    'has_ss' marks proteins carrying at least one mapped staple, the only
    proteins for which plain-vs-augmented comparisons are meaningful.
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import os
import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import config, long_path

DD = Path(config.CA_CLOUDS_DIR).parent


def catalogue():
    """SnakeVenom.csv (sequence, disulfide and GO annotations) or None."""
    here = Path(__file__).resolve()
    candidates = [os.environ.get("SNAKE_SSBOND_CSV"), config.METADATA_CSV]
    candidates += [d / "SnakeVenom.csv" for d in (*here.parents[:4], DD.parent, DD.parent.parent)]
    for p in candidates:
        if p and Path(p).exists():
            return Path(p)
    return None

DS = DD / "disulfide" 
ID_MAP = next((p for p in (DD / "id_map.csv", DS / "id_map.csv") if p.exists()), DD / "id_map.csv")


def feature_table(variant):
    """protein_features.csv of the standard / augmented disulfide run (first that exists)."""
    cands = [DD / "features" / variant / "protein_features.csv",
             DS / "features" / variant / "protein_features.csv",
             DS / variant / "protein_features.csv"]
    return next((p for p in cands if p.exists()), cands[0])
OUT_CSV = Path(config.HIGH_IMPACT_DIR) / "revision_ssbond_features.csv"

VALID = ["SNMP", "PLA", "serine", "3ft", "kallikrein", "kunitz"]

THREE2ONE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
             'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
             'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
             'TYR': 'Y', 'VAL': 'V'}

_CACHE: dict = {}


def id_map() -> dict:
    """stem -> uid; empty when there is no id_map.csv."""
    if "map" not in _CACHE:
        try:
            m = pd.read_csv(ID_MAP)
            _CACHE["map"] = dict(zip(m["stem"], m["uid"]))
        except Exception:
            _CACHE["map"] = {}
    return _CACHE["map"]


def track_id(stem: str) -> str:
    return id_map().get(stem, stem)


def read_ca(pdb):
    """(one-letter sequence, Ca coordinates) in residue order."""
    seq, coords, seen = [], [], set()
    with open(long_path(pdb)) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                key = (line[21], line[22:26].strip())
                if key in seen:
                    continue
                seen.add(key)
                seq.append(THREE2ONE.get(line[17:20].strip(), "X"))
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return "".join(seq), np.array(coords)


def parse_ss(s):
    if not isinstance(s, str):
        return []
    return [(int(a), int(b)) for a, b in re.findall(r'DISULFID\s+(\d+)\.\.(\d+)', s)]


def csv_annotator(path):
    """SnakeVenom.csv lookup: locate the protein sequence inside a catalogue
    sequence and shift the annotated cysteine positions accordingly."""
    csv = pd.read_csv(path)
    csv = csv[csv["Type"].isin(VALID)].copy()
    seqs = [s if isinstance(s, str) else "" for s in csv["Sequence"]]
    sss = csv["Disulfide_bond"].apply(parse_ss).tolist()

    def ann(cls, pid, pseq):
        for cs, ss in zip(seqs, sss):
            k = cs.find(pseq)
            if k >= 0:
                out = []
                for a, b in ss:
                    ia, ib = a - 1 - k, b - 1 - k
                    if 0 <= ia < len(pseq) and 0 <= ib < len(pseq) \
                            and pseq[ia] == 'C' and pseq[ib] == 'C':
                        out.append((ia, ib))
                return out, True
        return [], False

    return ann


def json_annotator(root):
    """Staple specs written next to the Ca clouds ('ca_pairs', 0-based)."""
    root = Path(root)

    def ann(cls, pid, pseq):
        for name in dict.fromkeys((track_id(pid), pid)):
            f = root / cls / f"{name}.json"
            if not long_path(f).exists():
                continue
            try:
                with open(long_path(f)) as fh:
                    d = json.load(fh)
            except Exception:
                continue
            pairs = []
            for a, b in d.get("ca_pairs", []):
                a, b = int(a), int(b)
                if a != b and 0 <= a < len(pseq) and 0 <= b < len(pseq):
                    pairs.append((a, b))
            return pairs, True
        return [], False

    return ann


def make_annotator(json_dir=None, csv_path=None):
    """Annotator trying the JSON staple files first, then SnakeVenom.csv."""
    sources = []
    if json_dir is not None and Path(json_dir).is_dir():
        sources.append(json_annotator(json_dir))
    if csv_path is not None and Path(csv_path).exists():
        sources.append(csv_annotator(csv_path))
    if not sources:
        return None

    def ann(cls, pid, pseq):
        for source in sources:
            pairs, matched = source(cls, pid, pseq)
            if matched:
                return pairs, True
        return [], False

    return ann


def _load_variant(variant: str) -> pd.DataFrame:
    df = pd.read_csv(feature_table(variant))[["Protein_ID", "Protein_Class", "H1_count", "H1_median",
                         "H1_sum", "H1_max"]]
    return df.rename(columns={"Protein_ID": "uid"})


def _chain_lengths() -> dict:
    """chain_length = number of Ca points = number of dim_0 bars in the standard barcode."""
    out = {}
    base = Path(config.PH_DIR)
    base = base / "standard" if (base / "standard").exists() else \
        DD / "processed_data" / "persistent_homology" / "standard"
    for jf in base.glob("*/barcodes/*.json"):
        try:
            with open(jf) as fh:
                out[jf.stem] = len(json.load(fh).get("dim_0", []))
        except Exception:
            continue
    return out


def _bonds() -> dict:
    out = {}
    for jf in Path(config.SS_BONDS_DIR).glob("*/*.json"):
        try:
            with open(jf) as fh:
                d = json.load(fh)
            out[jf.stem] = (int(d.get("n_ssbond", 0)), bool(d.get("ss_matched", False)))
        except Exception:
            continue
    return out


def build(write: bool = True) -> pd.DataFrame:
    """Assemble the Ripserer standard-vs-augmented table"""
    idm = pd.read_csv(ID_MAP)[["uid", "stem"]]
    std, aug = _load_variant("standard"), _load_variant("augmented")

    m = std.merge(aug, on="uid", suffixes=("_std", "_aug"))
    m["Protein_Class"] = m["Protein_Class_std"]
    m = m[m["Protein_Class"].isin(VALID)].copy()
    m = m.rename(columns={
        "H1_count_std": "h1c_std", "H1_median_std": "h1med_std",
        "H1_sum_std": "h1sum_std", "H1_max_std": "h1max_std",
        "H1_count_aug": "h1c_aug", "H1_median_aug": "h1med_aug",
        "H1_sum_aug": "h1sum_aug", "H1_max_aug": "h1max_aug",
    })

    cl, bd = _chain_lengths(), _bonds()
    m["chain_length"] = m["uid"].map(cl)
    m["n_ssbond"] = m["uid"].map(lambda u: bd.get(u, (0, False))[0])
    m["ss_matched"] = m["uid"].map(lambda u: bd.get(u, (0, False))[1])

    missing = int(m["chain_length"].isna().sum())
    if missing:
        warnings.warn(f"{missing} proteins lack a standard Ripserer barcode; "
                      "chain_length is NaN for them")

    m["logsize"] = np.log(m["chain_length"])
    m["h1c_ss"] = m["h1c_aug"] - m["h1c_std"]
    m["h1sum_ss"] = m["h1sum_aug"] - m["h1sum_std"]
    m["has_ss"] = m["ss_matched"] & (m["n_ssbond"] > 0)
    m["Protein_ID"] = m["uid"].map(dict(zip(idm["uid"], idm["stem"])))

    cols = ["Protein_ID", "Protein_Class", "chain_length", "n_ssbond", "ss_matched",
            "h1c_std", "h1med_std", "h1sum_std", "h1max_std",
            "h1c_aug", "h1med_aug", "h1sum_aug", "h1max_aug",
            "logsize", "h1c_ss", "h1sum_ss", "has_ss"]
    out = m[cols].reset_index(drop=True)

    if write:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        bak = OUT_CSV.with_name("revision_ssbond_features_backup.csv")
        if OUT_CSV.exists() and not bak.exists():
            OUT_CSV.rename(bak)
            print(f"[backup] previous table -> {bak.name}")
        out.to_csv(OUT_CSV, index=False)
        print(f"[ok] ssbond features -> {OUT_CSV} "
              f"({len(out)} proteins, {int(out['has_ss'].sum())} carry a disulfide staple)")
    return out


if __name__ == "__main__":
    df = build(write=True)
    sub = df[df["has_ss"]]
    print("\nDisulfide subsample (has_ss) per family:")
    print(pd.DataFrame({
        "n_total": df.groupby("Protein_Class").size(),
        "n_with_ssbond": sub.groupby("Protein_Class").size(),
    }).fillna(0).astype(int).to_string())
