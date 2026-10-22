"""
make_ca_clouds.py
Prepare the C-alpha inputs of the disulfide-augmented Ripserer experiment
(augmented_persistent_homology.jl) for the AlphaFold models. A C-alpha cloud is
sparse enough that a disulfide staple (Ca_i-Ca_j of two bonded cysteines, several
Angstrom apart) creates a genuinely new backbone loop, so the plain-vs-augmented
comparison is meaningful.

Per protein it writes, named with ssbond_features.track_id (uid when an id_map.csv
exists, otherwise the Protein_ID):
  <CA_CLOUDS_DIR>/<class>/<id>.csv    Ca coordinates (x,y,z), one row per residue
  <SS_BONDS_DIR>/<class>/<id>.json    {protein_id, class_, n_ssbond, ss_matched, ca_pairs}
                                      (0-based indices into the Ca cloud)

Disulfide positions come from SnakeVenom.csv by sequence matching. Experimental
structures get their clouds and staple specs from the PDB preparation script
(SSBOND records), so this script does nothing for SNAKE_DATASET=pdb.

Usage:
  python make_ca_clouds.py                 # SNAKE_DATASET defaults to alphafold
Nothing in clean_data / SnakeVenom.csv is modified.
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("SNAKE_DATASET", "alphafold")
from config import config, long_path
import ssbond_features as ssf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    if config.dataset in ("pdb", "experimental", "exp"):
        print("Experimental clouds and staple specs are written by the PDB preparation script; "
              "nothing to do here.")
        return
    catalogue = ssf.catalogue()
    if catalogue is None:
        sys.exit(f"SnakeVenom.csv not found (looked at {config.METADATA_CSV})")
    ann = ssf.csv_annotator(catalogue)

    feats = Path(config.VIZ_FEATURES_DIR) / "protein_features.csv"
    feat = pd.read_csv(feats)
    feat = feat[feat["Protein_Class"].isin(ssf.VALID)]
    out_ca, out_ss = Path(config.CA_CLOUDS_DIR), Path(config.SS_BONDS_DIR)
    print(f"Building Ca clouds + staple pairs for {len(feat)} proteins ...")
    n_cloud = n_matched = n_pairs = 0
    for _, r in feat.iterrows():
        cls, pid = r["Protein_Class"], r["Protein_ID"]
        pdb = Path(config.PDB_DATA_DIR) / cls / f"{pid}.pdb"
        if not long_path(pdb).exists():
            continue
        seq, X = ssf.read_ca(pdb)
        if len(X) < 4:
            continue
        pairs, matched = ann(cls, pid, seq)
        n_matched += int(matched)
        n_pairs += len(pairs)
        name = ssf.track_id(pid)
        (out_ca / cls).mkdir(parents=True, exist_ok=True)
        np.savetxt(long_path(out_ca / cls / f"{name}.csv"), X, delimiter=",", fmt="%.3f",
                   header="x,y,z", comments="")
        (out_ss / cls).mkdir(parents=True, exist_ok=True)
        with open(long_path(out_ss / cls / f"{name}.json"), "w") as fh:
            json.dump(dict(protein_id=pid, class_=cls, n_ssbond=len(pairs), ss_matched=matched,
                           ca_pairs=[list(p) for p in pairs]), fh)
        n_cloud += 1
    print(f"  Ca clouds written : {n_cloud} -> {out_ca}")
    print(f"  staple specs      : {n_cloud} -> {out_ss}")
    print(f"  matched proteins  : {n_matched}   Ca-Ca staples: {n_pairs}")


if __name__ == "__main__":
    main()
