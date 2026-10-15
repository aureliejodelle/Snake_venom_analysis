
"""
curate_experimental.py
Build a NON-REDUNDANT, quality-filtered representative set from the full
experimental download


1. Read PDB_computation/downloaded_structures/manifest_experimental.csv
2. Fetch each venom entity's RCSB sequence-cluster id at 90% identity
3. Group entities by (family, 90%-cluster) and keep ONE representative per
   cluster: prefer X-ray / cryo-EM at the best (lowest) resolution, then NMR.
4. Quality filter: keep a representative only if it is NMR, or X-ray/cryo-EM
   with resolution max-res (default 3.0 Angstrom).
5. Write downloaded_structures/manifest_curated.csv (same schema as the input,
   representatives only) for prepare_experimental_pdbs.py --manifest.

Usage:  python curate_experimental.py [--identity 90] [--max-res 3.0] [--keep-nmr]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
DL = PROJECT_ROOT / "PDB_computation" / "downloaded_structures"
HEADERS = {"User-Agent": "snake-venom-curate/1.0 (research use)"}
TIMEOUT = 120
# RCSB precomputed sequence clusters (one line per cluster, space-separated
# polymer-entity ids). The Data-API cluster field is deprecated/null, so we map
# our entities against this file.
CLUSTER_URL = "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{pct}.txt"


def fetch_clusters(session, target_ids: set, identity: int) -> dict:
    """entity_id -> cluster index, from the RCSB 90%(default) cluster file."""
    out = {}
    r = session.get(CLUSTER_URL.format(pct=identity), timeout=TIMEOUT, stream=True)
    r.raise_for_status()
    for idx, line in enumerate(r.iter_lines(decode_unicode=True)):
        if not line:
            continue
        members = line.split()
        hit = target_ids.intersection(members)
        for e in hit:
            out[e] = idx
    return out


def _method_class(m: str) -> str:
    m = (m or "").upper()
    if "X-RAY" in m or "DIFFRACTION" in m:
        return "xray"
    if "ELECTRON MICROSCOPY" in m or "EM" in m:
        return "em"
    if "NMR" in m:
        return "nmr"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--identity", type=int, default=90, choices=[100, 95, 90, 70, 50, 30])
    ap.add_argument("--max-res", type=float, default=3.0,
                    help="max resolution (A) for X-ray/EM representatives")
    ap.add_argument("--keep-nmr", action="store_true", default=True,
                    help="keep NMR structures (default: yes)")
    ap.add_argument("--drop-nmr", dest="keep_nmr", action="store_false")
    args = ap.parse_args()

    man = DL / "manifest_experimental.csv"
    if not man.exists():
        print(f"[error] {man} not found"); return 1
    rows = list(csv.DictReader(man.open(encoding="utf-8")))
    # only the six venom families are in this manifest already
    ent_ids = sorted({r["entity_id"] for r in rows})
    print(f"Fetching 90%-identity clusters for {len(ent_ids)} entities ...")

    session = requests.Session(); session.headers.update(HEADERS)
    cluster = fetch_clusters(session, set(ent_ids), args.identity)
    print(f"  mapped {len(cluster)}/{len(ent_ids)} entities to a {args.identity}% cluster "
          f"({len(ent_ids)-len(cluster)} singletons)")

    # rank: quality first, then method (xray/em<nmr), then resolution, then id
    def rank(r):
        cls = _method_class(r.get("method", ""))
        try:
            res = float(r.get("resolution") or 99)
        except ValueError:
            res = 99.0
        passes = (cls == "nmr" and args.keep_nmr) or (cls in ("xray", "em") and res <= args.max_res)
        morder = {"xray": 0, "em": 1, "nmr": 2}.get(cls, 3)
        return (0 if passes else 1, morder, res, r["pdb_id"])

    # group by (family, cluster); solo entities
    groups = defaultdict(list)
    for r in rows:
        cid = cluster.get(r["entity_id"])
        key = (r["family"], f"c{cid}" if cid else f"solo_{r['entity_id']}")
        groups[key].append(r)

    reps, dropped_quality = [], 0
    for key, members in groups.items():
        best = min(members, key=rank)
        # apply quality gate to the chosen representative
        cls = _method_class(best.get("method", ""))
        try:
            res = float(best.get("resolution") or 99)
        except ValueError:
            res = 99.0
        ok = (cls == "nmr" and args.keep_nmr) or (cls in ("xray", "em") and res <= args.max_res)
        if ok:
            reps.append(best)
        else:
            dropped_quality += 1

    # de-dupe to one row per pdb_id (a structure may host >1 venom entity)
    seen, rep_rows = set(), []
    for r in sorted(reps, key=lambda r: (r["family"], r["pdb_id"])):
        if r["pdb_id"] in seen:
            continue
        seen.add(r["pdb_id"]); rep_rows.append(r)

    out = DL / "manifest_curated.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rep_rows)

    fam = defaultdict(int)
    for r in rep_rows:
        fam[r["family"]] += 1
    print(f"\n[curate] {len(rows)} entities -> {len(groups)} clusters @ {args.identity}% "
          f"identity; {dropped_quality} clusters fail quality (>{args.max_res} A, non-NMR)")
    print(f"[curate] {len(rep_rows)} non-redundant representatives -> {out.name}")
    for k in ("PLA2", "3FTx", "SVMP", "serine", "kallikrein", "kunitz"):
        if fam.get(k):
            print(f"    {k:<12} {fam[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
