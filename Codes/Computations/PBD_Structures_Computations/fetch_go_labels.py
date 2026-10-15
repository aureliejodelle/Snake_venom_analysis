
"""
fetch_go_labels.py
Fetch Gene Ontology (GO) functional annotations for the experimental venom
entities from RCSB and build the per-structure function labels that
predict_function.py needs 

  toxin : entity has GO molecular-function 'toxin activity' (GO:0090729 & children)
  coag  : entity has a GO biological-process term containing 'coagulation'

Reads  data_experimental/manifest_experimental.csv  (entity_id per venom entity).
Writes experimental_data/results/function_labels.csv
       (Protein_ID = pdb_id, Protein_Class, toxin, coag, go_terms).

Usage:  python fetch_go_labels.py [--out <experimental_data>]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

CODES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODES_DIR.parent
MANIFEST = PROJECT_ROOT / "PDB_computation" / "downloaded_structures" / "manifest_experimental.csv"
DEFAULT_OUT = PROJECT_ROOT / "PDB_computation" / "experimental_data"

GRAPHQL_URL = "https://data.rcsb.org/graphql"
HEADERS = {"User-Agent": "snake-venom-go-labels/1.0 (research use)"}
CHUNK = 40
TIMEOUT = 60

GQL = """
query($ids:[String!]!){
  polymer_entities(entity_ids:$ids){
    rcsb_id
    rcsb_polymer_entity_container_identifiers{ entry_id }
    rcsb_polymer_entity_annotation{ type annotation_id name }
  }
}"""

FAMILY_KEY_TO_CLASS = {
    "PLA2": "PLA", "3FTx": "3ft", "SVMP": "SNMP",
    "serine": "serine", "kallikrein": "kallikrein", "kunitz": "kunitz",
}


def _post(session, ids, tries=4):
    for k in range(tries):
        try:
            r = session.post(GRAPHQL_URL, json={"query": GQL, "variables": {"ids": ids}},
                             timeout=TIMEOUT, headers=HEADERS)
            r.raise_for_status()
            return r.json().get("data", {}).get("polymer_entities") or []
        except requests.RequestException as exc:
            wait = 2 * (k + 1)
            print(f"    ! retry {k+1}/{tries} after {exc.__class__.__name__} ({wait}s)")
            time.sleep(wait)
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"[error] {MANIFEST} not found"); return 1

    # entity_id -> class, and the venom entity ids to query
    ent_class, ent_ids = {}, []
    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cls = FAMILY_KEY_TO_CLASS.get(row["family"])
            if not cls:
                continue
            e = row["entity_id"]
            ent_class[e] = cls
            ent_ids.append(e)
    ent_ids = sorted(set(ent_ids))
    print(f"Fetching GO for {len(ent_ids)} venom entities ...")

    session = requests.Session()
    session.headers.update(HEADERS)

    # pdb_id -> {class, toxin, coag, go set}
    agg = defaultdict(lambda: {"cls": "", "toxin": False, "coag": False, "go": set()})
    for i in range(0, len(ent_ids), CHUNK):
        chunk = ent_ids[i:i + CHUNK]
        for ent in _post(session, chunk):
            if not ent:
                continue
            e = ent["rcsb_id"]
            pid = (ent.get("rcsb_polymer_entity_container_identifiers") or {}
                   ).get("entry_id", e.split("_")[0]).upper()
            anns = ent.get("rcsb_polymer_entity_annotation") or []
            rec = agg[pid]
            rec["cls"] = rec["cls"] or ent_class.get(e, "")
            for a in anns:
                if a.get("type") != "GO":
                    continue
                name = (a.get("name") or "").lower()
                gid = a.get("annotation_id") or ""
                rec["go"].add(f'{gid}:{a.get("name","")}')
                if "toxin activity" in name or gid == "GO:0090729":
                    rec["toxin"] = True
                if "coagulation" in name:
                    rec["coag"] = True
        print(f"  {min(i + CHUNK, len(ent_ids))}/{len(ent_ids)}")
        time.sleep(0.1)

    out = Path(args.out) / "results" / "function_labels.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Protein_ID", "Protein_Class", "toxin", "coag", "go_terms"])
        for pid in sorted(agg):
            r = agg[pid]
            w.writerow([pid, r["cls"], int(r["toxin"]), int(r["coag"]),
                        " | ".join(sorted(r["go"]))])
    n = len(agg)
    ntox = sum(1 for r in agg.values() if r["toxin"])
    ncoag = sum(1 for r in agg.values() if r["coag"])
    print(f"\n[ok] {out}  ({n} structures; toxin={ntox}, coagulation={ncoag})")
    # per-class positives
    by = defaultdict(lambda: [0, 0, 0])
    for r in agg.values():
        by[r["cls"]][0] += 1
        by[r["cls"]][1] += int(r["toxin"])
        by[r["cls"]][2] += int(r["coag"])
    print("  class      n   toxin  coag")
    for c in ["3ft", "PLA", "serine", "SNMP", "kallikrein", "kunitz"]:
        if c in by:
            n_, t_, g_ = by[c]
            print(f"  {c:<10}{n_:>3}{t_:>7}{g_:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
