"""
download_all_experimental.py

Download EVERY experimental structure in the RCSB PDB for each snake-venom
protein family listed below. 
Method: the RCSB Search API is queried once for every experimental polymer
entity whose source organism lineage is Serpentes (NCBI taxId 8570).
Each entity is then classified into a family from its own
Pfam / InterPro / molecule-name annotation. Every entity classified into one
of the six requested families is downloaded. 

Families and their Pfam/InterPro signatures:
    PLA2        Phospholipase A2                PF00068
    3FTx        Three-finger toxin              PF00087
    SVMP        Snake-venom metalloproteinase   PF01421 / PF01562 / PF08516
    serine      Snake-venom serine protease     PF00089
    kallikrein  (serine-protease sub-type)      PF00089  + name contains 'kallikrein'
    kunitz      Kunitz-type                     PF00014

Output 
outputpath:Venom_Databases/data_experimental/:
    output: .cif or pdb files  (both formats when available)
    manifest_experimental.csv                       one row per downloaded entry

Usage

    python download_all_experimental.py                       # all families, cif+pdb
    python download_all_experimental.py --families PLA2 kunitz # subset
    python download_all_experimental.py --formats cif         # cif only
    python download_all_experimental.py --any-organism        # drop the Serpentes filter
    python download_all_experimental.py --list-only           # just report counts, no download
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests


SERPENTES_TAXID = "8570"
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
FILE_URL = "https://files.rcsb.org/download/{pdb}.{ext}"

CODES_DIR = Path(__file__).resolve().parent
BASE_DIR = CODES_DIR.parent   # output path                      
DEFAULT_OUT = BASE_DIR / "PDB_computation" / "downloaded_structures"

PAGE_ROWS = 500
GQL_CHUNK = 50
REQUEST_PAUSE = 0.12
TIMEOUT = 60
HEADERS = {"User-Agent": "snake-venom-experimental-collector/1.0 (research use)"}

# use some keys for classification like venom family (folder name), Pfam accessions
FAMILIES = {
    "PLA2": ("Phospholipase_A2", {"PF00068"},
             {"IPR001211", "IPR033112", "IPR033113", "IPR016090"},
             ["phospholipase a2", "phospholipase a-2", "pla2", "crotoxin",
              "ammodytoxin", "notexin", "agkistrodotoxin", "vipoxin"], None),
    "3FTx": ("Three_finger_toxin", {"PF00087"},
             {"IPR003571", "IPR018354", "IPR016054", "IPR045860"},
             ["three-finger", "three finger", "3ftx", "bungarotoxin",
              "cardiotoxin", "cobrotoxin", "erabutoxin", "fasciculin",
              "alpha-neurotoxin", "muscarinic toxin", "dendroaspin",
              "cytotoxin", "neurotoxin", "alpha-cobratoxin", "calciseptine",
              "mambalgin", "haditoxin", "candoxin", "irditoxin"], None),
    "SVMP": ("Snake_venom_metalloproteinase", {"PF01421", "PF01562", "PF08516"},
             {"IPR001590", "IPR024079", "IPR034027"},
             ["metalloproteinase", "metalloprotease", "reprolysin", "adamalysin",
              "jararhagin", "atrolysin", "acutolysin", "hemorrhagic", "fibrolase",
              "catrocollastatin"], None),
    "serine": ("Snake_venom_serine_protease", {"PF00089"},
               {"IPR001254", "IPR009003"},
               ["serine protease", "serine proteinase", "thrombin-like",
                "batroxobin", "gyroxin", "plasminogen activator", "venombin",
                "russelobin", "ancrod"], None),
    "kallikrein": ("Kallikrein", {"PF00089"}, {"IPR001254", "IPR009003"},
                   ["kallikrein"], "kallikrein"),
    "kunitz": ("Kunitz_type", {"PF00014"},
               {"IPR002223", "IPR036880", "IPR020901"},
               ["kunitz", "bpti", "dendrotoxin", "textilinin", "dtx",
                "chymotrypsin inhibitor", "calcicludine"], None),
}



# Search all experimental Serpentes polymer entities

def search_serpentes(session: requests.Session, any_organism: bool) -> list[str]:
    """All experimental polymer-entity ids for snakes (or all organisms)."""
    if any_organism:
        # Without an organism filter, restrict to protein entities so the crawl
        # is bounded; classification still narrows to the six families.
        query = {"type": "terminal", "service": "text", "parameters": {
            "attribute": "entity_poly.rcsb_entity_polymer_type",
            "operator": "exact_match", "value": "Protein"}}
    else:
        query = {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entity_source_organism.taxonomy_lineage.id",
            "operator": "exact_match", "value": SERPENTES_TAXID}}

    ids: list[str] = []
    start, total = 0, None
    while True:
        payload = {
            "query": query,
            "return_type": "polymer_entity",
            "request_options": {
                "paginate": {"start": start, "rows": PAGE_ROWS},
                "results_content_type": ["experimental"],
                "scoring_strategy": "combined",
            },
        }
        resp = session.get(SEARCH_URL, params={"json": json.dumps(payload)},
                           timeout=TIMEOUT)
        resp.raise_for_status()
        if resp.status_code == 204 or not resp.content:
            break
        data = resp.json()
        if total is None:
            total = data.get("total_count", 0)
            print(f"[search] {total} experimental polymer entities "
                  f"({'all organisms' if any_organism else 'Serpentes'})")
        batch = [r["identifier"] for r in data.get("result_set", [])]
        if not batch:
            break
        ids.extend(batch)
        start += len(batch)
        if start >= (total or 0):
            break
    return ids


def classify(meta: dict) -> "str | None":
    """Assign an entity's metadata to one of the six family keys, or None.

    Priority: Pfam id -> InterPro id -> molecule-name keyword. 'kallikrein' wins
    over 'serine' when the description says so.
    """
    pfam = set(meta.get("pfam", []))
    interpro = set(meta.get("interpro", []))
    name = (meta.get("description") or "").lower()
    # keyword hit for kallikrein takes precedence (it is a serine protease)
    if "kallikrein" in name:
        return "kallikrein"
    for key, (_folder, pfams, iprs, _kw, _must) in FAMILIES.items():
        if key == "kallikrein":
            continue
        if pfam & pfams or interpro & iprs:
            return key
    for key, (_folder, _pfams, _iprs, kws, _must) in FAMILIES.items():
        if key == "kallikrein":
            continue
        if any(kw in name for kw in kws):
            return key
    return None


GQL_QUERY = """
query($ids:[String!]!){
  polymer_entities(entity_ids:$ids){
    rcsb_id
    rcsb_polymer_entity{ pdbx_description pdbx_ec }
    rcsb_polymer_entity_annotation{ type annotation_id }
    rcsb_polymer_entity_container_identifiers{ entry_id auth_asym_ids }
    rcsb_entity_source_organism{ ncbi_scientific_name }
    entity_poly{ rcsb_sample_sequence_length }
  }
}"""

ENTRY_QUERY = """
query($ids:[String!]!){
  entries(entry_ids:$ids){
    rcsb_id
    exptl{ method }
    rcsb_entry_info{ resolution_combined }
  }
}"""


def fetch_entity_meta(session, entity_ids: list[str]) -> dict:
    out = {}
    for i in range(0, len(entity_ids), GQL_CHUNK):
        chunk = entity_ids[i:i + GQL_CHUNK]
        resp = session.post(GRAPHQL_URL,
                            json={"query": GQL_QUERY, "variables": {"ids": chunk}},
                            timeout=TIMEOUT)
        resp.raise_for_status()
        for ent in resp.json().get("data", {}).get("polymer_entities") or []:
            if not ent:
                continue
            pe = ent.get("rcsb_polymer_entity") or {}
            cid = ent.get("rcsb_polymer_entity_container_identifiers") or {}
            orgs = ent.get("rcsb_entity_source_organism") or []
            anns = ent.get("rcsb_polymer_entity_annotation") or []
            out[ent["rcsb_id"]] = {
                "pdb_id": cid.get("entry_id", ent["rcsb_id"].split("_")[0]),
                "auth_asym_ids": ";".join(cid.get("auth_asym_ids") or []),
                "description": pe.get("pdbx_description") or "",
                "ec": pe.get("pdbx_ec") or "",
                "pfam": [a["annotation_id"] for a in anns if a.get("type") == "Pfam"],
                "interpro": [a["annotation_id"] for a in anns
                             if a.get("type") == "InterPro"],
                "organism": "; ".join(o.get("ncbi_scientific_name") or ""
                                      for o in orgs if o).strip("; "),
                "seq_len": (ent.get("entity_poly") or {}).get(
                    "rcsb_sample_sequence_length"),
            }
    return out


def fetch_entry_meta(session, pdb_ids: list[str]) -> dict:
    out = {}
    for i in range(0, len(pdb_ids), GQL_CHUNK):
        chunk = pdb_ids[i:i + GQL_CHUNK]
        resp = session.post(GRAPHQL_URL,
                            json={"query": ENTRY_QUERY, "variables": {"ids": chunk}},
                            timeout=TIMEOUT)
        resp.raise_for_status()
        for e in resp.json().get("data", {}).get("entries") or []:
            if not e:
                continue
            methods = [m["method"] for m in (e.get("exptl") or []) if m.get("method")]
            res = (e.get("rcsb_entry_info") or {}).get("resolution_combined") or []
            out[e["rcsb_id"]] = {"method": "; ".join(methods),
                                 "resolution": min(res) if res else ""}
    return out



# Download
def download(session, pdb_id: str, dest: Path, ext: str) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    try:
        resp = session.get(FILE_URL.format(pdb=pdb_id, ext=ext),
                           timeout=TIMEOUT, headers=HEADERS)
    except requests.RequestException as exc:
        return f"error:{exc.__class__.__name__}"
    if resp.status_code != 200 or not resp.content:
        return f"error:http{resp.status_code}"
    dest.write_bytes(resp.content)
    time.sleep(REQUEST_PAUSE)
    return "downloaded"



# Main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output root (default: %(default)s)")
    ap.add_argument("--families", nargs="+", choices=list(FAMILIES),
                    default=list(FAMILIES), help="families to download (default: all)")
    ap.add_argument("--formats", nargs="+", choices=["cif", "pdb"],
                    default=["cif", "pdb"], help="file format(s) (default: cif pdb)")
    ap.add_argument("--any-organism", action="store_true",
                    help="drop the Serpentes filter (all organisms with these folds)")
    ap.add_argument("--list-only", action="store_true",
                    help="report per-family counts, download nothing")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    # 1. one crawl of all experimental snake polymer entities
    entity_ids = search_serpentes(session, args.any_organism)
    if not entity_ids:
        print("No entities returned -- aborting.")
        return 1

    # 2. metadata + classification into the six families
    meta = fetch_entity_meta(session, entity_ids)
    wanted = set(args.families)
    fam_entities: dict[str, list[str]] = {k: [] for k in FAMILIES}
    for e in entity_ids:
        if e not in meta:
            continue
        fam = classify(meta[e])
        if fam in wanted:
            fam_entities[fam].append(e)

    print("\n[classify] entities per requested family:")
    for k in args.families:
        n_ent = len(fam_entities[k])
        n_entry = len({meta[e]["pdb_id"] for e in fam_entities[k]})
        print(f"    {k:<12} {n_ent:>4} entities  /  {n_entry:>4} PDB entries")

    manifest_rows = []
    grand_total = 0

    # 3. per-family download
    for fam_key in args.families:
        folder = FAMILIES[fam_key][0]
        ents = fam_entities[fam_key]
        if not ents:
            continue
        pdb_ids = sorted({meta[e]["pdb_id"] for e in ents})
        grand_total += len(pdb_ids)
        entry_meta = {} if args.list_only else fetch_entry_meta(session, pdb_ids)

        fam_dir = out_root / folder
        if not args.list_only:
            fam_dir.mkdir(parents=True, exist_ok=True)

        for e in ents:
            m = meta[e]
            q = entry_meta.get(m["pdb_id"], {})
            manifest_rows.append(dict(
                family=fam_key, folder=folder, pdb_id=m["pdb_id"], entity_id=e,
                auth_asym_ids=m.get("auth_asym_ids", ""),
                description=m["description"], organism=m["organism"], ec=m["ec"],
                pfam=";".join(m.get("pfam", [])), seq_len=m["seq_len"],
                method=q.get("method", ""), resolution=q.get("resolution", "")))

        if args.list_only:
            continue
        for pid in pdb_ids:
            for ext in args.formats:
                status = download(session, pid, fam_dir / f"{pid}.{ext}", ext)
                if status.startswith("error") and not (ext == "pdb" and
                                                        status == "error:http404"):
                    print(f"    ! {pid}.{ext}: {status}")
        print(f"[download] {folder}: {len(pdb_ids)} entries -> {fam_dir}")

    # manifest
    man = out_root / "manifest_experimental.csv"
    with man.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["family", "folder", "pdb_id", "entity_id",
                                           "auth_asym_ids", "description", "organism",
                                           "ec", "pfam", "seq_len", "method",
                                           "resolution"])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"\n[summary] {grand_total} unique entries across "
          f"{len(args.families)} families")
    print(f"[report]  {man}  ({len(manifest_rows)} entity rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
