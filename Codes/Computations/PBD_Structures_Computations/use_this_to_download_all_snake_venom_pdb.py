#!/usr/bin/env pytho
"""
download_snake_venom_pdb.py
===========================

Download every available snake-venom protein structure from the RCSB Protein
Data Bank (PDB) and organise the files into one sub-folder per venom protein
family.

Pipeline
--------
1. Search the RCSB Search API for all polymer entities whose source organism
   falls under the taxonomic clade Serpentes (NCBI taxId 8570) -- i.e. snakes.
2. Fetch metadata (molecule name + Pfam / InterPro family annotations) for
   every entity in batches through the RCSB GraphQL Data API.
3. Classify each entity into a venom protein family using, in priority order,
   its Pfam id, its InterPro id, then a keyword match on the molecule name.
4. Download the structure file (mmCIF by default) for every matching PDB entry
   into  data/<Family>/  .  An entry that contains chains from more than one
   family is copied into each relevant family folder.
5. Write two CSV reports into the data/ folder:
       manifest.csv        one row per polymer entity
       family_summary.csv  entry counts per family

The script is safe to re-run: files that already exist on disk are skipped.

Usage
-----
    python download_snake_venom_pdb.py                 # full run, mmCIF files
    python download_snake_venom_pdb.py --format pdb    # legacy PDB format
    python download_snake_venom_pdb.py --no-download    # metadata / manifest only
    python download_snake_venom_pdb.py --limit 25       # quick test on 25 entities
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SERPENTES_TAXID = "8570"          # NCBI taxonomy id for the clade Serpentes

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
FILE_URL = "https://files.rcsb.org/download/{pdb}.{ext}"

# Directory layout: this file lives in  Venom_Databases/codes/ ,
# the data goes into  Venom_Databases/data/ .
CODES_DIR = Path(__file__).resolve().parent
BASE_DIR = CODES_DIR.parent
DATA_DIR = BASE_DIR / "PDB_computation" / "legacy_seqmatched_download"

PAGE_ROWS = 200          # search results per page
GQL_CHUNK = 50           # entities per GraphQL request
REQUEST_PAUSE = 0.15     # polite pause (s) between download requests
TIMEOUT = 60
HEADERS = {"User-Agent": "snake-venom-pdb-collector/1.0 (research use)"}

# --------------------------------------------------------------------------- #
# Family definitions
# --------------------------------------------------------------------------- #
# Each family is matched by (in priority order): Pfam ids, InterPro ids, then
# lower-cased keyword substrings found in the molecule name.  Order in the list
# matters -- the first family that matches wins, so more specific families are
# listed before broader ones.

FAMILIES: list[dict] = [
    {
        "name": "Phospholipase_A2",
        "pfam": {"PF00068"},
        "interpro": {"IPR001211", "IPR033112", "IPR033113", "IPR016090"},
        "keywords": ["phospholipase a2", "phospholipase a-2", "pla2", "crotoxin",
                     "ammodytoxin", "notexin", "agkistrodotoxin", "myotoxin ii",
                     "beta-bungarotoxin", "vipoxin"],
    },
    {
        "name": "Three_finger_toxin",
        "pfam": {"PF00087"},
        "interpro": {"IPR003571", "IPR018354", "IPR016054", "IPR045860"},
        "keywords": ["three-finger", "three finger", "3ftx", "bungarotoxin",
                     "cardiotoxin", "cobrotoxin", "erabutoxin", "fasciculin",
                     "alpha-neurotoxin", "muscarinic toxin", "dendroaspin",
                     "toxin alpha", "toxin b", "cytotoxin", "neurotoxin",
                     "alpha-cobratoxin", "calciseptine", "mambalgin",
                     "haditoxin", "candoxin", "irditoxin"],
    },
    {
        "name": "Snake_venom_metalloproteinase",
        "pfam": {"PF01421", "PF01562", "PF08516"},
        "interpro": {"IPR001590", "IPR024079", "IPR034027"},
        "keywords": ["metalloproteinase", "metalloprotease", "reprolysin",
                     "adamalysin", "jararhagin", "atrolysin", "acutolysin",
                     "vap1", "vap2", "hemorrhagic", "fibrolase",
                     "catrocollastatin"],
    },
    {
        "name": "Disintegrin",
        "pfam": {"PF00200"},
        "interpro": {"IPR001762"},
        "keywords": ["disintegrin", "echistatin", "kistrin", "trigramin",
                     "flavoridin", "contortrostatin", "trimestatin",
                     "salmosin", "obtustatin", "schistatin"],
    },
    {
        "name": "Snake_venom_serine_protease",
        "pfam": {"PF00089"},
        "interpro": {"IPR001254", "IPR009003"},
        "keywords": ["serine protease", "serine proteinase", "thrombin-like",
                     "batroxobin", "gyroxin", "plasminogen activator",
                     "kallikrein", "venombin", "russelobin", "ancrod"],
    },
    {
        "name": "C_type_lectin",
        "pfam": {"PF00059"},
        "interpro": {"IPR001304", "IPR016186", "IPR016187", "IPR018378"},
        "keywords": ["c-type lectin", "convulxin", "botrocetin", "flavocetin",
                     "rhodocetin", "ix/x-binding", "ix-binding", "x-binding",
                     "echicetin", "bothrojaracin", "aggretin", "snaclec"],
    },
    {
        "name": "Kunitz_type",
        "pfam": {"PF00014"},
        "interpro": {"IPR002223", "IPR036880", "IPR020901"},
        "keywords": ["kunitz", "bpti", "dendrotoxin", "textilinin",
                     "protease inhibitor", "dtx", "chymotrypsin inhibitor",
                     "calcicludine", "calciseptine"],
    },
    {
        "name": "Crotamine_myotoxin",
        "pfam": set(),
        "interpro": {"IPR000881"},   # beta-defensin fold
        "keywords": ["crotamine", "myotoxin a", "myotoxin i"],
    },
    {
        "name": "Bradykinin_potentiating_peptide",
        "pfam": set(),
        "interpro": set(),
        "keywords": ["bradykinin potentiating", "bradykinin-potentiating", "bpp"],
    },
    {
        "name": "Cathelicidin",
        "pfam": {"PF00666"},
        "interpro": {"IPR001894"},
        "keywords": ["cathelicidin"],
    },
    {
        "name": "Acetylcholinesterase",
        "pfam": {"PF00135"},
        "interpro": {"IPR000997", "IPR019826"},
        "keywords": ["acetylcholinesterase"],
    },
    {
        "name": "Prothrombin_activator",
        "pfam": set(),
        "interpro": set(),
        "keywords": ["prothrombin activator", "pseutarin", "factor v",
                     "coagulation factor"],
    },
    {
        "name": "CRISP",
        "pfam": {"PF00188"},
        "interpro": {"IPR014044", "IPR001283", "IPR035940"},
        "keywords": ["cysteine-rich secretory", "crisp", "ablomin", "triflin",
                     "natrin", "helothermine", "pseudechetoxin", "stecrisp"],
    },
    {
        "name": "L_amino_acid_oxidase",
        "pfam": {"PF01593"},
        "interpro": {"IPR002937"},
        "keywords": ["l-amino acid oxidase", "l-amino-acid oxidase", "laao",
                     "amino acid oxidase"],
    },
    {
        "name": "Nerve_growth_factor",
        "pfam": {"PF00243"},
        "interpro": {"IPR002072", "IPR020408"},
        "keywords": ["nerve growth factor", "ngf"],
    },
    {
        "name": "Natriuretic_peptide",
        "pfam": {"PF00212"},
        "interpro": {"IPR000663"},
        "keywords": ["natriuretic"],
    },
    {
        "name": "Cystatin",
        "pfam": {"PF00031"},
        "interpro": {"IPR000010"},
        "keywords": ["cystatin"],
    },
    {
        "name": "Waprin",
        "pfam": {"PF00095"},
        "interpro": {"IPR008197"},
        "keywords": ["waprin", "whey acidic protein"],
    },
    {
        "name": "VEGF",
        "pfam": {"PF00341"},
        "interpro": {"IPR000072"},
        "keywords": ["vascular endothelial growth factor", "vegf", "vammin"],
    },
    {
        "name": "Sarafotoxin",
        "pfam": set(),
        "interpro": {"IPR020475", "IPR001928"},
        "keywords": ["sarafotoxin", "endothelin"],
    },
    {
        "name": "Vespryn",
        "pfam": {"PF07648"},
        "interpro": set(),
        "keywords": ["ohanin", "vespryn", "thaicobrin"],
    },
    {
        "name": "Cobra_venom_factor",
        "pfam": set(),
        "interpro": set(),
        "keywords": ["cobra venom factor", "cobra-venom factor"],
    },
    {
        "name": "Nucleotidase",
        "pfam": {"PF02872", "PF00149"},
        "interpro": set(),
        "keywords": ["nucleotidase", "phosphodiesterase", "nucleotide pyrophosphatase"],
    },
]

# Fast lookup tables built from FAMILIES (first match wins -> preserve order).
PFAM_TO_FAMILY: dict[str, str] = {}
INTERPRO_TO_FAMILY: dict[str, str] = {}
for _fam in FAMILIES:
    for _p in _fam["pfam"]:
        PFAM_TO_FAMILY.setdefault(_p, _fam["name"])
    for _i in _fam["interpro"]:
        INTERPRO_TO_FAMILY.setdefault(_i, _fam["name"])

UNCLASSIFIED = "Unclassified"


# --------------------------------------------------------------------------- #
# Step 1 -- search
# --------------------------------------------------------------------------- #
def search_serpentes_entities(session: requests.Session, limit: int | None) -> list[str]:
    """Return the list of RCSB polymer-entity ids (e.g. '1ABT_1') for snakes."""
    ids: list[str] = []
    start = 0
    total = None
    while True:
        rows = PAGE_ROWS if limit is None else min(PAGE_ROWS, limit - len(ids))
        if rows <= 0:
            break
        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entity_source_organism.taxonomy_lineage.id",
                    "operator": "exact_match",
                    "value": SERPENTES_TAXID,
                },
            },
            "return_type": "polymer_entity",
            "request_options": {
                "paginate": {"start": start, "rows": rows},
                "results_content_type": ["experimental"],
                "scoring_strategy": "combined",
            },
        }
        resp = session.get(SEARCH_URL, params={"json": json.dumps(query)},
                           timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if total is None:
            total = data.get("total_count", 0)
            print(f"[search] {total} snake polymer entities in PDB")
        batch = [r["identifier"] for r in data.get("result_set", [])]
        if not batch:
            break
        ids.extend(batch)
        start += len(batch)
        if limit is not None and len(ids) >= limit:
            ids = ids[:limit]
            break
        if start >= total:
            break
    print(f"[search] collected {len(ids)} entity ids")
    return ids


# --------------------------------------------------------------------------- #
# Step 2 -- metadata
# --------------------------------------------------------------------------- #
GQL_QUERY = """
query($ids:[String!]!){
  polymer_entities(entity_ids:$ids){
    rcsb_id
    rcsb_polymer_entity{ pdbx_description pdbx_ec }
    rcsb_polymer_entity_annotation{ type annotation_id name }
    rcsb_polymer_entity_container_identifiers{ entry_id }
    rcsb_entity_source_organism{ ncbi_scientific_name }
    entity_poly{ rcsb_sample_sequence_length }
    rcsb_cluster_membership{ cluster_id identity }
  }
}"""

# entry-level query: experimental method + resolution + R-free for quality filter
ENTRY_QUERY = """
query($ids:[String!]!){
  entries(entry_ids:$ids){
    rcsb_id
    exptl{ method }
    rcsb_entry_info{ resolution_combined }
    refine{ ls_R_factor_R_free }
  }
}"""


def fetch_metadata(session: requests.Session, entity_ids: list[str]) -> list[dict]:
    """Batch-fetch metadata for every entity id via the GraphQL Data API."""
    out: list[dict] = []
    for i in range(0, len(entity_ids), GQL_CHUNK):
        chunk = entity_ids[i:i + GQL_CHUNK]
        resp = session.post(GRAPHQL_URL,
                            json={"query": GQL_QUERY, "variables": {"ids": chunk}},
                            timeout=TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
        for ent in payload.get("data", {}).get("polymer_entities") or []:
            if ent is None:
                continue
            anns = ent.get("rcsb_polymer_entity_annotation") or []
            pfam = [a["annotation_id"] for a in anns if a.get("type") == "Pfam"]
            interpro = [a["annotation_id"] for a in anns if a.get("type") == "InterPro"]
            go = [f'{a["annotation_id"]}:{a.get("name", "")}'
                  for a in anns if a.get("type") == "GO"]
            orgs = ent.get("rcsb_entity_source_organism") or []
            pe = ent.get("rcsb_polymer_entity") or {}
            clusters = {c["identity"]: c["cluster_id"]
                        for c in (ent.get("rcsb_cluster_membership") or [])}
            out.append({
                "entity_id": ent["rcsb_id"],
                "pdb_id": (ent.get("rcsb_polymer_entity_container_identifiers") or {}
                           ).get("entry_id", ent["rcsb_id"].split("_")[0]),
                "description": pe.get("pdbx_description") or "",
                "organism": "; ".join(
                    o.get("ncbi_scientific_name") or "" for o in orgs if o).strip("; ") or "",
                "ec": pe.get("pdbx_ec") or "",
                "go": go,
                "pfam": pfam,
                "interpro": interpro,
                "seq_len": (ent.get("entity_poly") or {}).get("rcsb_sample_sequence_length"),
                "clusters": clusters,
            })
        print(f"[metadata] {min(i + GQL_CHUNK, len(entity_ids))}/{len(entity_ids)}")
    return out


# --------------------------------------------------------------------------- #
# Step 3 -- classification
# --------------------------------------------------------------------------- #
def classify(record: dict) -> str:
    """Assign a venom family to one entity record."""
    for p in record["pfam"]:
        if p in PFAM_TO_FAMILY:
            return PFAM_TO_FAMILY[p]
    for ip in record["interpro"]:
        if ip in INTERPRO_TO_FAMILY:
            return INTERPRO_TO_FAMILY[ip]
    name = record["description"].lower()
    for fam in FAMILIES:
        for kw in fam["keywords"]:
            if kw in name:
                return fam["name"]
    return UNCLASSIFIED


# --------------------------------------------------------------------------- #
# Structural-quality metadata + curation (redundancy + quality filtering)
# --------------------------------------------------------------------------- #
def fetch_entry_quality(session: requests.Session,
                        pdb_ids: list[str]) -> dict[str, dict]:
    """Batch-fetch experimental method / resolution / R-free for each entry."""
    quality: dict[str, dict] = {}
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
            res_list = (e.get("rcsb_entry_info") or {}).get("resolution_combined") or []
            rfree = [r.get("ls_R_factor_R_free") for r in (e.get("refine") or [])
                     if r.get("ls_R_factor_R_free") is not None]
            quality[e["rcsb_id"]] = {
                "method": "; ".join(methods),
                "resolution": min(res_list) if res_list else None,
                "r_free": min(rfree) if rfree else None,
            }
        print(f"[quality] {min(i + GQL_CHUNK, len(pdb_ids))}/{len(pdb_ids)}")
    return quality


def _method_class(method: str) -> str:
    m = (method or "").upper()
    if "X-RAY" in m or "DIFFRACTION" in m:
        return "xray"
    if "ELECTRON MICROSCOPY" in m or "EM" in m:
        return "em"
    if "NMR" in m:
        return "nmr"
    return "other"


def passes_quality(q: dict, args) -> bool:
    """True if an entry meets the structural-quality cut-off for its method."""
    cls = _method_class(q.get("method", ""))
    res = q.get("resolution")
    if cls == "xray":
        return res is not None and res <= args.max_res_xray
    if cls == "em":
        return res is not None and res <= args.max_res_em
    if cls == "nmr":
        return args.keep_nmr
    return False        # unknown method -> excluded from curated set


# lower tuple sorts first == better representative
def _rep_rank(rec: dict) -> tuple:
    cls = _method_class(rec.get("method", ""))
    method_order = {"xray": 0, "em": 1, "nmr": 2}.get(cls, 3)
    res = rec.get("resolution")
    res = res if res is not None else 99.0
    rfree = rec.get("r_free")
    rfree = rfree if rfree is not None else 9.0
    return (0 if rec.get("quality_pass") else 1, method_order, res, rfree, rec["pdb_id"])


def curate(records: list[dict], args) -> None:
    """Assign quality_pass and is_representative flags in-place.

    Redundancy: entities sharing a cluster id at the chosen % identity are one
    group; the best-ranked member (see _rep_rank) is the representative.
    Entities without a cluster id (rare) are treated as their own group.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        if rec["family"] == UNCLASSIFIED and not args.include_unclassified:
            rec["is_representative"] = False
            continue
        cid = rec.get("clusters", {}).get(args.identity)
        key = f"c{cid}" if cid is not None else f"solo_{rec['entity_id']}"
        groups[key].append(rec)

    n_rep = n_curated = 0
    for members in groups.values():
        best = min(members, key=_rep_rank)
        for rec in members:
            rec["is_representative"] = rec is best
        n_rep += 1
        if best.get("quality_pass"):
            n_curated += 1
    print(f"[curate] {len(records)} entities -> {n_rep} sequence clusters "
          f"@ {args.identity}% identity; {n_curated} pass quality "
          f"(curated set size)")


def write_curated(records: list[dict], args) -> None:
    """Write curated_set.csv (representatives that pass quality) and,
    if requested, copy their structure files into data/curated/<Family>/."""
    curated = [r for r in records
               if r.get("is_representative") and r.get("quality_pass")]
    out = DATA_DIR / "curated_set.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb_id", "entity_id", "family", "description", "organism",
                    "method", "resolution", "r_free", "seq_len",
                    f"cluster{args.identity}", "ec", "pfam"])
        for r in sorted(curated, key=lambda r: (r["family"], r["pdb_id"])):
            w.writerow([r["pdb_id"], r["entity_id"], r["family"], r["description"],
                        r["organism"], r.get("method", ""), r.get("resolution", ""),
                        r.get("r_free", ""), r.get("seq_len", ""),
                        r.get("clusters", {}).get(args.identity, ""),
                        r["ec"], ";".join(r["pfam"])])
    print(f"[curate] wrote {out}  ({len(curated)} representatives)")

    if args.copy_curated:
        import shutil
        cur_dir = DATA_DIR / "curated"
        # wipe any previous curated copies so the folder matches this set exactly
        if cur_dir.exists():
            shutil.rmtree(cur_dir)
        n = 0
        for r in curated:
            dst_dir = cur_dir / r["family"]
            dst_dir.mkdir(parents=True, exist_ok=True)
            for ext in args.formats:
                src = DATA_DIR / r["family"] / f'{r["pdb_id"]}.{ext}'
                if src.exists():
                    shutil.copy2(src, dst_dir / src.name)
                    n += 1
        print(f"[curate] copied {n} files into {cur_dir}")


# --------------------------------------------------------------------------- #
# Step 4 -- download
# --------------------------------------------------------------------------- #
def download_structure(session: requests.Session, pdb_id: str, dest: Path,
                       ext: str) -> str:
    """Download one PDB structure file to *dest*.  Returns a status string."""
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    url = FILE_URL.format(pdb=pdb_id, ext=ext)
    try:
        resp = session.get(url, timeout=TIMEOUT, headers=HEADERS)
    except requests.RequestException as exc:
        return f"error:{exc.__class__.__name__}"
    if resp.status_code != 200 or not resp.content:
        return f"error:http{resp.status_code}"
    dest.write_bytes(resp.content)
    time.sleep(REQUEST_PAUSE)
    return "downloaded"


# --------------------------------------------------------------------------- #
# Disulfide extraction
# --------------------------------------------------------------------------- #
def _cif_disulfides(cif_path: Path) -> list[tuple]:
    """Return a list of disulfide bonds parsed from a mmCIF file.

    Each bond is (chain1, comp1, seq1, chain2, comp2, seq2) using author
    numbering, read from the _struct_conn category (conn_type_id == 'disulf').
    """
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    d = MMCIF2Dict(str(cif_path))

    def col(key: str) -> list:
        v = d.get(key)
        if v is None:
            return []
        return v if isinstance(v, list) else [v]

    types = col("_struct_conn.conn_type_id")
    if not types:
        return []
    c1, r1, s1 = (col("_struct_conn.ptnr1_auth_asym_id"),
                  col("_struct_conn.ptnr1_auth_comp_id"),
                  col("_struct_conn.ptnr1_auth_seq_id"))
    c2, r2, s2 = (col("_struct_conn.ptnr2_auth_asym_id"),
                  col("_struct_conn.ptnr2_auth_comp_id"),
                  col("_struct_conn.ptnr2_auth_seq_id"))
    bonds = []
    for i, t in enumerate(types):
        if t != "disulf":
            continue
        try:
            bonds.append((c1[i], r1[i], s1[i], c2[i], r2[i], s2[i]))
        except IndexError:
            continue
    return bonds


def extract_all_disulfides(entry_families: dict[str, set[str]], args) -> dict[str, int]:
    """Parse every downloaded .cif for disulfide bonds -> data/disulfides.csv.

    Returns a mapping pdb_id -> number of disulfide bonds, used to annotate the
    manifest.  mmCIF is used as the source (present for every entry).
    """
    ss_csv = DATA_DIR / "disulfides.csv"
    counts: dict[str, int] = {}
    rows: list[tuple] = []
    for pdb_id, fams in sorted(entry_families.items()):
        targets = set(fams)
        if not args.include_unclassified:
            targets.discard(UNCLASSIFIED)
        if not targets:
            continue
        family = sorted(targets)[0]
        cif_path = DATA_DIR / family / f"{pdb_id}.cif"
        if not cif_path.exists():
            continue
        try:
            bonds = _cif_disulfides(cif_path)
        except Exception as exc:                         # noqa: BLE001
            print(f"    ! disulfide parse failed for {pdb_id}: {exc}")
            continue
        counts[pdb_id] = len(bonds)
        for (ch1, cmp1, sq1, ch2, cmp2, sq2) in bonds:
            rows.append((pdb_id, family, ch1, cmp1, sq1, ch2, cmp2, sq2))
    with ss_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb_id", "family", "chain1", "res1", "seq1",
                    "chain2", "res2", "seq2"])
        w.writerows(rows)
    n_with = sum(1 for v in counts.values() if v)
    print(f"\n[disulfides] {len(rows)} bonds across {n_with} structures "
          f"-> {ss_csv}")
    return counts


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--formats", nargs="+", choices=["cif", "pdb"],
                    default=["cif", "pdb"],
                    help="structure file format(s) to download (default: cif pdb). "
                         "mmCIF is the complete/superset format; legacy PDB is "
                         "unavailable for very large structures.")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N entities (for testing)")
    ap.add_argument("--no-download", action="store_true",
                    help="build the manifest but do not download structures")
    ap.add_argument("--no-disulfides", action="store_true",
                    help="skip extraction of the disulfides.csv report")
    ap.add_argument("--include-unclassified", action="store_true",
                    help="also download entries that match no known family")
    # --- redundancy + quality filtering -----------------------------------
    ap.add_argument("--identity", type=int, default=90,
                    choices=[100, 95, 90, 70, 50, 30],
                    help="sequence-identity %% for redundancy clustering "
                         "(default 90); one representative kept per cluster")
    ap.add_argument("--max-res-xray", type=float, default=3.0,
                    help="max resolution (A) to keep an X-ray entry (default 3.0)")
    ap.add_argument("--max-res-em", type=float, default=4.0,
                    help="max resolution (A) to keep a cryo-EM entry (default 4.0)")
    ap.add_argument("--drop-nmr", dest="keep_nmr", action="store_false",
                    help="exclude NMR structures from the curated set "
                         "(default: keep them)")
    ap.add_argument("--copy-curated", action="store_true",
                    help="copy the curated representative structures into "
                         "data/curated/<Family>/")
    ap.add_argument("--no-curate", action="store_true",
                    help="skip redundancy + quality filtering entirely")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    # 1. search --------------------------------------------------------------
    entity_ids = search_serpentes_entities(session, args.limit)
    if not entity_ids:
        print("No entities found -- aborting.")
        return 1

    # 2. metadata ------------------------------------------------------------
    records = fetch_metadata(session, entity_ids)

    # 3. classify ------------------------------------------------------------
    for rec in records:
        rec["family"] = classify(rec)

    # map: pdb_id -> set of families it belongs to
    entry_families: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        entry_families[rec["pdb_id"]].add(rec["family"])

    fam_counter = Counter(rec["family"] for rec in records)
    print("\n[classify] entities per family:")
    for fam, n in fam_counter.most_common():
        print(f"    {fam:<32} {n}")

    # 3b. quality metadata + redundancy/quality curation ---------------------
    if not args.no_curate:
        pdb_ids = sorted({rec["pdb_id"] for rec in records})
        entry_quality = fetch_entry_quality(session, pdb_ids)
        for rec in records:
            q = entry_quality.get(rec["pdb_id"], {})
            rec["method"] = q.get("method", "")
            rec["resolution"] = q.get("resolution")
            rec["r_free"] = q.get("r_free")
            rec["quality_pass"] = passes_quality(q, args)
        curate(records, args)

    # 4. download ------------------------------------------------------------
    dl_stats: Counter = Counter()
    no_legacy_pdb: list[str] = []
    if not args.no_download:
        for pdb_id, fams in sorted(entry_families.items()):
            targets = set(fams)
            if not args.include_unclassified:
                targets.discard(UNCLASSIFIED)
            if not targets:
                continue
            fam_dirs = [DATA_DIR / fam for fam in sorted(targets)]
            for d in fam_dirs:
                d.mkdir(parents=True, exist_ok=True)
            # For each format: download once into the first family folder, then
            # copy the bytes into any additional folders the entry belongs to.
            for ext in args.formats:
                data_bytes: bytes | None = None
                got_error = False
                for fam_dir in fam_dirs:
                    dest = fam_dir / f"{pdb_id}.{ext}"
                    if dest.exists() and dest.stat().st_size > 0:
                        dl_stats[f"{ext}:cached"] += 1
                        if data_bytes is None:
                            data_bytes = dest.read_bytes()
                        continue
                    if data_bytes is None and not got_error:
                        status = download_structure(session, pdb_id, dest, ext)
                        dl_stats[f"{ext}:{status.split(':')[0]}"] += 1
                        if status.startswith("downloaded"):
                            data_bytes = dest.read_bytes()
                        else:
                            got_error = True
                            if ext == "pdb" and status == "error:http404":
                                no_legacy_pdb.append(pdb_id)
                            else:
                                print(f"    ! {pdb_id}.{ext}: {status}")
                    elif data_bytes is not None:
                        dest.write_bytes(data_bytes)
                        dl_stats[f"{ext}:copied"] += 1
        print(f"\n[download] {dict(dl_stats)}")
        if no_legacy_pdb:
            print(f"[download] {len(no_legacy_pdb)} entries have no legacy .pdb "
                  f"(too large) -- use their .cif: {', '.join(no_legacy_pdb[:12])}"
                  + (" ..." if len(no_legacy_pdb) > 12 else ""))

    # 5. disulfides ----------------------------------------------------------
    ss_by_entry: dict[str, int] = {}
    if not args.no_download and not args.no_disulfides:
        ss_by_entry = extract_all_disulfides(entry_families, args)

    # 6. reports -------------------------------------------------------------
    manifest = DATA_DIR / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb_id", "entity_id", "family", "description", "organism",
                    "ec", "go", "pfam", "interpro", "n_disulfides_entry",
                    "method", "resolution", "r_free", "seq_len",
                    f"cluster{args.identity}", "quality_pass", "is_representative"])
        for rec in sorted(records, key=lambda r: (r["family"], r["pdb_id"])):
            w.writerow([rec["pdb_id"], rec["entity_id"], rec["family"],
                        rec["description"], rec["organism"], rec["ec"],
                        " | ".join(rec["go"]),
                        ";".join(rec["pfam"]), ";".join(rec["interpro"]),
                        ss_by_entry.get(rec["pdb_id"], ""),
                        rec.get("method", ""), rec.get("resolution", ""),
                        rec.get("r_free", ""), rec.get("seq_len", ""),
                        rec.get("clusters", {}).get(args.identity, ""),
                        rec.get("quality_pass", ""),
                        rec.get("is_representative", "")])

    if not args.no_curate:
        write_curated(records, args)

    summary = DATA_DIR / "family_summary.csv"
    entry_fam_counter: Counter = Counter()
    for pdb_id, fams in entry_families.items():
        for fam in fams:
            entry_fam_counter[fam] += 1
    with summary.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "n_entities", "n_entries"])
        for fam in sorted(set(fam_counter) | set(entry_fam_counter)):
            w.writerow([fam, fam_counter.get(fam, 0), entry_fam_counter.get(fam, 0)])

    print(f"\n[report] wrote {manifest}")
    print(f"[report] wrote {summary}")
    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
