"""
prepare_experimental_pdbs.py
Adapter: turn the experimental PDB structures collected by
download_all_experimental.py  into the
exact input layout to feed the topology pipeline

So, per experimental entry, this script:
  1. picks ONE representative protein chain (longest amino-acid chain; ties ->
     alphabetical)
  2. writes a protein-only, single-chain PDB to its path
  3. writes that chain's C-alpha coordinates to path
  4. reads the entry's disulfide bonds from its own SSBOND records (mmCIF
     _struct_conn as fallback), keeps those internal to the chosen chain, maps
     them to 0-based C-alpha indices, and writes to path
  5. accumulates the per-protein cysteine and disulfide 
  6. writes <out>/manifest.csv.

    python prepare_experimental_pdbs.py                 # default in/out
    python prepare_experimental_pdbs.py --in  (data_experimental)
    python prepare_experimental_pdbs.py --out (experimental_data)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

CODES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODES_DIR.parent
DEFAULT_IN = PROJECT_ROOT / "PDB_computation" / "downloaded_structures"
DEFAULT_OUT = PROJECT_ROOT / "PDB_computation" / "experimental_data"

# download family key -> (short class, folder name under data_experimental/)
FAMILY_KEY_TO_CLASS = {
    "PLA2": "PLA", "3FTx": "3ft", "SVMP": "SNMP",
    "serine": "serine", "kallikrein": "kallikrein", "kunitz": "kunitz",
}
CLASS_ORDER = ["PLA", "3ft", "SNMP", "serine", "kallikrein", "kunitz"]

AA3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M', 'CSO': 'C', 'CSD': 'C', 'CME': 'C', 'OCS': 'C', 'CAS': 'C',
    'SEC': 'C', 'PCA': 'E', 'HYP': 'P', 'TPO': 'T', 'SEP': 'S', 'PTR': 'Y',
}
CYS_RESNAMES = {'CYS', 'CSO', 'CSD', 'CME', 'OCS', 'CAS', 'SEC'}


def long_path(p: Path) -> Path:
    p = Path(p)
    if sys.platform == "win32":
        s = str(p.resolve())
        if not s.startswith("\\\\?\\"):
            return Path("\\\\?\\" + s)
    return p


# PDB parsing
def parse_pdb(pdb_path: Path):
    """Parse a .pdb file.

    Returns (chains, ssbonds) where
      chains  = { chain_id: [ {resseq, icode, resname, is_aa, ca, lines}, ... ] }
                in file order, first altLoc only, model 1 only;
      ssbonds = [ (chain1, seq1, icode1, chain2, seq2, icode2), ... ]  from SSBOND.
    """
    chains = defaultdict(list)
    res_index = defaultdict(dict)
    seen_atom = defaultdict(set)
    ssbonds = []
    ended = False

    with open(long_path(pdb_path), "r", errors="replace") as fh:
        for line in fh:
            rec = line[:6]
            if rec == "SSBOND":
                # fixed columns per PDB spec
                c1, s1, i1 = line[15], line[17:21].strip(), line[21]
                c2, s2, i2 = line[29], line[31:35].strip(), line[35]
                if s1 and s2:
                    ssbonds.append((c1 if c1 != " " else "_", s1, i1,
                                    c2 if c2 != " " else "_", s2, i2))
                continue
            if rec == "ENDMDL":
                ended = True
                continue
            if ended or rec not in ("ATOM  ", "HETATM"):
                continue
            resname = line[17:20].strip()
            is_aa = resname in AA3TO1
            if rec == "HETATM" and not is_aa:
                continue
            altloc = line[16]
            atom = line[12:16].strip()
            if altloc not in (" ", "A", "1"):
                continue
            chain = line[21] if line[21] != " " else "_"
            resseq, icode = line[22:26].strip(), line[26]
            akey = (resseq, icode, atom)
            if akey in seen_atom[chain]:
                continue
            seen_atom[chain].add(akey)
            try:
                x, y, z = (float(line[30:38]), float(line[38:46]),
                           float(line[46:54]))
            except ValueError:
                continue
            rkey = (resseq, icode)
            if rkey not in res_index[chain]:
                res = {"resseq": resseq, "icode": icode, "resname": resname,
                       "is_aa": is_aa, "ca": None, "lines": []}
                chains[chain].append(res)
                res_index[chain][rkey] = res
            res = res_index[chain][rkey]
            res["lines"].append("ATOM  " + line[6:].rstrip("\n"))
            if atom == "CA":
                res["ca"] = (x, y, z)
    return chains, ssbonds


def parse_cif(cif_path: Path):
    """Fallback for entries with no legacy .pdb: parse chains + disulfides from
    mmCIF via BioPython. Same return shape as parse_pdb (icodes blank)."""
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from Bio.PDB import MMCIFParser

    chains = defaultdict(list)
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(long_path(cif_path)))
    model = next(iter(structure))                       # first model only
    for ch in model:
        cid = ch.id if ch.id.strip() else "_"
        for residue in ch:
            resname = residue.get_resname().strip()
            is_aa = resname in AA3TO1
            if not is_aa:
                continue
            ca = residue["CA"].get_coord() if "CA" in residue else None
            resseq = str(residue.id[1])
            icode = residue.id[2] if residue.id[2].strip() else " "
            lines = []
            for atom in residue:
                x, y, z = atom.get_coord()
                lines.append(
                    f"ATOM  {atom.serial_number:>5} {atom.get_name():<4}"
                    f"{resname:>3} {cid[0]}{resseq:>4}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}")
            chains[cid].append({"resseq": resseq, "icode": icode,
                                "resname": resname, "is_aa": True,
                                "ca": (tuple(ca) if ca is not None else None),
                                "lines": lines})
    # disulfides from _struct_conn
    ssbonds = []
    try:
        d = MMCIF2Dict(str(long_path(cif_path)))
        types = d.get("_struct_conn.conn_type_id")
        types = types if isinstance(types, list) else ([types] if types else [])

        def col(k):
            v = d.get(k)
            return v if isinstance(v, list) else ([v] if v is not None else [])
        c1, s1 = col("_struct_conn.ptnr1_auth_asym_id"), col("_struct_conn.ptnr1_auth_seq_id")
        c2, s2 = col("_struct_conn.ptnr2_auth_asym_id"), col("_struct_conn.ptnr2_auth_seq_id")
        for i, t in enumerate(types):
            if t == "disulf":
                try:
                    ssbonds.append((c1[i], s1[i], " ", c2[i], s2[i], " "))
                except IndexError:
                    pass
    except Exception:
        pass
    return chains, ssbonds


def pick_representative_chain(chains, target_lens=None):
    """

    Experimental entries are often complexes (a toxin bound to a large partner
    such as an acetylcholinesterase or an antibody). Choosing the longest chain
    would grab the partner, so when the venom entity's sequence length(s) are
    known (target_lens, from the download manifest), pick the amino-acid chain
    (>=4 C-alpha) whose resolved length best matches a venom length -- smallest
    relative error, ties broken by the longer chain. With no target, fall back
    to the longest chain.
    """
    cand = []
    for cid in sorted(chains.keys()):
        aa_res = [r for r in chains[cid] if r["is_aa"] and r["ca"] is not None]
        if len(aa_res) >= 4:
            cand.append((cid, aa_res))
    if not cand:
        return None, None

    targets = [t for t in (target_lens or []) if t]
    if targets:
        def rel_err(n):
            return min(abs(n - t) / t for t in targets)
        # smallest relative length error first, then longer chain
        cid, aa_res = min(cand, key=lambda c: (round(rel_err(len(c[1])), 3), -len(c[1])))
        return cid, aa_res

    cid, aa_res = max(cand, key=lambda c: len(c[1]))
    return cid, aa_res

# Main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_root", default=str(DEFAULT_IN),
                    help="download root (default: %(default)s)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="pipeline dataset root (default: %(default)s)")
    ap.add_argument("--manifest", default="manifest_experimental.csv",
                    help="manifest filename under --in (use manifest_curated.csv for "
                         "the non-redundant, quality-filtered representative set)")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out)
    manifest_in = in_root / args.manifest
    if not manifest_in.exists():
        print(f"[error] {manifest_in} not found -- run download_all_experimental.py "
              "(and curate_experimental.py for the curated manifest) first")
        return 1

    org_dir = out_root / "organized_pdb"
    ca_dir = out_root / "ca_clouds"
    ss_dir = out_root / "ss_bonds_ca"
    res_dir = out_root / "results"
    for d in (org_dir, ca_dir, ss_dir, res_dir):
        d.mkdir(parents=True, exist_ok=True)

    # plus pdb_id --venom entity seq_lens used to pick the right chain.
    entries = {}
    target_lens: "dict[str, list[int]]" = defaultdict(list)
    with manifest_in.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fam_key = row["family"]
            if fam_key not in FAMILY_KEY_TO_CLASS:
                continue
            pid = row["pdb_id"].upper()
            entries.setdefault(pid, (FAMILY_KEY_TO_CLASS[fam_key], row["folder"],
                                     fam_key, row.get("method", ""),
                                     row.get("resolution", ""),
                                     row.get("description", "")))
            try:
                sl = int(float(row.get("seq_len", "") or 0))
                if sl > 0:
                    target_lens[pid].append(sl)
            except ValueError:
                pass

    print(f"Preparing experimental dataset -> {out_root}")
    print(f"  {len(entries)} unique PDB entries from {manifest_in.name}")

    per_protein, manifest_rows = [], []
    counts = defaultdict(int)
    skipped = 0

    for pid, (cls, folder, fam_key, method, res, desc) in sorted(entries.items()):
        fam_dir = in_root / folder
        pdb_f = fam_dir / f"{pid}.pdb"
        cif_f = fam_dir / f"{pid}.cif"
        try:
            if pdb_f.exists() and pdb_f.stat().st_size > 0:
                chains, ssbonds = parse_pdb(pdb_f)
            elif cif_f.exists() and cif_f.stat().st_size > 0:
                chains, ssbonds = parse_cif(cif_f)
            else:
                skipped += 1
                continue
        except Exception as exc:                          # noqa: BLE001
            print(f"  [skip] {pid}: parse error: {exc}")
            skipped += 1
            continue

        chain_id, residues = pick_representative_chain(chains, target_lens.get(pid))
        if chain_id is None:
            skipped += 1
            continue

        (org_dir / cls).mkdir(parents=True, exist_ok=True)
        with open(long_path(org_dir / cls / f"{pid}.pdb"), "w") as fh:
            for r in residues:
                for l in r["lines"]:
                    fh.write(l + "\n")
            fh.write("END\n")

        (ca_dir / cls).mkdir(parents=True, exist_ok=True)
        idx_of = {}
        with open(long_path(ca_dir / cls / f"{pid}.csv"), "w", newline="") as fh:
            fh.write("x,y,z\n")
            for i, r in enumerate(residues):
                x, y, z = r["ca"]
                fh.write(f"{x},{y},{z}\n")
                idx_of[(r["resseq"], r["icode"])] = i
                idx_of.setdefault((r["resseq"], " "), i)   # tolerate blank icode

        seq = "".join(AA3TO1.get(r["resname"], "X") for r in residues)
        n = len(seq)
        n_cys = sum(1 for r in residues if r["resname"] in CYS_RESNAMES)

        pairs = []
        for (c1, s1, i1, c2, s2, i2) in ssbonds:
            if c1 != chain_id or c2 != chain_id:
                continue
            a = idx_of.get((s1, i1)) or idx_of.get((s1, " "))
            b = idx_of.get((s2, i2)) or idx_of.get((s2, " "))
            if a is not None and b is not None and a != b:
                pairs.append(sorted((a, b)))
        # de-dupe identical staples
        pairs = [list(p) for p in {tuple(p) for p in pairs}]

        (ss_dir / cls).mkdir(parents=True, exist_ok=True)
        with open(long_path(ss_dir / cls / f"{pid}.json"), "w") as fh:
            json.dump(dict(protein_id=pid, class_=cls, chain=chain_id,
                           n_ssbond=len(pairs), ss_matched=len(pairs) > 0,
                           ca_pairs=pairs), fh)

        per_protein.append(dict(Protein_ID=pid, Protein_Class=cls, family=fam_key,
                                chain=chain_id, chain_length=n, n_cys=n_cys,
                                cys_fraction=(n_cys / n if n else 0.0),
                                n_ssbond=len(pairs)))
        manifest_rows.append(dict(pdb_id=pid, Protein_Class=cls, family=fam_key,
                                  chain=chain_id, method=method, resolution=res,
                                  chain_length=n, n_cys=n_cys,
                                  cys_fraction=round(n_cys / n, 4) if n else 0.0,
                                  n_ssbond=len(pairs), description=desc))
        counts[cls] += 1

    _write_csv(res_dir / "cysteine_disulfide_per_protein.csv", per_protein,
               ["Protein_ID", "Protein_Class", "family", "chain",
                "chain_length", "n_cys", "cys_fraction", "n_ssbond"])
    _write_csv(out_root / "manifest.csv", manifest_rows,
               ["pdb_id", "Protein_Class", "family", "chain", "method",
                "resolution", "chain_length", "n_cys", "cys_fraction",
                "n_ssbond", "description"])

    print("\n[done] proteins prepared per class:")
    for cls in CLASS_ORDER:
        print(f"    {cls:<12} {counts.get(cls, 0)}")
    print(f"  skipped (no file / no chain): {skipped}")
    if counts.get("kallikrein", 0) == 0:
        print("  note: no experimental snake-venom kallikrein structures in the PDB.")
    print(f"\n  organized_pdb -> {org_dir}")
    print(f"  ca_clouds     -> {ca_dir}")
    print(f"  ss_bonds_ca   -> {ss_dir}")
    print(f"  tables        -> {res_dir}")
    return 0


def _write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[report] wrote {path}  ({len(rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())
