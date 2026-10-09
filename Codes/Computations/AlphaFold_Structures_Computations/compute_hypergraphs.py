from __future__ import annotations

"""
Compute Hypergraphs from Persistent Homology representatives


For each protein:
  - Each H1 cycle = one hyperedge
  - Nodes = CA atom indices appearing in the cycle edges

Output per protein (under config.HG_DIR/<class>/):
  - hyperedge_map
  - summary: protein-level stats

"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))  
from config import config, long_path
import os

def get_selected_classes():
    """Read PIPELINE_CLASSES env var. Returns list of classes or None (= all)."""
    raw = os.environ.get("PIPELINE_CLASSES", "").strip()
    return [c.strip() for c in raw.split(",") if c.strip()] if raw else None

# CONFIG
NUM_WORKERS = 8   # parallel threads

# CORE: representatives to hypergraph


def edges_to_hyperedges(
    representatives: List[List[List[int]]],
    protein_id: str,
) -> Tuple[List[Set[int]], List[int], List[int]]:
    """
    Convert PH cycle representatives to hyperedges

    Each hyperedge = set of all node indices appearing in the cycle.

    Returns:
        hyperedges     : list of sets of node indices
        all_nodes      : sorted list of unique node indices
        hyperedge_ids  : 1-based hyperedges
    """
    if not representatives:
        return [], [], []

    all_nodes: Set[int] = set()
    hyperedges: List[Set[int]] = []

    for cycle in representatives:
        hyperedge: Set[int] = set()
        for edge in cycle:
            for node in edge:
                if isinstance(node, (int, float)) and node > 0:
                    n = int(node)
                    all_nodes.add(n)
                    hyperedge.add(n)
        if hyperedge:
            hyperedges.append(hyperedge)

    sorted_nodes = sorted(all_nodes)
    hyperedge_ids = list(range(1, len(hyperedges) + 1))
    return hyperedges, sorted_nodes, hyperedge_ids


def build_incidence_matrix(
    hyperedges: List[Set[int]],
    all_nodes: List[int],
    hyperedge_ids: List[int],
) -> Tuple[None, Dict[int, List[int]]]:
    """
    Build sparse incidence matrix (nodes * hyperedges).

    Returns:
        edge_map   : {hyperedge_id: [node_indices]}
    """
    if not hyperedges or not all_nodes:
        n = len(all_nodes)
        return None, {}

    node_to_idx = {node: i for i, node in enumerate(all_nodes)}

    rows, cols = [], []
    for j, hyperedge in enumerate(hyperedges):
        for node in hyperedge:
            if node in node_to_idx:
                rows.append(node_to_idx[node])
                cols.append(j)

    n_nodes = len(all_nodes)
    n_edges = len(hyperedges)
    # Build hyperedge_id to list of actual node indices
    edge_map: Dict[int, List[int]] = {}
    for j, hid in enumerate(hyperedge_ids):
        edge_map[hid] = sorted(hyperedges[j])

    return None, edge_map

# PROCESS ONE PROTEIN


def process_protein(
    json_path: Path,
    class_name: str,
    dirs: dict,
) -> Dict:
    """
    Read representatives JSON, build hypergraph
    """
    protein_id = json_path.stem

    #Skip if already done 
    if long_path(dirs["hyperedge_map"] / f"{protein_id}.json").exists() and long_path(dirs["summary"]  / f"{protein_id}.json").exists():
        return {"protein_id": protein_id, "status": "skipped",
                "n_nodes": 0, "n_hyperedges": 0}

    try:
        with open(long_path(json_path)) as f:
            data = json.load(f)

        representatives = data.get("representatives", [])

        if not representatives:
            return {
                "protein_id": protein_id,
                "status":     "empty",
                "n_nodes":    0,
                "n_hyperedges": 0,
            }

        # Build hypergraph
        hyperedges, all_nodes, hyperedge_ids = edges_to_hyperedges(
            representatives, protein_id
        )
        _, edge_map = build_incidence_matrix(hyperedges, all_nodes, hyperedge_ids)

        
        with open(long_path(dirs["hyperedge_map"] / f"{protein_id}.json"), "w") as f:
            json.dump({str(k): v for k, v in edge_map.items()}, f, indent=2)

        
        summary = {
            "protein_id":   protein_id,
            "class":        class_name,
            "n_nodes":      len(all_nodes),
            "n_hyperedges": len(hyperedges),
        }
        with open(long_path(dirs["summary"] / f"{protein_id}.json"), "w") as f:
            json.dump(summary, f, indent=2)

        return {"protein_id": protein_id, "status": "ok", **summary}

    except Exception as e:
        return {
            "protein_id": protein_id,
            "status":     "failed",
            "error":      str(e),
            "n_nodes":    0,
            "n_hyperedges": 0,
        }

# PROCESS ONE CLASS

def process_class(class_name: str) -> Dict:
    """Process all representative JSON files for a protein class."""

    reps_dir = config.PH_DIR / class_name / "representatives"
    if not reps_dir.exists():
        print(f" Representatives dir not found: {reps_dir}")
        return {"total": 0, "ok": 0, "empty": 0, "failed": 0}

    json_files = sorted(reps_dir.glob("*.json"))
    if not json_files:
        print(f" No JSON files in {reps_dir}")
        return {"total": 0, "ok": 0, "empty": 0, "failed": 0}

    # Output subfolders for this class
    base_dir = config.HG_DIR / class_name
    dirs = {
        "hyperedge_map": base_dir / "hyperedge_map",
        "summary":       base_dir / "summary",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    stats = {"total": len(json_files), "ok": 0, "empty": 0, "failed": 0, "skipped": 0}
    lock  = threading.Lock()

    desc = f"  {class_name[:38]:<38}"
    with tqdm(total=len(json_files), desc=desc, unit="prot",
              bar_format="{l_bar}{bar:25}{r_bar}") as pbar:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = {
                ex.submit(process_protein, jf, class_name, dirs): jf
                for jf in json_files
            }
            for future in as_completed(futures):
                result = future.result()
                with lock:
                    stats[result["status"]] += 1
                pbar.update(1)
                pbar.set_postfix(
                    ok=stats["ok"], skip=stats["skipped"],
                    empty=stats["empty"], fail=stats["failed"],
                    refresh=False,
                )

    tqdm.write(
        f" {class_name}: "
        f"ok={stats['ok']} skipped={stats['skipped']} "
        f"empty={stats['empty']} failed={stats['failed']} "
        f"total={stats['total']}"
    )
    return stats

# MAIN

def main():
    print("=" * 60)
    print("SCRIPT 5: HYPERGRAPH COMPUTATION")
    print("=" * 60)
    print(f"Start  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input  : {config.PH_DIR}")
    print(f"Output : {config.HG_DIR}")
    print()

    if not config.PH_DIR.exists():
        print(f"PH directory not found: {config.PH_DIR}")
        print("   Run step 4 (persistent homology) first.")
        sys.exit(1)

    # Auto-detect classes and filter if classes was passed
    all_classes = sorted([
        d.name for d in config.PH_DIR.iterdir()
        if d.is_dir() and (d / "representatives").exists()
    ])

    if not all_classes:
        print(f"No class directories with representatives found in {config.PH_DIR}")
        sys.exit(1)

    selected = get_selected_classes()
    if selected:
        # Translate original class names (e.g. 'K+3(1)') to folder names (e.g. 'Kplus31')
        selected_folders = config.resolve_classes(selected)
        missing = [s for s, f in zip(selected, selected_folders) if f not in all_classes]
        if missing:
            print(f"These classes have no PH data yet: {', '.join(missing)}")
        classes = [c for c in all_classes if c in selected_folders]
        print(f"Filtering to selected classes: {', '.join(classes)}")
    else:
        classes = all_classes

    if not classes:
        print("No matching classes to process.")
        sys.exit(1)

    print(f"Processing {len(classes)} class(es): {', '.join(classes)}\n")

    overall = {"total": 0, "ok": 0, "empty": 0, "failed": 0}

    for class_name in classes:
        print(f"\n{'─' * 60}")
        print(f"CLASS: {class_name}")
        print("─" * 60)
        stats = process_class(class_name)
        for k in overall:
            overall[k] += stats[k]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Classes    : {len(classes)}")
    print(f"  Total      : {overall['total']}")
    print(f"  Success    : {overall['ok']}")
    print(f"  Empty (0 cycles) : {overall['empty']}")
    print(f"  Failed     : {overall['failed']}")
    print(f"\n Done. End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
