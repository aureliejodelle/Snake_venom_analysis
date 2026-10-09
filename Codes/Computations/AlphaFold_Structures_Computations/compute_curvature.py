from __future__ import annotations
"""
Compute Forman-Ricci Curvature
Computes the raw Forman-Ricci curvature for each hyperedge in each
protein's hypergraph.

Measure

  F(e) = 2|e| - D

where |e| = hyperedge size (number of nodes),
      D   = sum of node degrees across all nodes in the hyperedge.

Skips proteins whose output file already exists (incremental runs).
"""

import sys
import json
import os
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from config import config, long_path


def get_selected_classes():
    """Read PIPELINE CLASSES. Returns list of classes or None (= all)."""
    raw = os.environ.get("PIPELINE_CLASSES", "").strip()
    return [c.strip() for c in raw.split(",") if c.strip()] if raw else None

# CONFIG

NUM_WORKERS = 8

# Rebuild incidence matrix from hyperedge map

def load_hyperedge_map(hg_json: Path) -> Dict[int, List[int]]:
    """Load hyperedge map"""
    with open(long_path(hg_json)) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def build_incidence_matrix(
    edge_map: Dict[int, List[int]],
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Build dense incidence matrix from edges map.

    Returns:
        matrix    : np.ndarray shape (number of nodes, number of edges)
        all_nodes : sorted list of unique node indices
        edge_ids  : ordered list of hyperedge IDs (columns)
    """
    if not edge_map:
        return np.zeros((0, 0), dtype=np.float32), [], []

    all_nodes = sorted({n for nodes in edge_map.values() for n in nodes})
    edge_ids  = sorted(edge_map.keys())
    node_idx  = {n: i for i, n in enumerate(all_nodes)}

    mat = np.zeros((len(all_nodes), len(edge_ids)), dtype=np.float32)
    for j, eid in enumerate(edge_ids):
        for node in edge_map[eid]:
            if node in node_idx:
                mat[node_idx[node], j] = 1.0

    return mat, all_nodes, edge_ids

# CORE: Forman-Ricci curvature

def compute_curvature(
    matrix:   np.ndarray,
    edge_ids: List[int],
) -> Dict[str, float]:
    
    result = {}
    for j, eid in enumerate(edge_ids):
        nodes_in_e   = np.where(matrix[:, j] == 1)[0]
        edge_size    = len(nodes_in_e)
        node_degrees = [int(np.sum(matrix[k, :])) for k in nodes_in_e]
        D            = sum(node_degrees)
        result[str(eid)] = float(2 * edge_size - D)
    return result


# PROCESS ONE PROTEIN

def process_protein(
    protein_id: str,
    hg_json:    Path,
    out_dir:    Path,
) -> Dict:
    """
    Compute Forman-Ricci curvature for one protein and save output.
    """
    out_file = out_dir / f"{protein_id}.json"

    # Skip if already done 
    if long_path(out_file).exists():
        return {"protein_id": protein_id, "status": "skipped", "n_hyperedges": 0}

    try:
        edge_map = load_hyperedge_map(hg_json)
        if not edge_map:
            return {"protein_id": protein_id, "status": "empty", "n_hyperedges": 0}

        matrix, _, edge_ids = build_incidence_matrix(edge_map)
        curvature = compute_curvature(matrix, edge_ids)

        with open(out_file, "w") as f:
            json.dump(curvature, f, indent=2)

        return {"protein_id": protein_id, "status": "ok", "n_hyperedges": len(edge_ids)}

    except Exception as e:
        return {"protein_id": protein_id, "status": "failed",
                "error": str(e), "n_hyperedges": 0}


# PROCESS ONE CLASS

def process_class(class_name: str) -> Dict:
    """Process all proteins in a class."""

    hg_dir = config.HG_DIR / class_name / "hyperedge_map"
    if not hg_dir.exists():
        print(f"Hyperedge map dir not found: {hg_dir}")
        return {"total": 0, "ok": 0, "empty": 0, "failed": 0, "skipped": 0}

    proteins = sorted(f.stem for f in hg_dir.glob("*.json"))
    if not proteins:
        print(f"No JSON files in {hg_dir}")
        return {"total": 0, "ok": 0, "empty": 0, "failed": 0, "skipped": 0}

    out_dir = config.RC_DIR / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": len(proteins), "ok": 0, "empty": 0, "failed": 0, "skipped": 0}
    lock  = threading.Lock()

    desc = f"  {class_name[:38]:<38}"
    with tqdm(total=len(proteins), desc=desc, unit="prot",
              bar_format="{l_bar}{bar:25}{r_bar}") as pbar:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = {
                ex.submit(process_protein,
                          pid, hg_dir / f"{pid}.json", out_dir): pid
                for pid in proteins
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
        f"ok={stats['ok']}  skipped={stats['skipped']}  "
        f"empty={stats['empty']}  failed={stats['failed']}  "
        f"total={stats['total']}"
    )
    return stats


# MAIN

def main():
    print("=" * 60)
    print("=" * 60)
    print(f"Start  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input  : {config.HG_DIR}  (hyperedge maps)")
    print(f"Output : {config.RC_DIR}")
    print()

    if not config.HG_DIR.exists():
        print(f"Required directory not found: {config.HG_DIR}")
        print("   Run step 4 (hypergraphs) first.")
        sys.exit(1)

    all_classes = sorted([
        d.name for d in config.HG_DIR.iterdir()
        if d.is_dir() and (d / "hyperedge_map").exists()
    ])

    if not all_classes:
        print(f"No class directories found in {config.HG_DIR}")
        sys.exit(1)

    selected = get_selected_classes()
    if selected:
        selected_folders = config.resolve_classes(selected)
        missing = [s for s, f in zip(selected, selected_folders) if f not in all_classes]
        if missing:
            print(f"These classes have no hypergraph data: {', '.join(missing)}")
        classes = [c for c in all_classes if c in selected_folders]
        print(f"Filtering to selected classes: {', '.join(classes)}")
    else:
        classes = all_classes

    if not classes:
        print("No matching classes to process.")
        sys.exit(1)

    print(f"Processing {len(classes)} class(es): {', '.join(classes)}\n")

    overall = {"total": 0, "ok": 0, "empty": 0, "failed": 0, "skipped": 0}

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
    print(f"  Classes   : {len(classes)}")
    print(f"  Total     : {overall['total']}")
    print(f"  Success   : {overall['ok']}")
    print(f"  Skipped   : {overall['skipped']}  (already existed)")
    print(f"  Empty     : {overall['empty']}")
    print(f"  Failed    : {overall['failed']}")
    print(f"\n Done. End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
