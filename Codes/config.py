"""
config.py — Shared configuration for the Snake Venom PDB pipeline.
===================================================================
Single source of truth for all paths, constants, and helpers used
across every pipeline step.

Pipeline steps and their directories
--------------------------------------
  Step 1  organize_pdb        colabfold_output_results/ → data/organized_pdb/<type>/
  Step 2  clean_data          data/organized_pdb/<type>/ → data/clean_data/<type>/
  Step 3  persistent_homology data/clean_data/<type>/    → data/persistent_homology/<type>/
  Step 4  hypergraphs         data/persistent_homology/  → data/hypergraphs/<type>/
  Step 5  curvature           data/hypergraphs/          → data/ricci_curvature/<type>/

All paths are anchored to the directory containing this file,
so scripts work regardless of where they are invoked from.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional


def long_path(p: "Path | str") -> Path:
    """On Windows, prefix a path with ``\\\\?\\`` to bypass the 260-char
    MAX_PATH limit so ``open()``/``os.path.exists`` can reach long filenames.
    No-op on other platforms."""
    p = Path(p)
    if sys.platform == "win32":
        s = str(p.resolve())
        if not s.startswith("\\\\?\\"):
            return Path("\\\\?\\" + s)
    return p

# ---------------------------------------------------------------------------
# Anchor — absolute path of the folder containing this file
# e.g. /home/jupyter-akemme/colabfold_databases/codes/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Data root — one level up from codes/, inside colabfold_databases/
# e.g. /home/jupyter-akemme/colabfold_databases/snake_venom_data/
# Change _DATA if you want outputs somewhere else.
# ---------------------------------------------------------------------------
_DATA = _HERE.parent


# ===========================================================================
# Dataset profiles — one key switches EVERY path between the AlphaFold and PDB
# computations, so no script hardcodes a dataset location.
#
#   SNAKE_DATASET=alphafold   (predicted structures; split main_data/disulfide_data/data)
#   SNAKE_DATASET=pdb         (experimental structures; unified experimental_data/)
#
# A profile abstracts BOTH the location and the layout asymmetry (the AlphaFold
# all-atom set lives in main_data/, its Cα disulfide set in disulfide_data/,
# while the PDB set keeps everything under experimental_data/). Precedence:
#   explicit SNAKE_<X> env var  >  active profile  >  legacy default
# so an unset SNAKE_DATASET reproduces the historical behaviour exactly.
# ===========================================================================
# Physical computation roots (data lives here; scripts live in scripts/).
def _root(new: Path, old: Path) -> Path:
    """Data/<dataset> when it exists, else the earlier <dataset>_computation folder."""
    return new if new.exists() or not old.exists() else old


_AF  = _root(_DATA / "Data" / "AlphaFold_data", _DATA / "AlphaFold_computation")
_PDB = _root(_DATA / "Data" / "PDB_data", _DATA / "PDB_computation")


def _folder_label(folder: Path, default: str) -> str:
    """Read the per-folder dataset.json ('this folder's config') for its label."""
    f = folder / "dataset.json"
    try:
        import json
        return json.loads(f.read_text()).get("bfactor_label", default)
    except Exception:
        return default


def _dataset_profile(ds: str) -> dict:
    ds = (ds or "").strip().lower()
    if ds in ("pdb", "experimental", "exp"):
        b = _PDB / "experimental_data"
        m, d = b / "main_data", b / "disulfide_data"

        def pick(new: Path, old: Path) -> Path:
            """New layout when present, else the folder before it was reorganised."""
            return new if new.exists() or not old.exists() else old

        return {
            "ORGANIZED":   b / "organized_pdb",
            "CLEAN":       pick(m / "clean_data", b / "clean_data"),
            "PH":          pick(m / "processed_data" / "persistent_homology",
                                b / "processed_data" / "persistent_homology"),
            "HG":          pick(m / "processed_data" / "hypergraphs",
                                b / "processed_data" / "hypergraphs"),
            "RC":          pick(m / "processed_data" / "ricci_curvature",
                                b / "processed_data" / "ricci_curvature"),
            "FEATURES":    pick(m / "features", b / "features"),
            "RESULTS":     m / "results",
            "HIGH_IMPACT": m / "results",
            "DS_RESULTS":  d / "results",
            "PDB_DATA":    b / "organized_pdb",
            "CA":          pick(d / "ca_clouds", b / "ca_clouds"),
            "SS":          pick(d / "ss_bonds", b / "ss_bonds_ca"),
            "H1BARS":      pick(d / "h1_bars_ca", b / "h1_bars_ca"),
            "BFACTOR_LABEL": _folder_label(_PDB, "B-factor"),
        }
    if ds in ("alphafold", "af", "predicted", "colabfold"):
        return {
            "ORGANIZED":   _AF / "organized_pdb",
            "CLEAN":       _AF / "main_data" / "clean_data",
            "PH":          _AF / "main_data" / "processed_data" / "persistent_homology",
            "HG":          _AF / "main_data" / "processed_data" / "hypergraphs",
            "RC":          _AF / "main_data" / "processed_data" / "ricci_curvature",
            "FEATURES":    _AF / "main_data" / "features",
            "RESULTS":     _AF / "main_data" / "results",
            "HIGH_IMPACT": _AF / "main_data" / "results",
            "DS_RESULTS":  _AF / "disulfide_data" / "results",
            "PDB_DATA":    _AF / "data",
            "CA":          _AF / "disulfide_data" / "ca_clouds",
            "SS":          _AF / "disulfide_data" / "ss_bonds",
            "H1BARS":      _AF / "disulfide_data" / "h1_bars_ca",
            "BFACTOR_LABEL": _folder_label(_AF, "pLDDT"),
        }
    return {}   # no/unknown key -> legacy defaults preserved


class Config:
    """
    Configuration for the snake venom pipeline.
    Uses a plain class (not @dataclass) to avoid Python 3.9 mutable-default
    restrictions — all paths are set as normal attributes in __init__.

    Directory layout:

        colabfold_databases/
        ├── codes/                          ← _HERE  (this file lives here)
        │   ├── config.py
        │   ├── run_snake_venom_pipeline.py
        │   ├── SnakeVenom.csv
        │   └── colabfold_output_results/   ← raw ColabFold PDB output
        └── snake_venom_data/               ← _DATA  (all pipeline outputs)
            ├── organized_pdb/<type>/       ← Step 1
            ├── clean_data/<type>/          ← Step 2
            ├── persistent_homology/<type>/ ← Step 3
            ├── hypergraphs/<type>/         ← Step 4
            ├── ricci_curvature/<type>/     ← Step 5
            ├── logs/
            └── results/
    """

    def __init__(self) -> None:

        # ------------------------------------------------------------------ #
        # Step 0 — raw inputs (never created by the pipeline)                 #
        # ------------------------------------------------------------------ #
        self.COLABFOLD_DIR:   Path = _HERE / "colabfold_output_results"
        self.METADATA_CSV:    Path = _HERE / "SnakeVenom.csv"
        self.UNMATCHED_REPORT: Path = _HERE / "unmatched_report.csv"

        # ------------------------------------------------------------------ #
        # Step 1 — organize_pdb  (set profile-aware below, once _dir exists)  #
        # ------------------------------------------------------------------ #

        # Optional env overrides so the SAME code can process a parallel
        # dataset (e.g. disulfide_data) without editing paths. Set
        # SNAKE_CLEAN_DIR / SNAKE_PH_DIR / SNAKE_HG_DIR / SNAKE_RC_DIR /
        # SNAKE_FEATURES_DIR (the Julia PH step already reads PH_INPUT_ROOT /
        # PH_OUTPUT_ROOT for the same reason).
        # Active dataset profile (see _dataset_profile). Empty -> legacy defaults.
        self.dataset: str = os.environ.get("SNAKE_DATASET", "").strip().lower()
        _prof = _dataset_profile(self.dataset)

        def _dir(env_key: str, default: Path, prof_key: str | None = None) -> Path:
            """Precedence: explicit env var > active dataset profile > legacy default."""
            v = os.environ.get(env_key)
            if v:
                return Path(v)
            if prof_key and _prof.get(prof_key):
                return Path(_prof[prof_key])
            return default

        # ------------------------------------------------------------------ #
        # Step 2 — clean_data                                                 #
        # ------------------------------------------------------------------ #
        self.CLEAN_DIR: Path = _dir("SNAKE_CLEAN_DIR", _DATA / "clean_data", "CLEAN")

        # ------------------------------------------------------------------ #
        # Step 3 — persistent homology (Julia / Ripserer)                    #
        # ------------------------------------------------------------------ #
        self.PH_DIR: Path = _dir("SNAKE_PH_DIR", _DATA / "persistent_homology", "PH")

        self.PH_SUBDIR_PH:       str = "PH_1"
        self.PH_SUBDIR_REPS:     str = "representatives"
        self.PH_SUBDIR_BARCODES: str = "barcodes"

        self.JULIA_BIN:     str = os.environ.get("JULIA_BIN", "julia")
        self.JULIA_THREADS: int = int(os.environ.get("JULIA_THREADS", "4"))

        # ------------------------------------------------------------------
        # Step 4 — hypergraphs                                                #
        # ------------------------------------------------------------------ #
        self.HG_DIR: Path = _dir("SNAKE_HG_DIR", _DATA / "hypergraphs", "HG")

        # ------------------------------------------------------------------ #
        # Step 5 - Forman-Ricci curvature  F(e) = 2|e| - D                  #
        # ------------------------------------------------------------------ #
        self.RC_DIR: Path = _dir("SNAKE_RC_DIR", _DATA / "ricci_curvature", "RC")

        # step 6 - Features to be analysed
        #---------------------------------------------------------------------#
        self.VIZ_FEATURES_DIR: Path = _dir("SNAKE_FEATURES_DIR", _DATA / "features", "FEATURES")

        # step 7 - Cross-class comparison / statistical test outputs
        #---------------------------------------------------------------------#
        self.VIZ_COMPARE_DIR: Path = _dir("SNAKE_RESULTS_DIR", _DATA / "results", "RESULTS")

        # Organized per-class PDB directory the analysis scripts read (sequence
        # -> pI / cysteine / chain length / B-factor). Profile- and env-driven.
        self.PDB_DATA_DIR: Path = _dir("SNAKE_DATA_DIR", _DATA / "data", "PDB_DATA")

        # Step 1 output dir (profile-aware).
        self.ORGANIZED_DIR = _dir("SNAKE_ORGANIZED_DIR", _DATA / "organized_pdb", "ORGANIZED")

        # Disulfide (Cα) track locations, profile-driven.
        self.CA_CLOUDS_DIR: Path = _dir("SNAKE_CA_DIR", _DATA / "ca_clouds", "CA")
        self.SS_BONDS_DIR:  Path = _dir("SNAKE_SS_DIR", _DATA / "ss_bonds_ca", "SS")
        self.H1BARS_DIR:    Path = _dir("SNAKE_H1BARS_DIR", _DATA / "h1_bars_ca", "H1BARS")

        # B-factor column label: 'pLDDT' for AlphaFold models, 'B-factor' for
        # experimental structures. Profile-driven; SNAKE_BFACTOR_LABEL overrides.
        self.BFACTOR_LABEL: str = os.environ.get(
            "SNAKE_BFACTOR_LABEL", _prof.get("BFACTOR_LABEL", "pLDDT"))

        # ------------------------------------------------------------------ #
        # Misc                                                                 #
        # ------------------------------------------------------------------ #
        self.LOGS_DIR:    Path = _DATA / "logs"
        self.RESULTS_DIR: Path = _dir("SNAKE_RESULTS_DIR", _DATA / "results", "RESULTS")

        # All-atom structure-vs-topology analyses (formerly results/high_impact/)
        # now land in main_data/results/ so nothing lives in a separate legacy
        # folder. Override with SNAKE_HIGH_IMPACT_DIR.
        self.HIGH_IMPACT_DIR: Path = _dir("SNAKE_HIGH_IMPACT_DIR",
                                          _AF / "main_data" / "results", "HIGH_IMPACT")
        # Disulfide-track results: standard/, augmented/ and the disulfide-specific figures.
        self.DS_RESULTS_DIR: Path = _dir("SNAKE_DS_RESULTS_DIR", _DATA / "results" / "disulfide",
                                         "DS_RESULTS")
        self.NUM_WORKERS: int  = int(os.environ.get("PIPELINE_WORKERS", "8"))

        # Create all output directories on import
        self._makedirs()

    # ---------------------------------------------------------------------- #
    # Internal                                                                 #
    # ---------------------------------------------------------------------- #

    def _makedirs(self) -> None:
        """Create only the always-needed output dirs. The per-step processing
        dirs (clean_data, persistent_homology, hypergraphs, ricci_curvature,
        organized_pdb) are NOT created here — each pipeline step makes its own
        output dir on demand — so merely importing config no longer litters the
        repo root with empty folders (esp. now that real data lives under
        main_data/ and disulfide_data/)."""
        for d in [self.LOGS_DIR, self.RESULTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------- #
    # Helpers                                                                  #
    # ---------------------------------------------------------------------- #

    def available_types(self, root: Optional[Path] = None) -> List[str]:
        """Return sorted list of type folder names under *root*.
        Defaults to ORGANIZED_DIR (populated after step 1)."""
        root = root or self.ORGANIZED_DIR
        if not root.exists():
            return []
        return sorted(d.name for d in root.iterdir() if d.is_dir())

    def ph_reps_dir(self, type_name: str) -> Path:
        """Path to the representatives sub-folder for a given type."""
        return self.PH_DIR / type_name / self.PH_SUBDIR_REPS

    def hg_map_dir(self, type_name: str) -> Path:
        """Path to the hyperedge_map sub-folder for a given type."""
        return self.HG_DIR / type_name / "hyperedge_map"

    def resolve_classes(self, selected: list) -> list:
        """Pass-through — snake venom type names are plain strings."""
        return selected

    def validate_step(self, step: str) -> bool:
        """Check whether required inputs for *step* exist.

        step: 'organize_pdb' | 'clean_data' | 'persistent_homology'
              | 'hypergraphs' | 'curvature'
        """
        checks: dict = {
            "organize_pdb":        [self.COLABFOLD_DIR, self.METADATA_CSV],
            "clean_data":          [self.ORGANIZED_DIR],
            "persistent_homology": [self.CLEAN_DIR],
            "hypergraphs":         [self.PH_DIR],
            "curvature":           [self.HG_DIR],
        }
        missing = [p for p in checks.get(step, []) if not p.exists()]
        for p in missing:
            print(f"  [CONFIG] Missing required input: {p}")
        return len(missing) == 0

    def summary(self) -> str:
        """Return a formatted summary of the current configuration."""
        lines = [
            "Snake Venom Pipeline — Configuration",
            "=" * 50,
            f"  ColabFold input : {self.COLABFOLD_DIR}",
            f"  Metadata CSV    : {self.METADATA_CSV}",
            f"  Unmatched rpt   : {self.UNMATCHED_REPORT}",
            "",
            f"  Step 1 output   : {self.ORGANIZED_DIR}",
            f"  Step 2 output   : {self.CLEAN_DIR}",
            f"  Step 3 output   : {self.PH_DIR}",
            f"  Step 4 output   : {self.HG_DIR}",
            f"  Step 5 output   : {self.RC_DIR}",
            "",
            f"  Julia binary    : {self.JULIA_BIN}",
            f"  Julia threads   : {self.JULIA_THREADS}",
            f"  Python workers  : {self.NUM_WORKERS}",
            f"  Logs            : {self.LOGS_DIR}",
            f"  Results         : {self.RESULTS_DIR}",
            "=" * 50,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
config = Config()
