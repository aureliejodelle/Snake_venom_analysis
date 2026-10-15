# build_disulfide_data.sh

# Generate the full disulfide_data/ processed set (standard AND augmented C-alpha)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODES="$ROOT/scripts"
DD="$ROOT/disulfide_data"
DM="$ROOT/results/distance_matrices"
IDMAP="$DD/id_map.csv"

export PYTHONIOENCODING=utf-8
export ID_MAP="$IDMAP"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-4}"
export PIPELINE_WORKERS="${PIPELINE_WORKERS:-8}"
VARIANTS="${VARIANTS:-standard augmented}"

[ -f "$IDMAP" ] || python "$CODES/mapping/build_disulfide_id_map.py"

for V in $VARIANTS; do
  echo "==================== VARIANT: $V ===================="
  PH="$DD/processed_data/persistent_homology/$V"
  HG="$DD/processed_data/hypergraphs/$V"
  RC="$DD/processed_data/ricci_curvature/$V"
  FE="$DD/features/$V"
  mkdir -p "$PH" "$HG" "$RC" "$FE"

  echo "[1/4] full PH (Ripserer, Julia)"
  AUG_DM_ROOT="$DM/$V" PH_OUT_ROOT="$PH" julia "$CODES/full_ph_from_dm.jl"

  echo "[2/4] hypergraphs"
  SNAKE_PH_DIR="$PH" SNAKE_HG_DIR="$HG" python "$CODES/computation/compute_hypergraphs.py"

  echo "[3/4] Forman-Ricci curvature"
  SNAKE_HG_DIR="$HG" SNAKE_RC_DIR="$RC" python "$CODES/computation/compute_curvature.py"

  echo "[4/4] features -> CSV"
  SNAKE_PH_DIR="$PH" SNAKE_HG_DIR="$HG" SNAKE_RC_DIR="$RC" \
    SNAKE_FEATURES_DIR="$FE" python "$CODES/computation/extract_features.py"
done

echo "ALL DONE -> $DD"
