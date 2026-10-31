#!/usr/bin/env bash
# One-shot runner for the three protocols, on the node that holds the archive.
#
# It builds a minimal repository-shaped directory out of symlinks, so the scripts
# find what they expect without copying anything large:
#
#   $WORK/src/loop   -> the experiment code   (exp_core, agent, ontology)
#   $WORK/data/runs  -> the run artefacts     (code_*.py, meta_*.json, ...)
#   $WORK/server     -> these three scripts
#
# Usage
#   bash run_on_server.sh                # all three, year 2025 for decomp/blind
#   YEAR=2024 bash run_on_server.sh
#   ONLY=decompose bash run_on_server.sh
#
# Override anything with environment variables:
#   PY, CODE_DIR, RUNS_DIR, WORK, WIND_DATA_ROOT, OUT
set -u

PY=${PY:-$HOME/anaconda3/envs/llm_power_ts/bin/python}
[ -x "$PY" ] || PY=$(command -v python3)
CODE_DIR=${CODE_DIR:-$HOME/hzh_RSI/03_code}
RUNS_DIR=${RUNS_DIR:-$HOME/hzh_RSI/04_experiments/P3_knowledge_retention/results}
WORK=${WORK:-$HOME/p3_protocol}
OUT=${OUT:-$WORK/results}
YEAR=${YEAR:-2025}
ONLY=${ONLY:-all}

export WIND_DATA_ROOT=${WIND_DATA_ROOT:-$HOME/0NWXM}
export P3_RESULTS=${P3_RESULTS:-$RUNS_DIR}

die() { echo "[fatal] $*" >&2; exit 1; }
[ -d "$CODE_DIR" ] || die "CODE_DIR not found: $CODE_DIR"
[ -d "$RUNS_DIR" ] || die "RUNS_DIR not found: $RUNS_DIR"
[ -d "$WIND_DATA_ROOT" ] || die "WIND_DATA_ROOT not found: $WIND_DATA_ROOT (set it explicitly)"

mkdir -p "$WORK/src" "$WORK/data" "$OUT"
ln -sfn "$CODE_DIR" "$WORK/src/loop"
ln -sfn "$RUNS_DIR" "$WORK/data/runs"
[ -d "$WORK/server" ] || die "copy this repo's server/ to $WORK/server first"

echo "[env] python      $PY"
echo "[env] code        $CODE_DIR"
echo "[env] runs        $RUNS_DIR"
echo "[env] data root   $WIND_DATA_ROOT"
echo "[env] output      $OUT"
echo "[env] year        $YEAR"
echo

run() {
  local name=$1; shift
  echo "=================================================================="
  echo "[run] $name  $(date '+%F %T')"
  echo "=================================================================="
  ( cd "$WORK" && "$PY" "server/$name.py" "$@" ) 2>&1 | tee "$OUT/${name}_${YEAR}.log"
  local rc=${PIPESTATUS[0]}
  echo "[run] $name exit=$rc  $(date '+%F %T')"
  echo
  return $rc
}

rc=0
if [ "$ONLY" = all ] || [ "$ONLY" = decompose ]; then
  run decompose_floor --year "$YEAR" --out "$OUT" || rc=1
fi
if [ "$ONLY" = all ] || [ "$ONLY" = blind ]; then
  run blind_search --year "$YEAR" --draws 200 --out "$OUT" || rc=1
fi
if [ "$ONLY" = all ] || [ "$ONLY" = reconcile ]; then
  if [ -f "$WORK/official_${YEAR}.csv" ]; then
    extra=()
    [ -f "$WORK/available.csv" ] && extra+=(--available "$WORK/available.csv")
    [ -f "$WORK/cap_monthly.csv" ] && extra+=(--monthly "$WORK/cap_monthly.csv")
    run reconcile_audit --year "$YEAR" --official "$WORK/official_${YEAR}.csv" \
        "${extra[@]}" --out "$OUT" || rc=1
  else
    echo "[skip] reconcile_audit: no $WORK/official_${YEAR}.csv (farm_id,official_acc)"
    echo
  fi
fi

echo "=================================================================="
echo "[done] rc=$rc  outputs in $OUT"
ls -la "$OUT" 2>/dev/null | tail -n +2
exit $rc
