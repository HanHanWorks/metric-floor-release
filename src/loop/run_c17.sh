#!/bin/bash
# C17｜留出验证：3 臂 × 16 种子
# 用法: bash run_c17.sh [并行度, 默认6]
set -u
PAR=${1:-6}
PY=${PY:-python3}
cd "$(dirname "$0")" || exit 1
LOGD=${LOGD:-$(dirname "$0")/../../results/logs/c17}
mkdir -p "$LOGD"
TASKS=$(mktemp)
for arm in none station time; do
  for s in $(seq 0 15); do echo "$arm $s"; done
done > "$TASKS"
echo "[C17] $(wc -l < "$TASKS") runs, parallel=$PAR, start $(date '+%F %T')"
run_one() {
  arm=$1; s=$2
  $PY p3_agent_holdout.py --year 2024 --seed "$s" --holdout "$arm" \
      --iters 60 --stations 4 --tol 0.05 > "$LOGD/${arm}_s${s}.log" 2>&1
}
export -f run_one
export PY LOGD
cat "$TASKS" | xargs -P "$PAR" -n 2 bash -c 'run_one "$0" "$1"'
echo "[C17] done $(date '+%F %T')"
