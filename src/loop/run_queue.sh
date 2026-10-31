#!/bin/bash
# 统一排队执行器：等上一批空闲 → 顺序跑各阶段（每阶段内部并行）
# 用法: PY=python3 P3_RESULTS=./results nohup bash run_queue.sh > queue.log 2>&1 &
set -u
PAR=${PAR:-8}
PY=${PY:-python3}
cd "$(dirname "$0")" || exit 1
LOGD=${LOGD:-$(dirname "$0")/../../results/logs}
mkdir -p "$LOGD"

wait_idle() {
  while pgrep -u "$USER" -f "[p]3_agent_" > /dev/null; do sleep 20; done
}

run_phase() {          # $1=名字  $2=任务文件
  local name=$1 tf=$2
  echo "[QUEUE] phase=$name start $(date '+%F %T') tasks=$(wc -l < "$tf")"
  cat "$tf" | xargs -P "$PAR" -I{} bash -c '{}'
  echo "[QUEUE] phase=$name done  $(date '+%F %T')"
}

wait_idle

# ---------- P1: C17 station 臂（决定性） ----------
T=$(mktemp)
for s in 0 1 2 3 4 5 6 7; do
  echo "$PY p3_agent_holdout.py --year 2024 --seed $s --holdout station --iters 60 --tol 0.05 > $LOGD/c17/station_s$s.log 2>&1" >> "$T"
done
run_phase c17_station "$T"

# ---------- P2: C18 接受规则三档 ----------
T=$(mktemp)
for s in 0 1 2 3 4 5 6 7; do
  echo "$PY p3_agent_v3.py --year 2024 --seed $s --policy none --accept greedy --iters 40 > $LOGD/c18/greedy_s$s.log 2>&1" >> "$T"
  echo "$PY p3_agent_v3.py --year 2024 --seed $s --policy none --accept tol --tol 0.5 --iters 40 > $LOGD/c18/tol_s$s.log 2>&1" >> "$T"
  echo "$PY p3_agent_v3.py --year 2024 --seed $s --policy none --accept always --iters 40 > $LOGD/c18/always_s$s.log 2>&1" >> "$T"
done
run_phase c18_accept "$T"

# ---------- P3: C11 知识阶梯（仅新增四臂） ----------
T=$(mktemp)
for s in 0 1 2 3 4 5 6 7; do
  for pol in names effects code placebo oracle; do
    echo "$PY p3_agent_v3.py --year 2024 --seed $s --policy $pol --kb ${P3_RESULTS:-./results}/kb_v3_2024.json --accept greedy --iters 40 > $LOGD/c11/${pol}_s$s.log 2>&1" >> "$T"
  done
done
run_phase c11_ladder "$T"

# ---------- P4: C13 提议容量阶梯 ----------
T=$(mktemp)
for s in 0 1 2 3 4 5 6 7; do
  echo "$PY p3_agent_v3.py --year 2024 --seed $s --policy none --cap think --iters 40 > $LOGD/c13/think_s$s.log 2>&1" >> "$T"
  echo "$PY p3_agent_v3.py --year 2024 --seed $s --policy none --cap hot --iters 40 > $LOGD/c13/hot_s$s.log 2>&1" >> "$T"
done
run_phase c13_capacity "$T"
for s in 0 1 2 3 4 5 6 7; do
  echo "$PY p3_agent_v3.py --year 2024 --seed $s --policy none --cap bestk --bestk 4 --iters 40 > $LOGD/c13/bestk_s$s.log 2>&1"
done > /tmp/c13b.txt
run_phase c13_bestk /tmp/c13b.txt

echo "[QUEUE] ALL DONE $(date '+%F %T')"
