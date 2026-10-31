#!/usr/bin/env bash
# Build the floor-free counterfactual environment from the original code.
#
# Applies exactly the three edits the revision used (nothing else is changed):
#   1. exp_core: the score's denominator becomes the installed capacity when
#      P3_DENOM=cap, instead of max(P_M, 0.2 Cap). Both scoring paths.
#   2. p3_agent_v3: run artefacts write to $HOME/p3_nf/results.
#   3. p3_agent_v3: run tags are prefixed NF, so nothing collides with the
#      floored corpus.
#
# Everything else -- prompts, arms, seeds, sampling, acceptance rule, proposer --
# is untouched. The counterfactual therefore differs from the main corpus in the
# objective and nothing else.
#
# Usage:  bash make_floorfree.sh [CODE_DIR]   (default ~/hzh_RSI/03_code)
set -eu
SRC=${1:-$HOME/hzh_RSI/03_code}
DST=$HOME/p3_nf/03_code
mkdir -p "$DST" "$HOME/p3_nf/results"
cp "$SRC/exp_core.py" "$SRC/p3_agent_v3.py" "$SRC/p3_ontology.py" "$DST/"
cd "$DST"

"$HOME/anaconda3/envs/llm_power_ts/bin/python" - <<'PY'
s = open("exp_core.py", encoding="utf-8").read()
old1 = "    d = np.where(PM >= 0.2 * Cap, PM, 0.2 * Cap)"
new1 = ("    d = (np.full(np.shape(PM), Cap, dtype=float)\n"
        "         if os.environ.get(\"P3_DENOM\", \"floor\") == \"cap\"\n"
        "         else np.where(PM >= 0.2 * Cap, PM, 0.2 * Cap))")
old2 = "    d = np.where(PM >= 0.2 * C, PM, 0.2 * C)"
new2 = ("    d = (np.full(PM.shape, C, dtype=float)\n"
        "         if os.environ.get(\"P3_DENOM\", \"floor\") == \"cap\"\n"
        "         else np.where(PM >= 0.2 * C, PM, 0.2 * C))")
assert old1 in s and old2 in s
s = s.replace(old1, new1).replace(old2, new2)
open("exp_core.py", "w", encoding="utf-8").write(s)

a = open("p3_agent_v3.py", encoding="utf-8").read()
a = a.replace('os.path.expanduser("~/hzh_RSI/04_experiments/P3_knowledge_retention/results")',
              'os.path.expanduser("~/p3_nf/results")')
old3 = '    tag = "Y%d_%s%s_%s_s%d"'
assert old3 in a
a = a.replace(old3, '    tag = "NFY%d_%s%s_%s_s%d"')
a = a.replace('sys.path.insert(0, os.path.expanduser("~/hzh_RSI/03_code"))',
              'sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))')
open("p3_agent_v3.py", "w", encoding="utf-8").write(a)
print("patched exp_core.py and p3_agent_v3.py in", "$PWD")
PY

echo "Run the counterfactual with:"
echo "  P3_DENOM=cap $PY p3_agent_v3.py --year 2025 --seed 0 --policy nomem --iters 60"
echo "Then analyse with nofloor_analysis.py."
