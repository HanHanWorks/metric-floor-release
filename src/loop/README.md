# The search loop

These four modules are the experiment as it ran on the compute node. They are
released for inspection: the prompt, the sandbox, the acceptance rule and the
mechanism extractor are all here, so a reader can check what the optimizer could
and could not see.

They cannot be executed end to end from this repository, because `exp_core.py`
loads the metered output and the issued forecasts, which are withheld
(see `data/README.md`). Everything the loop produced — 106 final pipelines,
their per-iteration traces, and the tables built from them — is in `data/`.

| file | role |
|---|---|
| `p3_agent_v3.py` | the loop: proposes a `transform()` per iteration, executes it, scores it, keeps it if the score rises by more than `tau`. Arms select what (if anything) is injected into the prompt as prior knowledge |
| `p3_agent_holdout.py` | the same loop with acceptance decided on a validation subset and the change reported on a disjoint held-out subset (`station` or `time` split) |
| `p3_ontology.py` | the twelve-class mechanism ontology and the deterministic regular-expression extractor. No language model is involved in labelling |
| `exp_core.py` | data loading and the four losses, including the assessment rule with its floored denominator |

## What the proposer could see

Exactly `dt`, `PP` (issued forecast), `Cap`, and three calendar fields. Never the
metered output `PM`. Per iteration it received one scalar — the score on the
evaluation set — and the score history of the previous six iterations.

## Configuration

Credentials and the endpoint are read from the environment; nothing is
hard-coded:

```bash
export LLM_HOST=<host of an OpenAI-compatible chat-completions endpoint>
export LLM_API_KEY=<key>
export LLM_MODEL=deepseek-v4.1-flash      # default; the model used in the study
export WIND_DATA_ROOT=<directory holding the restricted series>
export P3_RESULTS=./results               # where run artefacts are written
```

Model settings used throughout: thinking disabled, 4,000-token output cap,
temperature `0.4 + 0.5 t/T` capped at 0.9.

## Reproducing a run, given the data

```bash
python p3_agent_v3.py       --year 2024 --seed 0 --iters 60 --stations 4 \
                            --accept greedy --policy none
python p3_agent_holdout.py  --year 2025 --seed 11 --holdout station \
                            --iters 60 --stations 4 --tol 0.05
```

`run_c17.sh` is the original batch driver for the 32 holdout runs;
`run_queue.sh` the queue that ran the campaigns in sequence. Both were edited
only to replace hard-coded server paths with environment variables.

## Expected input schema

`exp_core.load_station_data()` needs, per year, one row per farm and 15-minute
interval with `farm_id`, `data_time`, metered output, issued forecast and
installed capacity. The loader filters to wind farms with capacity above 10 MW,
requires a minimum number of days per farm (200, relaxed to 80 for the partial
2025 archive), and returns `dt, PM, PP, Cap, nm` per farm. See the function for
the exact column names in the utility's exports.
