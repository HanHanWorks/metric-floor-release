# Discrepancy log

No value printed in the current manuscript differs from what the released tables
give. `replication/verify_paper_numbers.py` reports **134/134 checks pass** with
no `diff` rows.

This file is kept because the first submission did differ, at the level of the
last printed digit, and a reader comparing the two versions will find the
differences here. All three were rounding slips or convention mismatches in the
manuscript, not errors in the analysis; all three were corrected in the revision.

| location | first submission | released tables | status |
|---|---|---|---|
| Table 6, 2025, farms inside that pipeline's search set | +1.11 | +1.1048 | corrected to **+1.10** |
| Table 8, holdout runs, inside the floor regime | +5.38 | +5.3747 | corrected to **+5.37** |
| Table 5 and §4.2, ratio of held-out to acceptance change, farm split | 0.53 | 0.525 (1.161 / 2.213) | corrected to **0.52** |

## One convention that needed stating, not correcting

Table 8's "all runs" row mixes years with unequal run counts. Pooling intervals
within a stratum gives −0.91 above the floor; the unweighted mean of per-run means
gives −0.95. Every per-year row and the holdout row are identical under both
conventions. The paper reports the pooled value and says which convention it uses;
`c21_regime.compute()` returns both (`all_runs`, `all_runs_run_mean`).

## One p-value that is discrete

With 16 paired observations the exact two-sided signed-rank p-value for the day
split is 6.1 × 10⁻⁵, which is the smallest this sample can produce short of the
floor of the null distribution. The paper reports 1 × 10⁻⁴, half an order high;
both round to "p < 0.001". The verification script tests this row by order of
magnitude.
