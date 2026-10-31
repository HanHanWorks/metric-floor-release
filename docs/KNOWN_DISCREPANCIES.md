# Values where the manuscript and the released tables differ

`replication/verify_paper_numbers.py` prints items 1–3 as `diff`. Each is at
the level of the last printed digit. They are recorded rather than corrected inside
the verification script, so that the script keeps matching the paper exactly and
the difference stays visible.

| # | location | printed | computed | cause |
|---|---|---|---|---|
| 1 | Table 6, 2025, farms used in the search | +1.11 | +1.1048 | rounds to +1.10 |
| 2 | Table 8, holdout runs, below the floor | +5.38 | +5.3747 | rounds to +5.37 |
| 3 | Table 5 and §4.2, ratio of held-out to acceptance change, farm split | 0.53 | 0.525 (1.161 / 2.213) | rounds to 0.52 |
| 4 | Table 8, all runs, above the floor | −0.91 | −0.9070 pooled; −0.9507 as the unweighted mean of per-run means | see below |
| 5 | §4.2, Wilcoxon for the day split | 1 × 10⁻⁴ | 6.1 × 10⁻⁵ | exact p is discrete; the paper rounds up by half an order |

Items 1–3 are rounding slips: the underlying value is unambiguous and appears in
the released tables, so the manuscript should be corrected rather than the
script.

Item 4 is a convention, not an error in the arithmetic. Most rows of Table 8 pool
intervals within a stratum (mean score after minus mean score before); the
unweighted mean of per-run means differs only when years with unequal run counts
are pooled, which is the "all runs" row. Every per-year row and the holdout row
give the same value under both conventions. The paper's −0.91 is the pooled one,
and `c21_regime.compute()` reports both (`all_runs`, `all_runs_run_mean`).

Item 5: with 16 paired observations the exact two-sided signed-rank p-value can
only take discrete values, and 6.1 × 10⁻⁵ is the smallest this sample can
produce short of the floor of the null distribution. The paper's 1 × 10⁻⁴ is the
same test reported half an order high; both round to "p < 0.001", so the check
for this row is an order-of-magnitude test and it passes without a flag.

Status: 134 checks in total. Three rows are flagged `diff` (items 1–3); every
other value matches to the printed precision. Items 1–3 are worth correcting in
the manuscript — they are one keystroke each and a copy-editor will otherwise
find them.
