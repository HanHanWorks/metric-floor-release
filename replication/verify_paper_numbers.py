# -*- coding: utf-8 -*-
"""Check every headline number in the manuscript against the released tables.

The expected values are the ones printed in the paper. Each row is compared
either exactly (counts), to the printed precision (means, SDs, correlations),
or by order of magnitude (p-values below 1e-3, which the paper rounds).

Exit status is 1 if any strict check fails, so this can gate a release.

Usage:
    python replication/verify_paper_numbers.py [--verbose]
"""
from __future__ import print_function

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src", "analysis"))

import c19_station_ext  # noqa: E402
import c20_holdout  # noqa: E402
import c21_regime  # noqa: E402
import c22_dm  # noqa: E402
import trace_tables  # noqa: E402


def fmt(x):
    if isinstance(x, float):
        if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e6):
            return "%.2e" % x
        return "%.4f" % x
    return str(x)


class Checker(object):
    def __init__(self):
        self.rows = []

    def exact(self, label, got, want, where):
        ok = int(got) == int(want)
        self.rows.append((ok, "ok" if ok else "FAIL", label, fmt(got), fmt(want), where))

    def close(self, label, got, want, where, tol=5e-3):
        ok = abs(float(got) - float(want)) <= tol
        self.rows.append((ok, "ok" if ok else "FAIL", label, fmt(got), fmt(want), where))

    def order(self, label, got, want, where):
        """Same order of magnitude for small p-values."""
        got, want = max(float(got), 1e-300), max(float(want), 1e-300)
        ok = abs(math.log10(got) - math.log10(want)) < 0.5
        self.rows.append((ok, "ok" if ok else "FAIL", label, fmt(got), fmt(want), where))

    def known(self, label, got, want, where, tol=0.02):
        """A value the paper prints to 1e-2 with a documented rounding slip.

        The check still runs (tolerance `tol`), but the row is marked `diff`
        so the slip stays visible in the output instead of being hidden.
        """
        diff = abs(float(got) - float(want))
        ok = diff <= tol
        mark = "ok" if diff <= 5e-3 else ("diff" if ok else "FAIL")
        self.rows.append((ok, mark, label, fmt(got), fmt(want), where))

    def report(self, verbose):
        n_ok = sum(1 for r in self.rows if r[0])
        width = max(len(r[2]) for r in self.rows) + 2
        print("%-6s %-*s %-14s %-14s %s" % ("", width, "check", "computed", "paper", "where"))
        print("-" * (width + 62))
        for ok, mark, label, got, want, where in self.rows:
            if ok and mark == "ok" and not verbose:
                continue
            print("%-6s %-*s %-14s %-14s %s" % (mark, width, label, got, want, where))
        print("-" * (width + 62))
        print("%d/%d checks pass" % (n_ok, len(self.rows)))
        return n_ok == len(self.rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    c = Checker()
    t4 = trace_tables.compute()["table4"]
    t9 = trace_tables.compute()["table9"]
    mar = trace_tables.compute()["marginal"]
    grp = trace_tables.compute()["mechanism_groups"]
    c19 = c19_station_ext.compute()
    c20 = c20_holdout.compute()
    c21 = c21_regime.compute()
    c22 = c22_dm.compute()

    # --- Table 4 ---------------------------------------------------------
    by_year = dict((r["year"], r) for r in t4["by_year"])
    for year, runs, iters, acc, rate, mean, sd in (
            (2023, 28, 1280, 366, 28.6, 0.90, 0.87),
            (2024, 36, 1760, 351, 19.9, 0.96, 1.35),
            (2025, 42, 2320, 485, 20.9, 1.15, 1.14),
            ("total", 106, 5360, 1202, 22.4, 1.02, 1.15)):
        r = by_year[year]
        c.exact("Table 4 %s runs" % year, r["runs"], runs, "Table 4")
        c.exact("Table 4 %s iterations" % year, r["iterations"], iters, "Table 4")
        c.exact("Table 4 %s accepted" % year, r["accepted"], acc, "Table 4")
        c.close("Table 4 %s rate %%" % year, 100 * r["acceptance_rate"], rate, "Table 4", 0.05)
        c.close("Table 4 %s mean dACC" % year, r["dacc_mean"], mean, "Table 4")
        c.close("Table 4 %s SD" % year, r["dacc_sd"], sd, "Table 4")
    c.close("range min", t4["dacc_min"], 0.01, "Section 4.1", 0.005)
    c.close("range max", t4["dacc_max"], 7.21, "Section 4.1", 0.005)
    c.exact("runs above their start", t4["above_start"], 106, "Section 4.1")
    nw = t4["non_windowed"]
    c.exact("Table 2 non-windowed runs", nw["runs"], 56, "Table 2")
    c.exact("Table 2 non-windowed iterations", nw["iterations"], 3360, "Table 2")
    c.exact("Table 2 non-windowed accepted", nw["accepted"], 733, "Table 2")
    c.close("Table 2 non-windowed rate %", 100 * nw["acceptance_rate"], 21.8, "Table 2", 0.05)

    # --- Table 9 ---------------------------------------------------------
    prev = dict((r["mechanism"], r) for r in t9["rows"])
    for mech, n, share in (
            ("M07_clip_capacity", 93, 87.7), ("M02_smoothing", 78, 73.6),
            ("M01_global_scale", 75, 70.8), ("M08_ramp_limit", 58, 54.7),
            ("M06_low_shrink", 42, 39.6), ("M11_calendar_correction", 40, 37.7),
            ("M03_median_filter", 14, 13.2), ("M04_spike_removal", 13, 12.3),
            ("M09_lag_blend", 12, 11.3), ("M12_piecewise", 12, 11.3),
            ("M05_floor", 11, 10.4), ("M10_quantile_stretch", 9, 8.5)):
        c.exact("Table 9 %s" % mech, prev[mech]["pipelines"], n, "Table 9")
        c.close("Table 9 %s share" % mech, 100 * prev[mech]["share"], share, "Table 9", 0.05)
    c.close("classes per pipeline, mean", t9["n_mech_mean"], 4.3, "Section 4.6", 0.05)
    c.exact("classes per pipeline, max", t9["n_mech_max"], 10, "Section 4.6")

    # --- marginal returns -------------------------------------------------
    c.exact("accepted without a new mechanism", mar["n_without_new_mechanism"], 953,
            "Section 4.6")
    c.close("share without a new mechanism", 100 * mar["share_without_new_mechanism"],
            79.3, "Section 4.6", 0.05)
    c.close("gain carried by first ten edits", 100 * mar["share_gain_first10"], 72,
            "Section 4.6", 1.0)

    # --- mechanism groups -------------------------------------------------
    g = dict((r["mechanism"], r) for r in grp["tests"])
    c.close("global rescaling, with", g["M01_global_scale"]["mean_with"], 3.92, "Section 4.6")
    c.close("global rescaling, without", g["M01_global_scale"]["mean_without"], 1.08,
            "Section 4.6")
    c.close("calendar correction, with", g["M11_calendar_correction"]["mean_with"], 4.33,
            "Section 4.6")
    c.close("calendar correction, without", g["M11_calendar_correction"]["mean_without"],
            2.26, "Section 4.6")
    c.close("low-output floor, with", g["M05_floor"]["mean_with"], 2.66, "Section 4.6")
    c.close("low-output floor, without", g["M05_floor"]["mean_without"], 3.18, "Section 4.6")
    c.order("low-output floor p", g["M05_floor"]["welch_p"], 0.60, "Section 4.6")
    r = grp["n_mechanism_classes_vs_floor_gain"]
    c.close("classes vs floor gain rho", r["rho"], 0.372, "Section 4.6")
    c.order("classes vs floor gain p", r["p"], 0.005, "Section 4.6")

    # --- Table 6 / Section 4.3 -------------------------------------------
    for year, unseen, seen, err in ((2023, 1.11, 1.14, 0.88), (2024, 0.79, 0.77, 0.65),
                                    (2025, 0.87, 1.10, 1.36)):
        y = c19["by_year"][year]
        c.close("Table 6 %d unseen dACC" % year, y["unseen"]["dacc"], unseen, "Table 6")
        c.close("Table 6 %d seen dACC" % year, y["seen"]["dacc"], seen, "Table 6")
        c.close("Table 6 %d unseen error %%" % year, y["unseen"]["dmase_pct"], err,
                "Table 6")
        c.exact("Table 6 %d unseen n" % year, y["unseen"]["n"], {2023: 80, 2024: 240, 2025: 640}[year],
                "Table 6")
    u = c19["run_level_unseen"]
    c.close("score up on unseen farms %", 100 * u["frac_score_up"], 94.6, "Section 4.3", 0.05)
    c.close("error worse %", 100 * u["frac_error_worse"], 64.3, "Section 4.3", 0.05)
    c.close("both %", 100 * u["frac_tradeoff"], 58.9, "Section 4.3", 0.05)
    c.close("rho score vs error", u["spearman"], 0.589, "Section 4.3")

    # --- Table 5 / Section 4.2 -------------------------------------------
    st, tm = c20["station"], c20["time"]
    c.exact("Table 5 farms n", st["n"], 16, "Table 5")
    c.close("Table 5 farms accept", st["d_accept_mean"], 2.21, "Table 5")
    c.close("Table 5 farms accept SD", st["d_accept_sd"], 1.23, "Table 5")
    c.close("Table 5 farms holdout", st["d_holdout_mean"], 1.16, "Table 5")
    c.close("Table 5 farms holdout SD", st["d_holdout_sd"], 1.49, "Table 5")
    c.close("Table 5 farms ratio", st["ratio"], 0.52, "Table 5")
    c.close("Table 5 farms t", st["t"], 3.12, "Table 5", 1e-2)
    c.order("Table 5 farms p", st["p"], 0.0070, "Table 5")
    c.order("Table 5 farms Wilcoxon", st["wilcoxon_p"], 0.0092, "Section 4.2")
    c.close("Table 5 farms paired t", st["paired_t"], 2.75, "Table 5", 1e-2)
    c.order("Table 5 farms paired p", st["paired_p"], 0.015, "Table 5")
    c.close("Table 5 days accept", tm["d_accept_mean"], 1.44, "Table 5")
    c.close("Table 5 days accept SD", tm["d_accept_sd"], 0.99, "Table 5")
    c.close("Table 5 days holdout", tm["d_holdout_mean"], 1.14, "Table 5")
    c.close("Table 5 days holdout SD", tm["d_holdout_sd"], 0.90, "Table 5")
    c.close("Table 5 days t", tm["t"], 5.04, "Table 5", 1e-2)
    c.close("Table 5 days ratio", tm["ratio"], 0.79, "Table 5", 5e-3)
    c.close("Table 5 days paired t", tm["paired_t"], 1.63, "Table 5", 1e-2)
    c.order("Table 5 days paired p", tm["paired_p"], 0.125, "Table 5")

    # --- Table 8 / Section 4.5 -------------------------------------------
    c.close("Table 8 all runs floor", c21["all_runs"]["floor"], 3.11, "Table 8")
    c.close("Table 8 all runs proportional", c21["all_runs"]["proportional"], -0.91,
            "Table 8")
    for year, floor, prop in ((2023, 3.45, -0.94), (2024, 2.79, -1.61), (2025, 3.19, -0.63)):
        y = c21["by_year"][year]
        c.close("Table 8 %d floor" % year, y["floor"], floor, "Table 8")
        c.close("Table 8 %d proportional" % year, y["proportional"], prop, "Table 8")
        c.close("Table 8 %d share below floor %%" % year, 100 * y["share_floor"],
                {2023: 61.1, 2024: 60.7, 2025: 56.4}[year], "Section 4.5", 0.1)
    c.close("Table 8 holdout floor", c21["holdout"]["floor"], 5.37, "Table 8")
    c.close("Table 8 holdout proportional", c21["holdout"]["proportional"], -2.10,
            "Table 8")
    c.close("within-run rho across regimes", c21["within_run"]["rho"], -0.54,
            "Section 4.5", 5e-3)
    c.close("share of intervals below floor %", 100 * c21["share_intervals_floor"],
            59, "Section 4.5", 0.5)

    # --- Table 7 / Section 4.4 -------------------------------------------
    for loss, pos, med, better, worse in (
            ("metric", 90.5, 3.02, 72.6, 4.5),
            ("se", 38.8, -1.00, 19.3, 37.5),
            ("ae", 46.8, -0.16, 19.0, 25.2)):
        r = c22["by_loss"][loss]
        c.exact("Table 7 %s n" % loss, r["n_unseen"], 960, "Table 7")
        c.close("Table 7 %s DM>0 %%" % loss, 100 * r["frac_positive"], pos, "Table 7", 0.05)
        c.close("Table 7 %s median DM" % loss, r["median_dm"], med, "Table 7", 5e-3)
        c.close("Table 7 %s significant better %%" % loss, 100 * r["frac_sig_better"],
                better, "Table 7", 0.05)
        c.close("Table 7 %s significant worse %%" % loss, 100 * r["frac_sig_worse"],
                worse, "Table 7", 0.05)

    ok = c.report(a.verbose)
    if not ok:
        print("\nSome checks differ from the printed values. Known rounding differences")
        print("are listed in docs/KNOWN_DISCREPANCIES.md.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
