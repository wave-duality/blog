# Paired significance of opening-regression improvements

These are held-out comparisons of frozen Generation 37 policy-logit regressions.
All regressions use the same 80,000 training boards and 20,000 test boards.
The downloaded board-error arrays are sufficient to recompute the paired tests.
Each key is STAGE__MODEL, with STAGE__seed and STAGE__target alongside it.
target is the mean squared within-decision-centered logit on that board.

Run from the CatanRL repository:
    .venv/bin/python opening_study_significance.py
To reuse already reconstructed errors in the default destination:
    .venv/bin/python opening_study_significance.py --reuse-errors

Requirements: the project Python environment (numpy, mpmath, torch and catan_rl
via the shared analysis helpers), original 100k candidate shards, and saved
regression results. No neural inference, model refitting, search or games are run.
The archive includes the analysis helpers for provenance; use the project tree
for their dependencies. Source and output hashes are in results.json.

df = 19,999. Each t-statistic uses paired mean-squared-error differences at the
board level. P-values use an approximate two-sided Student reference, followed
by Holm correction over the 28 unique comparisons. Tiny p-values are bounded
as <0.001; there is no claim of extreme-tail precision. Delta-R2 intervals are
pointwise paired ratio delta-method intervals, not simultaneous intervals.
The tests condition on the fixed fits and do not measure winning strength.
