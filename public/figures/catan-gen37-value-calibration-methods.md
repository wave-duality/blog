# Generation 37 implied-probability audit

This analysis reuses the **12,230 cached positions from 776 games** in `../audit/row_results.npz`. It performs no inference, new games, calibration fitting, or model selection. The ten-bin SVG and PNG are publication outputs; `stats.json` records full results, definitions, uncertainty, checks, and input hashes; `bins.csv` contains the plotted data. Reproduce with `.venv/bin/python run/analysis/blog-value-gen37/calibration-20261028/build.py` from the repository root.

## Target and weighting

For acting-player value v, the plotted implied probability is p = (v + 1) / 2. The binary recorded outcome is y = (z + 1) / 2, where z is +1 for an acting-player win and −1 for a loss. There are no draw positions in this corpus. This algebraic conversion does not establish calibration. Generation 37 was trained against a blend of 35% final outcome and 65% search value.

Every game has total weight one: a position in a game with n included positions has weight 1/n. Means, scores, and bin averages use these same weights. The weighted acting-player win fraction is 0.465350. The bins are fixed [0%, 10%), …, [90%, 100%], including 100% in the final bin. Each point is placed at its bin’s weighted mean implied probability, not at the bin midpoint. Raw position counts and distinct contributing game counts are supplied; a game can appear in multiple bins.

## Primary scores and the meaning of R²

- **Brier score:** 0.099813 (95% interval 0.092103–0.107798). This is the weighted mean of (p − y)²; lower is better.
- **Empirical constant baseline:** 0.248799 (95% interval 0.247334–0.249692). Always predicting this corpus’s weighted win fraction gives variance ȳ(1 − ȳ). This retrospective reference uses the evaluation outcomes; it is not a separately trained forecast.
- **Brier skill / outcome R²:** 0.598822 (95% interval 0.566607–0.629745). Defined as 1 − Brier / constant-baseline Brier, it is exactly the weighted per-position prediction R² against binary outcomes with the corpus mean as reference. The value means **59.9% less squared error than that constant reference**. It is not squared correlation, a fitted regression, a causal effect, or a claim of independent held-out generalization.
- **Ten-bin ECE:** 0.036854 (95% interval 0.026568–0.048232), or 3.69 percentage points. This is the weighted mean absolute gap between each bin’s observed frequency and mean implied probability. It depends on the bin definition and does not measure individual-position accuracy.

For completeness, **diagnostic weighted binned R²** is 0.987712 (95% interval 0.979200–0.992719): 1 − Σ W_b(ȳ_b − p̄_b)² / Σ W_b(ȳ_b − ȳ)². Here W_b is the sum of position weights in bin b. This compares bin means to the identity line; no regression is fitted. Binning averages away binary-outcome noise, so its much higher number must not be described as the fraction of individual outcomes explained. It is omitted from the headline figure statistics in favor of outcome R²/Brier skill.

## Uncertainty, verification, and scope

The 95% intervals are pointwise percentile intervals from **10,000 whole-game bootstrap resamples**, seed 20261028. Each resample draws 776 games with replacement, retains every selected game’s cached positions and original within-game weights, and recalculates bin statistics, prevalence, baseline, and all scores. Only vertical observed-win-rate intervals appear in the chart; intervals for mean implied probability are also in the JSON. These intervals describe sampling variability within this corpus, not training uncertainty, distribution shifts, or simultaneous coverage across ten bins.

Checks confirm all 776 game weights sum to one, the Brier score equals the earlier raw-value MSE divided by four, merging adjacent new bins reproduces all five original bins, and all bootstrap statistics are finite. Source files are untouched.

These positions are a selected, pre-existing convenience corpus of games collected after Generation 37 from the canonical Generation 38 data-generation run, with a neutral-prepared Generation 37 and frozen-opponent mixture. Selection includes recovered/completed games, Play-phase positions with at least one hidden opponent development card, and at most 16 evenly spaced eligible turns per game. There was no new outcome-based position selection for this audit. This describes agreement under the recorded continuation policies; it establishes neither opening calibration nor causal explanations of the network’s internals. No held-out calibration model was fitted or evaluated.
