# Generation 37: 100,000-board opening study

Analysis specification fixed before inspecting the large run's results (2026-10-29).

## Population and provenance

- Frozen original `run/scale/gen37.pt`, SHA-256 `fbc5209b17a14a1d7b649087efdb1503185c7dfab9d9f5b443d0957431272158`.
- 100,000 new generated boards, seeds 4,100,000 through 4,199,999. No overlap with the previous 1,000-board study or its gallery.
- One greedy opening trajectory per board: A1, B1, B2, A2, each followed by a greedy legal road. No search, training, or full-game rollouts.
- Train/test assignment is a fixed shuffled whole-board 80/20 split, seed 291037. All four stages of a board share the assignment.
- Batched float32 inference; close top-two legal scores (gap below 0.001) trigger a single-position recomputation. Lower action ID breaks exact ties. Source snapshots, compiled-engine hashes, checkpoint identity, and shard hashes are retained.
- Pilot positions are used only to validate legality, parity, and numerical implementation, not to select features or tune regularization.

## Regression targets

Each stage is analyzed separately. The target is the raw legal settlement logit minus the mean logit over that decision's legal candidates. Features are centered over precisely the same candidates. Each board has equal weight; its candidate rows share that weight.

Replicate the previous single-resource, five-resource, coverage, port, number-pattern, and expansion-context regressions. Also fit leave-one-resource-out variants of the five-resource model. Regularization remains training-RMS standardized ridge, alpha 0.001. Report raw per-unit coefficients, held-out R², exact top-choice agreement, and the score gap between the network's choice and the regression's choice.

New contextual controls add each resource's interactions with existing own production, lack of existing access, owned matching 2:1 ports, and geometric port proximity, plus resource-specific direct-port indicators and starting-card counts. Exact dependencies and constant columns are omitted in fixed feature order and listed explicitly; omitted coefficients are not estimates of zero importance.

## Availability and scarcity

Let P_r(v) be a candidate settlement's resource-r pips, S_r the sum of that resource's tile pips on the board, and M_r its mean candidate pips over currently legal corners. Reference supplies are 58 times resource tile count divided by 18.

- **Board scarcity:** add P_r(v) × (reference supply_r − S_r) for all five resources to the contextual model.
- **Legal availability:** in a separate model, add P_r(v) × (reference supply_r / 9 − M_r).
- At A1 all 54 corners are legal, so M_r = S_r / 9. The two standardized models must therefore produce equivalent predictions; this is an explicit validation check.
- A positive interaction coefficient associates greater scarcity with increased marginal policy priority for that resource, conditional on the fitted controls. It does not establish a causal effect.

## Selection enrichment

On each board, convert the selected candidate's five pip counts into resource fractions. Compare their equal-board means with these baselines:

1. Uniform over legal corners with positive total production. Exclude and count zero-production corners, whose resource fractions are undefined.
2. Uniform over legal corners matching the chosen corner's exact total pips and adjacent-hex count, including the chosen corner. Require at least three matching corners and report coverage. This is the primary baseline for describing resource mix beyond total productivity and coastal geometry.

Enrichment is the ratio of the mean selected fraction to the mean baseline fraction, not the mean of per-board ratios. Report stage-specific results and later-stage conditional summaries for resources the player already produces versus lacks. These descriptive statistics use all 100,000 boards because no predictor is fitted for them.

## Uncertainty and interpretation

- Training coefficient uncertainty: independent-board sandwich covariance for the fixed-ridge estimator, including the estimated RMS scaling. The retained feature set and analysis specification are held fixed.
- Held-out R² and paired changes in R²: board-level delta-method intervals conditional on the training fits.
- Enrichment: paired board-level log-ratio delta-method intervals; differences use paired board standard errors.
- Report ordinary 95% intervals. Additionally provide Bonferroni intervals for 40 resource × stage × scarcity-measure coefficients, and separately for 40 pairwise resource-substitution contrasts.
- Report supporting counts for conditional and matched subsets. Candidate rows are not independent samples.
- The study measures this checkpoint's preferences under this generator and greedy history distribution. It does not measure win probabilities, searched opening strength, optimal play, causal neuron mechanisms, or performance across independently trained models.

## Reproduction

```sh
.venv/bin/python opening_study_scale.py
.venv/bin/python opening_study_scale_analysis.py
```

The generator resumes with the identical command and immutable configuration. Detailed board renderings and activation traces can be regenerated for selected seeds from the saved board arrays and action history; the numerical corpus does not require a 100,000-page gallery.
