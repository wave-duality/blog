# Exact first-pick anticipation diagnostics

Frozen Generation 37, original 100,000 opening boards. No network training, search, or new game simulation. The original 80,000/20,000 training/test split is preserved.

## What the tests distinguish

For an A1 corner `a`, let `L(a)` be legal A2 sites before B moves, and `L(a,b1,b2)` the survivors after a legal B1/B2 pair. For pair-production utility `U(a,d)`, compute

\[
O(a)=\max_{d\in L(a)}U(a,d),\qquad
W(a)=\min_{(b_1,b_2)\text{ legal after }a}\max_{d\in L(a,b_1,b_2)}U(a,d).
\]

`O` is optimistic: the ideal complement survives. `W` is robust to any legal two-placement reply. Neither is a win probability. The adversary here minimizes A's stated proxy, irrespective of B's own payoff.

Two headline objectives:

1. **Total production:** `U(a,d)=pips(a)+pips(d)`.
2. **Resource-weighted production:** apply the original A1 five-resource regression coefficients to the combined resource-pip vector. Weights, wood through ore: `1.7747200306, 2.1109165217, 1.7631846016, 1.9765829979, 2.1193408057`. These are policy-score associations, not values of cards or winning probabilities.

The regression features use **future A2 production only**, subtracting the A1 contribution already represented by the baseline features. Descriptive maximizer/regret comparisons use the full A1+A2 objective. This avoids adding redundant A1 terms that would change the ridge penalty's geometry without adding information.

An exploratory third objective applies the original A2 resource/coverage/concentration regression to each possible complement. Its A1-dependent intercept is unidentifiable from within-position fits and is set to zero by convention. Cross-A1 comparisons under this proxy therefore depend on a modeling convention. Do not present it as calibrated position value or as the headline minimax test.

## Main results

On the **20,000 held-out boards**, how often does the selected A1 maximize the exact worst-case two-settlement score?

| A1 selection rule | Total-pip objective | Resource-weighted objective |
| :--- | ---: | ---: |
| Generation 37 | 64.18% | 49.10% |
| Local regression: immediate production, variety, concentration, adjacency | 80.75% | 72.41% |
| Choose maximum immediate total pips | 90.24% | 55.63% |
| Choose maximum immediate resource-weighted production | 83.29% | 84.79% |

Thus high agreement with a pip-minimax calculation by itself is **not evidence of planning**. Simple local choices already have high agreement. Nor do these results establish that the local rules play better Catan: they are optimized for narrow production proxies and are not evaluated by wins.

For comparison with the previous 100k-board statistic, the corresponding Generation 37 / local-regression / immediate-max-pip total-pip numbers on all boards are **63.803% / 80.522% / 90.160%**. Weighted-production minimax agreement on all boards is **48.930%** for Generation 37, **72.181%** for the local regression, and **84.688%** for immediate-max-weighted-production.

### Do future options still help explain A1 scores?

Yes, modestly. The following uses weighted A2 production, without the offset-dependent A2-score proxy:

| Regression features | Held-out R² |
| :--- | ---: |
| Local features | 0.874308 |
| Local + optimistic A2 production | 0.877942 |
| Local + worst-case A2 production | 0.879869 |
| Local + both | 0.882880 |
| Existing full context | 0.936937 |
| Full context + optimistic A2 production | 0.936983 |
| Full context + worst-case A2 production | 0.937415 |
| Full context + both | 0.937471 |

Beyond full context and optimistic production, worst-case production adds **ΔR² = 0.00048735**, 95% interval **[0.00043615, 0.00053856]**, paired **t = 18.653**, **df = 19,999**, Holm-adjusted **p = 5.87×10⁻⁷⁶** across the 24 supplemental comparisons. The effect is precise but small. Exact top-choice agreement rises from 62.03% to 62.67% when both production summaries are added to full context.

Among 1,215 held-out boards with another corner having the same exact five resource-pip counts and adjacent-hex count, the chosen A1 actually has slightly worse production fallback than the uniform matched pool (including itself): **−0.07160 pips** [−0.09003, −0.05318], or **−0.12433 weighted-score units** [−0.15476, −0.09391]. Ports, number patterns, and future geometry are not matched. This again argues against treating fallback production as the sole selection rule.

### Does the best second pick actually survive?

Across all 100k recorded opening sequences:

- A maximum-pip A2 complement survives B1/B2 on **42.348%** of boards [42.042%, 42.654%]. Ties count as surviving if any optimum remains legal.
- The best remaining A2 is on average rank **2.170** in the original pip ranking. Competition ranks are used: one plus the number of strictly better pre-B candidates.
- Recorded B placements remove **0.8362 pips** from A's best available complement on average. The deliberately most damaging legal pair removes **1.5097 pips**.
- The chosen A1 leaves **4.728** initially legal corners within one pip of its ideal complement, on average. These locations can overlap in what blocks them; they are not independent backups.
- The network's eventual A2 gives up another **1.1207 pips** relative to the maximum still available, consistent with caring about properties other than total production.

These describe recorded behavior and availability. They do not reveal an internal probability forecast or establish that the network literally expects the second-ranked corner.

## Statistical and implementation details

- Regressions target within-position-centered settlement logits, using equal weight per board and the original standardized ridge penalty `alpha=0.001`.
- Paired tests compare per-board mean squared prediction errors, with the board as the independent unit. R² intervals use paired influence estimates; tests are conditional on the fitted training coefficients and omit training-fit uncertainty.
- The 24 supplemental comparisons form one Holm family; the 11 broader comparisons including the exploratory A2 proxy form another. These are exploratory analyses, not preregistered experiments.
- The exact blocker shortcut agrees with exhaustive enumeration on 300 random A1/objective cases. Twenty witness reply sequences were replayed through the native game. All 54,000 first-shard total-pip results agree exactly with the preceding independently validated fallback analysis.
- The A2 proxy reconstructs its original feature equation to within `3.55e-15`. Original local/full baseline R² values reproduce to `1e-10`.
- Initial roads do not change setup settlement legality. Their strategic value and all later play are outside these narrow calculations.
- Local baselines break exact score ties by smallest vertex ID, matching the stored candidate ordering. Minimax agreement accepts all tied optimal A1s within absolute `1e-9`.

## Reproduction and outputs

From the CatanRL repository:

```sh
.venv/bin/python opening_anticipation_portfolios.py
.venv/bin/python run/analysis/opening-anticipation-20261102/portfolios/secondary.py
```

- `results.json`: all primary diagnostics, definitions, coefficients, and regressions.
- `secondary.json`: production-only regression comparisons, simple greedy baselines, and exact local matching.
- `board-diagnostics.npz`, `secondary-board-diagnostics.npz`: per-board statistics and split flags.
- `cache/shard-*.npz`: exact per-candidate optimistic/worst scores, witnesses, and counts.
- `heldout-errors.npz`: per-board prediction errors for the primary comparisons.
- `validation.json`: independent parity/native legality checks.

`opening_anticipation_portfolios.py` exposes `pair_utility` and `make_solver` for additional counterfactual board studies. It never calls the neural network. Study claims concern correlations with the frozen policy and exact maximization of explicitly defined proxies; they are not a solved game-theoretic strategy or a located neural planning circuit.
