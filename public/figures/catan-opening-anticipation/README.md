# Opening anticipation: Experiment 14

These files support three distinct exploratory analyses of frozen Generation 37:

- Exact production-fallback objectives across the original 100,000 boards, retaining the original 80,000/20,000 regression split.
- Bounded complete-opening trees on 128 held-out boards (527,556 endpoints), plus a first-eight-board width control (133,595 evaluation rows; overlapping endpoints may repeat).
- 500 balanced remote-board interventions holding focal local inputs and optimistic production scores fixed.

The checkpoint SHA-256 is fbc5209b17a14a1d7b649087efdb1503185c7dfab9d9f5b443d0957431272158.

Raw value-head scores are not calibrated win probabilities. Production proxies are not full-game payoffs. Worst-case opening values examine shortlisted B choices and greedy roads, not every legal continuation. Results do not establish an internal minimax algorithm.

## Reproduction

Run the scripts from the original CatanRL repository using its Python environment and native engine. They depend on the existing original 100k study corpus, checkpoint and helper modules; this archive does not duplicate those dependencies or model weights. Absolute local paths in the scripts should be adjusted if the repository is moved. The original seed selection and split manifest remain part of that corpus.

Main entry points:

1. opening_anticipation_portfolios.py, then portfolios/secondary.py
2. opening_anticipation_lookahead.py (see its --help and saved config.json), validate_opening_anticipation_lookahead.py, summarize_opening_anticipation_lookahead.py
3. opening_anticipation_remote.py
4. build_opening_anticipation_view.py

The archive contains the study scripts, summary methods, every examined opening tree with action histories and A2 scores, both remote boards and their scores, and per-board production diagnostics/errors. Exact per-candidate portfolio caches can be regenerated; they are not included.

All analyses are exploratory. Statistical intervals and test families are described in the corresponding methods. The rendered article presents a selected set of results; the detailed files retain other diagnostics and their limitations.
