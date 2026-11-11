# Generation 37 opening follow-up experiments (8–14)

Completed on 2026-11-02. These are frozen-policy experiments, not new training.
Checkpoint SHA-256: fbc5209b17a14a1d7b649087efdb1503185c7dfab9d9f5b443d0957431272158.

## Reproduction

Use the CatanRL repository, its installed native catan_rl extension, NumPy and PyTorch,
the original run/scale/gen37.pt checkpoint, and the existing 100,000-board corpus at
run/analysis/opening-study-100k-20261029. The model weights and 902MB corpus are not in
this archive. Copy the methods/*.py scripts to the repository root before running.
The selected intervention boards and complete continuation histories are included.

```
.venv/bin/python opening_followup_plans.py --pairs 2048
.venv/bin/python validate_followup_plans.py
.venv/bin/python opening_followup_ports_numbers.py --samples 2000 --patch-samples 128
.venv/bin/python opening_followup_hands_roads.py --n 2000 --patch-n 512 --readiness-n 1000
.venv/bin/python opening_followup_blocking_fallback.py
.venv/bin/python opening_followup_opponent_relocation.py --boards 2000
.venv/bin/python opening_followup_denial_continuations.py --pairs 1024
.venv/bin/python opening_followup_remote_context.py --boards 1000
```

All commands use one CPU thread per model. Use a separate output directory for pilots.
Per-study README files describe selection, masks, patches, seeds and limitations.
Plans uses 2,048 matched board pairs(4,096 full games); denial uses 1,024 pairs(2,048 full games).
All 6,144 reported games reached an actual terminal result before the 500-turn cap.
The earlier runtime pilot output is excluded from this archive and article.

## What the files contain

- plans/: matched portfolios, every complete rollout history, results and validation.
- ports_numbers/: reproducible board swaps, all scores and patches, and results.
- hands_roads/: legal reordered histories, controlled road comparisons and validation.
- blocking_fallback/: 100k-board regression results, exact-reply validation, relocation
  and remote-context interventions, and complete denial continuations.
- methods/: exact analysis source. Existing repository modules remain dependencies.

Raw policy logits, softmax preference, deterministic choice frequency and observed
winning outcomes are different quantities. The article labels them separately.
Pooled patch projections need not lie between 0 and 1 and do not add into semantic shares.
Intervals use whole boards; narrow intervals do not establish optimal Catan strategy.
