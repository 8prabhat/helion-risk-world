# Investigation log: what worked, what failed, what's still open

Single consolidated ledger, replacing four separate dated snapshot docs (`review_2026-07-01.md`,
`feature_onboarding_review_2026-07-11.md`, `model_training_diagnostics_2026-07-13.md`,
`model_training_diagnostics_2026-07-17.md` — all merged here and deleted; no content lost, just
de-duplicated and organized for scanning). **Purpose: before trying a new architecture, feature,
or label idea, check the FAILED section first** — several of these were tried more than once
across sessions before this log existed.

For the current, living picture of the system (not history), see `docs/architecture.md`,
`docs/failure_modes.md`, `docs/feature_engineering.md`, `docs/risk_management.md`,
`docs/ablation_plan.md`, and the README's "Current Status" section.

---

## Timeline

| Date | Session focus | Outcome |
|---|---|---|
| 2026-07-01 | 360° code review (5 independent read-only passes) | 2 critical, 11 high, 16 medium, 8 low findings — see §4 below |
| 2026-07-11 | Feature onboarding: alpha_data option-surface/breadth features, `OptionSurfaceEncoder` wired for the first time | Discovered H=48 beats H=192; root-caused a barrier-classifier collapse to `mae_head`/`mfe_head` |
| 2026-07-13 | Encoder dimensional-collapse root cause, anti-collapse fix, 2 real pipeline bugs | Fix implemented; honestly found NOT to move macro_f1 under a proper multi-seed re-test |
| 2026-07-15/17 | Cross-pair + ATM-delta + price-action features, second decomposed-architecture attempt, OOD bug, CVaR-dominance root cause | "Ceiling reached" verdict — stopped pursuing trading-readiness on this model/label combination |
| 2026-07-18 | Full-codebase profitability sweep: settlement multi-count bug, cost-model audit, prior-shift calibration fix, meta-labeling pipeline, utility-based checkpoint selection, 3-seed retrain | Verdict reopened then re-closed with better evidence — see §6 |

---

## 1. FAILED — do not retry without new information

1. **Decomposed touch/direction barrier head** (tried twice: before and after the 07-17 feature
   build-out). A 4-way ablation (Linear/MLP touch head × plain-BCE/asymmetric-loss) degenerates
   to a majority-class or near-constant touch probability every time, regardless of feature
   quality. Architecture-level issue, not feature-starvation.
2. **`barrier_mode: legacy`** as a fix for the H=192 majority-class collapse — doesn't fix it,
   just flips which class collapses (stop recall 87%, timeout recall 0%; macro_f1 got worse,
   0.189) (2026-07-11).
3. **"Just needs more epochs"** for the H=192 `mae_head`/`mfe_head` near-constant-output
   collapse — quantile/coverage metrics improved (rollout_mae 0.027→0.018) but barrier_macro_f1
   barely moved (0.236→0.239) and stop recall stayed at 0% (2026-07-11).
4. **Crude divide-by-training-class-weight probability correction** for calibration — drops
   Brier 0.547→0.345 but collapses argmax right back to "always timeout" (997/1000); demonstrates
   the mechanism but isn't a real fix (needs a properly-fit post-hoc recalibration instead)
   (2026-07-11).
5. **Relative variance floor for `OODHead`** (`std.clamp_min(0.25 * median(std))`) — made a
   second, more-collapsed retrain *worse*: full saturation to `ood_score=1.0` for all 1000 test
   rows, because the reference median was itself near zero (2026-07-17). Fixed instead with an
   absolute floor (`_MIN_STD=0.05`) — see §2 WORKED.
6. **Stacking ATM call/put delta on top of cross-pair features** — despite ranking #1/#2 of 124
   features by IC individually, joint classification got *worse*, not better, when both were fed
   to the model together (2026-07-16/17). Per-feature IC ranking is not sufficient evidence a
   feature will help once fused.
7. **1-minute bars / 2-hour decision horizon** — abandoned after direct cost/benefit assessment;
   not worth pursuing further (explicit user confirmation, pre-2026-07-13).
8. **Recalibrating a new fixed (even asymmetric) barrier stop/target multiplier** — a walk-forward
   check across six 4-month windows found the real stop/target touch ratio swings 0.82–1.51x with
   no consistent direction; any fixed multiplier recalibrated on one window just overfits to that
   window (2026-07-17).
9. **Risk-aversion (λ) tuning** as the fix for the zero-trade backtest result — swept λ from the
   strategy default toward near-zero on two independent unseen windows (test, val); only ever
   unlocks 1-2 trades, and every one of them loses money. Rules out miscalibration as the cause
   (2026-07-17). **⚠ Partially invalidated 2026-07-18**: the "every unlocked trade loses money"
   half of this evidence was measured with the broken settlement engine (§3 item 8) — the P&L
   of those unlocked trades is unreliable. The *trade-count* half (only 1-2 trades unlock even
   at near-zero λ) is decision-side and still stands.

## 2. WORKED — keep doing / reuse these

1. **Cross-pair beta/corr/relative-strength features** — macro_f1 **+31% relative**, the single
   biggest win recorded across all sessions. Fixes a real structural gap: `CrossAssetEncoder`
   mean-pools the time axis before cross-asset attention and cannot learn this relationship on
   its own (2026-07-15).
2. **H=48 relabeling** (from H=192) — unlocked option-surface features' peak signal window, beat
   every configuration tried up to that point including the original pre-onboarding baseline
   (macro_f1 0.474 vs. 0.424 baseline) (2026-07-11).
3. **Barrier rebalancing to `mult=1.0`** (from `mult=2.0`) — timeout dropped from ~82% to ~47%,
   fixed severe class imbalance without changing the horizon (2026-07-13).
4. **VICReg anti-collapse regularization** (`repr_var`/`repr_cov`) applied during supervised
   training, not just Stage-2 pretraining — a real, free, config-tunable fix. Multi-seed re-test
   showed it doesn't move macro_f1, but gives a modest, directionally-consistent calibration
   benefit (ECE 0.079→0.053, 2/3→3/3 gate passes) — kept as default (2026-07-13).
5. **Walk-forward-CV final-refit checkpoint-selection fix** + **`--seed` threading fix** — two
   real pipeline bugs, both fixed and verified (2026-07-13).
6. **Quantile-mode planner stop/target sizing** (`stop_target_mode="quantile"`) — real, unit- and
   integration-tested, structural improvement to the CVaR/edge ratio (~30x → ~6.6x), even though
   insufficient alone to produce a positive-expectancy trade count (2026-07-17).
7. **OOD absolute variance floor** (`_MIN_STD=0.05`) — fixed a real detector bug where
   near-collapsed latent dimensions dominated the Mahalanobis score (2026-07-17).
8. **`scripts/feature_ic_diagnostic.py`** — built from scratch (2026-07-11, no such tool existed
   before), extended to 124 features across every family (2026-07-17). Reusable for vetting any
   future feature or label-redesign idea before committing to a retrain.
9. **ATM call/put delta** — ranked #1/#2 of 124 features by walk-forward IC, the strongest
   individual signal found across all sessions, even though composing it with cross-pair features
   didn't help this round (see FAILED #6) (2026-07-16).
10. **Price-action + candle-pattern features** (12 + 8 new features, built in alpha_data) —
    validated via the IC diagnostic before trusting them, confirmed non-redundant with existing
    features (2026-07-15/16).

## 3. Bugs found & fixed (engineering, not ML-methodology)

1. `AlphaDataFuturesWindowBuilder` missing `quality_for_window` — would have crashed every
   real-data training run; added a permissive, history-availability-based implementation
   (2026-07-11).
2. Walk-forward-CV final refit never checkpoint-selected — trained on the full "pretest" split
   with `val_batches=[]`, so `HRWTrainer.fit()`'s best-checkpoint restoration never applied; the
   "final" model could be measurably worse than its own best point mid-training. Fixed to train
   on `"train"`/validate on `"val"` like every other path (2026-07-13).
3. `--seed` CLI flag never threaded into `TrainingConfig.seed` — silently a no-op for training
   reproducibility; every "different seed" comparison before this fix was byte-identical
   randomness (2026-07-13).
4. `OODHead` Mahalanobis score dominated by near-collapsed latent dimensions (only a `1e-6`
   epsilon floor) — saturated `ood_score`, crushing `PositionSizer`'s `(1-ood)` gate independent
   of real model signal (2026-07-17).
5. `prune_correlated` tuple-unpacking bug in `alpha_data/candle_patterns.py::compute_for_asset`
   — called `.to_parquet()` directly on a `(df, dropped)` tuple (2026-07-15/16).
6. `price_oi_signal` categorical-string crash in `feature_ic_diagnostic.py`'s ATM-series loader —
   fixed via an explicit score mapping applied before numpy assignment (2026-07-16/17).
7. `medium_frequency` strategy profile hardcoded to stale `decision_horizon_bars=192` after the
   project's H48 pivot — caused "artifact horizons (48,) do not include requested strategy
   horizon 192" backtest failures (2026-07-17).
8. **Backtest settlement multi-count bug (2026-07-18 — the most serious bug found in this repo
   to date).** The real-data step builders put the full H-bar forward label
   (`closes[i+H]/opens[i+1] − 1`) into `BacktestStep.realized_return`, and
   `BacktestEngine.run` settled `notional × realized_return` on EVERY bar via
   `PortfolioWorld.apply_fill` — including bars where the planner merely HELD
   (`NO_TRADE → target = current` keeps the full notional). A position held K bars therefore
   accrued K overlapping H-bar returns (~K× P&L overstatement in either direction), and a 1-bar
   hold realized a full 48-bar move. **Every backtest P&L/Sharpe/hit-rate number produced before
   this fix — including the λ-sweep "every unlocked trade loses money" evidence in §1 item 9 —
   was measured with a broken settlement engine and is unreliable.** (The *zero-trade* findings
   themselves are unaffected: n_trades is decision-side, computed before settlement.) Fixed by
   adding per-bar settlement legs (`carry_return` = prev-close→fill, `fill_to_mark_return` =
   fill→close) to `BacktestStep`, a `carry_return` leg to `apply_fill` (so EXIT realizes the
   final mark→fill move on the old notional — previously exits realized exactly 0), and wiring
   both real step builders. Regression tests poison `realized_return` with an absurd +50% label
   and assert it never reaches P&L.
9. **Barrier calibration could not correct the class-weight prior shift (2026-07-18).**
   Training uses class-weighted CE (auto-computed from label frequencies) which systematically
   inflates minority-class predicted probabilities (measured 2026-07-11: mean P(target)=27% vs.
   0.9% true base rate) — but the only fitted barrier correction was *temperature scaling*,
   which is structurally incapable of expressing a per-class prior shift. The planner's
   `exp_dW = p_stop·stop + p_target·target + …` consumed those distorted probabilities directly.
   Fixed: `PredictionCalibration.barrier_prior_offsets` — per-class log-prior offsets
   (`log(freq_c / mean_pred_c)`), fit with the same chronological-holdout-validation + 0.5
   shrinkage safeguards as the temperature, applied before temperature. Takes effect on the next
   calibration fit (existing artifacts carry no offsets).
10. `PredictionCalibration.apply()` dropped the `epistemic_calibrated` flag — rebuilt
   `ModelPrediction` without the field, silently defaulting the placeholder-epistemic marker
   back to `True` after calibration (2026-07-18).
11. `RewardScorer` scaled the edge by `fill_prob` but charged full CVaR regardless of fill —
   an unfilled order carries no position hence no tail risk, so this was a structural bias
   toward NO_TRADE that grew with fill uncertainty. CVaR is now scaled by `fill_prob` too;
   cost stays unscaled (conservative) (2026-07-18).

## 4. Still open / unverified — carried from the 2026-07-01 360° review

The review below predates the alpha_data migration (2026-07-08); file:line references to
`fetch_free_data.py`, `daily_context_loader.py`, `scripts/fetch_upstox.py`, etc. point at files
since retired or moved into the sibling `alpha_data` repo. The underlying methodology questions
are still worth a direct spot-check against current alpha_data code before treating any of these
as resolved just because the file moved — moving code doesn't fix a bug in it.

### Critical
- **C1 — same-day EOD regime leak.** Only `fii_dii_net` got a 1-day lag in the old
  `build_daily_context`; `usdinr`, `crude`, and NSE-bhavcopy-derived `atm_iv_pct`/`iv_skew_pct`/
  `pc_oi_ratio`/`basis` were stored under their own observation date with no lag — a 09:20 IST
  decision would train on day-D's closing values that don't exist yet live. Verify alpha_data's
  current daily-context pipeline lags *all* fields, not just FII/DII, before trusting any backtest
  that uses regime context.
- **C2 — live ATM IV/skew wiring is dead code.** `set_live_iv()` is never called from any
  production path. Still true as of 2026-07-17 (see README "Known Limitations" — ATM IV/skew
  always sourced from the historical, lagged `daily_context.parquet`, coverage only Jan
  2023–Jul 2024).

### High
- **H1 — RSSM train/serve distribution mismatch.** `filter()` was called with a T=1 window every
  time from a zero `initial_state`, while training rolls the recurrence over T=5+ real steps.
  *Partially addressed*: `WorldModelPredictor` now persists RSSM state across `predict_one` calls
  by default (README "Known Limitations"), but the underlying single-step-posterior-from-zero
  training/serving mismatch this finding describes has not been directly re-verified since.
- **H2 — continuous-futures roll boundary driven by file overlap, not expiry.** Adjustment ratio
  can be computed from non-contemporaneous prices when there's no true overlap window.
- **H3 — corporate-action blackout row removal creates an unguarded positional-index gap.** The
  barrier labeler's positional `t+1`/`t+H` indices silently jump across deleted-row gaps (HDFC
  merger blackout and any similar future event).
- **H4 — roll-gap bars are fabricated, not excluded, and the flag is never consumed.** A
  synthetic flat bar enters realized_vol/ATR/momentum/label computations around every roll.
- **H5 — no NaN/Inf guard in training loops.** A NaN loss from a data gap silently corrupts every
  parameter for the rest of a run with no error.
- **H6 — Stage-2 pretraining task near-identity by default.** `gap_bars=1` vs. `lookback_bars=96`
  means context/future windows overlap 99% — technically leak-free but defeats the objective's
  purpose.
- **H7 — `HeadTrainer` (documented Stage 4) is dead code.** No CLI script ever imports it; only
  exercised by unit tests.
- **H8 — backtest's "leakage-checked" run only performs the shallow static check.** The real
  per-row point-in-time and label-future checks are never invoked with real arguments.
- **H9 — epistemic-uncertainty gating hardcoded off for the plain forecaster.** Three downstream
  safety mechanisms (force-exit, risk-blocking, epistemic-scaled sizing) are silently inert for
  any deployment not using `model_kind="world_model"` (the default).
- **H10 — data-quality fail-safe defaults to off in code**, masked today only because the shipped
  `configs/paper_trading.yaml` happens to set it true.
- **H11 — `epistemic` is NaN when `n_samples=1`** — a legal call that produces a silently-false
  risk-shield comparison instead of a triggered flag.

### Medium
M1 OOD `_normalize_ood` leaks batch composition into individual scores · M2
`ExcursionBarrierHead` computes but discards `volatility_ratio` · M3 `RSSM.step_posterior` always
stochastic, no deterministic inference mode (repeated `predict_one` calls yield different outputs)
· M4 `event_calendar.py` dict-key collision on 2023-02-01 (Budget + FOMC) silently drops one label
· M5 `purged_cv.py` right-side embargo boundary computed but never applied · M6 expiry-calendar
holiday list stops 2024-11-15, later unlisted holidays could corrupt DTE/roll features · M7
`label.py` NaN-drop without re-validating bar contiguity (same class of bug as H3) · M8 *(stale as
of this review — RSSM path is now wired into `scripts/train.py` via `--model-kind world_model`)*
· M9 `RSSMLoss` fixed `beta=1.0`, sums instead of means over T — loss scale changes with horizon
count · M10 no optimizer-state persistence in checkpoints · M11 `HeadTrainer._head_parameters`
duplicated inline · M12 `world_model_metrics.py`'s KL-collapse/prior-coverage diagnostics never
called in production, so RSSM posterior collapse can occur silently · **M13 — FIXED 2026-07-16**:
the calibration gate is now enforced directly by `scripts/backtest.py`/`paper_trade.py` via
`--calibration-report`/`--allow-uncalibrated` and `_common.py::check_calibration_gate`, not just
by orchestration convention · M14 cold-start warm-up NaN→0 mismatch for rolling windows longer
than available history · M15 ATM IV/skew historical coverage gap (Jan 2023–Jul 2024 only) ·
M16 `ModelInputContract.assert_compatible` has no test coverage.

### Low
`RSSM.imagine` duplicates `RolloutEngine.rollout` (DRY) · `RSSMLoss.l_imag` not normalized per
valid-target count · `ModelConfig.fusion` allows unimplemented `"attention"`/`"moe"` values ·
`embargo_bars` not cross-validated against multi-horizon world-model training · positional
rolling windows assume uniform bar spacing · `corporate_actions.py` comment says trading days,
code uses calendar days · heuristic (non-model) backtest path is O(N²) over a long replay ·
`ModelPrediction` JSON dict keys need string-float parsing by non-Python consumers.

### What's solid (verified, not assumed) — from the 2026-07-01 review
Quantile monotonicity is structurally enforced · pinball/soft-coverage/heteroscedastic-NLL losses
are textbook-correct · Dreamer v2 KL balancing direction is correct · `FusionEncoder`
zero-padding for missing inputs is leak-safe · `fit_ood`/`OODHead.fit` correctly avoid gradient
contamination · chronological split methodology (`ChronoSplitManifest`, `PurgedKFold`,
`WalkForward`) is genuinely time-ordered with embargo at both boundaries · sample-weight plumbing
is fully wired to gradients · encoder freezing in `HeadTrainer` correctly excludes frozen params
from the optimizer · checkpoint/config coupling fails loud on mismatch, never silently wrong ·
barrier entry/exit mechanics and core rolling primitives are genuinely causal · quantile
coverage/Brier/ECE math is correct end-to-end · portfolio consequence model carries state forward
correctly · costs/spread/slippage/risk-shield vetoes are genuinely applied in the backtest, not
computed and discarded.

---

## 4b. Cost-model audit (2026-07-18) — real round-trip cost was ~2.4x too high

Direct measurement (not assumption): `round_trip_cost_frac(CostModelConfig())` returned
**31.69bps** before this fix. Two real bugs, not just conservative rounding:

1. `stt_rate=0.000625` was the OPTIONS-premium STT rate mislabeled onto a FUTURES cost
   model (real index-futures STT is ~0.02%, i.e. `0.0002` — ~3x lower), and
   `exchange_txn_rate=0.00035` was ~18x the real current NSE F&O futures transaction
   charge (~0.0019%, i.e. `0.000019`).
2. `ConservativeIndianCostModel.statutory()` charged the FULL stt_rate AND FULL
   stamp_duty_rate on every order regardless of side. Both are legally one-sided in
   India (STT on the sell leg only, stamp duty on the buy leg only) — charging both on
   both legs silently doubled their contribution to round-trip cost on top of bug #1.
3. `overnight_financing_rate_per_day=0.0008` (8bps/night) had no cited basis — standard
   NRML index-futures carrying at Indian discount brokers has no explicit daily
   financing charge (margin is blocked, which is already captured via
   `margin_fraction` sizing, but no interest is charged the way MTF/margin-funded
   equity delivery works). Reduced to `0.0001` as a small residual buffer, not zeroed,
   since this one is a judgment call rather than a hard-verified number.

**Fixed value: 12.98bps round trip** — realistic for a liquid BANKNIFTY futures contract.
This means the **25bps stress-test scenario is a genuine ~2x conservative margin above
real cost**, not a rate the strategy needs to memorize huge stop rules to overcome. Fixed
`statutory()` to be side-aware (STT only when `order.side == "sell"`, stamp duty only
when `order.side == "buy"`) and `round_trip_cost_frac()` to add STT/stamp duty once per
round trip instead of doubling them with the genuinely-symmetric components (brokerage,
exchange txn, GST, SEBI, spread, slippage). Regression tests assert the one-sided
behavior directly (`tests/test_cost_model.py`). This also lowers the labeling cost floor
(`round_trip_cost_frac` is reused as the triple-barrier cost floor and the quantile-mode
planner's `min_abs_return` floor) — a lower, more realistic floor lets smaller genuine
edges register as real barrier touches instead of being floored away, which should
increase realized signal density on retrain, not just backtest profitability.

## 5. Final status (as of 2026-07-18, revised)

**The 2026-07-17 "ceiling reached" verdict is partially reopened.** A full-codebase review on
2026-07-18 found the backtest settlement engine had been multi-counting P&L on held positions
the entire time (§3 item 8) — so while the *zero-trade decision behavior* and its CVaR-dominance
root cause remain valid findings, every backtest P&L number underpinning "the unlocked trades
lose money" was unreliable, and the barrier probabilities the planner scores with were consumed
without any correction for the class-weight-induced prior shift (§3 item 9, now fixed but
requiring a calibration refit to take effect). The honest current state: the model's *decision
layer* was being fed distorted probabilities and its *evaluation layer* was mis-measuring
outcomes — both now fixed, both requiring fresh runs (recalibrate, re-backtest) before the
ceiling question can be answered definitively. The label-redesign recommendation below still
stands as the highest-leverage direction, but it is no longer the *only* open lever.

Prior verdict (2026-07-17, for the record): the cross-pair + ATM-delta / `H=48` / `mult=1.0`
model had reached its ceiling — every fix applied that session (OOD bug, quantile-mode planner
sizing, λ calibration check) was real and tested and still didn't close the gap between the
model's directional edge and what triple-barrier CVaR requires to justify a trade on unseen
data. The open path forward is a **label redesign** (the barrier's symmetric CVaR structure and
its non-stationary touch skew are the best-evidenced structural constraint), not further
iteration on features or architecture against the current label definition.

### Post-fix re-validation (2026-07-18, fixed engine + prior-corrected calibration)

Retrained with the prior-offset calibration (`runs/hrw_h48_mult1_priorcal.pt`; fitted offsets
`stop −0.245 / target −0.481 / timeout +0.726` passed the holdout-generalization gate,
confirming the class-prior distortion was real). Test-split backtests on the FIXED settlement
engine, quantile stop/target mode:

| Artifact | λ | n_trades | hit_rate | total_return | at 5bps costs | at 25bps costs |
|---|---|---|---|---|---|---|
| old (retrain2) | default | 0 | — | 0 | — | — |
| old (retrain2) | 0.1 | 5 | 0.2 | −1.83% | — | — |
| old (retrain2) | 0.5 | 5 | 0.2 | −0.61% | — | — |
| **priorcal** | default | 0 | — | 0 | — | — |
| **priorcal** | 0.1 | 5 | **0.4** | **−1.11%** | **+0.48% (Sharpe +4.3)** | −0.21% |

Honest read: the calibration fix measurably improved trade quality (hit rate 0.2→0.4, loss
roughly halved at λ=0.1) but not to profitability under the current cost model, and n=5 trades
on one window is far too small to claim anything durable. The 5bps cost-sensitivity flipping
positive says the surviving trades' *gross* directional edge is slightly positive and the
margin is currently eaten by modeled costs + turnover (3.45x) — worth a direct audit of
`CostModelConfig` against real BANKNIFTY futures execution costs before drawing conclusions
from the current-cost numbers. Default-λ behavior is unchanged (0 trades): the corrected
probabilities shift mass toward timeout, which *lowers* expected edge as much as risk — the
CVaR-dominance structural constraint stands, and the label redesign remains the
highest-leverage open direction. See the current
`docs/*.md` files for the living picture and `scripts/feature_ic_diagnostic.py` /
the walk-forward touch-frequency methodology (§1 item 8 above) as reusable tools for evaluating
any new label definition's stationarity before committing to a full retrain.

---

## 6. Profitability push (2026-07-18): meta-labeling, checkpoint selection, 3-seed retrain

Full-codebase sweep specifically targeting "make this profitable at 25bps" (features, labels,
architecture, training, evaluation, backtesting). Two serious defects were found and fixed before
any new modeling work (§3 items 8-11 above: backtest settlement multi-count, uncorrected
class-prior shift, `fill_prob`-asymmetric CVaR). Then three structural changes, in order of
expected impact:

### 6.1 Cost-model audit — see §4b. Real round-trip cost corrected from ~31.7bps to **12.98bps**.

### 6.2 Cost-aware meta-labeling (new primary lever)

Replaced the "predict which barrier gets hit, then have the planner reverse-engineer a trading
decision from 3 noisy probabilities" pipeline with a direct question, per López de Prado's
meta-labeling framework: a momentum-based PRIMARY signal (`labeling/meta_labels.py::
primary_side_from_close`, trailing 12-bar momentum sign, deliberately simple and
model-independent) proposes a side; a new `MetaLabelHead` (binary, `heads/meta_label_head.py`)
predicts whether taking a trade in that direction — settled through the SAME triple-barrier exit
mechanics — would net more than round-trip cost. Real coverage on the regenerated
`labels_h48_mult1.parquet` (34,329 rows, corrected 12.98bps cost floor): **99.81% of rows have a
proposed primary side, and 42.07% of those clear the cost floor** (`mean_meta_label=0.4207`) — a
far better-posed, far less imbalanced target than the barrier head's ~9% minority classes, and one
that doesn't need class-reweighting (so it can't reintroduce the prior-shift problem in §3 item 9).
Wired end-to-end: label columns (`primary_side`/`meta_label`, `LABEL_SCHEMA_VERSION` 8→9) →
`ForecastBatch` → `ForecasterLoss`'s masked BCE term (NaN-sentinel rows excluded, never poison the
batch) → `ModelPrediction.meta_label_prob` → `PositionSizer`'s new multiplicative confidence gate
(neutral 1.0 whenever `primary_side==0` or the artifact predates this head — fully backward
compatible). 68 new tests across labels, model, loss, sizer, and calibration round-trip.

### 6.3 Utility-based checkpoint selection (`training/checkpoint_metrics.py`)

`HRWTrainer(..., checkpoint_metric=trading_utility_loss)` (opt-in, `--checkpoint-metric
trading_utility`) selects the best epoch/fold by "would this checkpoint's own meta-label
trade/no-trade rule have made money on held-out data" (net edge rate: +1 per correctly-taken
profitable trade, −1 per incorrectly-taken one, 0.0 neutral — matching NO_TRADE's own U=0 baseline
— when the model takes no trades at all) instead of composite validation loss. Default (`"loss"`)
preserves the exact original behavior; nothing changes unless explicitly requested.

### 6.4 Three-seed retrain + backtest (test split, quantile stop/target mode, fixed settlement engine)

Retrained seeds 7/13/21 on the regenerated meta-labeled data with `--checkpoint-metric
trading_utility`. All three completed 5-fold walk-forward-CV model selection successfully.
Backtested each across λ ∈ {default=3.0, 0.5, 0.1}:

| Seed | λ | n_trades | hit_rate | profit_factor | total_return | at 25bps |
|---|---|---|---|---|---|---|
| 7 | 3.0 (default) | 5 | 0.2 | 0.546 | −0.17% | −0.33% |
| 7 | 1.0 | 5 | 0.2 | 0.683 | −0.09% | −0.26% |
| 7 | 0.5 | 5 | **0.4** | **0.897** | **−0.03%** | −0.22% |
| 7 | 0.1 | 5 | 0.2 | 0.444 | −0.26% | −0.43% |
| 13 | 3.0 (default) | 5 | 0.4 | 0.388 | −0.89% | −1.07% |
| 13 | 0.5 / 0.1 | 4 | 0.0 | 0.000 | −2.35% | −2.49% |
| 21 | 3.0 (default) | 5 | 0.2 | 0.142 | −0.34% | −0.52% |
| 21 | 0.5 / 0.1 | 4 | 0.0 | 0.000 | −2.35% | −2.49% |

**Findings, stated plainly:**
- **A real, measurable behavioral change**: at the strategy's DEFAULT λ=3.0, all three seeds now
  take 5 trades — before this session's fixes, default λ produced ZERO trades every time (the
  entire premise of the earlier "ceiling reached" verdict). The meta-label/cost/calibration fixes
  did change decision behavior, not just diagnostics.
- **None of the 12 (seed × λ) combinations is profitable.** The single closest-to-breakeven
  result is seed 7 at λ=0.5 (profit_factor 0.897, total_return −0.03% — a near-wash, not a win).
- **Seed variance is still the dominant signal, not λ or the fixes.** At identical default λ,
  total_return ranges from −0.17% (seed 7) to −0.89% (seed 13), a >5x spread from the random seed
  alone — reproducing the exact 07-13 lesson ("run-to-run variance is comparable in magnitude to
  the effect being measured") on a completely different intervention.
- **Lower λ is not free money and is seed-dependent**: it improved seed 7 (best result at λ=0.5)
  but made seeds 13 and 21 collapse to an IDENTICAL worse outcome (4 trades, 0% hit rate, −2.35%,
  matching to 5 decimal places between the two seeds — likely the quantile-mode cost floor
  clipping multiple different models' sizing down to the same few extreme-quantile trades at low
  λ, not a coincidence worth over-interpreting further this session).
- **25bps remains unmet everywhere**, though by a smaller, more honestly-measured margin than
  before (real cost is 12.98bps, so 25bps is a ~2x stress multiplier on top of a small negative
  edge, not an arbitrary wall).
- **Not attempted this session**: a genuine multi-seed ENSEMBLE (averaging the 3 checkpoints'
  predictions at inference) — would need new `EnsembleModelRuntime`-style plumbing not built here.
  Given seed 7's markedly better behavior, averaging could plausibly help, but three single-seed
  backtests are not evidence an ensemble would — flagged as the natural next step, not claimed.

### 6.5 Honest final verdict (2026-07-18)

The session's premise — "what would make this profitable at 25bps" — surfaced and fixed real,
material defects: a backtest engine that mis-measured P&L on every multi-bar trade, a planner
consuming systematically distorted probabilities, an cost model overstating real friction by
~2.4x, and a training pipeline whose validation signal never matched the trading decision it was
meant to serve. Every one of those was a genuine bug, not a matter of interpretation, and each is
now fixed and tested (see §3 items 8-11, §4b, §6.2-6.3). Despite all four fixes compounding, real
backtests on the corrected engine, corrected costs, corrected calibration, and a genuinely
better-posed learning target still do not clear even the TRUE (12.98bps) cost, let alone 25bps —
and the seed-to-seed variance is large enough that no single number above should be treated as
"the" result. This is a materially stronger, more trustworthy negative result than the 07-17
verdict (which rested on a broken settlement engine and uncorrected probabilities), not a weaker
one — the remaining gap is real, has been measured correctly, and the two most promising
unexplored levers are (1) a genuine multi-seed ensemble and (2) the still-unattempted label
redesign (§5's original recommendation, still standing).
