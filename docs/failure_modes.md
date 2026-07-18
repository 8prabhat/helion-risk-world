# Failure modes & rejected designs

> Expands [`SPEC.md`](../SPEC.md) §30.

## Rejected designs (and why)
- Pure buy/sell classifier — no downside/account/execution reasoning.
- RL-first — unstable/unsafe before a validated world model + costs.
- Historical order book as V1 dependency — data not reliably available.
- Portfolio vars in market prediction — non-causal leakage.
- Ignoring cost/slippage — backtest toy → live loss machine.
- Profit-only evaluation — hides drawdown/tail/overtrading.
- ML overriding the Risk Shield — removes the safety floor.
- Random train/test split / future option-chain leakage — inflated metrics.
- God classes / duplicated feature logic / hardcoded broker logic — unmaintainable, train/serve drift.

## Operational failure modes (runtime-monitored)
calibration drift · OOD regime · data gaps · stale option chain · broker timeout · margin spike ·
drawdown breach · repeated rejected fills → each maps to a Risk-Shield reaction + a logged event.

## Discovered failure modes (training/backtest investigation, 2026-07-13 → 2026-07-17)

- **Dimensional-collapse-poisoned OOD detector.** `OODHead`'s Mahalanobis-style score
  divides by each latent dimension's fitted `std`. When VICReg anti-collapse regularization
  only *reduces* (doesn't eliminate) residual dimensional collapse in `z` — severity varies
  run-to-run, observed 14/128 to 104/128 collapsed dims across otherwise-identical retrains —
  a handful of near-zero-std dimensions dominate the averaged score, saturating `ood_score`
  toward 1.0 for ordinary inputs. Downstream, `PositionSizer`'s `(1-ood)` gate crushes
  position size independent of real model signal. A first fix (relative floor,
  `0.25 * median(std)`) made a second retrain *worse* (full saturation to exactly 1.0,
  because the median itself was near zero). Fixed with an absolute floor (`_MIN_STD=0.05`),
  justified by `FusionEncoder`'s `LayerNorm` anchoring per-sample scale to O(1). See
  `docs/investigation_log.md` §3 item 4.
- **CVaR-dominated zero-trade result.** The mean-CVaR planner objective
  (`U = E[ΔW] − λ·CVaR − Cost`) computes CVaR from a discrete outcome mixture where stop/target
  legs realize their *full fixed-magnitude* barrier return when triggered, while `E[ΔW]` is a
  small net quantity. With a symmetric barrier (`stop_mult = target_mult = 1.0`, frozen at
  training time), CVaR structurally dwarfs edge unless directional skew is large — it isn't.
  Root-caused as the primary reason a well-trained model produces zero trades on real
  backtests, ahead of cost assumptions or OOD gating (both checked and ruled out first). See
  `docs/risk_management.md` and `docs/investigation_log.md` §5.
- **Non-stationary barrier-touch skew.** Walk-forward measurement of the real stop/target
  touch-frequency ratio across six 4-month windows ranged 0.82–1.51 with no consistent
  direction (full-history average 1.06). A fixed barrier multiplier — symmetric or
  recalibrated-asymmetric — is fair only on average across the whole history; any single
  window can be very unfair, so recalibrating to a new fixed value just overfits to whichever
  window it's fit on. Motivated the opt-in `stop_target_mode="quantile"` planner sizing (reads
  live `return_quantiles` instead of a frozen multiplier), which measurably reduced the
  CVaR/edge ratio (~30x → ~6.6x) without fully resolving the zero-trade result. See
  `docs/investigation_log.md` §1 item 8 and §2 item 6.
- **Backtest settlement multi-count (2026-07-18).** The real-data step builders put the full
  H-bar forward label into `BacktestStep.realized_return`, and the engine settled
  `notional × realized_return` on every held bar — a K-bar hold accrued K overlapping H-bar
  returns, and exits realized nothing (the final close→fill move was dropped). Every backtest
  P&L/Sharpe/hit-rate produced before the fix was unreliable; zero-trade *counts* were
  unaffected (decision-side). Fixed with per-bar settlement legs
  (`carry_return`/`fill_to_mark_return`) + regression tests that poison the H-bar label and
  assert it never reaches P&L. Lesson: a settlement engine needs an invariant test tying a
  multi-bar trade's booked P&L to its entry-fill→exit-fill price move — decision-side tests
  can all pass while the accounting is wrong. See `docs/investigation_log.md` §3 items 8-11.
- **Decomposed touch/direction head degenerate collapse.** Splitting the barrier head into
  separate touch-probability and direction heads was tried twice (once before, once after a
  major feature build-out), across a 4-way ablation (Linear/MLP touch head × plain-BCE/
  asymmetric-loss). All variants degenerate to a majority-class or near-constant touch
  probability regardless of head capacity, loss weighting, or feature quality — ruling out
  architecture (not features) as the cause. See `docs/investigation_log.md` §1 item 1.
