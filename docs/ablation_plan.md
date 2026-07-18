# Ablation plan

> Expands [`SPEC.md`](../SPEC.md) §31. Toggle ONE factor; measure trading + calibration + no-trade
> metrics on the SAME walk-forward folds.

1. Option-surface encoder vs flattened-columns baseline.
2. Tri-plane vs dual-world (drop Execution Reality) vs market-only.
3. No-trade first-class vs forced always-in-market.
4. Uncertainty-gated sizing vs fixed sizing.
5. Stochastic Market World (S samples) vs deterministic single path.
6. Conformal/calibration envelope on vs off.
7. OOD quarantine on vs off.
8. Regime memory on vs off.
9. Cross-asset encoder on vs off.
10. Planner penalty sweep (CVaR / drawdown / execution weights).

Each ablation: fixed seed, identical folds, report delta in net Sharpe, max drawdown, ECE, quantile
coverage, and no-trade quality.

## Results: #10 executed (2026-07-16/17)

Ran as a `risk_aversion_lambda` sweep against the best-available model (cross-pair + ATM-delta,
`H=48/mult=1.0`) under the new `stop_target_mode="quantile"` planner sizing (see
`docs/risk_management.md`), on both the `test` and `val` splits (`--eval-split`), rather than the
originally-planned CVaR/drawdown/execution weight-component breakdown — the driving question
became "is the zero-trade result a risk-aversion calibration problem" rather than a general
sensitivity sweep, once §4 layers 1-4 (OOD, cost, CVaR structure, barrier non-stationarity) had
each been checked and only partially resolved the zero-trade result.

**Finding**: sweeping λ from the strategy default toward near-zero only ever unlocks 1-2 trades
total across either split, and every unlocked trade loses money (hit_rate 0, or 0.5 with
profit_factor 0.2). Replicated independently on two windows (test, val) to rule out single-window
noise. **λ miscalibration is ruled out** as the cause of the zero-trade result — the model's edge
does not clear its own CVaR estimate at any λ that isn't functionally zero. Full data and
methodology: `docs/investigation_log.md` §1 item 9.

The remaining ablation items (1-9) have not been executed as a systematic sweep; #9 (cross-asset
encoder on/off) is partially addressed by the cross-pair-feature investigation in
`docs/feature_engineering.md`, which found the encoder structurally cannot learn rolling
beta/covariance on its own regardless of on/off toggling — see `docs/architecture.md`.
