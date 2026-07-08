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
