# Risk management

> Expands [`SPEC.md`](../SPEC.md) §19, §22. The Risk Shield is deterministic; ML never overrides it.

## Hard rules (each a `RiskRuleProtocol`)
daily-loss-limit → block · max-drawdown → stop · free-margin floor → reduce/exit · uncertainty >
thr → no-trade · OOD > thr → no-trade · expected-return < total-cost → no-trade · event blackout →
no-trade/reduce · slippage-risk high → no-trade · exposure exceeded → reduce/deny · max trades/day →
no-trade · consecutive-losses → cool-down · execution-realism low → no-trade.

## Default-safe
If a rule path is unimplemented, the shield returns NO_TRADE rather than silently allowing a trade
(`RiskShield._no_trade`). Tests assert the override path and the NO_TRADE fallback.

## Risk metrics
VaR, CVaR, drawdown duration, daily-loss/exposure/margin breach counts, shield-intervention count,
consecutive-loss behaviour.

## CVaR dominance finding (2026-07-16/17)

The mean-CVaR planner objective (`U = E[ΔW] − λ·CVaR − Cost`, `RewardScorer.score()`) was found
to structurally dominate the model's edge on real backtests, producing **zero trades** on unseen
data even from the best-available model. `PortfolioWorld._build_dW_distribution` builds a
discrete outcome mixture where stop/target legs realize their **full fixed-magnitude** barrier
return when triggered, while `E[ΔW]` is a small net quantity — with a symmetric barrier
(`stop_mult = target_mult = 1.0`, frozen at training time), CVaR dwarfs edge unless directional
skew is large, and it isn't. This was root-caused ahead of both OOD gating (a real bug, fixed,
but insufficient alone — see `docs/architecture.md`) and cost-model assumptions (checked
directly, a small contributor, not the primary cause).

**Compounding cause**: a walk-forward check of the real stop/target touch-frequency ratio across
six 4-month windows found it non-stationary (0.82–1.51x, full-history average 1.06) — no single
fixed barrier multiplier, symmetric or recalibrated-asymmetric, is right most of the time.

**Fix implemented**: opt-in `stop_target_mode="quantile"` (`PlannerConfig`, `PortfolioWorld`,
`--stop-target-mode` CLI flag on `scripts/backtest.py`) sizes stop/target from the model's own
live `return_quantiles[0.1]`/`[0.9]` instead of the frozen multiplier — asymmetric and
regime-adaptive, recomputed every decision. Measured effect: CVaR/edge ratio improved ~30x → ~6.6x,
a real, tested, structural improvement — but still insufficient alone to produce a
positive-expectancy, non-trivial trade count on the held-out test window.

**Risk-aversion (λ) sweep, ruled out as the remaining cause**: swept `risk_aversion_lambda` from
the strategy default toward near-zero, both on the test split and independently replicated on the
val split (a different unseen window). Even near-zero λ only ever unlocks 1-2 trades total, and
every unlocked trade loses money:

| λ (val split) | n_trades | total_return | Sharpe | hit_rate | profit_factor |
|---|---|---|---|---|---|
| 1.0 | 1 | -0.01478 | -12.459 | 0 | 0 |
| 0.5 | 1 | -0.04135 | -12.492 | 0 | 0 |
| 0.1 | 2 | -0.03307 | -9.581 | 0.5 | 0.2 |

This rules out λ miscalibration — the model's directional edge does not clear its own CVaR
estimate at any reasonable risk-aversion setting. Full root-cause chain and final verdict:
`docs/investigation_log.md` §1 item 9 and §5.
