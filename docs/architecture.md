# Architecture deep-dive

> Companion to [`SPEC.md`](../SPEC.md) §11–§18. This file expands the model internals; the SPEC is
> the contract.

## Tri-plane overview

HelionRiskWorld separates three concerns that single-classifier systems conflate:

1. **Market World** — what futures are possible? (distributions, not a point)
2. **Portfolio World** — can *this account* afford the consequence?
3. **Execution Reality** — can the trade be filled profitably after costs?

The planner consumes all three; the Risk Shield can veto the planner.

## Forward pass

```
M_t ──► encoders ──► z_t ──► MarketWorld.rollout (S samples) ──► heads ──► ModelPrediction
```

- `z_t = Fusion(Temporal, CrossAsset, OptionSurface, Regime)` — market plane only.
- `rollout` produces `[S, B, |H|, d]`; the ensemble spread is the epistemic-uncertainty signal.
- Heads emit distributions per horizon (return quantiles, direction, vol, MAE/MFE, barrier, regime)
  plus uncertainty, OOD, and a learned execution-cost prior.

## Why compact modules (Mac Studio 64GB)

Prefer patch encoders / compact SSM / TCN over giant Transformers. Bounded option-chain windows,
offline feature caching, memory-mapped datasets, staged training, gradient accumulation,
checkpointing. Small (latent 128) and Medium (latent 256) variants only; Large is GPU-cluster.

## Latent dynamics

`z_{t+1} = f_θ(z_t, ε)`. V1 default: compact recurrent state-space block (`RecurrentStateSpaceDynamics`).
Pluggable via `LatentDynamicsProtocol` (swap GRU / SSM / latent stochastic dynamics without touching
the rollout engine — OCP).

## Causal boundary (critical)

Encoders accept only market-plane batches. `PortfolioState`/`ExecutionState` are structurally
unable to reach `forward()`. Enforced by schema typing, builder separation, the
`MARKET_FEATURE_NAMES`/`PORTFOLIO_FEATURE_NAMES` registries, and `tests/test_no_leakage.py`.

## `CrossAssetEncoder` pooling order (structural limitation, fixed via features not architecture)

`CrossAssetEncoder` mean-pools the time axis *before* running self-attention across assets.
That ordering means it structurally cannot learn a rolling beta/covariance relationship
between instruments — by the time attention runs, the temporal structure needed to infer
"how much does X move for a 1% move in Y" is already averaged away. Rather than restructure
the encoder, this was closed by feeding a precomputed rolling `cross_pair_beta`/`cross_pair_corr`/
`cross_pair_relative_strength` directly as candle features (`market_window_builder.py` columns
27-29, replacing three near-dead Kalman-filter columns) — the single largest accuracy win
measured in this repo's investigation history (macro_f1 +31% relative). See
`docs/investigation_log.md` §2 item 1.

## Decomposed touch/direction head — tried twice, reverted both times

The barrier head's 3-class (`stop`/`target`/`timeout`) design was twice replaced experimentally
with a decomposed pair of heads (touch probability + direction), most recently across a 4-way
ablation (Linear vs. MLP touch head × plain-BCE vs. asymmetric loss) after a major feature
build-out (cross-pair + ATM delta + price-action + candle-pattern features). All variants,
both times, degenerate to a majority-class or near-constant touch probability — this rules out
"the shared encoder just needs better features to feed a decomposed head" and points instead to
a training-dynamics issue specific to decomposing touch and direction on this label definition.
Current `BarrierHead` (single 3-class softmax) remains the production design. See
`docs/investigation_log.md` §1 item 1.

## `OODHead` variance-floor fix

`OODHead.fit()` computes a per-latent-dimension Mahalanobis-style `std` denominator. Because
VICReg anti-collapse regularization (`repr_var`/`repr_cov`, see `docs/investigation_log.md`
§2 item 4) only reduces, not eliminates, residual
dimensional collapse in `z`, near-zero-std dimensions used to dominate the averaged OOD score
regardless of actual anomalousness (only a `1e-6` epsilon floor existed). Fixed with an
**absolute** floor, `_MIN_STD = 0.05` (a *relative* floor tried first, `0.25 * median(std)`,
made a highly-collapsed retrain worse — full saturation to 1.0, because the reference median was
itself near zero — see `docs/investigation_log.md` §1 item 5). The absolute floor is justified by
`FusionEncoder`'s `nn.LayerNorm(latent_dim)` anchoring per-sample scale to O(1) regardless of
collapse severity.

## Planner: quantile-based stop/target sizing (opt-in)

`PortfolioWorld._build_dW_distribution` and `PlannerConfig` support an opt-in
`stop_target_mode: Literal["barrier_context", "quantile"]` (default `"barrier_context"`,
preserving prior behavior exactly). In `"quantile"` mode, stop/target legs are sized per-decision
from the model's own predicted `return_quantiles[0.1]`/`[0.9]` (via new
`ModelPrediction.quantile_stop_return`/`quantile_target_return` resolvers in
`schemas/prediction_schema.py`) instead of the frozen, symmetric `BarrierContext` multiplier
baked in at training time. Motivated by a walk-forward finding that the real stop/target
touch-frequency ratio is non-stationary (0.82–1.51 across six 4-month windows) — no fixed
multiplier can be right most of the time. Wired through `MPCPlanner.default(...)` and exposed via
`scripts/backtest.py --stop-target-mode {barrier_context,quantile}`. See
`docs/investigation_log.md` §2 item 6 for the measured effect (CVaR/edge ratio
~30x → ~6.6x) and its limits (still insufficient alone to produce a positive-expectancy trade
count on the unseen test window).
