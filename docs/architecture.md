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
