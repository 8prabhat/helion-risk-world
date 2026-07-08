# Training plan (10 stages)

> Expands [`SPEC.md`](../SPEC.md) §20.

1. **Data validation & leakage prevention.**
2. **Market-state pretraining** (self-supervised: masked window, future-latent, cross-asset
   consistency, regime contrastive, vol reconstruction). **Implemented locally — never imports
   msh_jepa** (SPEC §4). JEPA-like objectives, if used, are local or shared `quanthelion` utils.
3. **Multi-horizon forecasting** (heads).
4. **Latent Market World** (`z_t→z_{t+H}`; multi-step rollout eval).
5. **Portfolio World simulation** (historical paths × synthetic account profiles).
6. **Execution Reality calibration** (real logs where available; conservative otherwise).
7. **Planner evaluation** (conservative MPC — not RL).
8. **Walk-forward backtest** (purged splits, costs, expiry/event stress, regime breakdown).
9. **Paper trading** (full audit logging).
10. **(V3) microstructure expert.**

Determinism: fixed seeds, `embargo_bars >= max horizon`, checkpoints, MLflow/null tracker via the
adapter.
