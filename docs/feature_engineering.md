# Feature engineering

> Expands [`SPEC.md`](../SPEC.md) §10. ONE shared `FeatureBuilder` for train/backtest/paper (DRY).

## Groups
- **Market (candle):** log/simple returns, realized vol, ATR, volume z-score, rolling beta/corr,
  trend strength, range compression/expansion, gap size, time-of-day, day-of-week.
- **Futures:** price, volume, OI, ΔOI, basis, near/next spread, rollover.
- **Option surface (ATM-relative):** per token `ATM-N..ATM+N`: call/put OI, ΔOI, volume, IV,
  delta/gamma/theta/vega, moneyness, DTE. Derived: PCR, IV skew, gamma concentration, walls,
  max-pain proxy, expiry pressure, ATM/wing IV. Missing strikes are **masked**, never dropped.
- **Regime/event:** VIX (+percentile), expiry/event flags, blackout, RBI/Fed/CPI/budget/election,
  FII/DII, global cues, USDINR, crude.
- **Portfolio (Portfolio World/Planner/Sizer/Shield/backtest ONLY).**
- **Execution (Execution Reality/Planner ONLY).**

## Shared primitives (DRY)
`compute_returns, realized_vol, atr, oi_change, align_to_atm, map_expiry, transaction_cost,
estimate_slippage`. Train and backtest call the SAME functions — enforced by
`tests/test_no_leakage.py::test_train_backtest_feature_parity` (to be added with the builder).
