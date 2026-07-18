# Feature engineering

> Expands [`SPEC.md`](../SPEC.md) §10. ONE shared `FeatureBuilder` for train/backtest/paper (DRY).

## Groups
- **Market (candle):** log/simple returns, realized vol, ATR, volume z-score, rolling
  `cross_pair_beta`/`cross_pair_corr`/`cross_pair_relative_strength` (implemented 2026-07-15,
  see below — replaces an earlier aspirational "rolling beta/corr" placeholder that was actually
  a near-dead Kalman-trend triple), trend strength (ADX/DMI), range compression/expansion
  (Donchian/Keltner position, consolidation score), price-action (higher-high/lower-low, pivot
  distance, RSI-price divergence, wick %, gap-from-prev-close), candle-pattern strength scores
  (engulfing/doji/marubozu/hammer/shooting-star/inside-outside bar), gap size, time-of-day,
  day-of-week.
- **Futures:** price, volume, OI, ΔOI, basis, near/next spread, rollover.
- **Option surface (ATM-relative):** per token `ATM-N..ATM+N`: call/put OI, ΔOI, volume, IV,
  delta/gamma/theta/vega, moneyness, DTE. Derived: PCR, IV skew, gamma concentration, walls,
  max-pain proxy, expiry pressure, ATM/wing IV, and (2026-07-16) `atm_call_delta`/`atm_put_delta`
  broadcast into `SURFACE_CONTEXT_FEATURES` (10→12 wide). Missing strikes are **masked**, never
  dropped.
- **Regime/event:** VIX (+percentile), expiry/event flags, blackout, RBI/Fed/CPI/budget/election,
  FII/DII, global cues, USDINR, crude.
- **Portfolio (Portfolio World/Planner/Sizer/Shield/backtest ONLY).**
- **Execution (Execution Reality/Planner ONLY).**

## Cross-pair context (2026-07-15)

Rolling OLS beta, correlation, and relative return strength of each instrument against a
reference (bank constituents vs. BANKNIFTY_FUT; BANKNIFTY_FUT vs. NIFTY; NIFTY/FINNIFTY
zero-filled, no natural reference), computed in `alpha_data`'s `technical_features.py` and loaded
via `helion_risk_world/data/alpha_cross_pair_context.py`. Not a cosmetic addition:
`CrossAssetEncoder` mean-pools the time axis before cross-asset attention, so it structurally
cannot learn this relationship on its own — see `docs/architecture.md`. Measured **macro_f1 +31%
relative**, the largest single feature win recorded in this repo.

## ATM option greeks (2026-07-16)

`atm_call_delta`/`atm_put_delta`, sourced from alpha_data's per-expiry-cycle ATM greeks parquets
via `AlphaDataAtmGreeksLoader` (`data/alpha_option_chain.py`), with a local fallback computed
directly from the ATM-token row when cycle data is unavailable for a timestamp. **Ranked #1 and
#2 of 124 features by walk-forward IC** — the strongest individual signal found in this repo's
feature investigation. Note: composing it with cross-pair features made joint barrier
classification *worse*, not better, despite the strong marginal IC — see
`docs/investigation_log.md` §1 item 6. Treat per-feature IC ranking as necessary
but not sufficient evidence a feature will help once fused with others.

## Price-action and candle-pattern features (2026-07-15/16)

12 price-action concept-list entries (Donchian/Keltner position, consolidation score, price
acceleration, higher-high/lower-low, pivot distance, RSI-price divergence, wick %,
gap-from-prev-close) and 8 continuous candle-pattern strength scores (not binary flags:
`bullish_engulfing_score`, `bearish_engulfing_score`, `doji_score`, `marubozu_score`,
`hammer_score`, `shooting_star_score`, `inside_bar_score`, `outside_bar_score`), built in
`alpha_data/pipelines/technical_features.py` and `pipelines/candle_patterns.py` respectively.
Both were checked against real walk-forward IC (not assumed useful) before being trusted — see
`scripts/feature_ic_diagnostic.py`, now covering 124 features total across all families.

## IC diagnostic (`scripts/feature_ic_diagnostic.py`)

Walk-forward Spearman rank correlation between each feature and forward returns, extended this
session to cover candle/futures/regime/surface/priceaction/candlepattern/atmce/atmpe/atmpcr/
straddle/otm5ce/otm5pe families (124 features). The standard tool for vetting any new feature
before wiring it into training — reuse it for future feature or label-redesign work rather than
trusting a feature's apparent usefulness without a walk-forward check.

## Shared primitives (DRY)
`compute_returns, realized_vol, atr, oi_change, align_to_atm, map_expiry, transaction_cost,
estimate_slippage`. Train and backtest call the SAME functions — enforced by
`tests/test_no_leakage.py::test_train_backtest_feature_parity` (to be added with the builder).
