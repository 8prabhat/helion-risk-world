# Backtesting plan

> Expands [`SPEC.md`](../SPEC.md) §23.

- **Splits:** purged + embargoed walk-forward via `quanthelion.labels.embargo.make_purged_splits`.
  Never random train/test split.
- **No leakage:** no lookahead, no future option-chain, no survivorship bias; rollover + expired-
  contract mapping + corporate actions handled.
- **Costs:** brokerage, STT, exchange, GST, SEBI, stamp duty + spread/slippage; partial fills where
  possible. Shared with live (DRY).
- **Stress:** event-day and expiry-day stress tests; regime-wise breakdown.
- **Report separates:** gross vs net PnL, model accuracy, planner decisions, risk-shield
  interventions, good-trades-blocked vs bad-trades-blocked, no-trade quality, execution-cost impact,
  slippage sensitivity, regime-wise performance.
- **`leakage_report.py`** re-runs the §11.10 checks over the backtest window and fails loudly.
