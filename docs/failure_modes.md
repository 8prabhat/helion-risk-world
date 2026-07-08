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
