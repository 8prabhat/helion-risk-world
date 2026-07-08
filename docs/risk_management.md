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
