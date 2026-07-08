# Execution Reality Layer

> Expands [`SPEC.md`](../SPEC.md) §15. "Even with a good forecast, can this trade be executed
> profitably?"

## Components (each a Protocol — LSP/OCP)
`CostModelProtocol` (brokerage + STT + exchange + GST + SEBI + stamp duty + spread) ·
`SlippageModel` · `LiquidityModel` · `LatencyModel` · `FillSimulator` (fill/partial/reject).

## Realism score
- **high** — survives costs & fill assumptions → allow.
- **medium** — reduce size or require higher edge.
- **low** — block.

## Calibration tiers
V1: conservative assumptions (no historical depth). V2: calibrate against own fill logs. V3: LOB
microstructure expert. Statutory rates in `configs/*.yaml` are illustrative placeholders — replace
with calibrated values from broker contract notes / SEBI circulars before paper/live use.
