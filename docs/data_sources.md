# Data sources & availability

> Expands [`SPEC.md`](../SPEC.md) §8. Source→field→V-tier mapping lives in
> [`configs/data_sources.yaml`](../configs/data_sources.yaml).

## V1 (no historical tick/depth)

candles (5m) · open interest · futures OHLCV+OI · India VIX · expiry calendar · event calendar ·
FII/DII (daily) · global cues (USDINR, crude, SGX/Dow prior close) · option-chain snapshots *if
available* · live bid/ask (paper only) · simulated portfolio states · conservative cost model.

## Point-in-time discipline

Every record carries `ts` and `available_at`. Builders assert `available_at <= ts`. FII/DII is
daily-granularity and only available end-of-day — never stamp it intraday.

## IV / greeks reconstruction

When a vendor exposes only live greeks, reconstruct an IV history from option OHLC via
`quanthelion.options.black_scholes.implied_vol`; sanity-check supplied greeks against
`quanthelion.options.greeks`.

## V2 / V3

V2: self-collected option-chain snapshot history, own execution logs, live depth.
V3: vendor historical tick/depth, LOB expert. **Never** a V1 dependency.
