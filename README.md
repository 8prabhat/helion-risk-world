# HelionRiskWorld

Research code for a risk-aware trading model stack targeting BankNIFTY futures on NSE.

- **Market World**: encode market state → predict return distribution, volatility, barrier outcome, regime, uncertainty, OOD score.
- **Portfolio World**: estimate account-level consequence of acting on a prediction.
- **Execution Reality**: apply spread, slippage, fees, fill assumptions, and risk vetoes.

This repository is for research, simulation, backtesting, and dry-run paper trading only.

---

## Current Status (2026-07-18)

A full-codebase profitability sweep (features, labels, architecture, training,
evaluation, backtesting) found and fixed four real defects, then added a new
meta-labeling pipeline and re-validated on 3 retrained seeds:

1. **Backtest settlement multi-count** — the engine settled the full H-bar forward
   label against the position notional on *every held bar*, multi-counting P&L on any
   multi-bar hold. Fixed with per-bar settlement legs
   (`carry_return`/`fill_to_mark_return`) plus regression tests.
2. **Uncorrected class-prior shift in barrier probabilities** — class-weighted training
   provably inflates minority-class probabilities; the only fitted correction
   (temperature scaling) cannot express a per-class prior shift. Fixed with
   holdout-validated per-class log-prior offsets in `PredictionCalibration`.
3. **Cost model overstated real friction by ~2.4x** (31.7bps → corrected **12.98bps**
   round-trip) — a mislabeled options STT rate applied to futures, an 18x-too-high
   exchange charge, and one-sided taxes (STT/stamp duty) double-charged on both legs.
4. **Meta-labeling pipeline** (new): replaced the 3-class barrier-probabilities-then-CVaR
   pipeline with a direct, cost-aware binary question — "would a trade in the
   momentum-based primary side's direction clear round-trip cost" — via a new
   `MetaLabelHead`, wired into training, calibration, and a `PositionSizer` confidence
   gate. Paired with a utility-based checkpoint-selection option
   (`--checkpoint-metric trading_utility`).

**Result after all four fixes + retraining 3 seeds**: the strategy's default risk
setting now genuinely takes trades (5 per seed on the held-out test window — before
this session it took zero), a real behavioral change. But **none of 12 (seed × λ)
backtest combinations was profitable**; the closest to breakeven was profit_factor
0.897 (essentially a wash). Seed-to-seed variance remains larger than any single fix's
effect (total_return ranged −0.17% to −0.89% across seeds at identical settings). This
is a stronger, more trustworthy negative result than the prior "ceiling reached"
verdict (which rested on the broken settlement engine and uncorrected probabilities),
not a weaker one. **Two open, unexplored levers remain: a genuine multi-seed ensemble
predictor (not built this session) and the label redesign** (still the
highest-leverage structural direction). Full ledger of everything tried (what worked,
what failed, what's open — check before retrying an old idea): `docs/investigation_log.md`.

---

## Architecture Overview

```
RAW DATA (market-plane only — no portfolio or account fields)
    │
    ├── OHLCV bars: BANKNIFTY + universe (NIFTY, HDFCBANK, …)
    └── BankNIFTY futures continuous (for microstructure + labeling)

        ▼ Stage 0: assemble_data.py
        5-min bars, roll-gap fill, merger blackout drop, basis computation

        ▼ Stage 1: label.py
        Triple-barrier labels (stop / target / timeout) + uniqueness weights + heuristic regime labels

═══════════════════════════  MODEL FORWARD PASS  ═══════════════════════════

 CANDLE WINDOW [B, A, L, F]      FUTURES [B, T, 13]     REGIME CONTEXT [B, 20]
 A = assets  (universe size)     T = lookback            VIX, IV, expiry flags,
 L = lookback bars (96 default)  13 features per bar     FII/DII, event type
 F = 9 features per bar
        │                               │                        │
  TemporalEncoder             FuturesEncoder              RegimeEncoder
  (per-asset GRU,             (1-D Conv × 2 →             (2-layer MLP)
   mean-pool over assets)      global avg pool)
        │                               │                        │
   [B, d=128]                      [B, d=128]              [B, d=128]
        │                               │                        │
  CrossAssetEncoder                     │                        │
  (multi-head self-attn                 │                        │
   across assets → pool)               │                        │
        │                               │                        │
   [B, d=128]                          │                        │
        └─────────────────┬────────────┘────────────────────────┘
                          │
                  FusionEncoder (gated)
                  4-slot gate+candidate; zero-pads missing inputs
                          │
                     z_t [B, 128]
                          │
    ┌──────────┬──────────┬────────────┬───────────┬────────────┐
    │          │          │            │           │            │
  Return    Volatility  Barrier    Uncertainty  Regime       OOD
  Quantile  Head        Head       Head         Head         Head
  Head      (softplus,  (3-class:  (epistemic + (6-class     (fitted post-
  (5        strictly>0) stop/      aleatoric,   logits:      training on
  monotone              target/    softplus>0)  trend/range/ train latents)
  quantiles)            timeout)               event/hi-vol/
                                               lo-vol/chop)
    │          │          │            │           │            │
 [B, 5]      [B]       [B, 3]       [B, 2]      [B, 6]      [B, 1]
```

### Candle Features (F=30 per bar, `market_window_builder.py`)

| # | Feature | Computation |
|---|---------|-------------|
| 0 | `log_return` | log(close_t / close_{t-1}) |
| 1 | `hl_range` | (H−L)/C intrabar range fraction |
| 2 | `open_close_norm` | (C−O)/(ATR%·C) bar direction |
| 3 | `realized_vol_short` | 12-bar Rogers-Satchell OHLC vol (feature/label overhaul Phase 2) |
| 4 | `realized_vol_long` | 60-bar Rogers-Satchell OHLC vol |
| 5 | `atr_pct` | ATR/close, price-normalized |
| 6 | `bb_position` | Bollinger Band position, window=20 |
| 7 | `rsi_14` | RSI(14)/100 in [0,1] |
| 8 | `momentum_norm` | 12-bar momentum / (ATR%·C·√12) |
| 9 | `session_return` | (C − first_bar_close_today) / first_bar_close |
| 10 | `high_low_pos` | Position in 12-bar rolling H/L range, [0,1] |
| 11 | `volume_zscore` | 20-bar volume z-score (0 for NSE indices) |
| 12 | `oi_norm` | OI / 96-bar mean OI (0 for indices) |
| 13 | `d_oi_pct` | Fractional OI change (0 for indices) |
| 14–17 | `tod_sin/cos`, `dow_sin/cos` | Cyclic time-of-day / day-of-week encodings |
| 18 | `rel_log_return` | asset log_return − primary-asset log_return |
| 19 | `adx_14` | Trend strength (magnitude), ADX(14)/100 |
| 20 | `dmi_diff_14` | Trend direction (sign), (+DI−−DI)/100 |
| 21 | `variance_ratio_20` | Lo-MacKinlay VR(20) − 1.0; 0 = random walk |
| 22 | `vol_ratio_short_long` | realized_vol_short / realized_vol_long — vol-of-vol regime |
| 23 | `opening_range_position` | Causal position within the first-15-min opening range |
| 24 | `first_15min_return` | Cumulative return over the first 15 min, frozen after |
| 25 | `breadth` | Fraction of universe (excl. primary) with positive 12-bar return |
| 26 | `dispersion` | Cross-sectional std of universe 12-bar returns |
| 27 | `cross_pair_beta` | Rolling OLS beta vs. a reference instrument (2026-07-15) |
| 28 | `cross_pair_corr` | Rolling correlation vs. the same reference |
| 29 | `cross_pair_relative_strength` | Relative return strength vs. the reference |

`breadth`/`dispersion` are market-wide scalars broadcast identically into every asset's
row; `rel_log_return` is per-asset-differentiated. Both are filled in by
`FeatureBuilder` after stacking all universe symbols (see `feature_builder.py`).
`cross_pair_beta`/`cross_pair_corr`/`cross_pair_relative_strength` replaced the earlier
`kalman_trend`/`kalman_innovation_norm`/`kalman_trend_uncertainty` triple on 2026-07-15
(IC diagnostic found the Kalman columns near-dead: `kalman_trend_uncertainty` IC=0.0,
`kalman_innovation_norm`≈0.000, `kalman_trend` weak and redundant with `ema_dist`).
Sourced from `data/alpha_cross_pair_context.py` against alpha_data's per-pair
`{symbol}_vs_{reference}_5min.parquet` files (each bank constituent vs. BANKNIFTY_FUT,
BANKNIFTY_FUT vs. NIFTY; NIFTY/FINNIFTY zero-filled, no natural reference). This closes
a real structural gap in `CrossAssetEncoder`, which mean-pools the time axis *before*
cross-asset attention and so cannot learn rolling beta/covariance on its own — see
`docs/investigation_log.md` §2 (macro_f1 +31% relative from this
change alone, the largest single win recorded across every session). `ARTIFACT_VERSION` bumped
14→15 for this swap (same F=30 width, semantics-only).

### Futures Microstructure Features (F=14, `futures_window_builder.py`)

| # | Feature | Notes |
|---|---------|-------|
| 0 | `basis` | (close_fut − close_spot) / close_spot |
| 1 | `oi_norm` | OI normalised by 20-bar rolling mean |
| 2 | `d_oi` | Bar-to-bar OI change |
| 3 | `volume_zscore` | Futures volume z-score |
| 4 | `calendar_spread` | Near − next month; 0 outside a roll's near/next overlap window |
| 5 | `dte_norm` | Days to expiry / 30, in [0, 1]; uses NSE last-Thursday rule |
| 6 | `roll_flag` | 1.0 when within 5 bars of expiry |
| 7 | `d_oi_mag` | \|ΔOI\| magnitude |
| 8 | `oi_available` | 1.0 if OI has a real value this bar, 0.0 if missing/NaN (review Idea #5) |
| 9 | `oi_basis_interaction` | d_oi · sign(Δbasis) — genuine accumulation vs short-covering-driven basis moves (feature/label overhaul Phase 2) |
| 10–13 | `oi_flow_onehot` | long-buildup / short-covering / short-buildup / long-unwinding |

### Regime Context Features (K=22, `regime_builder.py`)

VIX level + percentile, ATM IV, IV skew, expiry flag, event-day flag, blackout flag,
FII/DII net flow (rolling 60-day z-score, not a fixed-divisor rescale — feature/label
overhaul Phase 0), USD/INR and crude oil (5-day rate-of-change, not raw level — both
are trending macro drivers, not mean-reverting flows), put-call OI ratio (rolling
z-score), basis, `usdinr_vol`/`crude_vol` (rolling realized-vol of the macro
rate-of-change series — cross-asset vol transmission signal, Phase 2), a
`regime_missing_mask` flag (1.0 when ATM IV/IV skew/PCR/basis are all unavailable —
review Idea #5, distinguishes "no signal" from "genuinely zero/neutral"), and 7-way
event-type one-hot.

### Target Variables (supervised labels)

| Target | Head | Loss |
|--------|------|------|
| Return quantiles (q10/25/50/75/90) | `ReturnQuantileHead` | Pinball |
| Future realized volatility | `VolatilityHead` | Huber |
| Barrier hit (stop / target / timeout) | `BarrierHead` | Cross-entropy |
| Epistemic + aleatoric uncertainty | `UncertaintyHead` | Heteroscedastic NLL |
| Market regime (6-class heuristic from primitives.regime_label) | `RegimeHead` | Cross-entropy |
| OOD score | `OODHead` | Fitted post-training (no gradient) |

### World Model (RSSM) — Research Path

```
z_t [B, d] → filter() → RSSMState(h_t, z_t)
                              ↓
                         imagine() → ensemble [S=16, B, H, deter+stoch]
                                          ↓
                              per-horizon return_head / vol_head / barrier_head
```

The RSSM prior is trained (KL to posterior) so ensemble spread is calibrated epistemic
uncertainty, not random noise.  Stage 3 training uses `WorldModelTrainer.encode_sequence()`
to produce `[T, B, embed_dim]` input sequences.

---

## CLI Workflow (Stages –1 to 7)

```
Stage -1  alpha_data (sibling repo)  → alpha_data/data/ohlcv/*.parquet  (Upstox API)
Stage -1  alpha_data (sibling repo)  → alpha_data/data/regime/daily_context.parquet
          (Yahoo Finance + NSE; RETIRED 2026-07-08: this repo's own fetch_upstox.py /
          fetch_free_data.py / fetch_nse_bhavcopy.py, all superseded — see
          alpha_data/docs/DATA_CATALOG.md)
Stage 0   assemble_data.py    → data/processed/banknifty_5min.parquet
Stage 1   label.py            → data/processed/labels.parquet
Stage 2   (encoder pretraining — optional, see pretrain_market_state.py / train.py)
Stage 3   (RSSM training      — research path, see train_world_model.py)
Stage 4   train.py / train_heads.py  → runs/forecaster.pt
Stage 5   calibrate.py        → exit 0 (PASS) or 1 (FAIL)
Stage 6   backtest.py         → runs/backtest/decisions.jsonl
Stage 7   predict.py          → JSON prediction on stdout
```

---

## Repository Layout

```
scripts/
  (fetch_upstox.py, fetch_free_data.py, fetch_nse_bhavcopy.py RETIRED 2026-07-08 —
   see alpha_data/scripts/backfill_all.py and alpha_data/pipelines/regime.py)
  assemble_data.py      Stage 0: merge raw parquets, handle roll gaps + merger blackout
  label.py              Stage 1: triple-barrier labels + uniqueness weights + regime labels
  train.py              Stage 2+4: optional encoder pretraining + forecaster training
  calibrate.py          Stage 5: calibration gate (exit 0=PASS / 1=FAIL)
  backtest.py           Stage 6: heuristic or model-backed backtest
  generate_report.py    Summarize backtest/paper outputs into durable JSON review payloads
  predict.py            Stage 7: emit one ModelPrediction as JSON
  paper_trade.py        Dry-run paper trading loop with audit logging, data fail-safe, + monitor_summary.json
  train_workflow.py     Full retraining workflow → calibration_report.json + report_summary.json + workflow_summary.json
  build_features.py     Dev helper: build one MarketBatch and cache as .npz
  validate_data.py      Sanity-check raw OHLCV files

src/helion_risk_world/
  config/               ModelConfig, DataConfig, TrainingConfig
  data/
    upstox_client.py            Upstox API wrapper (V3 analytics + ExpiredInstrumentApi)
    continuous_futures.py       Stitch monthly contracts → backward-adjusted continuous
    event_calendar.py           NSE/RBI/macro event calendar (RBI, Budget, Fed, CPI, Election)
    daily_context_loader.py     Load daily_context.parquet (USD/INR, crude, FII/DII)
    regime_context_builder.py   Assemble RegimeContext + EventContext at any timestamp
    market_window_builder.py    [A, L, F] candle tensors
    futures_window_builder.py   [T, 14] futures microstructure (FuturesEncoder input)
    expiry_calendar.py          NSE BankNIFTY expiry dates + DTE computation
    corporate_actions.py        HDFC merger blackout (2023-07-01 ± 5 bars)
    rollover.py                 Roll-gap detection in continuous futures series
    feature_builder.py          MarketBatch + FeatureBuilder (training/backtest/paper DRY)
    kalman_trend.py             2-state Kalman local-linear-trend filter (feature/label overhaul Phase 3)
    primitives.py               Stateless feature primitives + regime_label()
    regime_builder.py           RegimeContext → [K=22] feature vector
  encoders/
    temporal_encoder.py         [B, A, L, F] → [B, d]  per-asset GRU + pool
    cross_asset_encoder.py      [B, A, L, F] → [B, d]  self-attn across assets
    futures_encoder.py          [B, T, 14]  → [B, d]  1-D conv + global avg pool
    regime_encoder.py           [B, 22]     → [B, d]  MLP
    fusion_encoder.py           4-slot gated fusion  → z_t [B, d]
  heads/
    return_head.py              Monotone quantile regression [B, Q]
    volatility_head.py          Realized vol (> 0)  [B]
    barrier_head.py             3-class logits [B, 3]
    uncertainty_head.py         Epistemic + aleatoric [B, 2]
    regime_head.py              6-class logits [B, 6]
    ood_head.py                 OOD score fitted post-training [B, 1]
  model.py                      HRWForecaster, HRWWorldModel
  inference.py                  ForecasterPredictor, WorldModelPredictor
  worlds/
    rssm.py                     RSSM: prior/posterior/GRU/decode
    market_world.py             filter() + imagine() + per-horizon heads
    rollout_engine.py           @no_grad ensemble rollout
    portfolio_world.py          Portfolio-level consequence model
  losses/
    quantile_loss.py            Pinball loss (device-safe)
    composite_loss.py           ForecasterLoss: weighted sum
    rssm_loss.py                L_dyn + L_imag (Dreamer v2 KL balancing + free-bits)
  evaluation/
    calibration_metrics.py      compute() + CalibrationGate (PASS/FAIL)
    world_model_metrics.py      rollout MAE/RMSE + KL collapse + prior coverage
  labeling/
    barrier_labeler.py          BarrierLabeler (EWMA vol scaling + LabelRecord)
    uniqueness.py               apply_uniqueness_weights()
    purged_cv.py                PurgedKFold
  training/
    trainer.py                  HRWTrainer + ForecastBatch
    train_heads.py              HeadTrainer (freeze_encoder=True default); opt-in via
                                training.head_finetune_epochs / --head-finetune-epochs (review H7)
    train_world_model.py        WorldModelTrainer + encode_sequence()
  schemas/
    market_schema.py            MarketCandle, FuturesCandle, Regime, RegimeContext, EventContext
    label_schema.py             LabelRecord, Barrier
    prediction_schema.py        ModelPrediction, HorizonPrediction
tests/                          179 unit tests
```

---

## Install

```bash
cd helion_risk_world
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

```bash
ruff check .
pytest   # 179 tests, ~27 s
```

---

## Data Sources (Upstox + free only — no paid vendors)

| Data | Source | Script |
|---|---|---|
| OHLCV 5-min bars (all universe) | Upstox V3 analytics API | `alpha_data/scripts/backfill_all.py` (sibling repo) |
| India VIX 5-min intraday | Upstox (`NSE_INDEX|India VIX`) | `alpha_data/scripts/backfill_all.py` |
| BankNIFTY futures monthly contracts | Upstox ExpiredInstrumentApi | `alpha_data`'s `pipelines/futures_roll.py` |
| Continuous futures series | Stitched from monthly contracts | `alpha_data`'s `pipelines/futures_roll.py` |
| USD/INR daily (1-day lag) | Yahoo Finance (`USDINR=X`) | `alpha_data`'s `pipelines/regime.py` |
| WTI Crude daily (1-day lag) | Yahoo Finance (`CL=F`) | `alpha_data`'s `pipelines/regime.py` |
| FII/DII net flow (1-day lag) | NSE participant-OI archive | `alpha_data`'s `pipelines/macro.py` |
| Put/call OI ratio (1-day lag) | NSE participant-OI archive | `alpha_data`'s `pipelines/macro.py` |
| ATM IV, IV skew (1-day lag) | NSE bhavcopy, Jan 2023–Jul 2024 only | `alpha_data`'s `pipelines/regime.py` |

All `daily_context.parquet` columns above are shifted +1 calendar day at build time so
intraday bars never see same-day EOD values before they exist (see
`quanthelion.data.features.daily_context.assemble_daily_context`, RETIRED 2026-07-08 from
this repo's own `fetch_free_data.py::build_daily_context`). There is currently no live Upstox
option-chain client wired anywhere — `RegimeContextBuilder.set_live_iv()` is an
unused hook for a future live feed; ATM IV/skew are always sourced from the
historical daily_context parquet today, at both train and inference time.

### Data Layout

```
data/
  ohlcv/
    BANKNIFTY_5min.parquet              ← spot index
    NIFTY_5min.parquet
    INDIAVIX_5min.parquet               ← India VIX (replaces daily proxy)
    HDFCBANK_5min.parquet
    ICICIBANK_5min.parquet
    SBIN_5min.parquet
    AXISBANK_5min.parquet
    KOTAKBANK_5min.parquet
    BANKNIFTY_FUT_2310_5min.parquet     ← individual monthly contracts
    BANKNIFTY_FUT_2311_5min.parquet
    ...
    BANKNIFTY_FUT_continuous_5min.parquet   ← backward-adjusted continuous
  regime/
    daily_context.parquet               ← usdinr, crude, fii_dii_net (daily)
  processed/
    banknifty_5min.parquet              ← assembled by assemble_data.py
    labels.parquet                      ← labeled by label.py
```

Each parquet has a datetime index (IST, `Asia/Kolkata`) plus columns: `open`, `high`, `low`, `close`, `volume`, `oi`.

### Assembled Futures Parquet (for FuturesWindowBuilder and labeling)

Produced by `scripts/assemble_data.py`.  Contains 5-min bars with columns:
`open_fut`, `high_fut`, `low_fut`, `close_fut`, `volume_fut`, `oi_fut`,
`open_spot`, `high_spot`, `low_spot`, `close_spot`, `volume_spot`, `basis`, `roll_gap`.

---

## Quick Demo (no real data needed)

```bash
python scripts/train.py    --config configs/v1.yaml --demo
python scripts/backtest.py --config configs/v1.yaml --demo --model
python scripts/predict.py  --config configs/v1.yaml --demo
```

Dry-run checks:
```bash
python scripts/train.py     --config configs/v1.yaml --demo --dry-run
python scripts/calibrate.py --config configs/v1.yaml \
    --model-path runs/forecaster.pt \
    --labels-path /tmp/labels.parquet \
    --data-dir /tmp/data \
    --dry-run
```

---

## Real Workflow (step by step)

### Stage -1 — Fetch data (Upstox + free)

**RETIRED 2026-07-08**: all raw ingestion (OHLCV/futures via Upstox, USD/INR + crude
via Yahoo Finance, FII/DII/PCR + ATM IV/skew via NSE) moved to the sibling `alpha_data`
repo (`fetch_upstox.py`, `fetch_free_data.py`, `fetch_nse_bhavcopy.py` all deleted;
backed up in `src/helion_risk_world/data/.pre_quanthelion_migration_backup/` and
`scripts/.pre_quanthelion_migration_backup/`). Run from `alpha_data/` instead:

```bash
cd ../alpha_data
# Set credentials in .env (see alpha_data/README.md)
python scripts/backfill_all.py --from 2022-01-03 --to 2026-07-08
```

### Stage 0 — Assemble dataset

Raw OHLCV now lives in the shared alpha_data lake, not this repo's own `data/` —
see `alpha_data/docs/DATA_CATALOG.md`. `--out-path` still writes into this repo's
own `data/processed/` (helion-specific assembly choices, not raw data).

```bash
python scripts/assemble_data.py \
  --futures-path ../alpha_data/data/ohlcv/BANKNIFTY_FUT_continuous_5min.parquet \
  --spot-path    ../alpha_data/data/ohlcv/BANKNIFTY_5min.parquet \
  --out-path     data/processed/banknifty_5min.parquet
```

What it does: resample to 5-min, NaN-flag roll gaps, drop HDFC merger blackout bars
(±5 bars of 2023-07-01), inner-join spot + futures, compute basis.

### Stage 1 — Create labels

```bash
python scripts/label.py \
  --data-path  data/processed/banknifty_5min.parquet \
  --out-path   data/processed/labels.parquet \
  --H 12 --stop-mult 2.0 --target-mult 2.0
```

Output columns: `barrier`, `exit_return`, `realized_vol`, `label_realized_at`,
`horizon_bars`, `mae`, `sample_weight`, `regime` (heuristic from `primitives.regime_label`).

### Stage 4 — Train forecaster

```bash
python scripts/train.py \
  --config      configs/v1.yaml \
  --data-dir    data \
  --labels-path data/processed/labels.parquet

# Optional: enable Stage-2 self-supervised encoder pretraining before supervised fitting.
python scripts/train.py \
  --config          configs/v1.yaml \
  --data-dir        data \
  --labels-path     data/processed/labels.parquet \
  --pretrain-epochs 5 \
  --pretrain-gap-bars 2
```

Output: `runs/forecaster.pt` (weights + metadata bundle used by calibrate/backtest/predict).

### Stage 5 — Calibration gate

```bash
python scripts/calibrate.py \
  --config      configs/v1.yaml \
  --model-path  runs/forecaster.pt \
  --labels-path data/processed/labels.parquet \
  --data-dir    data
```

Exit code `0` = PASS (quantile coverage + barrier Brier + barrier ECE all within thresholds).
Exit code `1` = FAIL (blocks backtest promotion in CI).

### Stage 6 — Backtest

```bash
python scripts/backtest.py \
  --config    configs/backtest_banknifty.yaml \
  --real      \
  --data-dir  data \
  --model     \
  --model-path runs/forecaster.pt
```

### Stage 7 — Predict one timestamp

```bash
python scripts/predict.py \
  --config     configs/v1.yaml \
  --model-path runs/forecaster.pt \
  --data-dir   data \
  --timestamp  2026-06-27T15:25:00
```

---

## Important Config Knobs

| Key | Default | Effect |
|-----|---------|--------|
| `data.universe` | 8 BankNIFTY universe symbols | Symbols expected under `data/ohlcv/` |
| `data.lookback_bars` | 96 (≈ one NSE session) | Feature window length L |
| `horizons.horizon_steps` | [3, 6, 12] | 15/30/60-min forecasts at 5-min bars |
| `model.latent_dim` | 128 | Encoder and fusion embedding size d |
| `model.temporal_layers` | 2 | GRU depth in TemporalEncoder |
| `model.futures_conv_layers` | 2 | Conv depth in FuturesEncoder |
| `model.cross_asset_heads` | 4 | Multi-head attention heads |
| `training.lr` | 3e-4 | Adam learning rate |
| `training.pretrain_epochs` | 0 | Optional Stage-2 latent pretraining epochs before supervised fitting |
| `training.pretrain_gap_bars` | 12 | Gap between context and future windows used in Stage-2 pretraining |
| `training.max_epochs` | 50 | Training epochs |

---

## Known Limitations (V1)

- **calendar_spread is 0 outside a roll's overlap window**: `continuous_futures.py::build_continuous`
  now emits a `close_fut_next` column (the real, un-adjusted next-contract price) during the
  near/next overlap window around each roll (review Idea #6), and `futures_window_builder.py`
  computes `calendar_spread = (next - near) / near` from it. Since a contract only briefly
  overlaps its successor, this is still 0 for the large majority of bars — genuinely 0, not a
  placeholder — and 0 for any assembled parquet built before this change (no `close_fut_next`
  column). Re-run `assemble_data.py` against freshly-stitched continuous parquets to activate it.
- **Cross-asset attention is meaningful only with A ≥ 2**: with a single asset the attention
  is a no-op.  The default universe has 8 symbols; demo mode uses 1 asset for speed.
- **EVENT regime label**: `primitives.regime_label()` cannot detect event days from OHLCV alone.
  Wire the `EventContext.event_day_flag` from a calendar to get EVENT-labelled training rows.
- **No live ATM IV / IV skew feed**: `RegimeContextBuilder.set_live_iv()` exists as a hook but no
  Upstox option-chain client is implemented anywhere, and no caller invokes it. ATM IV/skew are
  always sourced from the historical, lagged `daily_context.parquet` (NSE bhavcopy coverage:
  Jan 2023–Jul 2024 only; forward-filled thereafter) at both train and inference time — including
  live paper trading. Pass `require_live_iv=True` to `RegimeContextBuilder` if a future caller needs
  to fail loudly instead of silently using this historical fallback.
- **RSSM state now persists across `predict_one` calls by default**: `WorldModelPredictor` used to
  reset the RSSM's recurrent state to zero on every call, discarding real bar-to-bar history the
  RSSM was trained on and undermining the "calibrated epistemic uncertainty" claim below. It now
  carries the RSSM `RSSMState` forward across successive calls on ascending timestamps (resetting
  at trading-day boundaries), controlled by `WorldModelPredictor(persist_state=...)` /
  `load_model_runtime(..., persist_state=...)` (default `True`; set `False` to recover the old
  reset-every-call behavior for A/B comparison). This is a genuine behavior change to the live/paper
  inference path — `fit_ood` calibration and paper-trading validation should be re-run against it
  before trusting calibrated thresholds under the new behavior.
- **RSSM path**: `HRWWorldModel` and `WorldModelTrainer` are test-covered but not yet wired
  into a CLI training script.
- **Option surface context (`SURFACE_CONTEXT_FEATURES`, 12-wide as of 2026-07-16)** is not shown
  in the architecture diagram above, which predates it. It includes ATM call/put delta
  (`atm_call_delta`/`atm_put_delta`, ranked #1/#2 of 124 features by IC — see
  `docs/investigation_log.md` §2 item 9), sourced via
  `data/alpha_option_chain.py::AlphaDataAtmGreeksLoader` with a local per-snapshot fallback.
  `ARTIFACT_VERSION` bumped 15→16 for this change (a genuine 10→12 shape change, unlike the
  same-width 14→15 cross-pair swap) — artifacts trained before this bump cannot be loaded
  (`state_dict` shape mismatch on `option_surface_encoder.context.0.weight`).
