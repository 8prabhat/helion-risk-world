# HelionRiskWorld — Technical Specification

**Core architecture:** Tri-Plane Risk-Aware Market World Model
**Status:** V1 design, implementation-ready. This revision folds in the world-model corrections
(formerly in `WORLD_MODEL_SPEC.md`) and is now the **single authoritative document**.
**Audience:** senior ML / quant engineers.
**Not financial advice. Not a guaranteed-profit system. Not an autonomous trading bot.**

---

## 0. How to read this document

This is the authoritative spec. Where it disagrees with code, the code is wrong until this document
is amended. `WORLD_MODEL_SPEC.md` is now **superseded and folded in here** — its corrections to the
model, data, labels, training and evaluation are integrated below (§9–§24) and cross-referenced in the
flaw register ([§31](#31-failure-modes--flaw-register)). Pseudocode lives in
[Appendix A](#appendix-a--corrected-pseudocode). Implementation order is in
[§35](#35-first-week-v1-implementation-plan).

Companion deep-dives live under `docs/`. This SPEC is the index and the contract; those docs expand
individual subsystems. **Read §3 first** — it is the honest thesis, and it gates everything else.

---

## 1. Executive summary

HelionRiskWorld (HRW) is a **risk-aware decision-intelligence research system** for Indian index and
derivatives markets (BankNIFTY-first; NIFTY, FINNIFTY and a small bank-stock basket as context). It is
built for research, simulation, backtesting, and paper trading — **not** for unattended live trading.

Most retail "AI trading" systems are a single classifier mapping `OHLCV → buy/sell`. They fail in
production for four reasons: (1) they predict a point, not a distribution, so they cannot reason about
downside; (2) they ignore the account — the same forecast implies different correct actions for a ₹50k
account vs a ₹50L account already in drawdown; (3) they ignore execution — an edge that does not
survive spread, slippage, brokerage and taxes is not an edge; (4) they were never validated for
**calibration** or against **costed baselines**, so their backtests are fiction.

HRW addresses all four with a **tri-plane** design plus a **calibration-first, falsification-first**
validation regime:

1. **Market World** — a *trained latent dynamics model* (RSSM) that learns how the market state
   actually evolves, autonomously and stochastically, and emits a **calibrated probabilistic** set of
   futures (return quantiles, volatility, barrier-hit, drawdown, regime, uncertainty, OOD).
2. **Portfolio World** — given a market future and the *current account state*, simulates the
   account-level consequence (capital, margin, exposure, PnL, drawdown). A **simulator, not a learned
   model** — because the account moves the portfolio, not the market.
3. **Execution Reality Layer** — estimates whether a trade can actually be filled profitably after
   spread, brokerage, taxes, slippage, latency and liquidity.

A conservative **mean–CVaR MPC planner** scores every candidate action — including a first-class
`NO_TRADE` — against a single, interpretable risk-aversion parameter and real execution cost, and a
deterministic **Risk Shield** can override the planner. Every decision emits a full audit record.

**The two things that make this a *world model* and not another classifier:** (a) the market dynamics
are *learned* — the latent transition is trained to predict the encoded future state with a calibrated
prior (not driven by injected `randn` noise); (b) success is measured as **calibration of the
predictive distribution first, PnL last**. A miscalibrated model is not permitted to trade, regardless
of backtest profit.

HRW depends on the **`quanthelion`** package as a reusable framework (config, tracking, interfaces,
options math, labels, conformal calibration, logging). It is *standalone* — not a fork, submodule, or
subclass of `msh_jepa`.

---

## 2. Problem statement

Given point-in-time Indian market data (5-minute candles + **futures** OI/basis + India VIX + event
flags + macro flow + optional V2 option-chain snapshots), decide at each decision step which of
`{NO_TRADE, ENTER_LONG, ENTER_SHORT, EXIT, REDUCE, INCREASE}` and which size to take, for a *given
account state and risk profile*, such that the chosen action maximizes a conservative
risk/cost/execution-adjusted objective over horizons of 15/30/60 minutes — while never violating hard
risk constraints, and only after the underlying predictive distribution has been shown to be
calibrated and the strategy has been shown to beat costed baselines.

Formally, at decision step `t` we want a policy `π(a_t | M_t, P_t, R)` with

```
M_t = market context (point-in-time features, market plane only, no portfolio leakage)
P_t = portfolio state (capital, position, drawdown, margin, ...)
R   = risk profile (limits, tolerances)
```

that is the `argmax` over admissible actions of an expected mean–CVaR, cost-adjusted utility, subject
to a deterministic admissibility filter (the Risk Shield).

The system must be honest about what is learnable: markets are near-efficient and noisy. The goal is
**not** to forecast price accurately. It is to (a) produce a *calibrated* distribution of futures —
above all of **volatility and tail/barrier events**, which are far more predictable than direction;
(b) decline most opportunities; and (c) act only when a risk-adjusted edge survives all costs and the
account can afford the risk.

### Explicit non-goals

- Not a price oracle. Not a guaranteed-profit system. Not unattended live execution.
- Not a single buy/sell classifier. Not a reinforcement-learning-first design.
- Not dependent on historical tick/order-book data in V1. Not dependent on historical option chains in
  V1 (options awareness is V2; V1 derivatives signal comes from **futures**).
- Not a multi-asset portfolio in V1 — V1 is an honest **single-position** (BankNIFTY futures) account
  simulator.

---

## 3. The honest thesis — where the edge actually is (read this first)

A world model is expensive. Before building one we state, plainly, what we believe is and is not
learnable on this data, and we make the project **falsifiable**.

**3.1 What is *not* predictable.** Intraday 5-minute index *returns* are near-zero-mean, heavy-tailed,
and have a signal-to-noise ratio close to zero. We do **not** expect a usable directional edge from
price history alone, and we will not pretend a deep model conjures one. Reconstructing prices, or
optimising directional accuracy, wastes capacity on noise. The headline of a naive "AI trader" — a
direction classifier — is therefore *dropped as a primary target*.

**3.2 What *is* predictable (the real targets).** In rough order of signal:

1. **Volatility and its regime.** Vol clusters (ARCH effect); realised vol is materially predictable at
   30–60 min horizons. Recent realised vol and futures activity carry this; India VIX adds a *forward
   implied* (30-day) regime signal — useful context, not an intraday realised-vol proxy.
2. **Tail / barrier probabilities.** Whether a stop or target is hit first, and the distribution of
   max adverse excursion, is more estimable than the signed return — and it is exactly what a
   risk-managed position needs.
3. **Order-flow / positioning** from **futures OI and basis.** Long buildup vs short covering
   (price–OI co-movement), basis compression/expansion, calendar-spread and rollover pressure are the
   only genuine microstructure in V1 (spot has no volume/OI — see §9.2).
4. **Cross-sectional breadth/dispersion** across bank constituents — broad vs narrow moves.
5. **Execution edge.** For a price-taker, *not losing to costs and adverse selection* is itself an
   edge. The Execution Reality plane is not overhead; it is arguably the most reliable alpha here.

So HRW is, honestly, a **volatility / tail / positioning / execution** model that *also* emits a
direction distribution — not a direction model dressed up.

**3.3 Calibration is the success metric, PnL is downstream.** The Market World's job is a *correct
predictive distribution*. Primary success = calibration (do predicted quantiles cover realised
outcomes at stated rates? PIT uniform? barrier ECE low?), measured **before** any trading metric. A
miscalibrated model cannot be risk-managed regardless of backtest PnL, and is barred from the planner
([§21 Stage 5 gate](#21-training-stages--gates)).

**3.4 Falsification-first.** Before the full RSSM cathedral is built, [§26](#26-baselines--kill-criteria)
defines cheap baselines (persistence, AR, **HAR-RV / GARCH vol + split-conformal quantiles**) and a
**kill gate**: if no PIT feature set produces OOS-stable, post-cost edge and if the world model cannot
beat a GARCH+conformal baseline on calibration, the complexity is not justified and the project is
re-scoped or stopped. This is a feature, not a failure.

---

## 4. Design philosophy

1. **A world model means a *trained* latent transition.** The dynamics `s_t → s_{t+1}` are learned to
   predict the encoded future state (JEPA-in-time), with a prior trained to match a posterior so that
   sampled rollouts are *calibrated*. Injected `randn` noise through an untrained cell is **not** a
   world model and is explicitly forbidden ([§14](#14-market-world--the-rssm-core)).
2. **Distributions over points.** Heads emit quantiles / probabilities, never a single number.
3. **Calibration before PnL.** A hard calibration gate precedes any trading evaluation.
4. **No-trade is the default.** Acting requires beating no-trade after every cost and risk penalty.
5. **Causal separation is sacred.** Portfolio variables never feed the market predictor ([§6](#6-non-negotiable-causal-boundary)).
6. **Execution realism is a first-class plane**, conservative by default — never silently optimistic.
7. **Determinism on top of ML.** A hard, auditable Risk Shield can always override the model.
8. **Counterfactual accounts.** The same market path is simulated against many account profiles.
9. **Shared feature definitions.** Train, backtest and paper import the *same* feature builder.
10. **Honest scoping.** V1 claims only what V1 data supports: single position, futures (not options),
    conservative costs. Anything else is V2/V3.
11. **Conservative-by-construction.** Every ambiguity resolves toward smaller size or no-trade.
12. **Explainability.** Every final decision carries a structured audit trail.

---

## 5. Why HelionRiskWorld is not msh_jepa

`msh_jepa` is a separate existing project (multi-scale hierarchical JEPA for pretraining). HRW **must
not** copy its internals, subclass it, wrap it, or import from it.

| Dimension | msh_jepa | HelionRiskWorld |
|---|---|---|
| Primary object | Self-supervised representation (JEPA) | Tri-plane risk-aware decision system |
| Output | Latent embeddings / pretraining loss | Calibrated distributions + account consequence + execution realism + audited action |
| Account awareness | None | First-class Portfolio World |
| Execution modeling | None | Dedicated Execution Reality Layer |
| Decision layer | None | mean–CVaR MPC planner + deterministic Risk Shield |
| Relationship | Sibling project | Sibling project (no code dependency) |

A JEPA-style *representation* objective is genuinely used here (Stage 2, §14.4 / §22), but it is
**reimplemented locally** in `helion_risk_world.training.pretrain_market_state` or pulled from shared
`quanthelion` utilities — never imported from `msh_jepa`. `tests/test_no_leakage.py::test_no_msh_jepa_import`
asserts the package never imports `msh_jepa`.

---

## 6. Non-negotiable causal boundary

**Portfolio variables must not be used as causal features for market prediction.** This follows from a
physical fact ([§3](#3-the-honest-thesis--where-the-edge-actually-is)): a retail/research account is a
**price-taker** — its action moves the *portfolio*, not BankNIFTY. Therefore the **market is
autonomous** (its dynamics take no action input) and the **portfolio is action-conditioned**.

```
WRONG:  OHLCV + OI + capital + risk_tolerance ──► future market return
RIGHT:
  Market World:    market features ──► future market DISTRIBUTION   (autonomous, learned)
  Portfolio World: market_distribution + portfolio_state + action ──► portfolio consequence (simulated)
  Planner:         market_forecast + portfolio_consequence + execution_cost + constraints ──► action
  Risk Shield:     validates / blocks action
```

Capital, risk tolerance, exposure, drawdown, margin, trade counts, consecutive losses belong **only**
to Portfolio World, Planner, Position Sizer, Risk Shield, Reward Scorer, Backtest Simulator.

### How the boundary is enforced (not just asserted)

- **Schema typing.** Market schemas (`MarketCandle`, `FuturesCandle`, `RegimeContext`, `EventContext`)
  carry market features only; `PortfolioState`, `ExecutionState` carry account/execution fields.
  Distinct Pydantic models, no shared mutable fields.
- **Builder separation.** `MarketWindowBuilder` / `FuturesFeatureBuilder` produce the *only* tensors
  the encoders consume. `PortfolioStateBuilder` tensors are consumed *only* by Portfolio World /
  Planner. Encoder `forward()` signatures literally cannot accept a `PortfolioState`.
- **Field registry.** `data/leakage_checks.py` holds `MARKET_FEATURE_NAMES` and
  `PORTFOLIO_FEATURE_NAMES` frozensets; `assert_no_portfolio_in_market(frame)` raises on intersection.
- **Test.** `tests/test_no_leakage.py::test_portfolio_fields_absent_from_market_encoder_inputs`.

These guards are implemented and tested today and remain mandatory.

---

## 7. Quanthelion dependency / integration

HRW uses `quanthelion` as a **framework dependency**, installed editable during development:

```bash
cd work/helion_risk_world
pip install -e ../quanthelion     # the reusable framework
pip install -e .                  # this project
```

### 7.1 Real Quanthelion surface (verified against the installed package)

| Need | Real symbol |
|---|---|
| Config load/merge/resolve | `quanthelion.config.{ConfigLoader, ConfigResolver, MaterializedConfig}` |
| Pluggable ABCs | `quanthelion.interfaces.{DataFetcher, FeatureBlock, FeatureMerger, Labeler, ModelTrainer, Calibrator, SignalGenerator, RiskFilter, ExecutionHandler, PipelineStep, Monitor, Validator, ...}` |
| Experiment tracking | `quanthelion.tracking.base.ExperimentTracker`; `null_tracker`, `mlflow_tracker` |
| Training utilities | `quanthelion.training.{CheckpointCallback, EarlyStopping, MetricsLogger, move_to_device, SAM, FocalLoss, SymbolWeightedSampler, SessionBatchSampler}` |
| Generic datasets | `quanthelion.data.{MultiSymbolSupervisedDataset, TemporalJitter}` |
| Options math | `quanthelion.options.black_scholes.{bs_price, bs_vega, implied_vol, greeks, time_to_expiry_years}` (V2 options path) |
| Risk gate | `quanthelion.risk.{ExecutionGate, GateResult}` |
| Portfolio tracking | `quanthelion.portfolio.PortfolioStateTracker` |
| Labels / purging | `quanthelion.labels.triple_barrier.make_triple_barrier_label`; `quanthelion.labels.embargo.{apply_embargo, make_purged_splits}` |
| Conformal calibration | `quanthelion.uncertainty.conformal.{ConformalCalibrator, AdaptiveConformalCalibrator}` |
| Logging | `quanthelion.utils.logging.{get_logger, configure_logging, bind_log_context}` |
| Reporting / errors | `quanthelion.reporting.base.StageReporter`; `quanthelion.core.errors.QuanthelionError` |

### 7.2 Gaps — symbols the brief assumed but that do NOT exist

Bridged by the local adapter layer (`src/helion_risk_world/integration/`):

| Assumed import | Reality | HRW resolution |
|---|---|---|
| `quanthelion.training.BaseTrainer` | absent | `integration.quanthelion_adapter.TrainerAdapter` wraps `HRWTrainer`, reuses `CheckpointCallback`/`EarlyStopping`/`MetricsLogger` |
| `quanthelion.data.DatasetProtocol` | absent | define locally in `schemas`; optionally subclass `MultiSymbolSupervisedDataset` |
| `quanthelion.metrics.MetricRegistry` | absent | local `evaluation.MetricRegistry` |
| `quanthelion.experiments.ExperimentRunner` | absent | `integration.quanthelion_adapter.ExperimentRunnerAdapter` |
| `quanthelion.logging.get_logger` | wrong path | adapter re-exports `quanthelion.utils.logging.get_logger` |

**Rule:** never invent a `quanthelion.*` import. If it is not in §7.1, it goes through `integration/`.

### 7.3 CLI reality

The real CLI is **stage-based**: `quanthelion run --stage <name> --env local --experiment <path>`. HRW
ships its own thin `scripts/*.py` entry points (§29) as the primary supported interface, and exposes an
*optional* `registry_hooks.py` so HRW stages *can* later be driven by `quanthelion run`.

---

## 8. Folder / package structure

`src`-layout, package `helion_risk_world`:

```
work/helion_risk_world/
  README.md  SPEC.md  pyproject.toml  requirements.txt  .gitignore
  configs/   v1.yaml  model_small.yaml  model_medium.yaml
             backtest_banknifty.yaml  paper_trading.yaml
             risk_profiles.yaml  data_sources.yaml
  docs/      architecture.md  data_sources.md  feature_engineering.md
             labeling.md  training_plan.md  backtesting_plan.md
             risk_management.md  execution_reality.md  failure_modes.md  ablation_plan.md
  src/helion_risk_world/
    integration/  quanthelion_adapter.py  registry_hooks.py
    config/       model_config.py data_config.py training_config.py
                  risk_config.py planner_config.py execution_config.py
    schemas/      market_schema.py futures_schema.py option_chain_schema.py
                  portfolio_schema.py execution_schema.py prediction_schema.py
                  label_schema.py action_schema.py
    data/         dataset.py feature_builder.py market_window_builder.py
                  futures_feature_builder.py option_surface_builder.py     # surface = V2
                  portfolio_state_builder.py execution_log_builder.py
                  corporate_actions.py rollover.py leakage_checks.py data_quality.py
    labeling/     barrier_labeler.py uniqueness.py sample_weights.py purged_cv.py
                  # barrier_labeler.py WRAPS quanthelion.labels.triple_barrier (no local re-scan)
    encoders/     temporal_encoder.py cross_asset_encoder.py futures_encoder.py
                  option_surface_encoder.py  regime_encoder.py fusion_encoder.py
    worlds/       market_world.py rssm.py latent_dynamics.py rollout_engine.py
                  portfolio_world.py
    execution/    execution_reality.py cost_model.py slippage_model.py
                  liquidity_model.py latency_model.py fill_simulator.py
    heads/        return_quantile_head.py volatility_head.py drawdown_head.py
                  barrier_head.py regime_head.py
    losses/       quantile_loss.py barrier_loss.py volatility_loss.py
                  rssm_loss.py repr_loss.py calibration_loss.py composite_loss.py
    planner/      mpc_planner.py action_sampler.py position_sizer.py
                  reward_scorer.py action_auditor.py management_loop.py
    risk/         risk_shield.py constraints.py exposure_manager.py
                  margin_simulator.py drawdown_guard.py event_blackout.py
    memory/       regime_memory.py calibration_memory.py drift_memory.py
    training/     pretrain_market_state.py train_world_model.py
                  train_heads.py train_planner.py trainer.py
    evaluation/   world_model_metrics.py calibration_metrics.py ml_metrics.py
                  trading_metrics.py risk_metrics.py no_trade_metrics.py baselines.py
    backtesting/  backtest_engine.py walk_forward.py combinatorial_cv.py
                  event_stress_test.py transaction_costs.py deflated_sharpe.py leakage_report.py
    paper_trading/ paper_engine.py broker_adapter_interface.py
                  execution_logger.py decision_logger.py
  scripts/   assemble_data.py validate_data.py build_features.py label.py
             pretrain.py train_world_model.py train_heads.py calibrate.py
             backtest.py paper_trade.py generate_report.py
  notebooks/ 01_data_exploration.ipynb 02_feature_validation.ipynb
             03_calibration_diagnostics.ipynb 04_backtest_analysis.ipynb
  tests/     test_shapes.py test_no_leakage.py test_data_quality.py
             test_corporate_actions.py test_triple_barrier.py test_uniqueness.py
             test_purged_cv.py test_rssm.py test_rollout_calibration.py
             test_risk_shield.py test_portfolio_world.py test_execution_reality.py
             test_backtest_engine.py test_no_trade_action.py test_deflated_sharpe.py
```

**Responsibility map (one job per module).** `encoders/` only encode. `worlds/` only simulate
dynamics. `execution/` only models cost/fill/liquidity. `heads/` only project latents to
distributions. `planner/` only ranks admissible actions. `risk/` only validates/blocks. `labeling/`
only builds path-correct targets and CV folds. No god classes.

---

## 9. Data reality (corrected) — what the model actually sees

### 9.1 Available real data (offline, validated to exist)

`quanthelion_workspace/data/ohlcv/<SYMBOL>_1min.parquet`, columns `open/high/low/close/volume/oi`,
tz-aware index, resampled to the 5-min base. **Verified ranges and lengths (not uniform — read §9.1a):**

| Group | Symbols | Volume | OI | Verified range / length | Use |
|---|---|---|---|---|---|
| Index spot | BANKNIFTY, NIFTY | **≡ 0** | **≡ 0** | 2022-01-03 → 2026-05-29, ~407k 1-min | price / return / vol features ONLY |
| Constituents | HDFCBANK, ICICIBANK, SBIN, AXISBANK, KOTAKBANK | real | **0** | 2022-01-03 → 2026-05-29, ~407k 1-min | cross-asset breadth / dispersion |
| Volatility | INDIAVIX | 0 | 0 | 2022-01-03 → 2026-05-29 | regime context |
| **Futures** | BANKNIFTY_FUT_<expiry> (+ `_continuous`) | **real** | **real** | **`_continuous`: 2024-10-01 → 2026-05-25, ~138k 1-min (≈ 28k 5-min)** | **the only real microstructure** |
| Macro | dii_cash_daily, fii_index_futures_daily (daily); pcr_banknifty_weekly (weekly) | — | — | daily / weekly | regime / flow context |

Live fetch (Upstox adapter, option-chain ingestion) exists in `quanthelion.ingestion` but needs
auth/market-hours and is **not** V1. V1 is the parquet history above.

### 9.1a 🔴 Data-window & sample-budget honesty (the linchpin caveat)

The corrected V1 thesis rests on **futures** microstructure (basis/OI-flow/calendar/rollover, §9.2),
but the continuous futures series only begins **2024-10-01** — about **1.6 years, ≈ 28k 5-min session
bars**, versus the **~4.4 years** of spot/constituent history. Any feature that needs the futures
(notably `basis = F − S`, futures OI-flow, calendar spread) is **undefined before Oct-2024**. This is
the single most consequential data fact in the spec, and it dictates the training split:

- **Pretrain wide, train narrow.** Stage-2 representation pretraining (§14.4) uses the **full
  2022→2026 spot/constituent history** (no futures features required) to learn a non-collapsed `E_φ`.
  The **RSSM dynamics (Stage 3), decode heads (Stage 4), calibration (Stage 5) and the backtest
  (Stage 8)** run **only on the ~1.6-yr futures window**, because those stages depend on the futures
  signal and on realizable futures PnL.
- **Sample-budget consequence.** ~28k 5-min bars, minus purge+embargo, split across folds, on a
  low-SNR target, is *small*. It bounds model capacity (favor the compact RSSM, `d ∈ {128}`, few
  params), bounds the number of walk-forward folds (≈ 3–5, not 8), and makes the **calibration gate
  (§21 Stage 5) the binding constraint** — a large model will appear calibrated in-sample and fail
  OOS. The spec treats "not enough data to calibrate honestly" as a valid **kill outcome** (§26), not
  a reason to shrink embargo or reuse test data.

### 9.2 🔴 The correction that matters: spot has no volume/OI

**BANKNIFTY/NIFTY spot `volume ≡ 0` and `oi ≡ 0`** (verified). Every feature derived from index
volume/OI is dead — `oi`, `d_oi`, `volume_zscore` on spot are identically zero. **The real
microstructure must come from the FUTURES**, which the original design never used. This single fact
reshapes the feature set (§12) and the encoder lineup (§13): the option-surface plane is **V2** (no
historical chain), and the V1 "derivatives signal" is **futures basis/OI**, not options.

### 9.3 Data source tiers

| Source | V-tier | Notes |
|---|---|---|
| Index/constituent OHLCV (5m) | V1 | spot volume/OI dead → price/return/vol only |
| Futures OHLCV + OI | V1 | **primary microstructure**: basis, OI flow, calendar, rollover |
| India VIX | V1 | level + rolling percentile |
| Expiry calendar | V1 | days-to-expiry, expiry flag |
| Event calendar | V1 | RBI/Fed/CPI/budget/election → blackout windows |
| FII/DII (daily), PCR (weekly) | V1 | EOD/EOW availability lag, lagged to publication |
| Global cues (SGX/Dow, USDINR, crude) | V1 | prior-close, point-in-time |
| Live bid/ask, option chain | V2 | paper-collected snapshots; activates option-surface plane |
| Own execution logs | V2 | calibrates Execution Reality |
| Historical tick / L2 depth | V3 | LOB expert — **not a V1/V2 dependency** |

### 9.4 🔴 Mandatory Stage-0 validation (was skipped)

- **Corporate actions.** Constituent series MUST be adjusted for splits/bonus and the **HDFC–HDFCBANK
  merger (~July 2023, inside the range)**. Unadjusted series produce discontinuous returns that corrupt
  the cross-asset encoder and labels. `data/corporate_actions.py` must verify adjustment (no
  unexplained |overnight return| outliers) and **fail loudly** otherwise.
- **Futures rollover.** The continuous series is built with documented roll logic (roll on expiry,
  OI-weighted or last-trading-day); basis/calendar features use the correct near/next contracts and are
  NaN-safe across rolls (`data/rollover.py`).
- **Availability lag.** FII/DII (daily) and PCR (weekly) are end-of-period; stamp `available_at` at
  publication, never before. Bars are labeled at **close** (`ts == available_at`).
- **Session handling.** Restrict to the NSE session; do not let overnight gaps enter rolling windows as
  if they were 5-minute moves (session-aware resampling).

---

## 10. Data schemas

Pydantic v2 models in `schemas/`. Every record is **point-in-time**: it carries `ts` and an explicit
`available_at` (when the value could first be known). Builders assert `available_at <= ts`.

| Schema | Plane | Key fields (abridged) |
|---|---|---|
| `MarketCandle` | Market | `symbol, ts, available_at, o,h,l,c,v` (spot v ignored) |
| `FuturesCandle` | Market | `+ expiry, fut_oi, d_oi, basis, calendar_spread, dte, roll_flag` |
| `RegimeContext` | Market | `vix, vix_pct, pcr_lagged, regime_onehot` |
| `EventContext` | Market | `expiry_flag, event_day_flag, blackout_active, event_type, fii_dii_net_lagged, usdinr, crude` |
| `OptionSurfaceSnapshot` | Market (**V2**) | `underlying, ts, available_at, atm_strike, strikes[ATM-N..ATM+N]`, per-strike call/put arrays |
| `PortfolioState` | Portfolio | `capital0, capital, cash, position, entry_price, realized_pnl, unrealized_pnl, daily_pnl, drawdown, margin_used, free_margin, exposure, risk_budget_used, trades_today, consecutive_losses` |
| `ExecutionState` | Execution | `bid, ask, spread, est_slippage, fill_prob, partial_fill_prob, reject_prob, latency_ms` |
| `LabelRecord` | Label | `ts, label_realized_at, horizon_bars, barrier{STOP/TARGET/TIMEOUT}, exit_return, exit_t, realized_vol, mae, uniqueness_weight` |
| `ModelPrediction` | Output | `return_quantiles{p10..p90}, vol, mae, barrier_probs{stop,target,timeout}, regime_probs, epistemic, aleatoric, ood_score` |
| `CandidateAction` | Planner | `action_type, size_fraction` |
| `RiskDecision` | Risk | `allowed, reason_code, adjusted_size, final_action` |
| `FinalDecision` | Audit | full audit record (§29) |

Future labels live in `LabelRecord` with `label_realized_at > ts`; builders assert it so a label can
never be a feature. Note `ModelPrediction` has **no `direction_probs`** — direction is dropped as a
target (§3, §15). `epistemic/aleatoric/ood` are **derived from the model's own distributions** (§16),
not separate prediction heads.

---

## 11. Labels (corrected) — path-aware, weighted, purged

The original used a **fixed-horizon return** label and a **forward-return-derived regime** label. Both
are wrong. Corrected labels are **path-aware (triple-barrier)** and the model predicts what the planner
actually consumes.

### 11.1 Triple-barrier labeling (de Prado AFML Ch. 3)

We **reuse `quanthelion.labels.triple_barrier.make_triple_barrier_label`** (verified to exist) as the
core labeler; HRW adds only what the framework lacks — concurrency/uniqueness weighting
(`labeling/uniqueness.py`) and futures-vol scaling. There is **no** reimplementation of the barrier
scan in HRW.

**🔴 Label on the traded instrument.** We trade **BankNIFTY futures**, so `e_t` and the realised path
are the **roll-aware futures continuous** price — not spot — so `exit_return` is the PnL a futures
position actually realises (spot-based labels would mislabel by the basis and ignore rolls). Entry
`e_t = fut_close_t`; local vol `σ_t` = EWMA of futures returns scaled to horizon `H`:

```
upper (take-profit):  e_t · (1 + u·σ_t)
lower (stop-loss):    e_t · (1 − d·σ_t)
vertical (timeout):   bar t + H
```

Scan the realised path over `(t, t+H]`; the **first** barrier touched yields the label:

```
first touch = upper  → barrier = TARGET (+1),  exit_return = price@touch/e_t − 1
first touch = lower  → barrier = STOP   (−1),  exit_return = price@touch/e_t − 1
no touch by t+H      → barrier = TIMEOUT (0),  exit_return = close_{t+H}/e_t − 1
```

Per-sample outputs: `barrier` (→ barrier head), `exit_return` (→ return-quantile head),
`realized_vol` over `[t,t+H]` (→ vol head), `mae = max_τ (e_t − price_τ)/e_t` (→ MAE/drawdown head),
`exit_t` (label span). `u, d` are config per instrument/regime (symmetric `u=d` default); `H` is the
**trade-management horizon** (the maximum life of a position), set to `max(horizon_bars)` so the
barrier label and the planner's holding loop (§19) share one timeframe.

### 11.2 Sample uniqueness & weighting (mandatory) — `labeling/uniqueness.py`

🔴 Overlapping labels (consecutive bars whose `[t, exit_t]` spans overlap) are **not independent**. For
each timestamp `τ`, concurrency `c_τ` = number of active labels; sample `i`'s uniqueness
`ū_i = mean_{τ∈[t_i,exit_i]} 1/c_τ`. Training and CV use **sample weights ∝ ū_i** and **sequential
bootstrap** (or decimation) for any i.i.d.-assuming step.

### 11.3 Purging & embargo (mandatory, in *both* train construction and CV) — `labeling/purged_cv.py`

🔴 A split must **purge** every training sample whose label window `[t, exit_t]` overlaps the test
window, and **embargo** an extra `embargo_bars ≥ H` after the test block, via
`quanthelion.labels.embargo.make_purged_splits`. Embargo only at a single boundary, with no intra-train
purge, is invalid (the original sin).

### 11.4 What is *not* a label

- ❌ **No fixed-horizon return label** (path-blind; mismatched with a barrier-managed position).
- ❌ **No forward-derived regime label** (circular). Regime is a property of the *current* state,
  inferred from *current* observables (vol level, trend, VIX, events) — it enters as **context** (the
  regime encoder); if a regime head is trained at all, its label is **state-derived from PAST data**.
- ❌ **No standalone 3-class direction label** — a lossy projection of a distribution the model already
  predicts (quantiles + barrier). Dropped.

### 11.5 Multi-horizon coherence (how single-H labels meet multi-horizon heads)

The decode heads (§15) emit predictions at `horizon_bars = {3, 6, 12}`, but a single triple-barrier
label has one exit. These reconcile as follows, with **no ambiguity about which label trains which
head**:

- **Barrier and MAE/drawdown heads** are trained on the single trade-management horizon `H = 12` (the
  position's life) — `barrier`, `exit_return`, `mae` from §11.1. These are the targets the planner's
  ΔW uses (§19), because they describe the actual managed trade.
- **Return-quantile and volatility heads** are produced **per horizon** `h ∈ {3,6,12}` against
  **horizon-specific terminal targets**: the realised futures return `fut_close_{t+h}/fut_close_t − 1`
  and realised vol over `[t,t+h]`. These give the planner a *term structure* of expected move and risk
  for sizing and uncertainty, and feed regime/OOD context.
- Both label families share the same purge/embargo and uniqueness weighting (§11.2–11.3), computed on
  the **longest** span `[t, max(exit_t, t+max(h))]` so no horizon leaks.

---

## 12. Feature engineering (futures-centric)

One shared `FeatureBuilder` (DRY) used by training, backtesting and paper. Computed point-in-time.

- **Spot (index):** log/simple returns, realised vol (rolling & EWMA), ATR, range
  compression/expansion, gap, time-of-day, day-of-week. *(No volume/OI features — dead, §9.2.)*
- **Futures (continuous + near/next):** `basis = F_t − S_t` (and % of spot), **futures OI and ΔOI**,
  futures volume & z-score, **calendar spread = F_near − F_next**, **rollover** (DTE, OI migration near
  expiry). **OI-flow classification** (the cheap real signal): price↑/OI↑ = *long buildup*,
  price↑/OI↓ = *short covering*, price↓/OI↑ = *short buildup*, price↓/OI↓ = *long unwinding* — encoded
  as a 4-state feature plus signed-ΔOI magnitude.
- **Cross-asset (constituents):** per-constituent returns/vol → cross-asset encoder; derived
  **breadth** (fraction up) and **dispersion** (cross-sectional return std) — broad vs concentrated.
- **Regime/event (slow):** VIX level + percentile; PCR (weekly, lagged); FII/DII (daily, lagged);
  expiry flag; event-day flags; global cues; USDINR; crude.
- **Portfolio (Portfolio World / Planner / Sizer / Shield / backtest only):** capital, position, PnL,
  drawdown, margin, exposure, risk budget, trade counts, consecutive losses.
- **Execution (Execution Reality / Planner only):** spread, slippage est, Indian statutory costs,
  latency, fill/partial/reject probabilities.
- **Options (V2 only):** ATM-relative per-strike OI/IV/greeks, PCR, IV skew, gamma walls, max-pain
  proxy. **Not present in V1.**

Shared DRY primitives in `data/feature_builder.py`: `compute_returns`, `realized_vol`, `atr`,
`futures_basis`, `oi_flow_state`, `calendar_spread`, `breadth_dispersion`, `transaction_cost`,
`estimate_slippage`. Backtest and live use the **same functions** — enforced by
`tests/test_no_leakage.py::test_train_backtest_feature_parity`.

---

## 13. Encoder architecture (market plane only)

`o_t → e_t ∈ ℝ^d` via modality encoders fused into `e_t`:

| Encoder | Purpose | V1 default |
|---|---|---|
| `TemporalEncoder` | multi-asset candle/return/vol window `[B,A,L,F]` | patch + compact SSM/GRU |
| `CrossAssetEncoder` | constituent + futures relations; broad vs concentrated; order-invariant | asset-attention matrix |
| `FuturesEncoder` *(new, required)* | basis, futures OI/ΔOI, calendar spread, rollover, OI-flow state | small temporal MLP/TCN |
| `RegimeEncoder` | slow regime/event context | embedding + MLP |
| `OptionSurfaceEncoder` *(V2)* | ATM-relative vol surface as a set | DeepSets → Set Transformer (wired, inactive in V1) |
| `FusionEncoder` | fuse all into `e_t` | gated fusion (V1), MoE (V2) |

Compactness (Mac Studio 64 GB): small modular blocks; latent `d ∈ {128, 256}`. The option chain is
never naively flattened to hundreds of static columns. All encoders satisfy
`EncoderProtocol.forward(batch) -> Tensor` with shape comments. **The FuturesEncoder replaces the
option-surface encoder as the V1 "derivatives" signal** — see §9.2.

---

## 14. Market World — the RSSM core (corrected)

🔴 The original `latent_dynamics.py` was a deterministic `GRUCell(noise, z)` with **arbitrary injected
noise** and **no training objective** — its rollout "ensemble" was init-noise, and "epistemic
uncertainty = ensemble spread" was meaningless. This is replaced by a proper **stochastic recurrent
state-space model (RSSM)** whose transition is a *learned distribution*, trained so that (a) it predicts
the next encoded state and (b) its spread is calibrated.

### 14.1 RSSM (PlaNet/Dreamer-style, autonomous, JEPA representation target) — `worlds/rssm.py`

```
h_t       = GRU(h_{t-1}, z_{t-1})                         # deterministic carry (no action: market exogenous)
prior:     p_θ(z_t | h_t)        = N(μ_p(h_t),  σ_p(h_t)²)            # used to IMAGINE / roll forward
posterior: q_φ(z_t | h_t, e_t)   = N(μ_q([h_t,e_t]), σ_q([h_t,e_t])²)  # used in TRAINING (sees the obs)
representation head: ê_t = D_ψ(h_t, z_t)                  # predicts the ENCODED rep (JEPA), not raw prices
```

`e_t = E_φ(o_t)` is the §13 encoder output. The model is **autonomous** — no action conditions the
market dynamics; the action enters only the Portfolio World (§17). Full latent state `s_t = (h_t, z_t)`.

**Why latent, not price space:** intraday returns are near-zero-mean and heavy-tailed; reconstructing
prices wastes capacity on noise. Predicting the *encoded* future state focuses capacity on predictable
structure (vol regime, futures-OI order-flow, cross-asset breadth). This is the whole reason to use a
world model here.

### 14.2 Training objective (Stage 3 — latent dynamics) — `losses/rssm_loss.py`

Run the RSSM over a real sequence with the **posterior** and minimise, per step:

```
L_dyn = Σ_t [ α·‖ D_ψ(s_t) − sg(e_t) ‖²                          # (i) representation prediction (JEPA)
            + β·KL( q_φ(z_t|h_t,e_t) ‖ p_θ(z_t|h_t) ) ]          # (ii) prior matches posterior
```

- **(i)** is world-model consistency in latent space: the state must reproduce the *encoded*
  observation. `sg(·)` stop-gradients the target; the encoder is frozen or slowly fine-tuned. Collapse
  is prevented by Stage-2 pretraining + a VICReg variance/covariance term retained on `{e_t}`.
- **(ii) KL** trains the **prior to match the posterior**, so at inference — when no future observation
  exists — **sampling the prior produces calibrated next-latents**. This was the missing ingredient.
  Use KL balancing / free-bits (Dreamer) to avoid posterior collapse on near-random-walk returns.

**Multi-step (imagination) consistency.** Roll the **prior** open-loop for `k = 1..K` from `s_t` and
require the imagined state to predict `e_{t+k}`:

```
L_imag = Σ_t Σ_{k=1}^{K} γ^k · ‖ D_ψ(roll_k(s_t)) − sg(e_{t+k}) ‖²
```

This makes horizons `> 1` valid (it is "evaluate multi-step rollout" turned into a training objective).

### 14.3 Rollout / imagination (inference) — `worlds/rollout_engine.py`

**Obtaining the current state `s_t` is itself a recurrence, not a single posterior call.** The
deterministic carry `h_t` is a function of the *whole* observed lookback, so inference first **rolls the
RSSM forward over the window** `o_{t-L..t}`, applying the posterior `q_φ(z_τ|h_τ,e_τ)` at each step to
update `(h_τ, z_τ)`, ending at `s_t = (h_t, z_t)`. Only then do we imagine `S` futures by sampling the
**prior** open-loop:

```
for s in 1..S:
  s_t^(s) = s_t
  for k in 1..H:
    z_{t+k}^(s) ~ p_θ( · | h_{t+k-1}^(s) )         # sample the LEARNED transition distribution
    h_{t+k}^(s) = GRU(h_{t+k-1}^(s), z_{t+k}^(s))
  collect s_{t+k}^(s) for k in horizons
→ ensemble [S, B, |H|, d]
```

Because the prior is trained (KL to posterior), ensemble spread is **calibrated predictive
uncertainty** (epistemic from the recurrence, aleatoric from `σ_p`), not arbitrary noise. The decode
heads (§15) run on every `(s, k)` member; the planner consumes the per-horizon distribution.

### 14.4 Staged learning, and the representation–task alignment risk

- **Stage 2 (representation pretrain, done):** pretrain `E_φ` by **future-latent prediction** with
  VICReg anti-collapse (`MarketStatePretrainer` / `LatentPredictionLoss`) → a predictable,
  non-collapsed `e_t` *before* dynamics train on it. Uses the **full 2022→2026 spot/constituent
  history** (no futures features needed — §9.1a "pretrain wide").
- **Stage 3 (dynamics):** train the RSSM (`L_dyn + L_imag`) on the pretrained representation, **on the
  ~1.6-yr futures window** ("train narrow"); fine-tune `E_φ` jointly at a small LR.

**🟠 The alignment risk (must be designed for, not assumed away).** A purely self-supervised VICReg
latent is optimised to *predict itself*, not to *carry exit-return/barrier signal* — high-variance,
decorrelated features need not be the trade-relevant ones. If `E_φ` is frozen through Stage 4, the
heads may decode from a representation that simply does not contain the answer. Mitigations, applied in
order: (a) **unfreeze `E_φ` for Stage-4 head training** at a small LR so the representation can
specialise toward the decode targets; (b) optionally add a **light auxiliary supervised term**
(quantile/barrier) during Stage-2/3 so the latent retains task signal from the start; (c) monitor a
**linear-probe baseline** — if a linear head on `e_t` already extracts most of the decodable signal,
the deep heads (and arguably the RSSM) are not adding value (ties into the §26 kill criteria).

---

## 15. Prediction (decode) heads — on the latent state

Heads decode trading-relevant quantities from the (rolled) latent `s_{t+k}`, trained on §11
path-correct, uniqueness-weighted labels.

| Head | Horizon(s) | Output | Loss | Label (§11) |
|---|---|---|---|---|
| Return quantiles | per `h ∈ {3,6,12}` | `q_{0.1..0.9}(h)` | pinball, monotone-enforced | terminal futures return over `[t,t+h]` |
| Volatility | per `h ∈ {3,6,12}` | `σ̂(h)` | Huber | realised vol over `[t,t+h]` |
| Barrier | single `H=12` | `P(stop/target/timeout)` | cross-entropy | triple-barrier `barrier` |
| MAE / drawdown | single `H=12` | `m̂ ≥ 0` | Huber | path MAE |
| Regime (optional) | at `t` | `P(regime)` | cross-entropy | **state-derived** regime (not forward-derived) |

The per-horizon return/vol heads give a **term structure** for sizing and uncertainty; the single-`H`
barrier/MAE heads describe the **actual managed trade** and feed the planner's ΔW (§19), exactly per
the §11.5 coherence rule.

**Direction head is removed** (subsumed by quantiles + barrier). **Choosing LONG vs SHORT** uses the
**asymmetry of the predicted `H`-horizon return-quantile distribution and the barrier probabilities**
(`P(target) − P(stop)` and the sign of the median/expected exit_return) — no separate direction head is
needed. **Uncertainty and OOD are not heads** — they come from the model's own distributions (§16).
Quantiles are enforced non-crossing within a horizon and monotone-coherent across horizons. Each head
is a small module behind `HeadProtocol`, ISP-clean (a return head knows nothing of brokers).

---

## 16. Uncertainty & OOD — from the model, not bolted on

- **Aleatoric:** the per-horizon spread of the decode distribution (quantile width) and prior `σ_p`.
- **Epistemic:** the **dispersion of the decode across the `S` rollout members** — now meaningful,
  because the prior is trained. Grows with horizon as the recurrence compounds.
- **OOD:** the **prior log-likelihood of the inferred latent**, `−log p_θ(z_t | h_t)` (or
  posterior↔prior KL at `t`). A state poorly explained by the learned dynamics is out-of-distribution.
  This replaces the post-hoc Gaussian-on-latents fit (which was calibrated to *untrained* noise). The
  Risk Shield's OOD quarantine fires on this signal.

---

## 17. Portfolio World — action-conditioned simulator

`worlds/portfolio_world.py`. Pure simulator: `Φ(P_t, a_t, market_future) → (P_{t+1}, Consequence)`.
Handles capital, realized/unrealized PnL, margin, free margin, drawdown, exposure, position size,
transaction costs, slippage assumptions, stop/target logic, daily risk limit, trade count, consecutive
losses. **Consumes** portfolio variables; **never** exposes them to the encoders (§6). Reuses
`quanthelion.portfolio.PortfolioStateTracker` for bookkeeping.

Corrections:

- 🟠 **ΔW is computed analytically from the decode heads — there is no price-path simulation.** The RSSM
  rolls out *latent* states and decodes *distributions*; it does **not** emit ordered intra-bar price
  trajectories, so PortfolioWorld must **not** pretend to walk a path through stop/target. Instead, the
  consequence of an action is the head-implied wealth-change distribution: `P(stop/target/timeout)` ×
  the corresponding exit-return outcomes (target → `+u·σ`, stop → `−d·σ`, timeout → the `exit_return`
  quantiles), converted to fraction-of-capital ΔW given size and cost. This is exactly what the model
  actually predicts, and it removes the old, unrealizable `sample_paths()` step.
- 🟠 **CVaR over the head-implied ΔW distribution**, not a 5-knot linear-tail extrapolation. With CVaR
  defined as a **positive loss magnitude** `CVaR_α[X] = −E[X | X ≤ q_α(X)]` (the expected shortfall in
  the worst-`α` tail, reported ≥ 0), it is trustworthy *because* §14 gives a calibrated distribution and
  enters the planner objective (§19) with a single sign. Where Monte-Carlo over head samples is used
  (e.g. mixing horizons), draw a **fresh RNG stream per decision step** — the original re-seeded the
  same generator every call, producing perfectly correlated draws; that is invalid.
- 🟡 **Multi-asset honesty.** V1 trades a single instrument (BankNIFTY futures). This is a
  **single-position account simulator** in V1; a true multi-asset book (netting, correlation) is V2. Do
  not claim diversification that isn't modeled.

**Counterfactual profiles (first-class).** The same market future is simulated against many accounts:
small/low-risk, balanced, large/conservative, aggressive, already-in-drawdown, near-daily-loss-limit,
near-exposure-limit, post-consecutive-losses. Identical forecasts imply different correct actions per
account — this is why the planner output is account-conditioned.

---

## 18. Execution Reality Layer (honest V1)

`execution/`. Answers: *even with a good forecast, can this trade be filled profitably?* Each component
behind a Protocol (LSP/OCP): `cost_model.py`, `slippage_model.py`, `liquidity_model.py`,
`latency_model.py`, `fill_simulator.py`. `execution_reality.py` orchestrates into a realism score.

- **Costs (known, not learned):** Indian **statutory** charges (brokerage, STT, exchange txn, SEBI,
  stamp, GST) are *published schedules* — encode them config-driven from the public rate cards (and a
  real contract note if available as a sanity check). These are **not** placeholders and do **not**
  require own trades. **Futures** cost schedule in V1 (options costs are V2).
- **Spread/slippage (assumed in V1):** V1 has no historical depth and no own fills → **conservative**
  notional-relative assumptions, explicitly flagged as assumptions; **calibrated against own fill logs
  only in V2**. Never present synthetic bid/ask as measured.
- **Latency:** at a 5-min bar cadence with decisions at bar close and fills at the next bar, intraday
  latency is effectively **unmodelable and ~negligible in V1** — `latency_model.py` returns a small
  constant placeholder. Genuine latency/queue modeling is a **V3 (tick)** concern; the module exists for
  interface stability, not because V1 can estimate it.
- **Realism score** ∈ {low, med, high} from microstructure (fill prob × liquidity) and edge-awareness
  (cost ÷ expected edge). **Default to *medium*, not *high*, when blind** — the original 0.95 default
  liquidity was optimistic. `low` → block; `medium` → reduce size / require higher edge; `high` → ok.

The realism layer must not silently inflate confidence: absence of depth ⟹ conservative, with the cost
(which *is* charged) doing the real work. V2 calibrates on own logs; V3 adds a microstructure/LOB
expert.

---

## 19. Planner — re-founded mean–CVaR objective

🔴 The original objective was **eight hand-tuned penalty weights** that needed repeated ad-hoc
rescaling and still wouldn't trade on real data. Replace with a **coherent mean–CVaR expected-utility**
rule with a single interpretable risk-aversion parameter:

```
U(a) = E[ΔW(a)] − λ · CVaR_α[ΔW(a)] − Cost(a)
       where CVaR_α[X] = −E[X | X ≤ q_α(X)] ≥ 0   (expected shortfall as a POSITIVE loss)
```

- `ΔW(a)` = change in account wealth from action `a` over the holding period, in **fraction-of-capital**
  units, from the Portfolio World's **head-implied** ΔW distribution (§17 — barrier-prob × exit-return
  heads; no path simulation).
- `λ` = a **single, stated risk-aversion** parameter (one number with meaning), not eight magic
  weights. CVaR at level `α` (e.g. 5%).
- **Sign convention (must hold everywhere):** `CVaR_α` is the worst-tail *loss reported as a positive
  number*, so a riskier action has larger `CVaR_α` and the `− λ·CVaR_α` term *reduces* `U`. Every
  occurrence of CVaR in this spec (§17, §19, Appendix A) uses this one definition.
- `Cost(a)` = real execution cost in the same units (§18).
- `NO_TRADE` has `U = 0`; a trade is taken iff `U(a*) > 0` — i.e. risk-adjusted edge beats cost.

The old "soft penalties" (uncertainty, concentration, overtrading) become **hard constraints in the
Risk Shield** or small explicit terms with stated units — never free parameters.

**Flow** (`planner/`): `ActionSampler` enumerates the **state-dependent** candidate set — when flat:
`{NO_TRADE, ENTER_LONG, ENTER_SHORT}` × sizes `{0,10,25,50,100}%` of the allowed risk unit; when in a
position: `{HOLD, EXIT, REDUCE}` (see "Action set while in a position" below) → `PortfolioWorld.step`
per candidate → `ExecutionReality.estimate` per candidate → `RewardScorer` computes `U(a)` → `argmax`
incl. the no-op baseline (`NO_TRADE` when flat, `HOLD` when in a position) → `ActionAuditor` records
every candidate's score and why the winner won.

🟠 **Decision cadence = holding period (management loop).** Because positions are barrier-managed over
`H` bars, the planner must **not** re-decide every bar on an `H`-bar horizon (the original did →
overlapping, incoherent positions). The planner **opens/sizes**; the position is then **managed to its
triple-barrier exit** (`planner/management_loop.py`); a new *entry* is evaluated only when flat.

**Action set while in a position (V1).** Once long/short, the only candidates each bar are `HOLD`,
`EXIT` (discretionary early close before the barrier — e.g. forecast flips sign or uncertainty spikes),
and `REDUCE` (cut size on a Risk-Shield or de-risk trigger). `INCREASE` and fresh `ENTER_*` are **not**
available mid-position in V1 — a single-position simulator (§17) does not pyramid; pyramiding/scaling is
a V2 multi-position concern. `ENTER_LONG/ENTER_SHORT/INCREASE` are therefore only live when flat. This
keeps the managed trade coherent with the single-`H` barrier label.

**Uncertainty-gated sizing**: size shrinks as epistemic uncertainty or OOD rise. Planner depends on
`CostModelProtocol` / `LatentDynamicsProtocol`, never concrete brokers (DIP).

---

## 20. Risk Shield — deterministic, hard, auditable (kept)

`risk/risk_shield.py` — the ML can **never** override it. Wraps `quanthelion.risk.ExecutionGate` /
`GateResult`. Ordered rules, each a `RiskRuleProtocol` (OCP):

```
max_drawdown_breached          -> stop system / force flatten
daily_loss_limit_breached      -> block new trades
free_margin_below_threshold    -> reduce or exit
exposure_limit_exceeded        -> reduce or deny
max_trades_per_day_reached     -> no trade
consecutive_losses_above_thr   -> cool-down mode
uncertainty_above_threshold    -> no trade
ood_score_above_threshold      -> no trade (OOD = prior NLL, §16)
event_blackout_active          -> no trade / reduced size
execution_realism_too_low      -> no trade
expected_utility <= 0          -> no trade
```

De-risking (EXIT/REDUCE/NO_TRADE) is **never** blocked. Sub-modules: `constraints.py`,
`exposure_manager.py`, `margin_simulator.py`, `drawdown_guard.py`, `event_blackout.py`. Returns
`RiskDecision(allowed, reason_code, adjusted_size, final_action)`.

---

## 21. Training stages & gates

Each stage has an **objective** and a numeric **gate to proceed**. Gates are kill-criteria: failing one
stops the pipeline, not silently degrades it.

| # | Stage | Objective | Gate to proceed |
|---|---|---|---|
| 0 | **Data assembly + validation** | corp-action adjustment, futures rollover, PIT stamping, session handling | adjustment & rollover checks pass; no unexplained return outliers |
| 1 | **Labeling** | triple-barrier `{barrier, exit_return, realized_vol, mae}` + uniqueness weights + purged/embargoed splits | label QA: no leakage; reasonable class balance |
| 2 | **Representation pretrain** *(done)* on full 2022→2026 history | future-latent prediction + VICReg (`L_repr`) | latent does not collapse (std bound); future-latent error < persistence baseline |
| 3 | **Latent dynamics (world model)** on the ~1.6-yr futures window | RSSM `L_dyn + L_imag` (§14.2) | **1-step & multi-step rollout error < persistence & AR baselines** (§23.1) |
| 4 | **Decode heads** | quantile/barrier/vol/MAE on rolled latents, path-correct, uniqueness-weighted; `E_φ` fine-tuned (§14.4) | heads converge OOS; **beat a linear probe on `e_t`** |
| 5 | **Calibration validation** | coverage, ECE, PIT on held-out (§23.2) | **HARD GATE: calibrated (coverage/PIT/ECE within tol) AND calibration-parity-or-better than the GARCH+conformal baseline while strictly *sharper* / better conditional coverage — else STOP, no trading** |
| 6 | **Portfolio + Execution calibration** | account transitions; statutory cost schedule; conservative slippage | cost sanity vs published rate cards |
| 7 | **Planner (mean–CVaR MPC)** | imagine futures → `U(a)` → Risk Shield; management-loop cadence | decisions audited; trades on real data |
| 8 | **Walk-forward backtest (+ CPCV for metric stability)** | contiguous WF for the PnL path; CPCV only for predictive-metric distributions; real costs, baselines, regime/event breakdown | **(a) DSR > 0 (true Sharpe > 0 after trial-deflation) AND (b) beats flat/buy-hold/random after costs by a paired block-bootstrap test** (§23.3) |
| 9 | **Paper trading** | dry-run, full audit, drift/calibration monitor | calibration stable live |

The **calibration gate (Stage 5)** is the spine: a trading world model that is not calibrated is not
permitted to proceed to PnL evaluation, regardless of backtest results. Note Stages 3–8 run on the
futures window (§9.1a), so the gates must be met on **few (≈3–5) folds of low-SNR data** — small enough
that "cannot be shown calibrated OOS" is a likely and acceptable kill outcome.

---

## 22. Loss functions (exact)

`losses/`, each behind `LossProtocol`, config-driven weights selected on a **validation fold** (not
hand-set):

```
# Stage 2 — representation (VICReg future-latent prediction):
L_repr = sim·smooth_l1(P(e_t), sg(e_{t+gap}))
       + var·[ var_hinge(P(e_t)) + var_hinge(e_{t+gap}) ]      # std → 1 per dim (anti-collapse)
       + cov·[ offdiag_cov(P(e_t)) + offdiag_cov(e_{t+gap}) ]  # decorrelate dims

# Stage 3 — dynamics (RSSM):
L_dyn  = α·‖D_ψ(s_t) − sg(e_t)‖²  +  β·KL(q_φ(z_t|h_t,e_t) ‖ p_θ(z_t|h_t))    # KL-balanced, free-bits
L_imag = Σ_k γ^k ‖D_ψ(roll_k(s_t)) − sg(e_{t+k})‖²

# Stage 4 — decode heads (uniqueness-weighted per sample):
L_head = w_q·pinball(q̂, exit_return) + w_b·CE(barrier̂, barrier)
       + w_v·Huber(σ̂, realized_vol)  + w_m·Huber(m̂, mae)
       + w_c·calibration(coverage)     # differentiable coverage penalty or conformal wrap
```

Reuse `quanthelion.training.FocalLoss` for imbalanced barrier where helpful. Weights live in
`configs/*.yaml` under `loss.weights`. The imagination depth **`K` must be ≥ `max(horizon_bars)`**
(i.e. ≥ 12) — otherwise the longest-horizon rollout is extrapolated beyond any trained imagination
step and its uncertainty is not calibrated. `γ < 1` down-weights the deepest (noisiest) steps.

---

## 23. Evaluation — calibration first, PnL last

`evaluation/` modules through a local `MetricRegistry` (OCP).

### 23.1 World-model metrics (Stage 3 gate)
- **1-step latent error** `‖D_ψ(s_t) − e_t‖` and **k-step** errors, vs baselines: persistence
  (`ê_{t+k}=e_t`) and a linear AR on `e`. **The model must beat both.**
- **Latent consistency**: posterior↔prior KL trends down; prior samples reconstruct held-out `e`.

### 23.2 Calibration metrics (Stage 5 gate — PRIMARY)
- **Quantile coverage** `ĉ_τ = (1/N)Σ 1{r_i ≤ q̂_τ}`; require `|ĉ_τ − τ|` small for all `τ`.
- **PIT**: `u_i = F̂(r_i | x_i)`; if calibrated `{u_i} ~ Uniform(0,1)` — KS / histogram test.
- **ECE** for barrier head; reliability diagrams.
- **Rollout coverage**: realised outcomes fall within ensemble predictive intervals at nominal rate.
- Optional **conformal** wrap (`quanthelion.uncertainty.conformal`) for finite-sample coverage and to
  drive the calibration risk-envelope (shrink size when coverage degrades).

### 23.3 Trading metrics (Stage 8 — only after calibration passes)

Two **distinct** statistical questions, two methods — do not conflate them:

- **(a) Is the strategy's own Sharpe real, or selection-bias luck?** → **Deflated Sharpe Ratio
  (de Prado)** (`backtesting/deflated_sharpe.py`): correct the observed Sharpe for the **number of
  trials** and non-normality (skew/kurtosis); report **PSR/DSR**, not raw Sharpe. DSR is computed on
  the **contiguous walk-forward** PnL path.
- **(b) Does it beat the do-nothing/naive alternatives after costs?** → **baselines** (always-flat,
  buy-and-hold, random-with-matched-turnover) compared by a **paired block-bootstrap test of the
  per-bar return differences** (block length ≥ holding period). DSR does **not** answer this — it
  deflates a single strategy's Sharpe; it is not a two-strategy comparison.
- **CPCV is for predictive/calibration-metric *distributions only*, not the PnL path.** Combinatorial
  purged CV (`backtesting/combinatorial_cv.py`) gives a spread of quantile-coverage / latent-error /
  barrier-ECE across many fold combinations — useful for robustness of the *model*. It is **not** used
  to compute the trading PnL, because a path-dependent account (running drawdown/margin, management
  loop) cannot be simulated across time-discontiguous stitched test blocks. The **PnL backtest is
  strictly contiguous walk-forward** (§24).
- Net vs gross PnL, max drawdown, Calmar, profit factor, turnover, holding time, **no-trade quality**
  (`no_trade_metrics.py`: avoided-loss fraction, regret of declined winners, calibration of the
  no-trade decision — first-class, and *not gameable by never trading*), risk-shield interventions,
  **regime/event-day breakdown**, **cost & slippage sensitivity** sweep.
- **Execution:** estimated-vs-actual slippage, fill-quality estimate, realism-score accuracy,
  cost-adjusted edge-survival rate.

---

## 24. Backtesting design

`backtesting/`. The **PnL backtest is a contiguous, purged, embargoed walk-forward** (multiple
sequential train→test folds via `quanthelion.labels.embargo.make_purged_splits`) — **never** a random
split, and **never** time-discontiguous CPCV blocks for the PnL path (a path-dependent account cannot
be simulated across stitched-together test windows; see §23.3). CPCV is reserved for *predictive-metric*
robustness only. Requirements: no lookahead, no future-data leakage, no survivorship bias, futures
rollover handling, expired-contract mapping, corporate-action handling, realistic Indian costs,
conservative spread/slippage, **management-loop cadence** (§19), event-day/expiry stress tests,
regime-wise breakdown. Given the ~1.6-yr futures window (§9.1a), expect **≈ 3–5 folds**, not 8.

Reports **separate gross vs net PnL**, calibration summary, planner decisions, risk-shield
interventions, bad-vs-good trades blocked, no-trade quality, execution-cost impact, slippage
sensitivity, regime-wise performance, **DSR (trial-deflated Sharpe)**, and the **paired-bootstrap
baseline comparison** (§23.3, two separate tests). `leakage_report.py` re-runs the leakage checks over
the backtest window and fails loudly on any violation.

---

## 25. Paper-trading design

`paper_trading/`. **Never silently executes.** Every decision emits the §29 audit record. Components:
dry-run `BrokerAdapterProtocol`, paper portfolio, `execution_logger`, `decision_logger`, risk-event
logger, calibration/drift monitor (shrinks size when live coverage degrades). Depends on
`BrokerAdapterProtocol`, never a concrete broker (DIP). Honest caveat: **paper ≠ live** — latency,
partial fills, and the exchange's view of your orders are only fully revealed live; V2/V3 close this
gap with own fill logs and a microstructure simulator.

---

## 26. Baselines & kill criteria (falsification-first)

The single most important honesty mechanism: **the world model must justify its complexity.** Before
and alongside the full build, run these and enforce the gates.

**Edge-existence probe (Stage 0.5, cheapest).** On purged folds, test whether *any* PIT feature set
(returns, futures basis/OI-flow, VIX, breadth) has OOS-stable, post-cost predictive power on BankNIFTY
at 5/30/60 min — via simple linear/GBM probes and a costed long/flat rule. *If nothing survives costs,
no architecture will rescue it — re-scope (e.g. to a pure vol/regime product) or stop.*

**Calibration baseline (Stage 5 gate).** A HAR-RV / GARCH vol model + split-conformal return quantiles
is the baseline. Because calibration has a ceiling (both can be well-calibrated), the RSSM is justified
only if it is **at least as calibrated AND strictly better on a *second* axis** — *sharper* intervals
at equal coverage, better **conditional (regime-wise) coverage**, or better **multi-horizon
coherence**. *If a 50-line GARCH + conformal matches it on every axis, the RSSM is not justified* — keep
the simpler model or fix the RSSM.

**Trading baselines (Stage 8 gate).** Two separate tests (§23.3): **(a)** the strategy's Sharpe must
survive **trial-deflation (DSR > 0)** on the contiguous walk-forward path; **(b)** it must beat
always-flat, buy-and-hold, and random-matched-turnover *after costs* by a **paired block-bootstrap** of
return differences. *Raw Sharpe on one slice is not evidence; DSR alone does not prove you beat
buy-and-hold.*

These kill gates are the project's spine of honesty. Passing them is the definition of success; a
clean "no edge survives costs" result is a valid, valuable outcome — not a failure to hide.

---

## 27. Coding standards / SOLID / DRY

Python 3.11+, `src`-layout, type hints everywhere, Pydantic/dataclasses for schemas/configs, `ruff` +
`black` + `mypy`/`pyright` + `pytest`, structured logging (`quanthelion.utils.logging.get_logger`),
deterministic seeds, config-driven, no hardcoded paths, no broker secrets, no notebook-only logic, no
silent failure, tensor shape assertions.

- **SRP:** one job per class (encoders encode, worlds simulate, shield validates, planner ranks).
- **OCP:** new encoder/loss/risk-rule/broker/cost-model/metric is pluggable via Protocols + registries.
- **LSP:** `EncoderProtocol, HeadProtocol, LossProtocol, RiskRuleProtocol, CostModelProtocol,
  LatentDynamicsProtocol, BrokerAdapterProtocol` make modules substitutable.
- **ISP:** return head ⊥ broker logic; broker adapter ⊥ losses; dataset ⊥ planner internals.
- **DIP:** `MPCPlanner → CostModelProtocol` (not `ZerodhaCostModel`); `PaperEngine →
  BrokerAdapterProtocol`; `Trainer → ModelProtocol`.
- **DRY:** one definition each for return/vol/ATR/basis/OI-flow/calendar/cost/slippage/
  walk-forward-split/leakage-check/metric, shared by train, backtest, paper.

---

## 28. Testing strategy

`tests/` with `pytest`. Tests must **prove**:

- portfolio variables never enter the Market World predictor,
- no `msh_jepa` import,
- triple-barrier labels are path-correct; uniqueness weights & purge/embargo are applied in *both*
  train construction and CV,
- the RSSM prior is *trained* — rollout spread tracks held-out coverage (`test_rollout_calibration`),
- the Risk Shield can override the planner; de-risking is never blocked,
- `NO_TRADE` is always in the candidate set; costs are applied before acceptance,
- future labels are never used as features,
- corporate-action adjustment is verified (`test_corporate_actions`),
- V1 runs with **no** historical tick/depth and **no** option chain,
- backtest reproducibility under fixed seed; DSR computed correctly (`test_deflated_sharpe`).

---

## 29. CLI design & audit record

Thin `scripts/` over the library:

```bash
cd work/helion_risk_world
python scripts/assemble_data.py      --config configs/v1.yaml   # + corp-action / rollover validation
python scripts/validate_data.py      --config configs/v1.yaml
python scripts/build_features.py     --config configs/v1.yaml
python scripts/label.py              --config configs/v1.yaml    # triple-barrier + uniqueness + folds
python scripts/pretrain.py           --config configs/v1.yaml    # Stage 2
python scripts/train_world_model.py  --config configs/v1.yaml    # Stage 3 (RSSM)
python scripts/train_heads.py        --config configs/v1.yaml    # Stage 4
python scripts/calibrate.py          --config configs/v1.yaml    # Stage 5 GATE
python scripts/backtest.py           --config configs/backtest_banknifty.yaml
python scripts/paper_trade.py        --config configs/paper_trading.yaml
python scripts/generate_report.py    --run-id <id>
```

Every script: `--config`, `--seed`, `--log-level`, `--dry-run`.

**Audit record (`FinalDecision`)** — emitted by every decision in backtest and paper:
`ts, market_summary, latent_regime, prediction (quantiles/vol/barrier/mae), epistemic, ood_score,
portfolio_state, candidates[], candidate_scores[U(a)], execution_realism, risk_shield_result,
final_action, reason_code, expected_cost, expected_cvar, expected_reward, rejected_actions[], result`.

---

## 30. Config example

YAML under `configs/`, loaded via `quanthelion.config.ConfigLoader`, validated into typed dataclasses:

```yaml
project: helion_risk_world
seed: 7
base_interval: "5min"
horizon_bars: [3, 6, 12]            # 15 / 30 / 60 min
universe:
  traded: BANKNIFTY_FUT             # single position in V1
  context: [NIFTY, HDFCBANK, ICICIBANK, SBIN, AXISBANK, KOTAKBANK]
data_window:                         # §9.1a pretrain-wide / train-narrow
  pretrain_range: ["2022-01-03", "2026-05-29"]   # spot/constituents (no futures features)
  model_range:    ["2024-10-01", "2026-05-25"]   # futures window: RSSM + heads + calib + backtest
label:  { kind: triple_barrier, instrument: BANKNIFTY_FUT_continuous,   # label on the traded instrument
          H: 12, u: 2.0, d: 2.0, vol_span: 50, uniqueness: true }       # H = max(horizon_bars)
model:  { latent_dim: 128, rssm: { stoch: 32, deter: 128 } }
loss:   { repr: { sim: 1.0, var: 1.0, cov: 0.04 },
          dyn:  { alpha: 1.0, beta: 1.0, free_bits: 1.0, kl_balance: 0.8 },   # KL terms live with the loss
          imag: { gamma: 0.95, K: 12 },                                       # K >= max(horizon_bars)
          head: { q: 1.0, barrier: 0.5, vol: 0.5, mae: 0.5, calibration: 0.4 } }
planner: { risk_aversion_lambda: 3.0, cvar_alpha: 0.05,   # CVaR as positive shortfall (§19)
           sizes: [0.0, 0.1, 0.25, 0.5, 1.0], cadence: management_loop }
calibration_gate: { max_coverage_err: 0.05, max_ece: 0.05,
                    must_match: garch_conformal, must_improve: [sharpness, conditional_coverage] }
backtest: { scheme: walk_forward, folds: 4, embargo_bars: 12,    # contiguous WF for PnL (~1.6yr → few folds)
            baselines: [flat, buy_hold, random_matched],
            significance: { dsr: true, baseline_test: paired_block_bootstrap } }
cpcv_metrics: { folds: 8, purge: true }   # CPCV used ONLY for predictive/calibration-metric stability
risk_profile: balanced
data_sources: configs/data_sources.yaml
```

---

## 31. Failure modes & flaw register

Rejected designs and the live flaws this spec fixes.

**Rejected designs (and why):** pure buy/sell classifier (no downside/account/execution reasoning, and
direction is near-unpredictable); RL-first (sample-inefficient, unsafe before a validated world model);
historical order book / option chain as a V1 dependency (data not available; blocks V1); portfolio
vars in market prediction (non-causal leakage); ignoring cost/slippage (turns a backtest toy into a
live loss machine); profit-only evaluation (hides drawdown/tail/overtrading and trial-selection bias);
ML overriding the Risk Shield; random or single-slice train/test split (leaks the future, no DSR); god
classes / duplicated feature logic; treating HRW as a guaranteed live system or copying msh_jepa.

**Flaw register — what was wrong in the implementation and how this spec prevents it:**

| # | Flaw (verified) | Sev | Remedy |
|---|---|---|---|
| F1 | World-model stages were stubs; "dynamics" was untrained `GRUCell(noise)` | 🔴 | §14 trained RSSM (prior/posterior + KL + multi-step); §21 Stages 2–3 gates |
| F2 | "Epistemic = ensemble spread" was init-noise | 🔴 | §14.2 KL trains prior → §14.3 calibrated ensemble; §16 |
| F3 | Labels were fixed-horizon return, path-blind | 🔴 | §11.1 triple-barrier |
| F4 | Vol/barrier labels never constructed (random in tests) | 🔴 | §11.1 vol/MAE/barrier from the path |
| F5 | Regime label derived from forward return (circular) | 🟠 | §11.4 regime is context; head label state-derived |
| F6 | Overlapping labels, no uniqueness weighting, no intra-train purge | 🔴 | §11.2–11.3 uniqueness + purge/embargo everywhere |
| F7 | Spot index volume/OI ≡ 0 → dead features; real microstructure unused | 🔴 | §9.2 / §12 futures basis/OI/calendar/rollover + OI-flow |
| F8 | Option-surface & some regime planes dead on real data | 🔴 | §9.3 futures replaces options in V1; options → V2 |
| F9 | Corporate actions (HDFC merger) not validated | 🔴 | §9.4 mandatory adjustment check |
| F10 | Planner = 8 hand-tuned magic weights; wouldn't trade | 🔴 | §19 single-λ mean–CVaR utility |
| F11 | Decision cadence ≠ holding period (re-decide every bar on H-bar label) | 🟠 | §19 barrier-managed management loop |
| F12 | CVaR from 5-knot linear-tail extrapolation | 🟠 | §17 CVaR (positive shortfall) over the head-implied ΔW distribution; §19 sign convention |
| F13 | Portfolio World re-seeds same RNG every step (correlated noise) | 🟠 | §17 independent stream per decision |
| F14 | OOD detector fit on untrained latents (noise) | 🟠 | §16 OOD = prior NLL under trained dynamics |
| F15 | Calibration absent (CalibrationLoss stub, no conformal) | 🟠 | §21 Stage-5 calibration gate; §23.2 |
| F16 | No real OOS: one contiguous slice, no baselines/DSR | 🔴 | §23.3 / §24 contiguous purged WF (CPCV for metrics only) + baselines (paired bootstrap) + DSR |
| F17 | "Portfolio World" / "derivatives-aware" oversold | 🟡 | §17, §9.3 honest scoping (single position; futures-not-options) |

**Operational failure modes monitored at runtime:** calibration drift, OOD regime, data gaps, stale
data, broker timeout, margin spike, drawdown breach, repeated rejected fills → each maps to a
Risk-Shield reaction and a logged event.

---

## 32. Ablation study plan

Toggle one factor, measure calibration + trading + no-trade metrics on the same purged folds:

1. RSSM (trained prior) vs deterministic single-path vs injected-noise (the old bug) — proves §14 matters.
2. Futures microstructure encoder vs price-only.
3. Triple-barrier labels vs fixed-horizon return.
4. Uniqueness weighting + purge on vs off.
5. Mean–CVaR single-λ planner vs the old 8-weight scorer.
6. Management-loop cadence vs re-decide-every-bar.
7. OOD = prior-NLL vs post-hoc Gaussian.
8. Conformal envelope on vs off.
9. Cross-asset breadth/dispersion on vs off.
10. λ / CVaR-α risk-aversion sweep.

---

## 33. V1 / V2 / V3 roadmap

- **V1 (this spec):** futures-based microstructure (~1.6-yr window, §9.1a); **RSSM world model trained &
  calibrated**; triple-barrier labels with uniqueness + purge; mean–CVaR planner with management-loop
  cadence; contiguous purged-WF evaluation (CPCV for metric stability) with baselines + DSR. Single
  position. **No options, no tick/depth.**
- **V2:** self-collected option-chain history → activate the option-surface plane; own fill logs →
  calibrate Execution Reality; live paper trading with calibration/drift monitors; multi-asset book.
- **V3:** vendor tick/depth → LOB microstructure expert; offline-RL refinement of the planner;
  advanced execution simulator; multi-strategy portfolio mode.

---

## 34. Open research questions (honest)

1. Does **any** PIT feature set (incl. futures basis/OI-flow) have OOS-stable, post-cost predictive
   power on BankNIFTY at 5/30-min horizons at all? *Answer this (§26) before trusting any architecture.*
2. Latent-dynamics family for low-SNR 5-min data: RSSM vs JEPA-only vs SSM — judged by **calibration**,
   not accuracy.
3. KL weighting / free-bits to avoid posterior collapse on near-random-walk returns.
4. Conformal recalibration cadence under regime shift (adaptive vs split).
5. Barrier-managed holding loop vs fixed-cadence decisions — which gives better cost-aware OOS results?
6. Reliable epistemic uncertainty without huge ensembles (rollout dispersion vs MC-dropout vs SWAG).
7. Measuring no-trade quality in a way that is not gamed by never trading.
8. Event-blackout policy: hard block vs size-down vs regime-specialised expert?
9. Multi-horizon quantile coherence: enforce monotone/coherent quantiles across 15/30/60 min.
10. Calibrating Execution Reality with sparse own-fill logs without overfitting to one broker.

---

## 35. First-week V1 implementation plan

Goal: a runnable, leak-checked V1 skeleton that validates data, builds futures features, makes
triple-barrier labels, trains a tiny RSSM + heads, passes a calibration smoke-check, and produces an
audited purged-WF backtest decision stream — end-to-end, conservative, no tick/option data.

- **Day 1 — Scaffolding & integration.** `pip install -e ../quanthelion && pip install -e .`. Wire
  `integration/quanthelion_adapter.py` to the §7.1 symbols; `test_quanthelion_adapter` +
  `test_no_msh_jepa_import`. Green CI on stubs.
- **Day 2 — Data reality & validation.** `assemble_data.py` (futures continuous + rollover),
  `corporate_actions.py` (HDFC-merger check), PIT stamping, session handling. Tests:
  `test_data_quality`, `test_corporate_actions`.
- **Day 3 — Schemas, leakage spine, labels.** All `schemas/*` (incl. `LabelRecord`);
  `data/leakage_checks.py`; `labeling/barrier_labeler.py` (wraps `quanthelion.labels.triple_barrier`)
  + `uniqueness.py` + `purged_cv.py`. Tests: `test_no_leakage`, `test_triple_barrier`,
  `test_uniqueness`, `test_purged_cv`.
- **Day 4 — Features (futures-centric) + tiny encoder.** `feature_builder.py` (basis/OI-flow/calendar/
  breadth), `FuturesEncoder` + `TemporalEncoder` + `FusionEncoder` → `e_t`; `test_shapes`,
  `test_train_backtest_feature_parity`.
- **Day 5 — RSSM + heads (tiny).** `worlds/rssm.py` (prior/posterior + KL), `rollout_engine.py`
  sampling the trained prior; quantile/barrier/vol heads; overfit a small slice. Tests: `test_rssm`,
  `test_rollout_calibration`.
- **Day 6 — Worlds, execution, planner, shield.** `PortfolioWorld.step` (head-implied ΔW, positive-CVaR
  shortfall), conservative `ExecutionReality`, mean–CVaR `RewardScorer`, `MPCPlanner` + management loop,
  `RiskShield`. Tests: `test_portfolio_world`, `test_execution_reality`, `test_no_trade_action`,
  `test_risk_shield`.
- **Day 7 — Calibration + purged-WF backtest + report.** `calibration_metrics.py` (coverage/PIT/ECE),
  `backtest_engine.py` (purged WF), `deflated_sharpe.py`, baselines, `leakage_report.py`,
  `generate_report.py`. Tests: `test_backtest_engine`, `test_deflated_sharpe`.

**Definition of done (week 1):** `pytest` green; the data→label→train→calibrate→backtest pipeline runs
end-to-end on a small cached slice; the backtest emits `FinalDecision` audit records and a DSR-vs-
baseline line; `leakage_report` passes; no `msh_jepa` import; no portfolio field reachable by any
encoder; no dependence on spot volume/OI, option chains, or tick data.

---

# Appendix A — Corrected pseudocode

> Illustrative, not final. Types reference `helion_risk_world.schemas`.

### `HelionRiskWorld.forward`

```python
def forward(self, market_batch: MarketBatch) -> ModelPrediction:
    # market_batch: market planes only (NO portfolio fields) — enforced by type
    e_t = self.fusion(                                      # encoded rep e_t, [B, d]
        temporal=self.temporal_encoder(market_batch.candles),
        cross=self.cross_asset_encoder(market_batch.cross_asset),
        futures=self.futures_encoder(market_batch.futures),     # basis/OI-flow/calendar/rollover
        regime=self.regime_encoder(market_batch.regime),
    )
    # h_t depends on the WHOLE window: roll the RSSM forward with the posterior over e_{t-L..t}
    s_t = self.rssm.filter(market_batch.window_e)          # -> (h_t, z_t); not a single-step posterior
    rollouts = self.rollout_engine.imagine(s_t, horizons=self.H, n_samples=self.S)  # sample trained PRIOR
    return self.heads(rollouts)                            # quantiles/vol/barrier/mae + unc/ood from dist
```

### `RSSM.imagine` (rollout samples the *trained prior* — not torch.randn)

```python
def imagine(self, s_t, horizons, n_samples) -> Tensor:     # -> [S, B, |H|, d]
    out = []
    for _ in range(n_samples):
        h, z, traj = s_t.h, s_t.z, {}
        for k in range(1, max(horizons) + 1):
            h = self.gru(h, z)                             # deterministic carry (autonomous, no action)
            z = self.prior(h).rsample()                    # LEARNED p_theta(z|h); calibrated by KL
            if k in horizons:
                traj[k] = self.repr_head(h, z)             # D_psi -> predicted encoded state ê_{t+k}
        out.append(stack([traj[k] for k in sorted(horizons)], dim=1))
    return stack(out, dim=0)                               # ensemble spread = calibrated epistemic unc
```

### `RSSM.filter` (infer current state from the observed window — used before imagine)

```python
def filter(self, window_e) -> State:                       # window_e = [e_{t-L}, ..., e_t]
    h, z = self.h0, self.z0
    for e_tau in window_e:                                 # roll the recurrence over the lookback
        h = self.gru(h, z)
        z = self.posterior(h, e_tau).rsample()            # posterior sees the observation
    return State(h=h, z=z)                                 # s_t = (h_t, z_t)
```

### `RSSM.loss` (Stage 3)

```python
def loss(self, seq_e):                                     # seq of encoded observations e_t
    h = self.h0; L_dyn = L_imag = 0
    states = []
    for t, e_t in enumerate(seq_e):
        prior = self.prior(h)                              # p_theta(z|h)
        post  = self.posterior(h, e_t)                     # q_phi(z|h,e_t)
        z = post.rsample()
        e_hat = self.repr_head(h, z)
        L_dyn += self.alpha * mse(e_hat, sg(e_t)) \
               + self.beta  * kl_balanced(post, prior, free_bits=self.fb)
        states.append((h, z)); h = self.gru(h, z)
    for t, (h_t, z_t) in enumerate(states):                # multi-step imagination consistency
        h_r, z_r = h_t, z_t
        for k in range(1, self.K + 1):
            h_r = self.gru(h_r, z_r); z_r = self.prior(h_r).rsample()
            if t + k < len(seq_e):
                L_imag += (self.gamma ** k) * mse(self.repr_head(h_r, z_r), sg(seq_e[t + k]))
    return L_dyn + L_imag
```

### `make_triple_barrier_labels` (illustrates the quanthelion labeler we WRAP + our uniqueness step)

```python
# close = roll-aware FUTURES continuous close (the traded instrument), sigma = EWMA futures vol
def make_triple_barrier_labels(close, sigma, u, d, H) -> list[LabelRecord]:
    labels = []
    for t in range(len(close) - 1):
        e = close[t]; up = e * (1 + u * sigma[t]); lo = e * (1 - d * sigma[t])
        barrier, exit_i, exit_px = "TIMEOUT", min(t + H, len(close) - 1), close[min(t + H, len(close)-1)]
        for i in range(t + 1, min(t + H, len(close) - 1) + 1):
            if close[i] >= up: barrier, exit_i, exit_px = "TARGET", i, up; break
            if close[i] <= lo: barrier, exit_i, exit_px = "STOP",   i, lo; break
        path = close[t + 1: exit_i + 1]
        labels.append(LabelRecord(ts=t, label_realized_at=exit_i, horizon_bars=H, barrier=barrier,
            exit_return=exit_px / e - 1, exit_t=exit_i,
            realized_vol=std(diff(log(path))) if len(path) > 1 else 0.0,
            mae=max((e - path.min()) / e, 0.0) if len(path) else 0.0, uniqueness_weight=None))
    return apply_uniqueness_weights(labels)                # ū_i from label-span concurrency
```

### `MPCPlanner.plan` (mean–CVaR, management-loop aware)

```python
def plan(self, pred: ModelPrediction, p: PortfolioState, R: RiskProfile) -> FinalDecision:
    if p.in_position and not self.management.exit_signal(p, pred):
        return self.auditor.record_hold(p, pred)           # barrier-managed: do not re-decide every bar
    candidates = self.sampler.enumerate(p, R)              # state-dependent; baseline (NO_TRADE/HOLD) incl.
    scored = []
    for a in candidates:
        # ΔW distribution is ANALYTIC from the heads (no price-path sim, §17):
        #   barrier_probs {stop,target,timeout} x {−d·σ, +u·σ, exit_return quantiles}, sized & costed
        _, cons = self.portfolio_world.step(p, a, pred.heads, R)
        ce = self.execution_reality.estimate(to_order(a, p), pred.market_snapshot)
        # cvar_dW is a POSITIVE shortfall (§19); riskier -> larger cvar -> lower U
        u  = cons.exp_dW - R.lam * cons.cvar_dW - ce.total_cost      # U(a) = E[ΔW] − λ·CVaR − Cost
        scored.append((a, u if a is not baseline(p) else 0.0, cons, ce))
    best = max(scored, key=lambda s: s[1])
    decision = self.risk_shield.validate(best, p, R, pred)  # may force NO_TRADE / REDUCE / EXIT
    return self.auditor.record(decision, scored, pred, p)
```

### `RiskShield.validate`

```python
def validate(self, best, p, R, pred) -> RiskDecision:
    for rule in self.rules:                                # RiskRuleProtocol, ordered; de-risking never blocked
        d = rule.check(best.action, p, R, pred)
        if not d.allowed:
            return RiskDecision(allowed=False, reason_code=d.reason_code,
                                adjusted_size=d.fallback_size, final_action=d.fallback_action)
    return RiskDecision(allowed=True, reason_code="OK",
                        adjusted_size=best.action.size_fraction, final_action=best.action)
```

### `BacktestEngine.run` (purged WF, baselines, DSR)

```python
def run(self, cfg) -> BacktestReport:
    folds = make_purged_splits(index, n_splits=cfg.folds, embargo=cfg.embargo)   # CONTIGUOUS WF (not CPCV)
    audits, returns = [], []
    for train_idx, test_idx in folds:
        model = self.trainer.fit(self.features[train_idx], self.labels[train_idx])  # uniqueness-weighted
        assert self.calibration.passes(model, self.val_of(train_idx))               # Stage-5 GATE
        port = PaperPortfolio(cfg.account)
        for t in test_idx:
            mbatch = self.feature_builder.build_window(t)   # point-in-time, market plane only
            assert_no_portfolio_in_market(mbatch)
            pred = model.forward(mbatch)
            decision = self.planner.plan(pred, port.state, cfg.risk_profile)  # management-loop cadence
            fill = self.cost_sim.apply(decision, market_at(t))                # real costs + slippage
            port.update(fill); audits.append(decision); returns.append(port.step_return)
    return self.reporter.build(audits, returns,
        baselines=self.baselines.run(index),                       # flat / buy-hold / random
        dsr=deflated_sharpe(returns, n_trials=cfg.trials),         # (a) is own Sharpe real?
        baseline_test=paired_block_bootstrap(returns, baselines),  # (b) beats baselines after costs?
        leakage=self.leakage_report.run())
```

---

*End of specification. Companion deep-dives live in `docs/`. This document is the contract and now
supersedes `WORLD_MODEL_SPEC.md`, whose corrections are folded in above.*
