# Stock Screener Engine (Indian Equities)

Production-oriented, modular stock screener engine for Indian equity markets.
Generates two signal families from public market and fundamental data:

- **Long-term investment candidates** — quality + value + governance composite
- **Short-term swing trade candidates** — trend + momentum + event catalyst composite

The core research and scoring engine is **broker-agnostic by design**.  Zerodha
and ICICI Breeze adapters exist as optional modules and are disabled by default.

---

## What This Engine Implements

### Domain Model (granular entity types)
| Entity | Update frequency | Fields |
|---|---|---|
| `MarketSnapshot` | Daily / intraday | OHLCV, delivery ratio, market cap |
| `FundamentalsSnapshot` | Quarterly | PE, ROE, D/E, FCF margin, growth rates |
| `GovernanceSnapshot` | Quarterly | Promoter holding, insider scores, audit opinion |
| `StockSnapshot` | Convenience | Unified flattened type for demos / legacy code |

### Feature Engine (`core/feature_specs.py` + `core/features.py`)
- Sector-relative + rolling valuation normalization (PE/PB z-score context)
- Explicit `earnings_stability` and `leverage_trend` risk features
- 26 named features organised in independent category methods
- Each category is pure-function: no IO, independently testable
- Feature constants live in `feature_specs.py`; no magic strings
- Backward-compatible `compute_from_snapshot()` wraps `StockSnapshot` for demos

### Scoring Engine (modular layer + compatibility facade)
- Modular layer (new):
  - `core/scoring_base.py`
  - `core/scoring_long_term.py`
  - `core/scoring_swing.py`
  - `core/scoring_risk.py`
  - `core/scoring_ranking.py`
  - `core/signal_generator.py`
  - `core/signal_schemas.py`
  - `core/explainability_engine.py`
  - `core/feature_access.py`, `core/normalizers.py`, `core/validators.py`
- Compatibility facade (existing imports still work):
  - `core/scoring.py`
  - `core/signals.py`
  - `core/explainability.py`
- **All weights and thresholds are configurable from YAML**.
- Regime-aware configurable profile switching (`bull` / `bear` / `sideways`).
- Optional calibration-driven prior auto-tuning from IC/decay/turnover diagnostics.

```yaml
# defaults.yaml — scoring section
scoring:
  long_term_min_score: 24.0
  swing_min_score: 28.0
  long_term_weights:
    growth_quality: 18.0
    profitability_quality: 17.0
    ...
  swing_weights:
    trend_strength: 20.0
    momentum_strength: 18.0
    ...
  risk_weights:
    liquidity_risk: 0.20
    volatility_risk: 0.20
    leverage_risk: 0.20
    earnings_instability_risk: 0.15
    event_uncertainty_risk: 0.15
    governance_risk: 0.10
  ranking:
    top_k_long_term: 25
    top_k_swing: 25
```

### Signal Explainability (`core/explainability.py`)
- Every `SignalResult` carries a `SignalExplanation` with:
  - `top_positive_drivers` (human-readable labels, e.g. `"growth quality: 14.40"`)
  - `top_negative_drivers` (risk-prefix components)
  - `holding_horizon`, `entry_logic`, `invalidation_logic`
  - `risk_flags` list

### Data Source Interfaces (`data_sources/base/interfaces.py`)
| Interface | Purpose |
|---|---|
| `MarketDataProvider` | Daily OHLCV + unified snapshot |
| `FinancialsProvider` | Fundamentals + governance snapshots |
| `FilingsProvider` | BSE/NSE regulatory filings |
| `NewsProvider` | Financial news + sentiment |
| `ExchangeAdapter` | Corporate actions + announcements |
| `TextEventProvider` | Generic text events + sentiment |
| `BrokerAdapter` | Optional execution layer |

### Adapter Packages
```
data_sources/
  market/
    mock_market_data.py         # MockIndianMarketDataProvider
    mock_fundamentals.py        # MockFinancialsProvider
  filings/
    mock_filings.py             # MockFilingsProvider
  news/
    mock_news.py                # MockNewsProvider
  exchange/
    nse_adapter.py              # NSEExchangeAdapter (implements ExchangeAdapter)
  broker/
    zerodha_adapter.py          # disabled by default
    breeze_adapter.py           # disabled by default
```

### Model Layer (`models/`)
- `ScorerProtocol` — structural `typing.Protocol`; any callable matching `.score(fv) → (float, dict)` satisfies it without explicit inheritance (ML models, ensembles, rule-based scorers all interchangeable)
- `LongTermModel.with_weights(LongTermWeights(...))` — convenience factory
- `SwingModel.with_weights(SwingWeights(...))` — convenience factory

### Storage
- `LocalFileStorage` — raw/clean/features/signals CSV+JSON under configurable `root_dir`
- `SQLiteStore` — features, scores, signals tables with proper primary keys:
  - `features (symbol, as_of)` — composite PK, upsert safe
  - `scores (symbol, as_of)` — composite PK, upsert safe
  - `signals (symbol, category, run_date)` — composite PK, **no duplicate rows on re-run**

### Portfolio Construction Adapter
- Deterministic post-ranking adapter (`execution/portfolio_adapter.py`)
- Constraint layer: max positions, sector caps, liquidity floor, single-name cap
- Outputs target shares/weights and rejection reasons for dropped candidates

Example portfolio config with separate long/swing overrides:

```yaml
scoring:
  ranking:
    portfolio:
      enabled: true
      max_positions_long: 12
      max_positions_swing: 10
      max_sector_positions: 3
      min_avg_daily_volume: 1000000
      max_single_position_weight: 0.12
      capital_base: 1000000

      # Shared defaults used if strategy-specific overrides are not set
      min_position_notional: 25000
      sector_target_weights:
        IT: 0.35
        Banking: 0.25
        Pharma: 0.20
        Energy: 0.20
      sector_target_tolerance: 0.05

      # Long-term specific overrides
      long_min_position_notional: 40000
      long_sector_target_weights:
        IT: 0.40
        Banking: 0.20
        Pharma: 0.20
        Energy: 0.20

      # Swing specific overrides
      swing_min_position_notional: 15000
      swing_sector_target_weights:
        IT: 0.30
        Banking: 0.35
        Pharma: 0.20
        Energy: 0.15
```

---

## Repository Layout

```
stock_screener_engine/
  config/             YAML + env settings, ScoringWeightsSettings
  core/
    entities.py       MarketSnapshot, FundamentalsSnapshot, GovernanceSnapshot, ...
    feature_specs.py  Named constants for all 20 feature keys
    features.py       FeatureEngine with independent category methods
    scoring.py        LongTermScorer, SwingScorer, configurable weights dataclasses
    explainability.py ExplanationEngine (single _top_components helper, no duplicates)
    signals.py        SignalGenerator — sector threaded through to SignalResult
    engine.py         ResearchEngine — accepts optional FinancialsProvider
    universe.py       UniverseSelector
    ranking.py        rank_by_long_term, rank_by_swing
  data_sources/
    base/interfaces.py  All provider + adapter ABCs
    market/           mock_market_data, mock_fundamentals
    filings/          mock_filings
    news/             mock_news
    exchange/         NSEExchangeAdapter (implements ExchangeAdapter)
    text/             MockTextEventProvider
    broker/           Zerodha + Breeze (disabled by default)
  models/
    protocols.py      ScorerProtocol
    long_term_model.py  LongTermModel.with_weights(...)
    swing_model.py    SwingModel.with_weights(...)
  pipelines/          daily_batch, feature_refresh, intraday_update, signal_generation
  storage/            local_files, sqlite_store (dedup-safe)
  execution/          order abstraction, execution router
  backtest/           cross_sectional, walk_forward, event_study scaffolds
  monitoring/         data_quality, health, signal_drift
docs/                 architecture, setup, extension guide
examples/             run_demo.py
tests/
  conftest.py         shared fixtures (AppSettings, FeatureVector, snapshots, ScoreCard)
  test_features.py    FeatureEngine — granular path + StockSnapshot compat
  test_scoring.py     configurable weights, risk flags, edge cases
  test_explainability.py  ExplanationEngine + _pretty_name regression
  test_universe.py    UniverseSelector liquidity filter
  test_engine_pipeline.py  end-to-end ResearchEngine
  test_config.py      settings loading and env overlay
  test_broker_optional.py  broker graceful-failure
```

---

## Quick Start

**Requirements**: Python 3.9+

```bash
# 1. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install the package in editable mode
pip install -e .

# 3. Optionally copy the env template
cp .env.example .env

# 4. Run demo pipeline
python examples/run_demo.py
```

Demo writes outputs under `data/`:
```
data/features/      feature vectors (CSV/JSON)
data/signals/       signal results
data/metadata.db    SQLite — features, scores, signals tables
```

---

## Configuration

Default config: `stock_screener_engine/config/defaults.yaml`

All settings are overridable with environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `SSE_ENV` | `dev` | Environment tag |
| `SSE_LOG_LEVEL` | `INFO` | Python log level |
| `SSE_STORAGE_ROOT` | `./data` | Output directory |
| `SSE_SQLITE_PATH` | `./data/metadata.db` | SQLite file |
| `SSE_ENABLE_ZERODHA` | `false` | Enable Zerodha broker |
| `SSE_ENABLE_BREEZE` | `false` | Enable Breeze broker |
| `SSE_MIN_LIQUIDITY` | `1000000` | Volume filter threshold |
| `SSE_MARKET_PROVIDER` | `nse_http` | Market data provider |
| `SSE_NEWS_PROVIDER` | `free_rss` | News source provider |
| `SSE_LLM_PROVIDER` | `heuristic` | LLM backend (`heuristic`, `openai`, `anthropic`) |
| `SSE_LLM_API_KEY_ENV` | `OPENAI_API_KEY` | Env var name that stores LLM API key |
| `SSE_LLM_AUDIT_PATH` | `./data` | Root path for low-confidence LLM audit logs |

### Scoring weights

Override any individual weight in `defaults.yaml` under `scoring.long_term_weights`
or `scoring.swing_weights` or `scoring.risk_weights`. Ranking cutoffs are under
`scoring.ranking`.

The `ResearchEngine` picks all of these up automatically via `AppSettings`.

### Signal Output Semantics

Each signal is built from:
1. Long-term category score (0-100)
2. Swing category score (0-100)
3. Risk penalty (0-max_risk_penalty)
4. Final score = category score - risk penalty

Outputs expose:
- Positive/negative driver contributions
- Missing feature hints
- Deterministic rejection reasons
- Horizon tag (`6-24 months` or `3-15 trading days`)

### LLM-Assisted Event Intelligence

The text pipeline now supports a hybrid event-intelligence path:

- Rule-based classification, event extraction, and sentiment remain available as the deterministic baseline.
- Optional LLM-assisted extractors can enrich document classification, event normalization, sentiment, and management-tone signals.
- All LLM outputs are normalized into typed schemas before they affect features, scoring, or explainability.
- Low-confidence LLM outputs can fall back to the rule pipeline.

Default config lives in `stock_screener_engine/config/defaults.yaml` under `llm:`:

```yaml
llm:
  enabled: false
  provider: heuristic
  model: heuristic-finance-v1
  base_url: https://api.openai.com
  api_key_env: OPENAI_API_KEY
  timeout_seconds: 30
  min_confidence: 0.55
  fallback_to_rules: true
  enable_management_tone: true
  audit_low_confidence: true
  audit_path: ./data
```

The shipped `heuristic` provider is deterministic and offline. It is intended as a provider-agnostic stub for testing and local development.

Supported real-provider wiring:

- OpenAI-style endpoints via `provider: openai` (or OpenAI-compatible gateway URL via `base_url`)
- Anthropic messages API via `provider: anthropic`

Startup now validates LLM provider credentials strictly:

- If `SSE_ENABLE_LLM_EXTRACTION=true` and provider is `openai` or `anthropic`, startup fails fast unless the env var named by `SSE_LLM_API_KEY_ENV` is present and non-empty.
- This prevents silently running with missing provider keys.

When `audit_low_confidence: true`, low-confidence LLM decisions are appended as JSONL artifacts under:

- `data/llm_audit/YYYY-MM-DD/low_confidence.jsonl`

Ingestion health reports are also written for operational monitoring under:

- `data/ingestion_health/YYYY-MM-DD/ingestion_health.jsonl`

Each report includes per-adapter and source-level (`news`, `filings`) fetch counts, failure counts, document counts, and latency (ms).

### Free News Sources

Deployment defaults use free, public RSS ingestion (no paid key required):

- Google News RSS search feeds per symbol (`free_rss` provider)
- Exchange announcements for filing-like event ingestion

This keeps the runtime disconnected from mock sources while preserving deterministic fallback behavior for LLM extraction.

---

## Pipelines

| Pipeline | Trigger | Purpose |
|---|---|---|
| `DailyBatchPipeline` | EOD | Full feature → score → signal cycle |
| `IntradayUpdatePipeline` | During market hours | Refresh swing-sensitive stack |
| `FeatureRefreshPipeline` | On-demand | Recompute features only |
| `SignalGenerationPipeline` | On-demand | Regenerate signals from cached features |

---

## Optional Broker Adapters

Both broker adapters are disabled by default and fail gracefully when credentials
are absent.

**Zerodha (Kite):**
```
SSE_ENABLE_ZERODHA=true
SSE_ZERODHA_API_KEY=...
SSE_ZERODHA_API_SECRET=...
SSE_ZERODHA_ACCESS_TOKEN=...
```

**ICICI Breeze:**
```
SSE_ENABLE_BREEZE=true
SSE_BREEZE_API_KEY=...
SSE_BREEZE_API_SECRET=...
SSE_BREEZE_SESSION_TOKEN=...
```

---

## Running Tests

```bash
pytest -q
```

All tests are offline — no network calls, no broker credentials required.

## Setup And Run Commands

```bash
# create env + install
python -m venv .venv && source .venv/bin/activate
pip install -e .

# run main demo
python examples/run_demo.py

# run modular scoring demo
python examples/scoring_framework_demo.py

# run LLM-assisted event intelligence demo
python examples/llm_event_intelligence_demo.py

# run tests
pytest -q
```

## Modular Scoring Demo

Run a standalone demo that scores a mini universe with missing-data handling,
risk penalties, explainability, and ranking:

```bash
python examples/scoring_framework_demo.py
```

## LLM Event Intelligence Demo

Run a side-by-side comparison of the research engine with and without the LLM-assisted text pipeline:

```bash
python examples/llm_event_intelligence_demo.py
```

The demo prints:

- Aggregated structured text features per symbol
- Long-term and swing score deltas with LLM assistance enabled

---

## Extending the Engine

### Plug in a real market data provider
Implement `MarketDataProvider` (and optionally `FinancialsProvider`) from
`data_sources/base/interfaces.py`, then pass it to `ResearchEngine`.

### Plug in an ML scorer
Any object with `.score(fv: FeatureVector) -> tuple[float, dict[str, float]]`
satisfies `ScorerProtocol`.  Pass it as `scorer=...` to `LongTermModel` or
`SwingModel`, or directly to `LongTermScorer`/`SwingScorer`.

### Add a new feature
1. Add a constant to `core/feature_specs.py`
2. Add it to the relevant `frozenset` group
3. Implement it in the matching `_xxx_features()` method in `core/features.py`
4. Add a weight entry in `LongTermWeights` or `SwingWeights` in `core/scoring.py`

---

## Future Roadmap

- Real NSE/BSE ingest adapters (bhavcopy, SEBI filings API)
- Financial statement parser with quality checks (Ind AS awareness)
- Transcript / news ingestion with transformer-based event extraction
- Portfolio and risk overlays + execution simulation
- ML ranker training pipeline and model registry
- Signal drift monitoring dashboards + alerting

