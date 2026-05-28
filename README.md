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
  conviction_weights:
    signal_agreement: 0.20
    data_completeness: 0.20
    source_confidence: 0.10
    backtest_evidence: 0.10
    sector_regime_confirmation: 0.15
    risk_resilience: 0.25
  ranking:
    top_k_long_term: 25
    top_k_swing: 25
```

Conviction is now an auditable cross-horizon support score. The base strength is
the average of risk-adjusted long-term and swing scores; it is then discounted by
signal agreement, data completeness, source confidence, backtest evidence,
sector/regime confirmation, and residual risk resilience. The report payload
includes the full `conviction_*` breakdown.

At runtime the engine injects source-confidence features from snapshot, feature,
freshness, and reconciliation quality checks. If a calibration/backtest artifact
exists at `scoring.calibration_auto_tune.report_path`, the same run also injects
global backtest evidence such as IC, top-quantile hit rate, and calibration
support into conviction scoring.

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
    mock_market_data.py         # deterministic test/dev provider
    sqlite_market_data_provider.py
    yfinance_market_data_provider.py
    broker_market_data_provider.py
  financials/
    sqlite_financials_provider.py
  filings/
    exchange_filings_provider.py
    mock_filings.py
  news/
    free_news_provider.py
    mock_news.py
  exchange/
    nse_http_adapter.py
    bse_http_adapter.py
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
- `MarketDataStore` — canonical security master, market calendar, OHLCV bars, and corporate actions:
  - `security_master (symbol, exchange)` — canonical identifiers and sector/industry tags
  - `market_calendar (venue, session_date)` — trading/holiday/special-session state
  - `ohlcv_bars (venue, symbol, ts, interval)` — queryable market history
  - `corporate_actions` — split/bonus/dividend action records with adjustment support
- `SQLiteMarketDataProvider` — research-facing canonical provider:
  - reads stored security master and OHLCV directly into scans/deep dives
  - uses adjusted history for features/backtests and unadjusted latest close for executable snapshots
  - emits freshness warnings or hard failures depending on canonical freshness settings
- `financial_statements` + `SQLiteFinancialsProvider` — point-in-time fundamentals:
  - stores period-end and filing-date aware financial statement rows
  - prefers four-quarter TTM when available, with annual fallback
  - computes ROE, ROCE, margins, leverage, cash-flow conversion, and YoY growth
  - computes PE/PB only when point-in-time market-cap data is available
  - supplies same-sector canonical peer context for sector-relative PE/PB z-scores
- `PeerComparisonBuilder` — sector-relative research ranking:
  - ranks valuation, quality, growth, balance-sheet risk, and composite peer position
  - powers `peer-report`, deep-dive peer sections, and sector peer leader views
- `equity_valuations` — point-in-time market-cap/share-count facts for valuation:
  - `as_of`, `market_cap`, `shares_outstanding`, free-float market cap, enterprise value
  - financial values should use the same currency/unit scale as statement rows
- `shareholding_patterns` — point-in-time ownership and governance facts:
  - promoter, FII, DII, and public holding percentages by period-end and filing date
  - drives promoter change, institutional ownership, and governance proxy features

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
    feature_specs.py  Named constants for feature keys
    features.py       FeatureEngine with independent category methods
    scoring.py        LongTermScorer, SwingScorer, configurable weights dataclasses
    explainability.py ExplanationEngine (single _top_components helper, no duplicates)
    signals.py        SignalGenerator — sector threaded through to SignalResult
    engine.py         ResearchEngine — accepts optional FinancialsProvider
    universe.py       UniverseSelector
    ranking.py        rank_by_long_term, rank_by_swing
  data_sources/
    base/interfaces.py  All provider + adapter ABCs
    market/           mock, canonical SQLite, HTTP, yfinance, broker-backed providers
    financials/       point-in-time SQLite provider and ingestion
    filings/          exchange, mock, and null filing providers
    news/             free RSS, generic, and mock providers
    exchange/         NSE/BSE HTTP ingestion adapters
    text/             text event providers
    broker/           Zerodha + Breeze (disabled by default)
  documents/          local document loading, classification, fact/commentary extraction
  llm/                optional validated LLM extraction and provider adapters
  models/
    protocols.py      ScorerProtocol
    long_term_model.py  LongTermModel.with_weights(...)
    swing_model.py    SwingModel.with_weights(...)
  pipelines/          data, document, backtest, signal, calibration, and daily workflows
  research/           company deep-dive and peer-comparison report packages
  reporting/          professional signal reports
  sector/             sector intelligence and peer leader views
  storage/            local files, signal SQLite, canonical market data store
  execution/          order abstraction, execution router
  backtest/           labels, ranking evaluation, costs, event study, calibration
  monitoring/         data quality, factor quality, invalidation, reconciliation, drift
examples/             run_demo.py
tests/                offline unit and integration tests
```

---

## Quick Start

**Requirements**: Python 3.11 from the shared Quant Engines environment.

```bash
# 1. From this repo, activate the shared environment one folder up
source ../.venv/bin/activate

# 2. Install the package into the shared environment
python -m pip install -e ".[all]"

# Optional: Yahoo Finance and broker SDK data sources
python -m pip install -e ".[market,broker]"

# 3. Optionally copy the env template
cp .env.example .env

# 4. Run demo pipeline
python examples/run_demo.py
```

Demo writes outputs under `data/`:
```
data/features/      feature vectors (CSV/JSON)
data/signals/       signal results
data/quality/       pipeline quality reports
data/metadata.db    SQLite — features, scores, signals tables
```

For real research runs, keep data, universes, plans, reports, PDFs, and other
research artifacts outside the git repo. A recommended local setup:

```bash
export SSE_STORAGE_ROOT="$HOME/stock_quant_engine_data"
export SSE_SQLITE_PATH="$SSE_STORAGE_ROOT/market.db"
mkdir -p "$SSE_STORAGE_ROOT/universe"
```

`README.md` is the only documentation-style file intended to live in git.
Local planning, architecture notes, and dated reviews live under the ignored
`documentation/` folder.

This repository should not have its own project-specific virtual environment for
normal development. The canonical runtime is `../.venv`, shared by the sibling
quant engines so integration work can converge on one portfolio-manager stack.

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
| `SSE_MARKET_PROVIDER` | `nse_http` | Market data provider (`canonical`, `nse_http`, `yfinance`, `zerodha`, `icici_breeze`, `mock`) |
| `SSE_FINANCIALS_PROVIDER` | `none` | Financials provider (`none`, `canonical`, `mock`); canonical is auto-used for canonical market scans |
| `SSE_NEWS_PROVIDER` | `free_rss` | News source provider |
| `SSE_CANONICAL_ADJUSTED_HISTORY` | `true` | Use split/bonus-adjusted stored bars for historical features |
| `SSE_CANONICAL_STRICT_FRESHNESS` | `false` | Block scans when canonical bars are stale instead of warning |
| `SSE_CANONICAL_MAX_STALENESS_DAYS` | `3` | Maximum allowed canonical bar staleness in strict mode |
| `SSE_INCLUDE_CROSS_SECTIONAL_FEATURES` | `true` | Add universe/sector-relative ranks, feature coverage, liquidity percentile, and research-readiness features before scoring |
| `SSE_CALIBRATION_REPORT_PATH` | `$SSE_STORAGE_ROOT/calibration/calibration_report_latest.json` | Latest conviction/backtest evidence artifact |
| `SSE_INDIANAPI_API_KEY` | empty | Optional IndianAPI key used by `indianapi-probe` |
| `SSE_INDIANAPI_STOCK_BASE_URL` | `https://stock.indianapi.in` | IndianAPI stock/company endpoint base URL |
| `SSE_INDIANAPI_ANALYST_BASE_URL` | `https://analyst.indianapi.in` | IndianAPI analyst endpoint base URL |
| `SSE_FMP_API_KEY` | empty | Optional Financial Modeling Prep key used by `fmp-probe`; sent as the `apikey` query parameter |
| `SSE_FMP_BASE_URL` | `https://financialmodelingprep.com/stable` | FMP stable API base URL |
| `SSE_FINEDGE_API_KEY` | empty | Optional FinEdge key used by `finedge-probe` / `finedge-inspect`; sent as the `token` query parameter |
| `SSE_FINEDGE_BASE_URL` | `https://data.finedgeapi.com` | FinEdge API base URL |
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
- Cross-sectional and sector-relative context: momentum rank, quality rank, value rank, feature coverage, liquidity percentile, and research-readiness score

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

Deployment defaults use free/public sources where possible:

- NSE public HTTP endpoints for OHLCV (`nse_http`)
- Google News RSS search feeds per symbol (`free_rss` provider)
- Exchange announcements for filing-like event ingestion

This keeps the runtime disconnected from mock sources while preserving deterministic fallback behavior for LLM extraction.

Yahoo Finance is available through `SSE_MARKET_PROVIDER=yfinance` after installing
the `market` extra. Zerodha Kite and ICICI Breeze can also be used as alternate
market data sources with `SSE_MARKET_PROVIDER=zerodha` or
`SSE_MARKET_PROVIDER=icici_breeze` after installing the `broker` extra and
setting the corresponding credentials.

---

## Pipelines

| Pipeline | Trigger | Purpose |
|---|---|---|
| `DailyBatchPipeline` | EOD | Full feature → score → signal cycle |
| `IntradayUpdatePipeline` | During market hours | Refresh swing-sensitive stack |
| `FeatureRefreshPipeline` | On-demand | Recompute features only |
| `SignalGenerationPipeline` | On-demand | Regenerate signals from cached features |
| `DataCollectionPipeline` | Scheduled / on-demand | Canonical exchange OHLCV/events/shareholding collection |
| `DataFoundationPipeline` | Scheduled / on-demand | Persist security master, calendar, OHLCV, corporate actions, quality and source reconciliation |
| `BacktestReadinessPipeline` | On-demand | Verify multi-year OHLCV, forward-return labels, security metadata, and factor-data coverage |
| `BacktestDatasetPipeline` | On-demand | Generate forward-return labels and evaluate technical/engine score panels with costs and sector-neutral diagnostics |
| `DocumentIntelligencePipeline` | On-demand | Local PDF/text document parsing, section detection, fact extraction |

### Intelligence Engines

| Engine | Module | Current status |
|---|---|---|
| Stock Signal Engine | `core/`, `pipelines/`, `reporting/` | Implemented with professional JSON/table/Markdown reports |
| Company Deep-Dive Research Engine | `research/company_deepdive/` | Report assembly implemented; peer/segment detail expands as data improves |
| PDF / Document Intelligence Engine | `documents/`, `pipelines/document_pipeline.py` | Text/PDF loader, sections, facts, commentary, quality warnings |
| Sector Intelligence Engine | `sector/` | Sector scoring, stance, drivers, risks, best expressions |
| Evaluation Engine | `backtest/`, `pipelines/backtest_dataset.py` | Forward-return labels, ranking IC, quantile spreads, turnover, costs, and sector-neutral diagnostics |

---

## Optional Broker Adapters

Both broker adapters are disabled by default and fail gracefully when credentials
are absent.

**Zerodha (Kite):**
```
SSE_MARKET_PROVIDER=zerodha
SSE_ENABLE_ZERODHA=true
SSE_ZERODHA_API_KEY=...
SSE_ZERODHA_API_SECRET=...
SSE_ZERODHA_ACCESS_TOKEN=...
```

**ICICI Breeze:**
```
SSE_MARKET_PROVIDER=icici_breeze
SSE_ENABLE_BREEZE=true
SSE_BREEZE_API_KEY=...
SSE_BREEZE_API_SECRET=...
SSE_BREEZE_SESSION_TOKEN=...
SSE_BREEZE_SYMBOL_MAP_PATH=./data/broker/breeze_symbol_map.csv
```

The Breeze adapter resolves NSE symbols to ICICI `stock_code` values with a
local ignored CSV cache. Manual overrides can be added with columns such as
`symbol,stock_code,source`; otherwise the adapter learns mappings from Breeze
`get_names` and persists them under `SSE_BREEZE_SYMBOL_MAP_PATH`.

---

## Running Tests

```bash
source ../.venv/bin/activate
python -m pytest -q
```

All tests are offline — no network calls, no broker credentials required.

## Setup And Run Commands

```bash
# activate shared Quant Engines env + install
source ../.venv/bin/activate
python -m pip install -e ".[all]"

# run main demo
python examples/run_demo.py

# run modular scoring demo
python examples/scoring_framework_demo.py

# run LLM-assisted event intelligence demo
python examples/llm_event_intelligence_demo.py

# run tests
python -m pytest -q
```

## CLI Commands

After `pip install -e .`, the `stock-engine` command is available:

```bash
stock-engine scan --mode full --format json
stock-engine scan --mode swing --format table
stock-engine scan --source canonical --mode full --format table
stock-engine analyze RELIANCE
stock-engine deepdive RELIANCE --format markdown
stock-engine document-ingest --symbol RELIANCE --file annual_report.pdf --document-type annual_report
stock-engine sector-rankings --format markdown
stock-engine sector-report --sector "CapitalGoods"
stock-engine sector-report --sector "IT" --include-peers --format markdown
stock-engine peer-report RELIANCE --as-of 2026-05-01 --format markdown
stock-engine security-master-ingest --file securities.csv
stock-engine data-foundation --start 2026-01-01 --end 2026-01-31 --symbols RELIANCE,TCS
stock-engine data-foundation --source yfinance --lookback-years 5 --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv"
stock-engine data-quality --lookback-years 5 --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv"
stock-engine data-entitlements --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --format markdown
stock-engine data-source-coverage --lookback-years 5 --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --format markdown
stock-engine data-source-priority --format markdown
stock-engine data-readiness --mode long-term-scan --lookback-years 5 --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --format markdown
stock-engine scan --source canonical --mode full --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --readiness-check warn --format table
stock-engine scan --source canonical --mode full --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --readiness-check enforce --format json
stock-engine refresh-market --source zerodha --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --lookback-days 10 --batch-size 25 --retries 2 --run-scan
stock-engine broker-health --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --sources zerodha,icici_breeze --lookback-days 10 --retries 2 --primary-source zerodha --lagged-sources icici_breeze --format table
stock-engine indianapi-probe --symbols RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK --check stock,financials,shareholding,analyst,forecasts,history --format table
stock-engine fmp-probe --symbols RELIANCE --check smoke --format table
stock-engine fmp-probe --symbols AAPL --check smoke,income_statement --limit 1 --exact-symbols --format table
stock-engine fmp-probe --symbols RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK --check all --timeout-seconds 5 --retries 0 --format table
stock-engine finedge-probe --symbols ITC,RELIANCE,HDFCBANK --check smoke --format table
stock-engine finedge-probe --symbols ITC,RELIANCE,HDFCBANK --check fundamentals --statement-type s --statement-code pl --period annual --format table
stock-engine finedge-inspect --symbols ITC,RELIANCE,HDFCBANK --check fundamentals --statement-type s --statement-code pl --period annual --format table
stock-engine finedge-inspect --symbols ITC,RELIANCE,HDFCBANK --check ownership --shareholding-period quarterly --format table
stock-engine finedge-factor-export --symbols ITC,RELIANCE,HDFCBANK --as-of 2026-05-28 --output-root "$SSE_STORAGE_ROOT/factors/finedge_trial" --sections financials,valuations,shareholding --format table
stock-engine finedge-inspect --symbols HDFCBANK --check ratios,basic_financials --statement-type s --statement-code pl --ratio-type pr --format table
stock-engine finedge-factor-export --symbols HDFCBANK --as-of 2026-05-28 --output-root "$SSE_STORAGE_ROOT/factors/finedge_banking_trial_v3" --sections banking --format table
stock-engine finedge-onboarding-plan --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --end 2026-05-28 --format markdown
stock-engine factor-qa --symbols ITC,RELIANCE,HDFCBANK --as-of 2026-05-28 --format table
stock-engine backtest-readiness --lookback-years 5 --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv"
stock-engine backtest-labels --lookback-years 5 --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --horizons 5,20,60
stock-engine technical-backtest --lookback-years 5 --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --universe-policy eligible_history --horizons 5,20,60
stock-engine engine-backtest --lookback-years 5 --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --universe-policy eligible_history --score-type swing --horizons 5,20,60
stock-engine conviction-calibrate --lookback-years 5 --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --universe-policy eligible_history --horizons 5,20,60
stock-engine financials-ingest --symbol RELIANCE --file financials.csv --as-of 2026-05-01
stock-engine valuation-ingest --symbol RELIANCE --file valuations.csv --as-of 2026-05-01
stock-engine shareholding-ingest --symbol RELIANCE --file shareholding.csv --as-of 2026-05-01
stock-engine factor-template --output-root "$SSE_STORAGE_ROOT/factors/nifty50" --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --as-of 2026-05-18
stock-engine factor-ingest --root "$SSE_STORAGE_ROOT/factors/nifty50" --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --as-of 2026-05-18 --min-coverage 0.8
stock-engine explain RELIANCE
stock-engine export-report RELIANCE --format markdown
```

Canonical workflow:

```bash
stock-engine security-master-ingest --file securities.csv
stock-engine data-foundation --source yfinance --lookback-years 5 --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv"
stock-engine data-entitlements --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --format markdown
stock-engine data-source-coverage --lookback-years 5 --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --format markdown
stock-engine refresh-market --source zerodha --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --lookback-days 10 --run-scan
stock-engine backtest-readiness --lookback-years 5 --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv"
stock-engine backtest-labels --lookback-years 5 --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv"
stock-engine technical-backtest --lookback-years 5 --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv"
stock-engine engine-backtest --lookback-years 5 --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --score-type swing
stock-engine conviction-calibrate --lookback-years 5 --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv"
stock-engine financials-ingest --symbol RELIANCE --file financials.csv --as-of 2026-05-01
stock-engine valuation-ingest --symbol RELIANCE --file valuations.csv --as-of 2026-05-01
stock-engine shareholding-ingest --symbol RELIANCE --file shareholding.csv --as-of 2026-05-01
stock-engine factor-template --output-root "$SSE_STORAGE_ROOT/factors/nifty50" --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --as-of 2026-05-18
stock-engine factor-ingest --root "$SSE_STORAGE_ROOT/factors/nifty50" --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --as-of 2026-05-18 --min-coverage 0.8
stock-engine factor-qa --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" --as-of 2026-05-18 --format markdown
stock-engine peer-report RELIANCE --as-of 2026-05-01 --format markdown
stock-engine scan --source canonical --mode full --format table
stock-engine analyze RELIANCE --source canonical
```

Security master CSV columns:
`symbol, exchange, isin, series, company_name, sector, industry, listing_date, delisting_date, active, lot_size, tick_size, source`.

`--universe-file` accepts either the same security-master CSV shape or a simple
one-symbol-per-line file. If sectors and industries are present, they are
persisted into the canonical security master during `data-foundation`.

Backtest universe policies:

- `current`: use every symbol in the supplied universe, even if the listing has partial history.
- `eligible_history`: exclude symbols that fail the requested minimum history and forward-label gates.

Backtest artifacts are written under the external storage root, for example:

- `$SSE_STORAGE_ROOT/backtest/forward_return_labels.csv`
- `$SSE_STORAGE_ROOT/backtest/technical_ranking_scores.csv`
- `$SSE_STORAGE_ROOT/backtest/technical_ranking_evaluation.json`
- `$SSE_STORAGE_ROOT/backtest/engine_swing_scores.csv`
- `$SSE_STORAGE_ROOT/backtest/engine_swing_evaluation.json`
- `$SSE_STORAGE_ROOT/calibration/calibration_report_latest.json`

`technical-backtest` evaluates a transparent first-pass price/volume score.
`engine-backtest` evaluates the actual engine scoring stack historically
(`swing`, `long_term`, or `conviction`). Both reports include gross and net
quantile metrics, turnover, Spearman-style rank IC, sector-neutral IC, and a
configurable Indian cash-equity cost model. Override costs with:

```bash
stock-engine engine-backtest --round-trip-cost-bps 35 --slippage-bps 5
```

`conviction-calibrate` runs the engine backtest and writes the latest compact
evidence artifact used by future scans. By default that artifact follows
`$SSE_STORAGE_ROOT/calibration/calibration_report_latest.json`; override it with
`SSE_CALIBRATION_REPORT_PATH` or `--output-path`.

Engine backtests are point-in-time factor-aware: when canonical financial
statements, valuations, and shareholding rows exist with filing/as-of dates
available before each historical scoring date, those factors flow into the
historical feature stack. Reports include `factor_coverage` so it is clear
whether a calibration was fundamental-aware or price-only.

Official index constituent CSVs, such as the NSE Indices Nifty 50 constituent
file, should be downloaded into `$SSE_STORAGE_ROOT/universe/` and kept outside
the git repository.

Current Nifty 50 bootstrap baseline, generated from the external storage root:

- Data foundation: 50/50 symbols, 60,166 daily OHLCV rows from 2021-05-18 to 2026-05-18.
- Backtest readiness: 48/50 symbols have at least 1,000 daily bars; JIOFIN and TMPV are flagged for short listed history.
- Forward-return labels: 176,248 labels across 5/20/60-bar horizons.
- First technical baseline: negative IC across 5/20/60 bars, so the naive technical score is not alpha-positive.
- Engine swing baseline: negative gross and sector-neutral IC across 5/20/60 bars after applying the current engine score to the history-eligible universe. This is an honest diagnostic: the evaluation harness is working, and the next research task is improving factor definitions and adding point-in-time fundamental/valuation/ownership data.

Financial statement CSV columns:
`period_end, filing_date, statement_type, revenue, ebit, net_income, operating_cash_flow, capex, total_debt, equity, total_assets, current_assets, current_liabilities, interest_expense, source_id`.

Valuation CSV columns:
`as_of, market_cap, shares_outstanding, free_float_market_cap, enterprise_value, currency, source_id`.

Shareholding CSV columns:
`period_end, filing_date, promoter_pct, fii_pct, dii_pct, public_pct, source_id`.

For universe-scale point-in-time factor loading, prefer the bulk external
workflow:

```bash
stock-engine factor-template \
  --output-root "$SSE_STORAGE_ROOT/factors/nifty50" \
  --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" \
  --as-of 2026-05-18

# Fill the generated CSVs from licensed/vendor/filing-derived sources, then:
stock-engine factor-ingest \
  --root "$SSE_STORAGE_ROOT/factors/nifty50" \
  --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" \
  --as-of 2026-05-18 \
  --sections financials,valuations,shareholding \
  --min-coverage 0.8

# Optional bank/NBFC-specific factor load for Financial Services names.
stock-engine factor-ingest \
  --root "$SSE_STORAGE_ROOT/factors/finedge_banking_trial" \
  --symbols HDFCBANK \
  --as-of 2026-05-28 \
  --sections banking \
  --min-coverage 1.0

stock-engine factor-qa \
  --universe-file "$SSE_STORAGE_ROOT/universe/nifty50.csv" \
  --as-of 2026-05-18 \
  --format table
```

The generated files are:

- `financials.csv`: symbol-level statement rows with period-end and filing-date.
- `valuations.csv`: market-cap/share-count/enterprise-value facts by as-of date.
- `shareholding.csv`: promoter/FII/DII/public ownership by period-end and filing-date.
- `banking.csv`: optional bank/NBFC factors such as NII, NIM, advances/deposit growth, CASA, GNPA/NNPA, provision coverage, credit cost, CAR/CET1, cost-income, ROA/ROE, and loan/deposit ratio.

The FinEdge banking mapper joins statement rows with `ratios` and
`basic_financials` to populate bank-specific fields. In the current free/basic
trial, HDFCBANK produced usable NIM, GNPA/NNPA, CET1, cost-income, ROA/ROE, NII,
growth, and loan/deposit metrics; ICICIBANK/SBIN/AXISBANK/KOTAKBANK returned
401 responses on financial endpoints, so broad bank coverage likely needs paid
or expanded vendor access.

`factor-ingest` writes a coverage report under the external storage root and
blocks the report-level `passed` flag unless the selected sections meet
`--min-coverage`. Use `--sections financials,shareholding` for a partial factor
load that intentionally excludes valuation rows. Use `--sections banking` when
loading bank/NBFC-specific factors separately for Financial Services names.

`factor-qa` is the review gate before broad scans or backtests. It reads the
canonical store and shows, per symbol, the latest financial statement, valuation
date, shareholding period, derived PE/PB/ROE/cash-flow ratios, and mapping
warnings such as stale valuation rows, missing share count, unit-mismatch
signals, or bank/NBFC-specific factor gaps. When `banking.csv` has been ingested,
it also reports bank-specific QA metrics, metric coverage, and a composite
banking quality score.

`data-entitlements` records source plan metadata: enabled domains, allowed
symbols, credential env names, rate limits when known, storage rights,
redistribution rights, commercial-use status, known limits, and next actions.
`data-source-coverage` is the aggregate source-readiness report. It reads the
canonical DB plus the latest ignored quality/vendor artifacts and writes
`data_source_coverage_report.json` and `data_source_coverage_report.md` under
`$SSE_STORAGE_ROOT/quality`. It separates market coverage, factor coverage,
broker-source health, FinEdge entitlement coverage, and still-open gaps such as
corporate actions, events/documents, and historical constituents. The key
difference is that it now reports both actual coverage and entitled coverage, so
FinEdge Basic's 3-symbol universe is treated as a plan boundary rather than a
mapper failure.

`data-source-priority` is the canonical source hierarchy by data domain. It
keeps the engine's source choices explicit: exchange and broker data lead
market/identity coverage, FinEdge is the paid-data candidate for fundamentals,
valuation, ownership, banking factors, and documents, and yfinance/FMP/IndianAPI
remain fallback or paused sources until proven. The report is written under
`$SSE_STORAGE_ROOT/quality`, outside git.

`data-readiness` applies hard coverage gates on top of `data-source-coverage`.
Modes include `swing-scan`, `long-term-scan`, `deep-research`, and `backtest`.
For example, a swing scan can pass with strong security master and OHLCV
coverage, while long-term research blocks until financials, valuations, and
shareholding reach configured thresholds.

`scan` is readiness-aware. By default it uses `--readiness-check warn`, attaches
the relevant data-readiness gate to JSON output, and prepends readiness warnings
to table/markdown output when a required domain is below threshold. Use
`--readiness-check enforce` to block production scans until the gate passes, or
`--readiness-check off` for low-level diagnostics.

`finedge-onboarding-plan` creates an ignored local checklist and command
sequence for the paid FinEdge rollout. It does not call FinEdge; it summarizes
current gaps, questions to confirm before paying, post-subscription probe/export
commands, ingest/QA steps, and success criteria.

When bank/NBFC factors are available, scan/analyze/report flows carry them into
feature vectors and signal reports. Financial Services names with missing or
sparse `banking.csv` coverage receive lower symbol-level source confidence, so
generic industrial ratios do not create false conviction for banks.

The legacy `python main.py screen` and `python main.py analyze RELIANCE` flows remain available.

## Output Schema

Professional stock signal reports include:

- Identity: symbol, company, sector, industry, market-cap category, liquidity class
- Signal summary: long-term/swing/final score, risk penalty, confidence, rank, horizon
- Technical metrics: trend, momentum, relative strength, volatility, volume participation, setup status
- Fundamental metrics: growth, profitability, leverage, cash-flow quality where available
- Valuation metrics: PE/PB, sector/history z-score proxies, earnings yield, valuation risk
- Peer context: sector PE/PB z-score and valuation position versus covered peers
- Cross-sectional context: universe/sector momentum ranks, quality/value ranks, liquidity percentile, feature coverage, and research readiness
- Event/NLP metrics: event scores, sentiment, management tone, uncertainty, governance risk
- Risk metrics: liquidity, volatility, leverage, valuation, earnings, event/governance, missing-data risk
- Conviction metrics: score strength, cross-horizon agreement, data/source confidence, backtest support, regime confirmation, risk resilience
- Explanation: positive drivers, negative drivers, why selected/rejected, monitorables, invalidation logic

Missing data is explicitly marked as unavailable or included in `missing_data_warnings`; the engine should not invent financials or document facts.

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

## Current Verification And Roadmap

Verified locally on 2026-05-25:

- `../.venv/bin/python -m pytest -q`: 210 passing tests
- `../.venv/bin/python -m ruff check .`: passes
- `../.venv/bin/python -m mypy stock_screener_engine`: passes
- `../.venv/bin/stock-engine --help` works from outside this repo

Recent maintenance:

- Consolidated local docs under ignored `documentation/`.
- Fixed `.gitignore` so local research artifacts stay ignored but the source package `stock_screener_engine/research/` can be tracked.
- Corrected short-score handling for symmetric text features so neutral sentiment is not treated as bearish.
- Aligned this repo with the shared parent Quant Engines Python 3.11 environment.
- Cleared the full package type-checking gate under the shared environment.

Highest-priority roadmap:

1. Add lineage metadata to features, scores, reports, calibration rows, and backtest rows.
2. Populate external point-in-time Nifty 50 fundamentals, valuation, shareholding, and corporate-action data.
3. Upgrade walk-forward evaluation from planner scaffolding to a full rolling train/test harness.
4. Add mandatory bias audit reports for survivorship, look-ahead, stale data, universe drift, and factor coverage.
5. Build an event-driven backtester with fills, costs, slippage, corporate actions, stops, and live invalidation events.
6. Add sector-specific factor packs and evaluate them separately before global weighting.
7. Add model governance for any trained ranker: version, feature set, training window, validation metrics, calibration date, and fallback behavior.
8. Keep broker execution paper-only until risk controls, audit trails, reconciliation, and current regulatory/broker checks are complete.
