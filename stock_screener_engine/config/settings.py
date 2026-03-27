"""Centralized runtime settings with YAML + env overlay support."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class StorageSettings:
    root_dir: str
    sqlite_path: str


@dataclass(frozen=True)
class FeatureSettings:
    include_sentiment: bool
    include_event_signals: bool
    include_regime_features: bool
    min_liquidity_threshold: float


@dataclass(frozen=True)
class BrokerIntegrationSettings:
    enabled: bool
    api_key_env: str
    api_secret_env: str
    token_env: str

    def credentials(self) -> dict[str, str | None]:
        return {
            "api_key": os.getenv(self.api_key_env),
            "api_secret": os.getenv(self.api_secret_env),
            "token": os.getenv(self.token_env),
        }


@dataclass(frozen=True)
class IntegrationSettings:
    zerodha: BrokerIntegrationSettings
    breeze: BrokerIntegrationSettings


@dataclass(frozen=True)
class LongTermWeightsSettings:
    growth_quality:        float = 18.0
    profitability_quality: float = 17.0
    balance_sheet_health:  float = 15.0
    cash_flow_quality:     float = 12.0
    valuation_sanity:      float = 12.0
    governance_proxy:      float = 10.0
    event_catalyst:        float = 8.0
    regime_tailwind:       float = 8.0


@dataclass(frozen=True)
class SwingWeightsSettings:
    trend_strength:          float = 20.0
    momentum_strength:       float = 18.0
    relative_strength_proxy: float = 14.0
    volatility_regime:       float = 12.0
    volume_confirmation:     float = 12.0
    event_catalyst:          float = 12.0
    sentiment_score:         float = 12.0


@dataclass(frozen=True)
class RiskWeightsSettings:
    liquidity_risk: float = 0.20
    volatility_risk: float = 0.20
    leverage_risk: float = 0.20
    earnings_instability_risk: float = 0.15
    event_uncertainty_risk: float = 0.15
    governance_risk: float = 0.10
    text_uncertainty_risk: float = 0.05


@dataclass(frozen=True)
class NlpSettings:
    enabled: bool = False
    enable_sentiment: bool = True
    enable_event_extraction: bool = True
    lookback_days: int = 30
    decay_half_life_days: float = 14.0
    high_impact_threshold: float = 0.65


@dataclass(frozen=True)
class LlmSettings:
    enabled: bool = False
    provider: str = "heuristic"
    model: str = "heuristic-finance-v1"
    base_url: str = "https://api.openai.com"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: int = 30
    min_confidence: float = 0.55
    fallback_to_rules: bool = True
    enable_management_tone: bool = True
    audit_low_confidence: bool = True
    audit_path: str = "./data"


@dataclass(frozen=True)
class RuntimeDataSettings:
    market_provider: str = "nse_http"
    market_universe: list[str] = field(default_factory=list)
    news_provider: str = "free_rss"
    filings_provider: str = "exchange_announcements"
    transcripts_provider: str = "none"


@dataclass(frozen=True)
class RegimeSwitchingSettings:
    enabled: bool = True
    bull_threshold: float = 0.25
    bear_threshold: float = -0.25


@dataclass(frozen=True)
class CalibrationAutoTuneSettings:
    enabled: bool = False
    report_path: str = "./data/calibration/calibration_report_latest.json"
    learning_rate: float = 0.30


@dataclass(frozen=True)
class PortfolioConstructionSettings:
    enabled: bool = True
    max_positions_long: int = 12
    max_positions_swing: int = 10
    max_sector_positions: int = 3
    min_avg_daily_volume: float = 1_000_000.0
    max_single_position_weight: float = 0.12
    capital_base: float = 1_000_000.0
    min_position_notional: float = 25_000.0
    sector_target_weights: dict[str, float] = field(default_factory=dict)
    sector_target_tolerance: float = 0.05
    long_min_position_notional: float | None = None
    swing_min_position_notional: float | None = None
    long_sector_target_weights: dict[str, float] | None = None
    swing_sector_target_weights: dict[str, float] | None = None


@dataclass(frozen=True)
class RankingSettings:
    top_k_long_term: int = 25
    top_k_swing: int = 25
    portfolio: PortfolioConstructionSettings = PortfolioConstructionSettings()


@dataclass(frozen=True)
class ScoringSettings:
    long_term_min_score: float
    swing_min_score: float
    max_risk_penalty: float
    short_min_score: float = 58.0
    long_term_weights: LongTermWeightsSettings = LongTermWeightsSettings()
    swing_weights: SwingWeightsSettings = SwingWeightsSettings()
    risk_weights: RiskWeightsSettings = RiskWeightsSettings()
    regime_switching: RegimeSwitchingSettings = RegimeSwitchingSettings()
    long_term_regime_profiles: dict[str, dict[str, float]] = field(default_factory=dict)
    swing_regime_profiles: dict[str, dict[str, float]] = field(default_factory=dict)
    calibration_auto_tune: CalibrationAutoTuneSettings = CalibrationAutoTuneSettings()
    ranking: RankingSettings = RankingSettings()


@dataclass(frozen=True)
class AppSettings:
    environment: str
    log_level: str
    storage: StorageSettings
    features: FeatureSettings
    nlp: NlpSettings
    llm: LlmSettings
    runtime_data: RuntimeDataSettings
    integrations: IntegrationSettings
    scoring: ScoringSettings


def _to_bool(text: str | None, default: bool) -> bool:
    if text is None:
        return default
    return text.lower().strip() in {"1", "true", "yes", "on"}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        content = yaml.safe_load(fh) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Expected dict config at {path}")
    return content


def load_settings(config_path: str | None = None) -> AppSettings:
    default_path = Path(__file__).with_name("defaults.yaml")
    path = Path(config_path) if config_path else default_path
    raw = _read_yaml(path)

    storage_raw = raw.get("storage", {})
    features_raw = raw.get("features", {})
    integrations_raw = raw.get("integrations", {})
    nlp_raw = raw.get("nlp", {})
    llm_raw = raw.get("llm", {})
    runtime_data_raw = raw.get("runtime_data", {})
    scoring_raw = raw.get("scoring", {})

    zerodha_raw = integrations_raw.get("zerodha", {})
    breeze_raw = integrations_raw.get("breeze", {})

    return AppSettings(
        environment=os.getenv("SSE_ENV", raw.get("environment", "dev")),
        log_level=os.getenv("SSE_LOG_LEVEL", raw.get("log_level", "INFO")),
        storage=StorageSettings(
            root_dir=os.getenv("SSE_STORAGE_ROOT", storage_raw.get("root_dir", "./data")),
            sqlite_path=os.getenv("SSE_SQLITE_PATH", storage_raw.get("sqlite_path", "./data/metadata.db")),
        ),
        features=FeatureSettings(
            include_sentiment=_to_bool(os.getenv("SSE_INCLUDE_SENTIMENT"), features_raw.get("include_sentiment", True)),
            include_event_signals=_to_bool(
                os.getenv("SSE_INCLUDE_EVENT_SIGNALS"), features_raw.get("include_event_signals", True)
            ),
            include_regime_features=_to_bool(
                os.getenv("SSE_INCLUDE_REGIME_FEATURES"), features_raw.get("include_regime_features", True)
            ),
            min_liquidity_threshold=float(
                os.getenv("SSE_MIN_LIQUIDITY", str(features_raw.get("min_liquidity_threshold", 1_000_000)))
            ),
        ),
        nlp=NlpSettings(
            enabled=_to_bool(os.getenv("SSE_ENABLE_NLP"), nlp_raw.get("enabled", False)),
            enable_sentiment=_to_bool(os.getenv("SSE_ENABLE_NLP_SENTIMENT"), nlp_raw.get("enable_sentiment", True)),
            enable_event_extraction=_to_bool(os.getenv("SSE_ENABLE_NLP_EVENT_EXTRACTION"), nlp_raw.get("enable_event_extraction", True)),
            lookback_days=int(os.getenv("SSE_NLP_LOOKBACK_DAYS", str(nlp_raw.get("lookback_days", 30)))),
            decay_half_life_days=float(os.getenv("SSE_NLP_DECAY_HALF_LIFE", str(nlp_raw.get("decay_half_life_days", 14.0)))),
            high_impact_threshold=float(os.getenv("SSE_NLP_HIGH_IMPACT_THRESHOLD", str(nlp_raw.get("high_impact_threshold", 0.65)))),
        ),
        llm=LlmSettings(
            enabled=_to_bool(os.getenv("SSE_ENABLE_LLM_EXTRACTION"), llm_raw.get("enabled", False)),
            provider=str(os.getenv("SSE_LLM_PROVIDER", llm_raw.get("provider", "heuristic"))),
            model=str(os.getenv("SSE_LLM_MODEL", llm_raw.get("model", "heuristic-finance-v1"))),
            base_url=str(os.getenv("SSE_LLM_BASE_URL", llm_raw.get("base_url", "https://api.openai.com"))),
            api_key_env=str(os.getenv("SSE_LLM_API_KEY_ENV", llm_raw.get("api_key_env", "OPENAI_API_KEY"))),
            timeout_seconds=int(os.getenv("SSE_LLM_TIMEOUT_SECONDS", str(llm_raw.get("timeout_seconds", 30)))),
            min_confidence=float(os.getenv("SSE_LLM_MIN_CONFIDENCE", str(llm_raw.get("min_confidence", 0.55)))),
            fallback_to_rules=_to_bool(os.getenv("SSE_LLM_FALLBACK_TO_RULES"), llm_raw.get("fallback_to_rules", True)),
            enable_management_tone=_to_bool(os.getenv("SSE_LLM_ENABLE_MANAGEMENT_TONE"), llm_raw.get("enable_management_tone", True)),
            audit_low_confidence=_to_bool(os.getenv("SSE_LLM_AUDIT_LOW_CONFIDENCE"), llm_raw.get("audit_low_confidence", True)),
            audit_path=str(os.getenv("SSE_LLM_AUDIT_PATH", llm_raw.get("audit_path", "./data"))),
        ),
        runtime_data=RuntimeDataSettings(
            market_provider=str(os.getenv("SSE_MARKET_PROVIDER", runtime_data_raw.get("market_provider", "nse_http"))),
            market_universe=_parse_csv_symbols(
                os.getenv("SSE_MARKET_UNIVERSE"),
                runtime_data_raw.get("market_universe", []),
            ),
            news_provider=str(os.getenv("SSE_NEWS_PROVIDER", runtime_data_raw.get("news_provider", "free_rss"))),
            filings_provider=str(
                os.getenv("SSE_FILINGS_PROVIDER", runtime_data_raw.get("filings_provider", "exchange_announcements"))
            ),
            transcripts_provider=str(
                os.getenv("SSE_TRANSCRIPTS_PROVIDER", runtime_data_raw.get("transcripts_provider", "none"))
            ),
        ),
        integrations=IntegrationSettings(
            zerodha=BrokerIntegrationSettings(
                enabled=_to_bool(os.getenv("SSE_ENABLE_ZERODHA"), zerodha_raw.get("enabled", False)),
                api_key_env=zerodha_raw.get("api_key_env", "SSE_ZERODHA_API_KEY"),
                api_secret_env=zerodha_raw.get("api_secret_env", "SSE_ZERODHA_API_SECRET"),
                token_env=zerodha_raw.get("access_token_env", "SSE_ZERODHA_ACCESS_TOKEN"),
            ),
            breeze=BrokerIntegrationSettings(
                enabled=_to_bool(os.getenv("SSE_ENABLE_BREEZE"), breeze_raw.get("enabled", False)),
                api_key_env=breeze_raw.get("api_key_env", "SSE_BREEZE_API_KEY"),
                api_secret_env=breeze_raw.get("api_secret_env", "SSE_BREEZE_API_SECRET"),
                token_env=breeze_raw.get("session_token_env", "SSE_BREEZE_SESSION_TOKEN"),
            ),
        ),
        scoring=ScoringSettings(
            long_term_min_score=float(scoring_raw.get("long_term_min_score", 55.0)),
            swing_min_score=float(scoring_raw.get("swing_min_score", 58.0)),
            max_risk_penalty=float(scoring_raw.get("max_risk_penalty", 30.0)),
            short_min_score=float(scoring_raw.get("short_min_score", 58.0)),
            long_term_weights=_parse_lt_weights(scoring_raw.get("long_term_weights", {})),
            swing_weights=_parse_swing_weights(scoring_raw.get("swing_weights", {})),
            risk_weights=_parse_risk_weights(scoring_raw.get("risk_weights", {})),
            regime_switching=_parse_regime_switching(scoring_raw.get("regime_switching", {})),
            long_term_regime_profiles=_parse_weight_profiles(scoring_raw.get("long_term_regime_profiles", {})),
            swing_regime_profiles=_parse_weight_profiles(scoring_raw.get("swing_regime_profiles", {})),
            calibration_auto_tune=_parse_calibration_auto_tune(scoring_raw.get("calibration_auto_tune", {})),
            ranking=_parse_ranking(scoring_raw.get("ranking", {})),
        ),
    )


def _parse_lt_weights(d: dict) -> LongTermWeightsSettings:
    defaults = LongTermWeightsSettings()
    return LongTermWeightsSettings(
        **{k: float(d.get(k, getattr(defaults, k))) for k in defaults.__dataclass_fields__}
    )


def _parse_swing_weights(d: dict) -> SwingWeightsSettings:
    defaults = SwingWeightsSettings()
    return SwingWeightsSettings(
        **{k: float(d.get(k, getattr(defaults, k))) for k in defaults.__dataclass_fields__}
    )


def _parse_risk_weights(d: dict) -> RiskWeightsSettings:
    defaults = RiskWeightsSettings()
    return RiskWeightsSettings(
        **{k: float(d.get(k, getattr(defaults, k))) for k in defaults.__dataclass_fields__}
    )


def _parse_ranking(d: dict) -> RankingSettings:
    defaults = RankingSettings()
    portfolio_defaults = defaults.portfolio
    return RankingSettings(
        top_k_long_term=int(d.get("top_k_long_term", defaults.top_k_long_term)),
        top_k_swing=int(d.get("top_k_swing", defaults.top_k_swing)),
        portfolio=PortfolioConstructionSettings(
            enabled=bool(d.get("portfolio", {}).get("enabled", portfolio_defaults.enabled)),
            max_positions_long=int(d.get("portfolio", {}).get("max_positions_long", portfolio_defaults.max_positions_long)),
            max_positions_swing=int(d.get("portfolio", {}).get("max_positions_swing", portfolio_defaults.max_positions_swing)),
            max_sector_positions=int(d.get("portfolio", {}).get("max_sector_positions", portfolio_defaults.max_sector_positions)),
            min_avg_daily_volume=float(d.get("portfolio", {}).get("min_avg_daily_volume", portfolio_defaults.min_avg_daily_volume)),
            max_single_position_weight=float(d.get("portfolio", {}).get("max_single_position_weight", portfolio_defaults.max_single_position_weight)),
            capital_base=float(d.get("portfolio", {}).get("capital_base", portfolio_defaults.capital_base)),
            min_position_notional=float(d.get("portfolio", {}).get("min_position_notional", portfolio_defaults.min_position_notional)),
            sector_target_weights={
                str(k): float(v)
                for k, v in d.get("portfolio", {}).get("sector_target_weights", {}).items()
            },
            sector_target_tolerance=float(d.get("portfolio", {}).get("sector_target_tolerance", portfolio_defaults.sector_target_tolerance)),
            long_min_position_notional=_parse_optional_float(d.get("portfolio", {}).get("long_min_position_notional")),
            swing_min_position_notional=_parse_optional_float(d.get("portfolio", {}).get("swing_min_position_notional")),
            long_sector_target_weights=_parse_optional_weight_map(d.get("portfolio", {}).get("long_sector_target_weights")),
            swing_sector_target_weights=_parse_optional_weight_map(d.get("portfolio", {}).get("swing_sector_target_weights")),
        ),
    )


def _parse_regime_switching(d: dict) -> RegimeSwitchingSettings:
    defaults = RegimeSwitchingSettings()
    return RegimeSwitchingSettings(
        enabled=bool(d.get("enabled", defaults.enabled)),
        bull_threshold=float(d.get("bull_threshold", defaults.bull_threshold)),
        bear_threshold=float(d.get("bear_threshold", defaults.bear_threshold)),
    )


def _parse_weight_profiles(d: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for regime, payload in d.items():
        if not isinstance(payload, dict):
            continue
        out[str(regime)] = {str(k): float(v) for k, v in payload.items()}
    return out


def _parse_calibration_auto_tune(d: dict) -> CalibrationAutoTuneSettings:
    defaults = CalibrationAutoTuneSettings()
    return CalibrationAutoTuneSettings(
        enabled=bool(d.get("enabled", defaults.enabled)),
        report_path=str(d.get("report_path", defaults.report_path)),
        learning_rate=float(d.get("learning_rate", defaults.learning_rate)),
    )


def _parse_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return float(value)
    return None


def _parse_optional_weight_map(value: object) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        return None
    return {str(k): float(v) for k, v in value.items()}


def _parse_csv_symbols(text: str | None, fallback: object) -> list[str]:
    if text:
        return [x.strip().upper() for x in text.split(",") if x.strip()]
    if isinstance(fallback, list):
        return [str(x).strip().upper() for x in fallback if str(x).strip()]
    return []
