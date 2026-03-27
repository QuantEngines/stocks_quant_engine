from __future__ import annotations

from pathlib import Path

import pytest

from stock_screener_engine.config.settings import load_settings
from stock_screener_engine.config.startup_validation import validate_startup_settings


def test_load_default_settings() -> None:
    settings = load_settings()
    assert settings.environment
    assert settings.storage.root_dir
    assert settings.scoring.long_term_min_score > 0
    assert settings.scoring.regime_switching.bull_threshold > settings.scoring.regime_switching.bear_threshold
    assert settings.scoring.ranking.portfolio.max_positions_long > 0
    assert settings.scoring.ranking.portfolio.min_position_notional >= 0
    assert settings.scoring.ranking.portfolio.long_min_position_notional is None
    assert settings.scoring.ranking.portfolio.swing_sector_target_weights is None


def test_portfolio_per_strategy_overrides_parse(tmp_path: Path) -> None:
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(
        """
environment: dev
log_level: INFO
storage:
  root_dir: ./data
  sqlite_path: ./data/metadata.db
features:
  include_sentiment: true
  include_event_signals: true
  include_regime_features: true
  min_liquidity_threshold: 1000000
integrations:
  zerodha:
    enabled: false
  breeze:
    enabled: false
scoring:
  long_term_min_score: 24.0
  swing_min_score: 28.0
  max_risk_penalty: 30.0
  ranking:
    portfolio:
      min_position_notional: 25000
      sector_target_weights:
        IT: 0.4
      long_min_position_notional: 40000
      swing_min_position_notional: 15000
      long_sector_target_weights:
        IT: 0.6
        Banking: 0.4
      swing_sector_target_weights:
        IT: 0.2
        Pharma: 0.8
""".strip(),
        encoding="utf-8",
    )

    settings = load_settings(str(cfg))
    p = settings.scoring.ranking.portfolio
    assert p.min_position_notional == 25000.0
    assert p.long_min_position_notional == 40000.0
    assert p.swing_min_position_notional == 15000.0
    assert p.long_sector_target_weights == {"IT": 0.6, "Banking": 0.4}
    assert p.swing_sector_target_weights == {"IT": 0.2, "Pharma": 0.8}


def test_startup_validation_fails_when_llm_key_missing(monkeypatch) -> None:
    settings = load_settings()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    strict = settings.__class__(
        environment=settings.environment,
        log_level=settings.log_level,
        storage=settings.storage,
        features=settings.features,
        nlp=settings.nlp,
        llm=settings.llm.__class__(
            enabled=True,
            provider="openai",
            model="gpt-4.1-mini",
            base_url="https://api.openai.com",
            api_key_env="OPENAI_API_KEY",
            timeout_seconds=30,
            min_confidence=settings.llm.min_confidence,
            fallback_to_rules=settings.llm.fallback_to_rules,
            enable_management_tone=settings.llm.enable_management_tone,
            audit_low_confidence=settings.llm.audit_low_confidence,
            audit_path=settings.llm.audit_path,
        ),
        runtime_data=settings.runtime_data,
        integrations=settings.integrations,
        scoring=settings.scoring,
    )

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        validate_startup_settings(strict)


def test_startup_validation_passes_when_llm_key_present(monkeypatch) -> None:
    settings = load_settings()
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    strict = settings.__class__(
        environment=settings.environment,
        log_level=settings.log_level,
        storage=settings.storage,
        features=settings.features,
        nlp=settings.nlp,
        llm=settings.llm.__class__(
            enabled=True,
            provider="openai",
            model="gpt-4.1-mini",
            base_url="https://api.openai.com",
            api_key_env="OPENAI_API_KEY",
            timeout_seconds=30,
            min_confidence=settings.llm.min_confidence,
            fallback_to_rules=settings.llm.fallback_to_rules,
            enable_management_tone=settings.llm.enable_management_tone,
            audit_low_confidence=settings.llm.audit_low_confidence,
            audit_path=settings.llm.audit_path,
        ),
        runtime_data=settings.runtime_data,
        integrations=settings.integrations,
        scoring=settings.scoring,
    )

    validate_startup_settings(strict)
