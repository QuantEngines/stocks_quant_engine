"""Create deterministic config snapshots for CI reproducibility checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from stock_screener_engine.config.settings import load_settings


def main() -> None:
    settings = load_settings()
    payload = {
        "environment": settings.environment,
        "log_level": settings.log_level,
        "storage": {
            "root_dir": settings.storage.root_dir,
            "sqlite_path": settings.storage.sqlite_path,
        },
        "features": {
            "include_sentiment": settings.features.include_sentiment,
            "include_event_signals": settings.features.include_event_signals,
            "include_regime_features": settings.features.include_regime_features,
            "min_liquidity_threshold": settings.features.min_liquidity_threshold,
        },
        "scoring": {
            "long_term_min_score": settings.scoring.long_term_min_score,
            "swing_min_score": settings.scoring.swing_min_score,
            "max_risk_penalty": settings.scoring.max_risk_penalty,
            "long_term_weights": vars(settings.scoring.long_term_weights),
            "swing_weights": vars(settings.scoring.swing_weights),
        },
    }

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()

    output = {
        "config_hash_sha256": digest,
        "settings": payload,
    }

    out_dir = Path(".artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "config_snapshot.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
