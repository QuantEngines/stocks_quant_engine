"""Run daily live invalidation against open broker positions.

Usage
-----
python tools/run_live_invalidation_daily.py
python tools/run_live_invalidation_daily.py --config stock_screener_engine/config/defaults.yaml
"""

from __future__ import annotations

import argparse
import json

from stock_screener_engine.app import run_live_invalidation_daily
from stock_screener_engine.config.settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily live invalidation runner")
    parser.add_argument("--config", dest="config_path", default=None, help="Optional YAML config path")
    args = parser.parse_args()

    settings = load_settings(config_path=args.config_path)
    result = run_live_invalidation_daily(settings)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
