"""Minimal runnable demo for long-term and swing ranking outputs."""

from __future__ import annotations

import json

from stock_screener_engine.app import run_demo


if __name__ == "__main__":
    result = run_demo()
    print(json.dumps(result, indent=2))
