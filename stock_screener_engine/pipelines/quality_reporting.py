"""Shared quality-report helpers for production pipeline runs."""

from __future__ import annotations

from typing import Any, Mapping


def build_pipeline_quality_report(output: Mapping[str, Any], pipeline_name: str) -> dict[str, Any]:
    flags = dict(output.get("quality_flags", {}))
    category_passes = [
        bool(payload.get("passed"))
        for payload in flags.values()
        if isinstance(payload, dict) and "passed" in payload
    ]
    passed = bool(category_passes) and all(category_passes)
    return {
        "pipeline": pipeline_name,
        "run_at": output.get("run_at"),
        "as_of": output.get("as_of"),
        "passed": passed,
        "symbols_requested": output.get("symbols_requested"),
        "symbols_selected": output.get("symbols_selected"),
        "symbols_with_features": output.get("symbols_with_features"),
        "quality_flags": flags,
    }
