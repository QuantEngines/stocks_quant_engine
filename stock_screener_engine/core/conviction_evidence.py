"""Evidence plumbing for conviction scoring.

This module converts pipeline quality reports and backtest/calibration
artifacts into normalized feature values consumed by ``ConvictionScorer``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


def load_backtest_evidence(path: str | Path) -> tuple[dict[str, float], dict[str, object]]:
    """Load global model evidence from a calibration/backtest artifact.

    Supported payloads include ``ModelCalibrationPipeline`` reports and the
    richer engine-backtest reports that contain horizon metrics.
    """

    report_path = Path(path)
    if not report_path.exists():
        return {}, {"backtest_evidence_loaded": False, "path": str(report_path)}

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        return {}, {
            "backtest_evidence_loaded": False,
            "path": str(report_path),
            "error": str(exc),
        }

    if not isinstance(payload, dict):
        return {}, {"backtest_evidence_loaded": False, "path": str(report_path), "error": "non-dict payload"}

    engine_evaluation = _mapping(payload.get("evaluation"))
    report = _mapping(payload.get("report")) or engine_evaluation or payload
    net_ic = _mean_numeric_map(_mapping(payload.get("net_quantile_ic")) or _mapping(engine_evaluation.get("net_quantile_ic")))
    gross_ic = _mean_numeric_map(_mapping(report.get("quantile_ic")))
    ic = net_ic if net_ic is not None else gross_ic
    decay = _mean_numeric_map(_mapping(report.get("decay")))
    hit_rate = _mean_horizon_metric(
        _mapping(payload.get("net_horizon_metrics"))
        or _mapping(engine_evaluation.get("net_horizon_metrics"))
        or _mapping(payload.get("gross_horizon_metrics"))
        or _mapping(engine_evaluation.get("gross_horizon_metrics")),
        "top_quantile_hit_rate",
    )
    spread = _mean_horizon_metric(
        _mapping(payload.get("net_horizon_metrics"))
        or _mapping(engine_evaluation.get("net_horizon_metrics"))
        or _mapping(payload.get("gross_horizon_metrics"))
        or _mapping(engine_evaluation.get("gross_horizon_metrics")),
        "avg_quantile_spread",
    )
    walk_forward = _walk_forward_score(payload)

    evidence: dict[str, float] = {}
    if ic is not None:
        evidence["backtest_information_coefficient"] = ic
        evidence["factor_ic"] = ic
    if hit_rate is not None:
        evidence["backtest_hit_rate"] = hit_rate
    if walk_forward is not None:
        evidence["walk_forward_score"] = walk_forward
    elif ic is not None or decay is not None:
        evidence["walk_forward_score"] = _persistence_score(ic=ic, decay=decay)
    if spread is not None:
        evidence["calibration_score"] = _spread_score(spread)
    elif evidence:
        evidence["calibration_score"] = _combined_evidence_score(evidence)

    diagnostics = {
        "backtest_evidence_loaded": bool(evidence),
        "path": str(report_path),
        "mean_ic": ic,
        "mean_decay": decay,
        "mean_hit_rate": hit_rate,
        "mean_quantile_spread": spread,
        "features": sorted(evidence),
    }
    return evidence, diagnostics


def quality_confidence_features(
    snapshot_quality: Mapping[str, object],
    feature_quality: Mapping[str, object],
    freshness_quality: Mapping[str, object] | None,
) -> dict[str, float]:
    """Convert data-quality reports into source-confidence features."""

    snapshot_score = _quality_pass_score(snapshot_quality)
    feature_score = _quality_pass_score(feature_quality)
    coverage = _metric_score(snapshot_quality, "coverage", default=snapshot_score)
    freshness_score = _freshness_score(freshness_quality)
    reconciliation_score = _reconciliation_score(freshness_quality)

    source_confidence = _clamp01(
        0.30 * snapshot_score
        + 0.20 * coverage
        + 0.20 * feature_score
        + 0.20 * freshness_score
        + 0.10 * reconciliation_score
    )
    market_confidence = _clamp01(0.40 * snapshot_score + 0.30 * coverage + 0.30 * freshness_score)

    return {
        "source_confidence": source_confidence,
        "market_data_confidence": market_confidence,
        "data_source_confidence": source_confidence,
        "source_reconciliation_score": reconciliation_score,
    }


def symbol_source_features(values: Mapping[str, float]) -> dict[str, float]:
    """Derive symbol-level source confidence from feature availability."""

    fundamental_keys = (
        "growth_quality",
        "profitability_quality",
        "balance_sheet_health",
        "cash_flow_quality",
        "valuation_sanity",
        "pe_ratio",
        "debt_to_equity",
    )
    populated = [
        abs(float(values.get(key, 0.0))) > 1e-12
        for key in fundamental_keys
        if key in values
    ]
    fundamental_confidence = sum(1 for flag in populated if flag) / len(fundamental_keys)
    banking_applicable = (_finite_float(values.get("banking_sector_applicable")) or 0.0) >= 0.5
    banking_coverage = _clamp01(_finite_float(values.get("banking_metric_coverage")) or 0.0)
    if banking_applicable:
        fundamental_confidence = (0.65 * fundamental_confidence) + (0.35 * banking_coverage)
    transcript_quality = _finite_float(values.get("transcript_quality_signal"))
    out = {
        "fundamental_data_confidence": _clamp01(fundamental_confidence),
        "financials_data_confidence": _clamp01(fundamental_confidence),
    }
    if banking_applicable:
        out["banking_data_confidence"] = banking_coverage
    if transcript_quality is not None and transcript_quality > 0:
        out["document_quality_score"] = _clamp01(transcript_quality)
        out["document_extraction_quality"] = _clamp01(transcript_quality)
    return out


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _mean_numeric_map(values: Mapping[str, object]) -> float | None:
    nums = [_finite_float(value) for value in values.values()]
    nums = [value for value in nums if value is not None]
    return sum(nums) / len(nums) if nums else None


def _mean_horizon_metric(metrics: Mapping[str, object], key: str) -> float | None:
    vals: list[float] = []
    for row in metrics.values():
        if not isinstance(row, dict):
            continue
        value = _finite_float(row.get(key))
        if value is not None:
            vals.append(value)
    return sum(vals) / len(vals) if vals else None


def _walk_forward_score(payload: Mapping[str, object]) -> float | None:
    explicit = _finite_float(payload.get("walk_forward_score"))
    if explicit is not None:
        return _clamp01(explicit)
    mean_ic = _finite_float(payload.get("mean_ic"))
    mean_hit_rate = _finite_float(payload.get("mean_hit_rate"))
    if mean_ic is None and mean_hit_rate is None:
        return None
    ic_score = _ic_score(mean_ic or 0.0)
    hit_score = _clamp01(mean_hit_rate or 0.0)
    return _clamp01(0.50 * ic_score + 0.50 * hit_score)


def _persistence_score(ic: float | None, decay: float | None) -> float:
    vals = [_ic_score(value) for value in (ic, decay) if value is not None]
    return sum(vals) / len(vals) if vals else 0.5


def _combined_evidence_score(evidence: Mapping[str, float]) -> float:
    vals: list[float] = []
    if "backtest_information_coefficient" in evidence:
        vals.append(_ic_score(evidence["backtest_information_coefficient"]))
    if "backtest_hit_rate" in evidence:
        vals.append(_clamp01(evidence["backtest_hit_rate"]))
    if "walk_forward_score" in evidence:
        vals.append(_clamp01(evidence["walk_forward_score"]))
    return sum(vals) / len(vals) if vals else 0.5


def _spread_score(value: float) -> float:
    return _clamp01(0.5 + value * 5.0)


def _ic_score(value: float) -> float:
    return _clamp01((value + 0.05) / 0.15)


def _quality_pass_score(report: Mapping[str, object]) -> float:
    if not report:
        return 0.5
    if bool(report.get("passed")):
        return 1.0
    issues = report.get("issues")
    issue_count = len(issues) if isinstance(issues, list) else 1
    return max(0.15, 0.65 - min(0.50, issue_count * 0.10))


def _metric_score(report: Mapping[str, object], key: str, default: float) -> float:
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return _clamp01(default)
    value = _finite_float(metrics.get(key))
    return _clamp01(default if value is None else value)


def _freshness_score(report: Mapping[str, object] | None) -> float:
    if not report:
        return 0.7
    if bool(report.get("passed")):
        return 1.0
    issues = report.get("issues")
    warnings = report.get("warnings")
    issue_count = len(issues) if isinstance(issues, list) else 1
    warning_count = len(warnings) if isinstance(warnings, list) else 0
    return max(0.20, 0.80 - issue_count * 0.20 - warning_count * 0.05)


def _reconciliation_score(report: Mapping[str, object] | None) -> float:
    if not report:
        return 0.7
    issues = report.get("issues")
    warnings = report.get("warnings")
    issue_count = len(issues) if isinstance(issues, list) else 0
    warning_count = len(warnings) if isinstance(warnings, list) else 0
    if issue_count == 0 and warning_count == 0:
        return 1.0
    return max(0.20, 1.0 - issue_count * 0.25 - warning_count * 0.05)


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
