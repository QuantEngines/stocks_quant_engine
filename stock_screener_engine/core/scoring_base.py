"""Base scoring primitives shared by long-term, swing, and risk engines."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CategoryScore:
    name: str
    score_0_1: float
    weight: float
    contribution: float
    missing_features: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScoringResult:
    total_score: float
    categories: list[CategoryScore]
    component_map: dict[str, float]
    missing_features: list[str]


def combine_categories(categories: list[CategoryScore], scale: float = 100.0) -> ScoringResult:
    total_weight = sum(max(0.0, c.weight) for c in categories)
    weighted_sum = sum(c.contribution for c in categories)
    normalized = 0.0 if total_weight <= 0 else (weighted_sum / total_weight)
    total_score = max(0.0, min(scale, normalized * scale))

    component_map: dict[str, float] = {c.name: c.contribution for c in categories}
    missing: set[str] = set()
    for c in categories:
        missing.update(c.missing_features)

    return ScoringResult(
        total_score=total_score,
        categories=categories,
        component_map=component_map,
        missing_features=sorted(missing),
    )
