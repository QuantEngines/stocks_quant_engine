"""Safe feature access abstraction for scoring modules.

Centralizes numeric coercion, missing-feature tracking, and defensive defaults
so category scorers remain clean and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass
class FeatureAccessor:
    values: Mapping[str, float] | None
    missing: set[str] = field(default_factory=set)
    invalid: set[str] = field(default_factory=set)

    def get(self, key: str, default: float = 0.0) -> float:
        if self.values is None or key not in self.values:
            self.missing.add(key)
            return default

        raw = self.values.get(key)
        if raw is None:
            self.missing.add(key)
            return default

        try:
            return float(raw)
        except (TypeError, ValueError):
            self.invalid.add(key)
            return default

    def get_non_negative(self, key: str, default: float = 0.0) -> float:
        return max(0.0, self.get(key, default=default))

    def get_bounded(self, key: str, lo: float, hi: float, default: float = 0.0) -> float:
        value = self.get(key, default=default)
        return max(lo, min(hi, value))

    def missing_features(self) -> list[str]:
        return sorted(self.missing)

    def invalid_features(self) -> list[str]:
        return sorted(self.invalid)
