"""Sector-relative and rolling valuation normalization utilities."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from statistics import mean, median, pstdev


@dataclass
class RollingSectorValuationNormalizer:
    """Computes robust z-scores for PE/PB within sector and rolling history."""

    window: int = 252
    _history_pe: dict[str, deque[float]] = field(default_factory=dict)
    _history_pb: dict[str, deque[float]] = field(default_factory=dict)

    def normalize(
        self,
        sector_by_symbol: Mapping[str, str],
        pe_by_symbol: Mapping[str, float],
        pb_by_symbol: Mapping[str, float],
    ) -> dict[str, dict[str, float]]:
        grouped_pe: dict[str, list[float]] = defaultdict(list)
        grouped_pb: dict[str, list[float]] = defaultdict(list)

        for symbol, sector in sector_by_symbol.items():
            bucket = sector or "UNKNOWN"
            pe = float(pe_by_symbol.get(symbol, 0.0))
            pb = float(pb_by_symbol.get(symbol, 0.0))
            if pe > 0:
                grouped_pe[bucket].append(pe)
            if pb > 0:
                grouped_pb[bucket].append(pb)

        out: dict[str, dict[str, float]] = {}
        for symbol, sector in sector_by_symbol.items():
            bucket = sector or "UNKNOWN"
            pe = float(pe_by_symbol.get(symbol, 0.0))
            pb = float(pb_by_symbol.get(symbol, 0.0))

            out[symbol] = {
                "sector_pe_zscore": self._robust_z(pe, grouped_pe.get(bucket, [])) if pe > 0 else 0.0,
                "sector_pb_zscore": self._robust_z(pb, grouped_pb.get(bucket, [])) if pb > 0 else 0.0,
                "rolling_pe_zscore": self._rolling_z(pe, self._history_pe.get(bucket)) if pe > 0 else 0.0,
                "rolling_pb_zscore": self._rolling_z(pb, self._history_pb.get(bucket)) if pb > 0 else 0.0,
            }

        self._append_history(grouped_pe, grouped_pb)
        return out

    def _append_history(self, grouped_pe: Mapping[str, Iterable[float]], grouped_pb: Mapping[str, Iterable[float]]) -> None:
        for sector, values in grouped_pe.items():
            nums = [v for v in values if v > 0]
            if not nums:
                continue
            hist = self._history_pe.setdefault(sector, deque(maxlen=self.window))
            hist.append(float(median(nums)))

        for sector, values in grouped_pb.items():
            nums = [v for v in values if v > 0]
            if not nums:
                continue
            hist = self._history_pb.setdefault(sector, deque(maxlen=self.window))
            hist.append(float(median(nums)))

    def _rolling_z(self, value: float, history: deque[float] | None) -> float:
        if history is None or len(history) < 5:
            return 0.0
        h = list(history)
        std = pstdev(h)
        if std <= 1e-9:
            return 0.0
        return (float(value) - mean(h)) / std

    def _robust_z(self, value: float, peers: list[float]) -> float:
        if len(peers) < 3:
            return 0.0
        med = median(peers)
        mad = median([abs(v - med) for v in peers])
        if mad <= 1e-9:
            std = pstdev(peers)
            if std <= 1e-9:
                return 0.0
            return (float(value) - mean(peers)) / std
        robust_std = 1.4826 * mad
        return (float(value) - med) / robust_std
