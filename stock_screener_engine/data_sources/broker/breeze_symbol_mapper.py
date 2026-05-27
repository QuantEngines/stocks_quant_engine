"""ICICI Breeze stock-code mapping helpers."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class BreezeSymbolResolution:
    symbol: str
    stock_code: str
    exchange_code: str = "NSE"
    source: str = "raw_symbol"
    token: str = ""
    company_name: str = ""
    error: str = ""

    def as_payload(self) -> dict[str, str]:
        return {
            "symbol": self.symbol,
            "stock_code": self.stock_code,
            "exchange_code": self.exchange_code,
            "source": self.source,
            "token": self.token,
            "company_name": self.company_name,
            "error": self.error,
        }


class BreezeSymbolMapper:
    """Resolve NSE trading symbols to Breeze `stock_code` values.

    Breeze cash APIs frequently expect ICICI's short stock codes rather than
    raw NSE symbols.  The mapper uses a local CSV cache first, then Breeze's
    `get_names` helper when an authenticated client is available.
    """

    def __init__(
        self,
        map_path: str | Path | None = None,
        exchange_code: str = "NSE",
        persist: bool = True,
    ) -> None:
        self.exchange_code = exchange_code.strip().upper() or "NSE"
        self.map_path = Path(map_path).expanduser() if map_path else _default_map_path()
        self.persist = persist
        self._cache: dict[str, BreezeSymbolResolution] = {}
        self._load()

    @classmethod
    def from_env(cls) -> "BreezeSymbolMapper":
        return cls(map_path=os.getenv("SSE_BREEZE_SYMBOL_MAP_PATH"))

    def resolve(self, symbol: str, client: object | None = None) -> BreezeSymbolResolution:
        normalized = _normalize_symbol(symbol)
        if not normalized:
            return BreezeSymbolResolution(symbol="", stock_code="", exchange_code=self.exchange_code, error="empty symbol")
        if normalized in self._cache:
            return self._cache[normalized]

        learned = self._resolve_with_client(normalized, client)
        self._cache[normalized] = learned
        if self.persist and learned.source == "breeze_get_names":
            self.save()
        return learned

    def save(self) -> None:
        if not self.map_path:
            return
        self.map_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [resolution.as_payload() for _, resolution in sorted(self._cache.items())]
        with self.map_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["symbol", "stock_code", "exchange_code", "source", "token", "company_name", "error"],
            )
            writer.writeheader()
            writer.writerows(rows)

    def _load(self) -> None:
        if not self.map_path or not self.map_path.exists():
            return
        with self.map_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                resolution = _resolution_from_row(row, self.exchange_code)
                if resolution.symbol and resolution.stock_code:
                    self._cache[resolution.symbol] = resolution

    def _resolve_with_client(self, symbol: str, client: object | None) -> BreezeSymbolResolution:
        get_names = getattr(client, "get_names", None)
        if not callable(get_names):
            return BreezeSymbolResolution(symbol=symbol, stock_code=symbol, exchange_code=self.exchange_code)
        try:
            payload = get_names(self.exchange_code, symbol)
        except Exception as exc:  # noqa: BLE001 - mapper must degrade gracefully
            return BreezeSymbolResolution(
                symbol=symbol,
                stock_code=symbol,
                exchange_code=self.exchange_code,
                error=str(exc)[:500],
            )
        if not isinstance(payload, Mapping):
            return BreezeSymbolResolution(symbol=symbol, stock_code=symbol, exchange_code=self.exchange_code)
        error = _first(payload, "Error", "error", "message", "Message")
        stock_code = _first(payload, "isec_stock_code", "stock_code", "ShortName", "short_name").strip().upper()
        if not stock_code:
            return BreezeSymbolResolution(
                symbol=symbol,
                stock_code=symbol,
                exchange_code=self.exchange_code,
                error=error[:500],
            )
        return BreezeSymbolResolution(
            symbol=symbol,
            stock_code=stock_code,
            exchange_code=str(_first(payload, "exchange_code") or self.exchange_code).strip().upper(),
            source="breeze_get_names",
            token=_first(payload, "isec_token", "token"),
            company_name=_first(payload, "company name", "company_name", "CompanyName"),
            error=error[:500],
        )


def _default_map_path() -> Path:
    root = Path(os.getenv("SSE_STORAGE_ROOT", "./data")).expanduser()
    return root / "broker" / "breeze_symbol_map.csv"


def _resolution_from_row(row: Mapping[str, object], exchange_code: str) -> BreezeSymbolResolution:
    symbol = _normalize_symbol(_first(row, "symbol", "nse_symbol", "exchange_stock_code"))
    stock_code = _normalize_symbol(_first(row, "stock_code", "breeze_code", "isec_stock_code", "icici_code"))
    return BreezeSymbolResolution(
        symbol=symbol,
        stock_code=stock_code,
        exchange_code=str(_first(row, "exchange_code") or exchange_code).strip().upper(),
        source=str(_first(row, "source") or "map_file").strip() or "map_file",
        token=_first(row, "token", "isec_token"),
        company_name=_first(row, "company_name", "company name"),
        error=_first(row, "error"),
    )


def _normalize_symbol(value: object) -> str:
    return str(value or "").strip().upper()


def _first(row: Mapping[str, object], *keys: str) -> str:
    lower = {str(k).strip().lower(): v for k, v in row.items()}
    for key in keys:
        value = row.get(key)
        if value is None:
            value = lower.get(key.strip().lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""
