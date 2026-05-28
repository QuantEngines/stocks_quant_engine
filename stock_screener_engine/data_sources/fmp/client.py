"""Financial Modeling Prep client and coverage probe.

The probe is deliberately diagnostic: it resolves Indian symbols, tests
endpoint coverage, and summarizes payload shape without persisting canonical
facts. FMP should only become a factor source after symbol coverage, plan
access, and point-in-time filing quality are verified.
"""

from __future__ import annotations

import gzip
import json
import logging
import time
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Union
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


JSONPayload = Union[dict[str, Any], list[Any]]
FetchFn = Callable[[str, Mapping[str, Any], Mapping[str, str]], JSONPayload]

DEFAULT_BASE_URL = "https://financialmodelingprep.com/stable"
LOGGER = logging.getLogger(__name__)
VALID_CHECKS = {
    "search",
    "profile",
    "quote",
    "income_statement",
    "balance_sheet",
    "cash_flow",
    "ratios",
    "key_metrics",
    "enterprise_values",
    "market_cap",
    "shares_float",
    "price_history",
    "analyst_estimates",
    "ratings",
    "grades",
    "transcripts",
}
DEFAULT_CHECKS = [
    "search",
    "profile",
    "income_statement",
    "balance_sheet",
    "cash_flow",
    "ratios",
    "key_metrics",
    "enterprise_values",
    "market_cap",
    "shares_float",
    "price_history",
    "analyst_estimates",
    "transcripts",
]


@dataclass(frozen=True)
class FMPRequest:
    name: str
    path: str
    params: dict[str, Any]


class FMPClient:
    """Small JSON client for FMP endpoints used by the coverage probe."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        timeout_seconds: int = 20,
        retries: int = 1,
        retry_delay_seconds: float = 0.5,
        fetch_fn: FetchFn | None = None,
    ) -> None:
        self.base_url = _clean_base_url(base_url)
        self.api_key = api_key or ""
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.retries = max(0, int(retries))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        self._fetch_fn = fetch_fn

    def search_symbol(self, query: str) -> JSONPayload:
        return self._get(FMPRequest("search_symbol", "/search-symbol", {"query": query}))

    def search_name(self, query: str) -> JSONPayload:
        return self._get(FMPRequest("search_name", "/search-name", {"query": query}))

    def profile(self, symbol: str) -> JSONPayload:
        return self._symbol_endpoint("profile", "/profile", symbol)

    def quote(self, symbol: str) -> JSONPayload:
        return self._symbol_endpoint("quote", "/quote", symbol)

    def income_statement(self, symbol: str, *, period: str = "annual", limit: int = 5) -> JSONPayload:
        return self._statement_endpoint("income_statement", "/income-statement", symbol, period, limit)

    def balance_sheet(self, symbol: str, *, period: str = "annual", limit: int = 5) -> JSONPayload:
        return self._statement_endpoint("balance_sheet", "/balance-sheet-statement", symbol, period, limit)

    def cash_flow(self, symbol: str, *, period: str = "annual", limit: int = 5) -> JSONPayload:
        return self._statement_endpoint("cash_flow", "/cash-flow-statement", symbol, period, limit)

    def ratios(self, symbol: str, *, period: str = "annual", limit: int = 5) -> JSONPayload:
        return self._statement_endpoint("ratios", "/ratios", symbol, period, limit)

    def key_metrics(self, symbol: str, *, period: str = "annual", limit: int = 5) -> JSONPayload:
        return self._statement_endpoint("key_metrics", "/key-metrics", symbol, period, limit)

    def enterprise_values(self, symbol: str, *, period: str = "annual", limit: int = 5) -> JSONPayload:
        return self._statement_endpoint("enterprise_values", "/enterprise-values", symbol, period, limit)

    def market_cap(self, symbol: str) -> JSONPayload:
        return self._symbol_endpoint("market_cap", "/market-capitalization", symbol)

    def shares_float(self, symbol: str) -> JSONPayload:
        return self._symbol_endpoint("shares_float", "/shares-float", symbol)

    def price_history(
        self,
        symbol: str,
        *,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> JSONPayload:
        params: dict[str, Any] = {"symbol": symbol}
        if from_date:
            params["from"] = from_date.isoformat()
        if to_date:
            params["to"] = to_date.isoformat()
        return self._get(FMPRequest("price_history", "/historical-price-eod/full", params))

    def analyst_estimates(self, symbol: str, *, period: str = "annual", limit: int = 10) -> JSONPayload:
        return self._get(
            FMPRequest(
                "analyst_estimates",
                "/analyst-estimates",
                {"symbol": symbol, "period": period, "page": 0, "limit": limit},
            )
        )

    def ratings(self, symbol: str) -> JSONPayload:
        return self._symbol_endpoint("ratings", "/ratings-snapshot", symbol)

    def grades(self, symbol: str) -> JSONPayload:
        return self._symbol_endpoint("grades", "/grades-consensus", symbol)

    def transcripts(self, symbol: str) -> JSONPayload:
        return self._symbol_endpoint("transcripts", "/earning-call-transcript-dates", symbol)

    def _symbol_endpoint(self, name: str, path: str, symbol: str) -> JSONPayload:
        return self._get(FMPRequest(name, path, {"symbol": symbol}))

    def _statement_endpoint(self, name: str, path: str, symbol: str, period: str, limit: int) -> JSONPayload:
        return self._get(
            FMPRequest(
                name,
                path,
                {"symbol": symbol, "period": period, "limit": max(1, int(limit))},
            )
        )

    def _get(self, request: FMPRequest) -> JSONPayload:
        params = {key: value for key, value in request.params.items() if value is not None and str(value) != ""}
        if self.api_key:
            params["apikey"] = self.api_key
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": "stock-screener-engine/0.1 FMP probe",
        }
        if self._fetch_fn is not None:
            return self._fetch_fn(_url_for(self.base_url, request.path), params, headers)

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._urlopen_json(_url_for(self.base_url, request.path), params, headers)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                if self.retry_delay_seconds:
                    time.sleep(self.retry_delay_seconds)
        raise RuntimeError(f"{request.name} failed: {_safe_error(last_error)}") from last_error

    def _urlopen_json(self, url: str, params: Mapping[str, Any], headers: Mapping[str, str]) -> JSONPayload:
        query = urlencode({k: v for k, v in params.items() if v is not None and str(v) != ""})
        full_url = f"{url}?{query}" if query else url
        req = Request(full_url, headers=dict(headers))
        with urlopen(req, timeout=self.timeout_seconds) as resp:  # noqa: S310 - explicit user-configured API endpoint.
            raw = _decode_response(resp.read(), resp.headers.get("Content-Encoding", ""))
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, (dict, list)):
            raise json.JSONDecodeError("JSON payload is not an object or list", raw.decode("utf-8"), 0)
        return parsed


class FMPProbe:
    """Probe FMP coverage for a symbol set without storing canonical facts."""

    def __init__(
        self,
        client: FMPClient,
        *,
        period: str = "annual",
        limit: int = 5,
        price_start: date | None = None,
        price_end: date | None = None,
        exact_symbols: bool = False,
    ) -> None:
        self.client = client
        self.period = period
        self.limit = max(1, int(limit))
        self.price_start = price_start
        self.price_end = price_end
        self.exact_symbols = exact_symbols
        self._halt_reason = ""

    def run(
        self,
        symbols: Sequence[str],
        checks: Sequence[str],
        symbol_names: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        normalized_symbols = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
        normalized_checks = normalize_fmp_checks(checks)
        symbol_reports = []
        for symbol in normalized_symbols:
            if self._halt_reason:
                symbol_reports.append(self._skipped_symbol_report(symbol, normalized_checks, self._halt_reason))
                continue
            LOGGER.info("FMP probe starting for %s (%d checks)", symbol, len(normalized_checks))
            symbol_reports.append(self._probe_symbol(symbol, normalized_checks, symbol_names or {}))
        coverage = _coverage(symbol_reports, normalized_checks)
        return {
            "pipeline": "fmp_probe",
            "run_at": datetime.utcnow().isoformat() + "Z",
            "symbols_requested": len(normalized_symbols),
            "checks": normalized_checks,
            "passed": bool(normalized_symbols) and any(v["ok"] > 0 for v in coverage.values()),
            "coverage": coverage,
            "symbol_reports": symbol_reports,
            "recommendations": _recommendations(coverage, normalized_checks),
        }

    def _probe_symbol(
        self,
        symbol: str,
        checks: Sequence[str],
        symbol_names: Mapping[str, str],
    ) -> dict[str, Any]:
        candidate_symbols = [symbol.strip().upper()] if self.exact_symbols else _symbol_variants(symbol)
        report: dict[str, Any] = {
            "symbol": symbol,
            "candidate_symbols": candidate_symbols,
            "checks": {},
        }
        if "search" in checks:
            search_result = _capture(lambda: self._search(symbol, symbol_names.get(symbol, "")))
            search_payload = _strip_payload(search_result)
            search_candidates = _search_candidates(search_payload, original_symbol=symbol)
            if search_candidates:
                resolved_symbols = [row["symbol"] for row in search_candidates if row.get("symbol")]
                candidate_symbols = _dedupe([*resolved_symbols, *candidate_symbols])
                report["resolved_symbol"] = candidate_symbols[0]
                report["search_candidates"] = search_candidates[:8]
                report["candidate_symbols"] = candidate_symbols
            report["checks"]["search"] = search_result
            if not search_result["ok"] and _is_rate_limit_error(str(search_result.get("error", ""))):
                self._halt_reason = "FMP rate limit reached; stop this probe and retry later with fewer symbols/checks."

        for check in checks:
            if check == "search":
                continue
            if self._halt_reason:
                report["checks"][check] = _skipped_result(self._halt_reason)
                continue
            LOGGER.info("FMP probe %s: %s", symbol, check)
            result = self._probe_endpoint(check, candidate_symbols)
            report["checks"][check] = result
            if result.get("ok") and result.get("resolved_symbol"):
                report.setdefault("resolved_symbol", result["resolved_symbol"])

        report["usable_sections"] = [
            name for name, result in report["checks"].items()
            if isinstance(result, dict) and result.get("ok")
        ]
        report["ok"] = bool(report["usable_sections"])
        return report

    def _skipped_symbol_report(self, symbol: str, checks: Sequence[str], reason: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "candidate_symbols": [symbol.strip().upper()] if self.exact_symbols else _symbol_variants(symbol),
            "checks": {check: _skipped_result(reason) for check in checks},
            "usable_sections": [],
            "ok": False,
            "skipped": True,
            "skip_reason": reason,
        }

    def _search(self, symbol: str, company_name: str) -> JSONPayload:
        payloads: list[Any] = []
        errors: list[str] = []
        for query in _dedupe([symbol, symbol.replace("-", " "), company_name]):
            if not query:
                continue
            for fetch in (self.client.search_symbol, self.client.search_name):
                try:
                    payload = fetch(query)
                except RuntimeError as exc:
                    errors.append(f"{fetch.__name__}:{query}: {_safe_error(exc)}")
                    continue
                if isinstance(payload, list):
                    payloads.extend(payload)
                elif isinstance(payload, dict):
                    payloads.append(payload)
        if not payloads and errors:
            raise RuntimeError(f"search failed across endpoints: {', '.join(errors[:6])}")
        return payloads

    def _probe_endpoint(self, check: str, candidate_symbols: Sequence[str]) -> dict[str, Any]:
        if check == "profile":
            return self._capture_first(check, candidate_symbols, self.client.profile)
        if check == "quote":
            return self._capture_first(check, candidate_symbols, self.client.quote)
        if check == "income_statement":
            return self._capture_first(
                check,
                candidate_symbols,
                lambda symbol: self.client.income_statement(symbol, period=self.period, limit=self.limit),
            )
        if check == "balance_sheet":
            return self._capture_first(
                check,
                candidate_symbols,
                lambda symbol: self.client.balance_sheet(symbol, period=self.period, limit=self.limit),
            )
        if check == "cash_flow":
            return self._capture_first(
                check,
                candidate_symbols,
                lambda symbol: self.client.cash_flow(symbol, period=self.period, limit=self.limit),
            )
        if check == "ratios":
            return self._capture_first(
                check,
                candidate_symbols,
                lambda symbol: self.client.ratios(symbol, period=self.period, limit=self.limit),
            )
        if check == "key_metrics":
            return self._capture_first(
                check,
                candidate_symbols,
                lambda symbol: self.client.key_metrics(symbol, period=self.period, limit=self.limit),
            )
        if check == "enterprise_values":
            return self._capture_first(
                check,
                candidate_symbols,
                lambda symbol: self.client.enterprise_values(symbol, period=self.period, limit=self.limit),
            )
        if check == "market_cap":
            return self._capture_first(check, candidate_symbols, self.client.market_cap)
        if check == "shares_float":
            return self._capture_first(check, candidate_symbols, self.client.shares_float)
        if check == "price_history":
            return self._capture_first(
                check,
                candidate_symbols,
                lambda symbol: self.client.price_history(
                    symbol,
                    from_date=self.price_start,
                    to_date=self.price_end,
                ),
            )
        if check == "analyst_estimates":
            return self._capture_first(
                check,
                candidate_symbols,
                lambda symbol: self.client.analyst_estimates(symbol, period=self.period, limit=self.limit),
            )
        if check == "ratings":
            return self._capture_first(check, candidate_symbols, self.client.ratings)
        if check == "grades":
            return self._capture_first(check, candidate_symbols, self.client.grades)
        if check == "transcripts":
            return self._capture_first(check, candidate_symbols, self.client.transcripts)
        return {"ok": False, "error": f"unsupported FMP check '{check}'", "summary": {}}

    def _capture_first(
        self,
        check: str,
        candidate_symbols: Sequence[str],
        fetch: Callable[[str], JSONPayload],
    ) -> dict[str, Any]:
        errors: list[str] = []
        for symbol in candidate_symbols:
            result = _capture(lambda value=symbol: fetch(value))
            _strip_payload(result)
            if result["ok"]:
                result["resolved_symbol"] = symbol
                result["attempted_symbols"] = _dedupe([*candidate_symbols])
                return result
            error = result.get("error") or "empty payload"
            errors.append(f"{symbol}: {error}")
            if _is_rate_limit_error(str(error)):
                self._halt_reason = "FMP rate limit reached; stop this probe and retry later with fewer symbols/checks."
                return {
                    "ok": False,
                    "error": f"{check} stopped after {symbol}: {error}",
                    "summary": {},
                    "attempted_symbols": [symbol],
                    "terminal_error": "rate_limit",
                }
            if _is_terminal_variant_error(str(error)):
                return {
                    "ok": False,
                    "error": f"{check} stopped after {symbol}: {error}",
                    "summary": {},
                    "attempted_symbols": [symbol],
                    "terminal_error": "auth_or_plan",
                }
        return {
            "ok": False,
            "error": f"{check} failed across symbol variants: {'; '.join(errors[:6])}",
            "summary": {},
            "attempted_symbols": _dedupe([*candidate_symbols]),
        }


def default_price_window(today: date | None = None) -> tuple[date, date]:
    end = today or date.today()
    return end - timedelta(days=90), end


def _capture(fetch: Callable[[], JSONPayload]) -> dict[str, Any]:
    try:
        payload = fetch()
        summary = _summarize_payload(payload)
        summary["_payload"] = payload
        payload_error = _payload_error(payload)
        ok = _payload_has_data(payload)
        return {
            "ok": ok,
            "error": "" if ok else payload_error,
            "summary": summary,
        }
    except Exception as exc:  # noqa: BLE001 - probe must continue across endpoints/symbols.
        return {
            "ok": False,
            "error": _safe_error(exc),
            "summary": {},
        }


def _skipped_result(reason: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": reason,
        "summary": {},
        "skipped": True,
    }


def _strip_payload(result: dict[str, Any]) -> JSONPayload | None:
    summary = result.get("summary")
    if not isinstance(summary, dict):
        return None
    payload = summary.pop("_payload", None)
    return payload if isinstance(payload, (dict, list)) else None


def _summarize_payload(payload: JSONPayload) -> dict[str, Any]:
    if isinstance(payload, list):
        first = payload[0] if payload and isinstance(payload[0], dict) else {}
        summary: dict[str, Any] = {
            "type": "list",
            "item_count": len(payload),
            "first_item_keys": sorted(first.keys())[:30] if isinstance(first, dict) else [],
        }
        dates = [
            str(item.get("date") or item.get("fillingDate") or item.get("calendarYear") or "")
            for item in payload
            if isinstance(item, dict)
        ]
        if dates:
            summary["sample_periods"] = _dedupe([value for value in dates if value])[:8]
        symbols = [
            str(item.get("symbol") or "")
            for item in payload
            if isinstance(item, dict) and item.get("symbol")
        ]
        if symbols:
            summary["sample_symbols"] = _dedupe(symbols)[:8]
        return summary

    keys = sorted(str(key) for key in payload.keys())
    return {
        "type": "dict",
        "top_level_keys": keys[:40],
        "top_level_key_count": len(keys),
    }


def _payload_has_data(payload: JSONPayload) -> bool:
    if isinstance(payload, list):
        return bool(payload)
    if not payload:
        return False
    if _payload_error(payload):
        return False
    return True


def _payload_error(payload: JSONPayload) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("Error Message", "error", "message", "detail"):
        value = payload.get(key)
        if value:
            return str(value)
    if set(payload.keys()) == {"info"}:
        return str(payload.get("info") or "info-only response")
    return ""


def _is_rate_limit_error(error: str) -> bool:
    text = error.lower()
    return "http error 429" in text or "too many requests" in text or "rate limit" in text


def _is_terminal_variant_error(error: str) -> bool:
    text = error.lower()
    return _is_rate_limit_error(error) or "http error 401" in text or "http error 403" in text


def _search_candidates(payload: JSONPayload | None, *, original_symbol: str) -> list[dict[str, str]]:
    if not isinstance(payload, list):
        return []
    rows: list[dict[str, str]] = []
    original = original_symbol.upper()
    for item in payload:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        row = {
            "symbol": symbol,
            "name": str(item.get("name") or item.get("companyName") or "").strip(),
            "exchange": str(item.get("exchange") or item.get("exchangeShortName") or "").strip(),
            "exchangeShortName": str(item.get("exchangeShortName") or "").strip(),
            "currency": str(item.get("currency") or "").strip(),
            "country": str(item.get("country") or "").strip(),
        }
        rows.append(row)
    return sorted(
        _unique_candidate_rows(rows),
        key=lambda row: (
            0 if row["symbol"] == f"{original}.NS" else 1,
            0 if row["symbol"].startswith(original) else 1,
            0 if row.get("exchangeShortName", "").upper() in {"NSE", "NSEI"} else 1,
            0 if row.get("country", "").upper() in {"IN", "INDIA"} else 1,
            row["symbol"],
        ),
    )


def _unique_candidate_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append({key: str(value) for key, value in row.items()})
    return out


def _coverage(symbol_reports: Sequence[Mapping[str, Any]], checks: Sequence[str]) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    total = len(symbol_reports)
    for check in checks:
        ok = 0
        errors: list[str] = []
        resolved: list[str] = []
        for report in symbol_reports:
            check_result = report.get("checks", {}).get(check) if isinstance(report.get("checks"), dict) else None
            if isinstance(check_result, Mapping) and check_result.get("ok"):
                ok += 1
                if check_result.get("resolved_symbol"):
                    resolved.append(str(check_result["resolved_symbol"]))
            elif isinstance(check_result, Mapping) and check_result.get("error"):
                errors.append(str(check_result["error"]))
        coverage[check] = {
            "ok": ok,
            "total": total,
            "coverage": round(ok / total, 4) if total else 0.0,
            "sample_resolved_symbols": _dedupe(resolved)[:8],
            "sample_errors": _dedupe(errors)[:5],
        }
    return coverage


def _recommendations(coverage: Mapping[str, Mapping[str, Any]], checks: Sequence[str]) -> list[str]:
    recs: list[str] = []
    core_checks = [
        check
        for check in ("income_statement", "balance_sheet", "cash_flow", "ratios", "key_metrics")
        if check in checks
    ]
    if core_checks and all(float(coverage.get(check, {}).get("coverage", 0.0)) >= 0.8 for check in core_checks):
        recs.append("FMP fundamentals look strong enough for a small point-in-time reconciliation test.")
    if "price_history" in checks and float(coverage.get("price_history", {}).get("coverage", 0.0)) >= 0.8:
        recs.append("FMP price history can be compared against canonical broker/yfinance bars for split/adjustment policy.")
    if "analyst_estimates" in checks and float(coverage.get("analyst_estimates", {}).get("coverage", 0.0)) > 0.0:
        recs.append("Treat analyst estimates as a medium-trust satellite factor until timestamp and revision history are verified.")
    if "transcripts" in checks and float(coverage.get("transcripts", {}).get("coverage", 0.0)) > 0.0:
        recs.append("Transcript date coverage can seed the document intelligence backlog, but Indian company transcript text must be verified.")
    if not recs:
        recs.append("FMP did not return enough usable payloads in this probe; check API key, symbol suffixes, and plan coverage.")
    return recs


def normalize_fmp_checks(checks: Sequence[str]) -> list[str]:
    """Normalize public FMP probe check names."""
    out: list[str] = []
    for raw in checks:
        value = raw.strip().lower().replace("-", "_")
        if not value:
            continue
        if value == "all":
            return list(DEFAULT_CHECKS)
        if value == "smoke":
            for expanded in ("search", "profile"):
                if expanded not in out:
                    out.append(expanded)
            continue
        if value in {"statements", "financials"}:
            for expanded in ("income_statement", "balance_sheet", "cash_flow"):
                if expanded not in out:
                    out.append(expanded)
            continue
        if value not in VALID_CHECKS:
            raise ValueError(f"unsupported FMP check '{raw}'")
        if value not in out:
            out.append(value)
    return out or list(DEFAULT_CHECKS)


def _symbol_variants(symbol: str) -> list[str]:
    cleaned = symbol.strip().upper()
    if not cleaned:
        return []
    if "." in cleaned:
        base = cleaned.split(".", maxsplit=1)[0]
        return _dedupe([cleaned, base, f"{base}.NS", f"{base}.BO"])
    return _dedupe([f"{cleaned}.NS", f"{cleaned}.BO", cleaned])


def _decode_response(raw: bytes, encoding: str) -> bytes:
    encoding = encoding.lower().strip()
    if "gzip" in encoding:
        return gzip.decompress(raw)
    if "deflate" in encoding:
        try:
            return zlib.decompress(raw)
        except zlib.error:
            return zlib.decompress(raw, -zlib.MAX_WBITS)
    return raw


def _url_for(base_url: str, path: str) -> str:
    return f"{_clean_base_url(base_url)}/{path.lstrip('/')}"


def _clean_base_url(base_url: str) -> str:
    return str(base_url or DEFAULT_BASE_URL).rstrip("/")


def _dedupe(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _safe_error(exc: object) -> str:
    if isinstance(exc, HTTPError):
        return f"HTTP Error {exc.code}: {exc.reason}"
    if isinstance(exc, URLError):
        return f"URL Error: {exc.reason}"
    return str(exc) if exc else "unknown error"
