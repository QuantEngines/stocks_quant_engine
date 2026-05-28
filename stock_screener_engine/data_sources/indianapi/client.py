"""Lightweight IndianAPI client and coverage probe.

The probe intentionally avoids writing canonical facts. IndianAPI is useful
enough to test, but source confidence should be earned by coverage and
reconciliation before it feeds backtests or live conviction.
"""

from __future__ import annotations

import gzip
import json
import time
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Union
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


JSONPayload = Union[dict[str, Any], list[Any]]
FetchFn = Callable[[str, Mapping[str, Any], Mapping[str, str]], JSONPayload]

DEFAULT_STOCK_BASE_URL = "https://stock.indianapi.in"
DEFAULT_ANALYST_BASE_URL = "https://analyst.indianapi.in"
VALID_CHECKS = {
    "search",
    "stock",
    "financials",
    "shareholding",
    "analyst",
    "forecasts",
    "history",
    "corporate_actions",
    "announcements",
    "news",
    "trending",
    "nse_most_active",
    "bse_most_active",
    "price_shockers",
    "week_52",
    "ipo",
}
MARKET_LEVEL_CHECKS = {
    "news",
    "trending",
    "nse_most_active",
    "bse_most_active",
    "price_shockers",
    "week_52",
    "ipo",
}


@dataclass(frozen=True)
class IndianAPIRequest:
    name: str
    base_url: str
    path: str
    params: dict[str, Any]


class IndianAPIClient:
    """Small JSON client for IndianAPI endpoints used by the probe."""

    def __init__(
        self,
        *,
        stock_base_url: str = DEFAULT_STOCK_BASE_URL,
        analyst_base_url: str = DEFAULT_ANALYST_BASE_URL,
        api_key: str | None = None,
        timeout_seconds: int = 20,
        retries: int = 1,
        retry_delay_seconds: float = 0.5,
        fetch_fn: FetchFn | None = None,
    ) -> None:
        self.stock_base_url = _clean_base_url(stock_base_url)
        self.analyst_base_url = _clean_base_url(analyst_base_url)
        self.api_key = api_key or ""
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.retries = max(0, int(retries))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        self._fetch_fn = fetch_fn

    def stock_details(self, symbol: str, *, query_terms: Sequence[str] | None = None) -> JSONPayload:
        errors: list[str] = []
        for term in _query_terms(symbol, query_terms):
            try:
                return self._get(
                    IndianAPIRequest(
                        name=f"stock:name:{term}",
                        base_url=self.stock_base_url,
                        path="/stock",
                        params={"name": term},
                    )
                )
            except RuntimeError as exc:
                errors.append(_safe_error(exc))
        raise RuntimeError(f"stock lookup failed across name params: {'; '.join(errors[:4])}")

    def industry_search(self, query: str) -> JSONPayload:
        return self._get(
            IndianAPIRequest(
                name=f"industry_search:{query}",
                base_url=self.stock_base_url,
                path="/industry_search",
                params={"query": query},
            )
        )

    def historical_data(
        self,
        symbol: str,
        *,
        period: str = "1yr",
        data_filter: str = "price",
        query_terms: Sequence[str] | None = None,
    ) -> JSONPayload:
        errors: list[str] = []
        for value in _query_terms(symbol, query_terms):
            try:
                return self._get(
                    IndianAPIRequest(
                        name=f"historical_data:stock_name:{value}",
                        base_url=self.stock_base_url,
                        path="/historical_data",
                        params={"stock_name": value, "period": period, "filter": data_filter},
                    )
                )
            except RuntimeError as exc:
                errors.append(_safe_error(exc))
        raise RuntimeError(f"historical_data failed across documented/example params: {'; '.join(errors[:4])}")

    def historical_stats(
        self,
        symbol: str,
        *,
        stats: str,
        query_terms: Sequence[str] | None = None,
    ) -> JSONPayload:
        errors: list[str] = []
        for value in _query_terms(symbol, query_terms):
            try:
                return self._get(
                    IndianAPIRequest(
                        name=f"historical_stats:{stats}:stock_name:{value}",
                        base_url=self.stock_base_url,
                        path="/historical_stats",
                        params={"stock_name": value, "stats": stats},
                    )
                )
            except RuntimeError as exc:
                errors.append(_safe_error(exc))
        for value in _query_terms(symbol, query_terms):
            try:
                return self._get(
                    IndianAPIRequest(
                        name=f"statement:{stats}:stock_name:{value}",
                        base_url=self.stock_base_url,
                        path="/statement",
                        params={"stock_name": value, "stats": stats},
                    )
                )
            except RuntimeError as exc:
                errors.append(_safe_error(exc))
        raise RuntimeError(f"historical_stats/statement:{stats} failed across params: {'; '.join(errors[:5])}")

    def corporate_actions(self, symbol: str, *, query_terms: Sequence[str] | None = None) -> JSONPayload:
        return self._stock_name_endpoint(
            path="/corporate_actions",
            name="corporate_actions",
            symbol=symbol,
            query_terms=query_terms,
        )

    def recent_announcements(self, symbol: str, *, query_terms: Sequence[str] | None = None) -> JSONPayload:
        return self._stock_name_endpoint(
            path="/recent_announcements",
            name="recent_announcements",
            symbol=symbol,
            query_terms=query_terms,
        )

    def news(self) -> JSONPayload:
        return self._simple_stock_endpoint(path="/news", name="news")

    def trending(self) -> JSONPayload:
        return self._simple_stock_endpoint(path="/trending", name="trending")

    def nse_most_active(self) -> JSONPayload:
        return self._simple_stock_endpoint(path="/NSE_most_active", name="nse_most_active")

    def bse_most_active(self) -> JSONPayload:
        return self._simple_stock_endpoint(path="/BSE_most_active", name="bse_most_active")

    def price_shockers(self) -> JSONPayload:
        return self._simple_stock_endpoint(path="/price_shockers", name="price_shockers")

    def week_52_high_low(self) -> JSONPayload:
        return self._simple_stock_endpoint(path="/fetch_52_week_high_low_data", name="week_52")

    def ipo(self) -> JSONPayload:
        return self._simple_stock_endpoint(path="/ipo", name="ipo")

    def target_price(self, stock_id: str) -> JSONPayload:
        return self._analyst_endpoint(
            name="stock_target_price",
            path="/stock_target_price",
            params={"stock_id": stock_id},
        )

    def forecasts(
        self,
        stock_id: str,
        *,
        measure_code: str = "EPS",
        period_type: str = "Annual",
        data_type: str = "Actuals",
        age: str = "Current",
    ) -> JSONPayload:
        return self._analyst_endpoint(
            name="stock_forecasts",
            path="/stock_forecasts",
            params={
                "stock_id": stock_id,
                "measure_code": measure_code,
                "period_type": period_type,
                "data_type": data_type,
                "age": age,
            },
        )

    def _get(self, request: IndianAPIRequest) -> JSONPayload:
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": "stock-screener-engine/0.1 IndianAPI probe",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key

        if self._fetch_fn is not None:
            return self._fetch_fn(_url_for(request.base_url, request.path), request.params, headers)

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._urlopen_json(_url_for(request.base_url, request.path), request.params, headers)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                if self.retry_delay_seconds:
                    time.sleep(self.retry_delay_seconds)
        raise RuntimeError(f"{request.name} failed: {_safe_error(last_error)}") from last_error

    def _stock_name_endpoint(
        self,
        *,
        path: str,
        name: str,
        symbol: str,
        query_terms: Sequence[str] | None = None,
    ) -> JSONPayload:
        errors: list[str] = []
        for value in _query_terms(symbol, query_terms):
            try:
                return self._get(
                    IndianAPIRequest(
                        name=f"{name}:stock_name:{value}",
                        base_url=self.stock_base_url,
                        path=path,
                        params={"stock_name": value},
                    )
                )
            except RuntimeError as exc:
                errors.append(_safe_error(exc))
        raise RuntimeError(f"{name} failed across stock_name params: {'; '.join(errors[:4])}")

    def _simple_stock_endpoint(self, *, path: str, name: str) -> JSONPayload:
        return self._get(
            IndianAPIRequest(
                name=name,
                base_url=self.stock_base_url,
                path=path,
                params={},
            )
        )

    def _analyst_endpoint(self, *, name: str, path: str, params: dict[str, Any]) -> JSONPayload:
        errors: list[str] = []
        bases = _dedupe([self.analyst_base_url, self.stock_base_url])
        for base_url in bases:
            try:
                return self._get(
                    IndianAPIRequest(
                        name=f"{name}:{base_url}",
                        base_url=base_url,
                        path=path,
                        params=params,
                    )
                )
            except RuntimeError as exc:
                errors.append(_safe_error(exc))
        raise RuntimeError(f"{name} failed across analyst/stock bases: {'; '.join(errors[:4])}")

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


class IndianAPIProbe:
    """Probe IndianAPI endpoint coverage for a symbol set."""

    def __init__(self, client: IndianAPIClient) -> None:
        self.client = client

    def run(
        self,
        symbols: Sequence[str],
        checks: Sequence[str],
        symbol_names: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        normalized_symbols = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
        normalized_checks = _normalize_checks(checks)
        symbol_checks = [check for check in normalized_checks if check not in MARKET_LEVEL_CHECKS]
        if _needs_search(symbol_checks) and "search" not in symbol_checks:
            symbol_checks = ["search", *symbol_checks]
        market_checks = [check for check in normalized_checks if check in MARKET_LEVEL_CHECKS]
        effective_checks = [*symbol_checks, *market_checks]
        symbol_reports = [
            self._probe_symbol(symbol, symbol_checks, symbol_names or {})
            for symbol in normalized_symbols
        ]
        market_report = self._probe_market(market_checks)
        coverage = _coverage(symbol_reports, symbol_checks)
        coverage.update(_market_coverage(market_report, market_checks))
        return {
            "pipeline": "indianapi_probe",
            "run_at": datetime.utcnow().isoformat() + "Z",
            "symbols_requested": len(normalized_symbols),
            "checks": effective_checks,
            "passed": bool(normalized_symbols) and any(v["ok"] > 0 for v in coverage.values()),
            "coverage": coverage,
            "market_report": market_report,
            "symbol_reports": symbol_reports,
            "recommendations": _recommendations(coverage, normalized_checks),
        }

    def _probe_symbol(
        self,
        symbol: str,
        checks: Sequence[str],
        symbol_names: Mapping[str, str],
    ) -> dict[str, Any]:
        query_terms = _query_terms(symbol, [symbol_names.get(symbol, "")])
        report: dict[str, Any] = {"symbol": symbol, "query_terms": query_terms, "checks": {}}
        stock_payload: JSONPayload | None = None
        search_payload: JSONPayload | None = None
        search_candidates: list[dict[str, str]] = []
        stock_id = symbol

        if _needs_search(checks):
            search_result = _capture(lambda: self.client.industry_search(query_terms[0] if query_terms else symbol))
            report["checks"]["search"] = search_result
            if search_result["ok"]:
                search_payload = _strip_payload(search_result)
                search_candidates = _search_candidates(search_payload, symbol=symbol)
                if search_candidates:
                    best = search_candidates[0]
                    stock_id = best.get("id") or stock_id
                    query_terms = _query_terms(
                        symbol,
                        [
                            best.get("commonName", ""),
                            best.get("exchangeCodeNsi", ""),
                            best.get("exchangeCodeBse", ""),
                            best.get("nseRic", ""),
                            best.get("bseRic", ""),
                            *query_terms,
                        ],
                    )
                    report["resolved_stock_id"] = stock_id
                    report["resolved_query_terms"] = query_terms
                    report["search_candidates"] = search_candidates[:5]
            else:
                _strip_payload(search_result)

        if _needs_stock_details(checks):
            stock_result = _capture(lambda: self.client.stock_details(symbol, query_terms=query_terms))
            report["checks"]["stock"] = stock_result
            if stock_result["ok"]:
                stock_payload = _strip_payload(stock_result)
                stock_id = _extract_stock_id(stock_payload, fallback=symbol)
                stock_result["stock_id"] = stock_id
            else:
                _strip_payload(stock_result)
        elif "stock" in checks:
            stock_result = _capture(lambda: self.client.stock_details(symbol, query_terms=query_terms))
            _strip_payload(stock_result)
            report["checks"]["stock"] = stock_result

        if "financials" in checks:
            financials_result = _capture(
                lambda: self.client.historical_stats(symbol, stats="quarter_results", query_terms=query_terms)
            )
            _strip_payload(financials_result)
            report["checks"]["financials"] = financials_result
        if "shareholding" in checks:
            shareholding_result = _capture(
                lambda: self.client.historical_stats(
                    symbol,
                    stats="shareholding_pattern_quarterly",
                    query_terms=query_terms,
                )
            )
            _strip_payload(shareholding_result)
            report["checks"]["shareholding"] = shareholding_result
        if "history" in checks:
            history_result = _capture(lambda: self.client.historical_data(symbol, query_terms=query_terms))
            _strip_payload(history_result)
            report["checks"]["history"] = history_result
        if "corporate_actions" in checks:
            actions_result = _capture(lambda: self.client.corporate_actions(symbol, query_terms=query_terms))
            _strip_payload(actions_result)
            report["checks"]["corporate_actions"] = actions_result
        if "announcements" in checks:
            announcements_result = _capture(lambda: self.client.recent_announcements(symbol, query_terms=query_terms))
            _strip_payload(announcements_result)
            report["checks"]["announcements"] = announcements_result
        if "analyst" in checks:
            analyst_result = _capture(lambda: self.client.target_price(stock_id))
            _strip_payload(analyst_result)
            report["checks"]["analyst"] = analyst_result
        if "forecasts" in checks:
            forecasts_result = _capture(lambda: self.client.forecasts(stock_id))
            _strip_payload(forecasts_result)
            report["checks"]["forecasts"] = forecasts_result

        report["usable_sections"] = [
            name for name, result in report["checks"].items()
            if isinstance(result, dict) and result.get("ok")
        ]
        report["ok"] = bool(report["usable_sections"])
        return report

    def _probe_market(self, checks: Sequence[str]) -> dict[str, Any]:
        report: dict[str, Any] = {"checks": {}}
        for check in checks:
            if check == "news":
                result = _capture(self.client.news)
            elif check == "trending":
                result = _capture(self.client.trending)
            elif check == "nse_most_active":
                result = _capture(self.client.nse_most_active)
            elif check == "bse_most_active":
                result = _capture(self.client.bse_most_active)
            elif check == "price_shockers":
                result = _capture(self.client.price_shockers)
            elif check == "week_52":
                result = _capture(self.client.week_52_high_low)
            elif check == "ipo":
                result = _capture(self.client.ipo)
            else:
                continue
            _strip_payload(result)
            report["checks"][check] = result
        report["usable_sections"] = [
            name for name, result in report["checks"].items()
            if isinstance(result, dict) and result.get("ok")
        ]
        report["ok"] = bool(report["usable_sections"])
        return report


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


def _strip_payload(result: dict[str, Any]) -> JSONPayload | None:
    summary = result.get("summary")
    if not isinstance(summary, dict):
        return None
    payload = summary.pop("_payload", None)
    return payload if isinstance(payload, (dict, list)) else None


def _summarize_payload(payload: JSONPayload) -> dict[str, Any]:
    if isinstance(payload, list):
        return {
            "type": "list",
            "item_count": len(payload),
            "first_item_keys": sorted(payload[0].keys())[:25] if payload and isinstance(payload[0], dict) else [],
        }
    keys = sorted(str(key) for key in payload.keys())
    summary: dict[str, Any] = {
        "type": "dict",
        "top_level_keys": keys[:40],
        "top_level_key_count": len(keys),
    }
    if "datasets" in payload and isinstance(payload["datasets"], list):
        datasets = payload["datasets"]
        summary["dataset_count"] = len(datasets)
        summary["dataset_metrics"] = [
            str(item.get("metric") or item.get("label") or "")
            for item in datasets[:10]
            if isinstance(item, dict)
        ]
        summary["value_count"] = sum(
            len(item.get("values", []))
            for item in datasets
            if isinstance(item, dict) and isinstance(item.get("values"), list)
        )
    else:
        metric_periods = [
            len(value)
            for value in payload.values()
            if isinstance(value, dict)
        ]
        if metric_periods:
            summary["metric_count"] = len(metric_periods)
            summary["max_period_count"] = max(metric_periods)
    summary["available_sections"] = [
        key
        for key in [
            "companyProfile",
            "currentPrice",
            "financials",
            "keyMetrics",
            "analystView",
            "recosBar",
            "riskMeter",
            "shareholding",
            "stockCorporateActionData",
            "recentNews",
            "priceTarget",
            "priceTargetSnapshots",
            "recommendation",
            "recommendationSnapshots",
        ]
        if key in payload and payload.get(key) not in (None, {}, [])
    ]
    return summary


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
    if payload.get("error"):
        return str(payload.get("error"))
    if payload.get("detail"):
        return str(payload.get("detail"))
    if set(payload.keys()) == {"info"}:
        return str(payload.get("info") or "info-only response")
    return ""


def _extract_stock_id(payload: JSONPayload | None, *, fallback: str) -> str:
    if not isinstance(payload, dict):
        return fallback.upper()
    for key in ("id", "stock_id", "stockId", "tickerId", "ticker_id", "symbol", "exchangeCodeNsi"):
        value = payload.get(key)
        if value:
            return str(value).strip()
    reusable = payload.get("stockDetailsReusableData")
    if isinstance(reusable, dict):
        for key in ("id", "stock_id", "stockId", "tickerId"):
            value = reusable.get(key)
            if value:
                return str(value).strip()
    return fallback.upper()


def _search_candidates(payload: JSONPayload | None, *, symbol: str) -> list[dict[str, str]]:
    if not isinstance(payload, list):
        return []
    candidates: list[dict[str, str]] = []
    normalized_symbol = symbol.strip().upper()
    for item in payload:
        if not isinstance(item, dict):
            continue
        candidate = {
            "id": str(item.get("id") or "").strip(),
            "commonName": str(item.get("commonName") or "").strip(),
            "exchangeCodeNsi": str(item.get("exchangeCodeNsi") or "").strip().upper(),
            "exchangeCodeBse": str(item.get("exchangeCodeBse") or "").strip(),
            "nseRic": str(item.get("nseRic") or "").strip(),
            "bseRic": str(item.get("bseRic") or "").strip(),
        }
        if any(candidate.values()):
            candidates.append(candidate)
    return sorted(
        candidates,
        key=lambda row: (
            0 if row.get("exchangeCodeNsi") == normalized_symbol else 1,
            0 if normalized_symbol in row.get("commonName", "").upper() else 1,
        ),
    )


def _coverage(symbol_reports: Sequence[Mapping[str, Any]], checks: Sequence[str]) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    total = len(symbol_reports)
    for check in checks:
        ok = 0
        errors: list[str] = []
        for report in symbol_reports:
            check_result = report.get("checks", {}).get(check) if isinstance(report.get("checks"), dict) else None
            if isinstance(check_result, Mapping) and check_result.get("ok"):
                ok += 1
            elif isinstance(check_result, Mapping) and check_result.get("error"):
                errors.append(str(check_result["error"]))
        coverage[check] = {
            "ok": ok,
            "total": total,
            "coverage": round(ok / total, 4) if total else 0.0,
            "sample_errors": _dedupe(errors)[:5],
        }
    return coverage


def _market_coverage(market_report: Mapping[str, Any], checks: Sequence[str]) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    report_checks = market_report.get("checks")
    checks_map = report_checks if isinstance(report_checks, Mapping) else {}
    for check in checks:
        check_result = checks_map.get(check)
        ok = 1 if isinstance(check_result, Mapping) and check_result.get("ok") else 0
        errors = []
        if isinstance(check_result, Mapping) and check_result.get("error"):
            errors.append(str(check_result["error"]))
        coverage[check] = {
            "ok": ok,
            "total": 1,
            "coverage": float(ok),
            "sample_errors": _dedupe(errors)[:5],
        }
    return coverage


def _recommendations(coverage: Mapping[str, Mapping[str, Any]], checks: Sequence[str]) -> list[str]:
    recs: list[str] = []
    if "stock" in checks and float(coverage.get("stock", {}).get("coverage", 0.0)) < 0.8:
        recs.append("Resolve symbol/API-key/base-URL issues before using IndianAPI for canonical data.")
    for check in ("financials", "shareholding", "history"):
        if check in checks and float(coverage.get(check, {}).get("coverage", 0.0)) >= 0.8:
            recs.append(f"{check} coverage looks high enough for a small reconciliation test.")
    if "analyst" in checks and float(coverage.get("analyst", {}).get("coverage", 0.0)) > 0.0:
        recs.append("Treat analyst targets/recommendations as medium-trust factors, not direct trading signals.")
    if not recs:
        recs.append("IndianAPI did not return enough usable payloads in this probe; check API key and endpoint base URLs.")
    return recs


def _normalize_checks(checks: Sequence[str]) -> list[str]:
    out: list[str] = []
    for raw in checks:
        value = raw.strip().lower()
        if not value:
            continue
        if value == "all":
            return [
                "stock",
                "search",
                "financials",
                "shareholding",
                "analyst",
                "forecasts",
                "history",
                "corporate_actions",
                "announcements",
                "news",
                "trending",
                "nse_most_active",
                "bse_most_active",
                "price_shockers",
                "week_52",
                "ipo",
            ]
        if value not in VALID_CHECKS:
            raise ValueError(f"unsupported IndianAPI check '{raw}'")
        if value not in out:
            out.append(value)
    return out or ["stock"]


def _query_terms(symbol: str, query_terms: Sequence[str] | None = None) -> list[str]:
    terms: list[str] = []
    for term in list(query_terms or []) + [symbol, symbol.replace("-", " "), symbol.title()]:
        cleaned = " ".join(str(term or "").strip().split())
        if not cleaned:
            continue
        if cleaned.upper() == "HDFCBANK":
            cleaned = "HDFC Bank"
        elif cleaned.upper() == "ICICIBANK":
            cleaned = "ICICI Bank"
        elif cleaned.upper() == "RELIANCE":
            cleaned = "Reliance"
        elif cleaned.upper() == "BHARTIARTL":
            cleaned = "Bharti Airtel"
        if cleaned not in terms:
            terms.append(cleaned)
    return terms


def _needs_search(checks: Sequence[str]) -> bool:
    return any(check not in MARKET_LEVEL_CHECKS for check in checks)


def _needs_stock_details(checks: Sequence[str]) -> bool:
    return any(check in checks for check in ("stock", "analyst", "forecasts"))


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


def _clean_base_url(value: str) -> str:
    cleaned = str(value or "").strip().rstrip("/")
    return cleaned or DEFAULT_STOCK_BASE_URL


def _url_for(base_url: str, path: str) -> str:
    return f"{base_url}{path if path.startswith('/') else '/' + path}"


def _safe_error(exc: Exception | None) -> str:
    if exc is None:
        return "unknown error"
    text = str(exc)
    return text[:500]


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
