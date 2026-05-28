"""FinEdge client and coverage probe.

The probe is diagnostic only. It tests endpoint coverage, summarizes response
shape, and writes quality reports through the app layer; it does not ingest
canonical factors or raw vendor data.
"""

from __future__ import annotations

import gzip
import json
import logging
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

DEFAULT_BASE_URL = "https://data.finedgeapi.com"
LOGGER = logging.getLogger(__name__)

VALID_CHECKS = {
    "stock_symbols",
    "company_profile",
    "financials",
    "segment_revenue",
    "notes",
    "ratios",
    "financial_metrics",
    "basic_financials",
    "quote",
    "daily_quotes",
    "daily_price_ratios",
    "annual_price_ratios",
    "shareholding_pattern",
    "shareholding_summary",
    "ownership_current",
    "ownership_history",
    "beneficial_owners",
    "declarations",
    "corporate_actions",
    "dividends",
    "announcements",
    "credit_ratings",
    "investor_presentations",
    "investor_call_transcripts",
    "results_calendar",
    "ipo_calendar",
    "index_master",
    "index_market_history",
    "index_valuation_history",
    "health",
}
MARKET_LEVEL_CHECKS = {
    "stock_symbols",
    "results_calendar",
    "ipo_calendar",
    "index_master",
    "index_market_history",
    "index_valuation_history",
    "health",
}
DEFAULT_CHECKS = [
    "stock_symbols",
    "company_profile",
    "financials",
    "ratios",
    "daily_quotes",
    "shareholding_pattern",
    "corporate_actions",
]


@dataclass(frozen=True)
class FinEdgeRequest:
    name: str
    path: str
    params: dict[str, Any]


class FinEdgeClient:
    """Small JSON client for documented FinEdge REST endpoints."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        timeout_seconds: int = 8,
        retries: int = 0,
        retry_delay_seconds: float = 0.5,
        fetch_fn: FetchFn | None = None,
    ) -> None:
        self.base_url = _clean_base_url(base_url)
        self.api_key = api_key or ""
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.retries = max(0, int(retries))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        self._fetch_fn = fetch_fn

    def stock_symbols(self) -> JSONPayload:
        return self._get(FinEdgeRequest("stock_symbols", "/api/v1/stock-symbols", {}))

    def company_profile(self, symbol: str) -> JSONPayload:
        return self._symbol_endpoint("company_profile", f"/api/v1/company-profile/{symbol}")

    def stock_search(self, *, group: str = "industry", value: str = "Cigarettes") -> JSONPayload:
        return self._get(FinEdgeRequest("stock_search", "/api/v1/stock-search", {"group": group, "value": value}))

    def financials(self, symbol: str, *, statement_type: str, statement_code: str, period: str) -> JSONPayload:
        return self._statement_endpoint(
            "financials",
            f"/api/v1/financials/{symbol}",
            statement_type=statement_type,
            statement_code=statement_code,
            period=period,
        )

    def segment_revenue(self, symbol: str, *, statement_type: str, statement_code: str, period: str) -> JSONPayload:
        return self._statement_endpoint(
            "segment_revenue",
            f"/api/v1/segment-revenue/{symbol}",
            statement_type=statement_type,
            statement_code=statement_code,
            period=period,
        )

    def notes(self, symbol: str, *, statement_type: str, period: str) -> JSONPayload:
        return self._get(
            FinEdgeRequest(
                "notes",
                f"/api/v1/notes/{symbol}",
                {"statement_type": statement_type, "period": period},
            )
        )

    def ratios(self, symbol: str, *, statement_type: str, ratio_type: str) -> JSONPayload:
        return self._ratio_endpoint("ratios", f"/api/v1/ratios/{symbol}", statement_type, ratio_type)

    def financial_metrics(self, symbol: str, *, statement_type: str, ratio_type: str) -> JSONPayload:
        return self._ratio_endpoint(
            "financial_metrics",
            f"/api/v1/financial-metrics/{symbol}",
            statement_type,
            ratio_type,
        )

    def basic_financials(self, symbol: str, *, statement_type: str, statement_code: str) -> JSONPayload:
        return self._get(
            FinEdgeRequest(
                "basic_financials",
                f"/api/v1/basic-financials/{symbol}",
                {"statement_type": statement_type, "statement_code": statement_code},
            )
        )

    def quote(self, symbols: Sequence[str]) -> JSONPayload:
        return self._get(FinEdgeRequest("quote", "/api/v1/quote", {"symbol": [s for s in symbols if s]}))

    def daily_quotes(self, symbol: str, *, from_date: int | None = None, to_date: int | None = None) -> JSONPayload:
        params: dict[str, Any] = {}
        if from_date is not None:
            params["from"] = from_date
        if to_date is not None:
            params["to"] = to_date
        return self._get(FinEdgeRequest("daily_quotes", f"/api/v1/daily-quotes/{symbol}", params))

    def daily_price_ratios(
        self,
        symbol: str,
        *,
        statement_type: str,
        from_date: int | None = None,
        to_date: int | None = None,
    ) -> JSONPayload:
        params: dict[str, Any] = {"statement_type": statement_type}
        if from_date is not None:
            params["from"] = from_date
        if to_date is not None:
            params["to"] = to_date
        return self._get(FinEdgeRequest("daily_price_ratios", f"/api/v1/daily-price-ratios/{symbol}", params))

    def annual_price_ratios(self, symbol: str, *, statement_type: str) -> JSONPayload:
        return self._get(
            FinEdgeRequest(
                "annual_price_ratios",
                f"/api/v1/annual-price-ratios/{symbol}",
                {"statement_type": statement_type},
            )
        )

    def shareholding_pattern(self, symbol: str, *, period: str) -> JSONPayload:
        return self._shareholding_endpoint("shareholding_pattern", f"/api/v1/shareholdings/pattern/{symbol}", period)

    def shareholding_summary(self, symbol: str, *, period: str) -> JSONPayload:
        return self._shareholding_endpoint("shareholding_summary", f"/api/v1/shareholdings/summary/{symbol}", period)

    def ownership_current(self, symbol: str) -> JSONPayload:
        return self._symbol_endpoint("ownership_current", f"/api/v1/shareholdings/ownership-current/{symbol}")

    def ownership_history(self, symbol: str, *, period: str) -> JSONPayload:
        return self._shareholding_endpoint(
            "ownership_history",
            f"/api/v1/shareholdings/ownership-history/{symbol}",
            period,
        )

    def beneficial_owners(self, symbol: str, *, period: str) -> JSONPayload:
        return self._shareholding_endpoint(
            "beneficial_owners",
            f"/api/v1/shareholdings/beneficial-owners/{symbol}",
            period,
        )

    def declarations(self, symbol: str, *, period: str) -> JSONPayload:
        return self._shareholding_endpoint("declarations", f"/api/v1/shareholdings/declaration/{symbol}", period)

    def corporate_actions(
        self,
        *,
        symbol: str | None = None,
        action: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> JSONPayload:
        return self._get(
            FinEdgeRequest(
                "corporate_actions",
                "/api/v1/corporate-actions/all",
                {"symbol": symbol, "action": action, "from_date": from_date, "to_date": to_date},
            )
        )

    def dividends(self, symbol: str) -> JSONPayload:
        return self._symbol_endpoint("dividends", f"/api/v1/dividend/{symbol}")

    def announcements(self, *, symbol: str | None = None, from_date: str | None = None, to_date: str | None = None) -> JSONPayload:
        return self._event_endpoint("announcements", "/api/v1/corp-announcements", symbol, from_date, to_date)

    def credit_ratings(self, *, symbol: str | None = None, from_date: str | None = None, to_date: str | None = None) -> JSONPayload:
        return self._event_endpoint("credit_ratings", "/api/v1/credit-ratings", symbol, from_date, to_date)

    def investor_presentations(
        self,
        *,
        symbol: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> JSONPayload:
        return self._event_endpoint("investor_presentations", "/api/v1/investor-presentations", symbol, from_date, to_date)

    def investor_call_transcripts(
        self,
        *,
        symbol: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> JSONPayload:
        return self._event_endpoint("investor_call_transcripts", "/api/v1/investor-call-transcripts", symbol, from_date, to_date)

    def results_calendar(self, *, from_date: str | None = None, to_date: str | None = None) -> JSONPayload:
        return self._get(
            FinEdgeRequest("results_calendar", "/api/v1/results-calendar", {"from_date": from_date, "to_date": to_date})
        )

    def ipo_calendar(self, *, from_date: str | None = None, to_date: str | None = None) -> JSONPayload:
        return self._get(FinEdgeRequest("ipo_calendar", "/api/v1/ipo-calendar", {"from_date": from_date, "to_date": to_date}))

    def index_master(self) -> JSONPayload:
        return self._get(FinEdgeRequest("index_master", "/api/v1/index/master", {}))

    def index_market_history(self, *, index_symbol: str, from_date: str, to_date: str) -> JSONPayload:
        return self._get(
            FinEdgeRequest(
                "index_market_history",
                "/api/v1/index/market-price/historical",
                {"index_symbol": index_symbol, "from_date": from_date, "to_date": to_date},
            )
        )

    def index_valuation_history(self, *, index_symbol: str, from_date: str, to_date: str) -> JSONPayload:
        return self._get(
            FinEdgeRequest(
                "index_valuation_history",
                "/api/v1/index/valuation/historical",
                {"index_symbol": index_symbol, "from_date": from_date, "to_date": to_date},
            )
        )

    def health(self) -> JSONPayload:
        return self._get(FinEdgeRequest("health", "/healthcheck", {}))

    def _symbol_endpoint(self, name: str, path: str) -> JSONPayload:
        return self._get(FinEdgeRequest(name, path, {}))

    def _statement_endpoint(
        self,
        name: str,
        path: str,
        *,
        statement_type: str,
        statement_code: str,
        period: str,
    ) -> JSONPayload:
        return self._get(
            FinEdgeRequest(
                name,
                path,
                {"statement_type": statement_type, "statement_code": statement_code, "period": period},
            )
        )

    def _ratio_endpoint(self, name: str, path: str, statement_type: str, ratio_type: str) -> JSONPayload:
        return self._get(
            FinEdgeRequest(name, path, {"statement_type": statement_type, "ratio_type": ratio_type})
        )

    def _shareholding_endpoint(self, name: str, path: str, period: str) -> JSONPayload:
        return self._get(FinEdgeRequest(name, path, {"period": period}))

    def _event_endpoint(
        self,
        name: str,
        path: str,
        symbol: str | None,
        from_date: str | None,
        to_date: str | None,
    ) -> JSONPayload:
        return self._get(
            FinEdgeRequest(name, path, {"symbol": symbol, "from_date": from_date, "to_date": to_date})
        )

    def _get(self, request: FinEdgeRequest) -> JSONPayload:
        params = {key: value for key, value in request.params.items() if value is not None and str(value) != ""}
        if self.api_key:
            params["token"] = self.api_key
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": "stock-screener-engine/0.1 FinEdge probe",
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
        query = urlencode(params, doseq=True)
        full_url = f"{url}?{query}" if query else url
        req = Request(full_url, headers=dict(headers))
        with urlopen(req, timeout=self.timeout_seconds) as resp:  # noqa: S310 - explicit user-configured API endpoint.
            raw = _decode_response(resp.read(), resp.headers.get("Content-Encoding", ""))
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, (dict, list)):
            raise json.JSONDecodeError("JSON payload is not an object or list", raw.decode("utf-8"), 0)
        return parsed


class FinEdgeProbe:
    """Probe FinEdge endpoint coverage for a symbol set."""

    def __init__(
        self,
        client: FinEdgeClient,
        *,
        statement_type: str = "s",
        statement_code: str = "pl",
        period: str = "annual",
        ratio_type: str = "pr",
        metrics_ratio_type: str = "gr",
        shareholding_period: str = "quarterly",
        from_date: str | None = None,
        to_date: str | None = None,
        index_symbol: str = "NIFTY 50",
    ) -> None:
        self.client = client
        self.statement_type = statement_type
        self.statement_code = statement_code
        self.period = period
        self.ratio_type = ratio_type
        self.metrics_ratio_type = metrics_ratio_type
        self.shareholding_period = shareholding_period
        self.from_date = from_date
        self.to_date = to_date
        self.index_symbol = index_symbol
        self._halt_reason = ""

    def run(self, symbols: Sequence[str], checks: Sequence[str]) -> dict[str, Any]:
        normalized_symbols = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
        normalized_checks = normalize_finedge_checks(checks)
        symbol_checks = [check for check in normalized_checks if check not in MARKET_LEVEL_CHECKS]
        market_checks = [check for check in normalized_checks if check in MARKET_LEVEL_CHECKS]
        market_report = self._probe_market(market_checks)
        symbol_reports = []
        for symbol in normalized_symbols:
            if self._halt_reason:
                symbol_reports.append(_skipped_symbol_report(symbol, symbol_checks, self._halt_reason))
                continue
            LOGGER.info("FinEdge probe starting for %s (%d checks)", symbol, len(symbol_checks))
            symbol_reports.append(self._probe_symbol(symbol, symbol_checks))
        coverage = _coverage(symbol_reports, symbol_checks)
        coverage.update(_market_coverage(market_report, market_checks))
        return {
            "pipeline": "finedge_probe",
            "run_at": datetime.utcnow().isoformat() + "Z",
            "symbols_requested": len(normalized_symbols),
            "checks": normalized_checks,
            "passed": any(v["ok"] > 0 for v in coverage.values()),
            "coverage": coverage,
            "market_report": market_report,
            "symbol_reports": symbol_reports,
            "recommendations": _recommendations(coverage, normalized_checks),
        }

    def _probe_symbol(self, symbol: str, checks: Sequence[str]) -> dict[str, Any]:
        report: dict[str, Any] = {"symbol": symbol, "checks": {}}
        for check in checks:
            if self._halt_reason:
                report["checks"][check] = _skipped_result(self._halt_reason)
                continue
            LOGGER.info("FinEdge probe %s: %s", symbol, check)
            result = self._probe_endpoint(check, symbol)
            report["checks"][check] = result
        report["usable_sections"] = [
            name for name, result in report["checks"].items()
            if isinstance(result, dict) and result.get("ok")
        ]
        report["ok"] = bool(report["usable_sections"])
        return report

    def _probe_endpoint(self, check: str, symbol: str) -> dict[str, Any]:
        if check == "company_profile":
            return self._capture_check(check, lambda: self.client.company_profile(symbol))
        if check == "financials":
            return self._capture_check(
                check,
                lambda: self.client.financials(
                    symbol,
                    statement_type=self.statement_type,
                    statement_code=self.statement_code,
                    period=self.period,
                ),
            )
        if check == "segment_revenue":
            return self._capture_check(
                check,
                lambda: self.client.segment_revenue(
                    symbol,
                    statement_type=self.statement_type,
                    statement_code=self.statement_code,
                    period=self.period,
                ),
            )
        if check == "notes":
            return self._capture_check(
                check,
                lambda: self.client.notes(symbol, statement_type=self.statement_type, period=self.period),
            )
        if check == "ratios":
            return self._capture_check(
                check,
                lambda: self.client.ratios(symbol, statement_type=self.statement_type, ratio_type=self.ratio_type),
            )
        if check == "financial_metrics":
            return self._capture_check(
                check,
                lambda: self.client.financial_metrics(
                    symbol,
                    statement_type=self.statement_type,
                    ratio_type=self.metrics_ratio_type,
                ),
            )
        if check == "basic_financials":
            return self._capture_check(
                check,
                lambda: self.client.basic_financials(
                    symbol,
                    statement_type=self.statement_type,
                    statement_code=self.statement_code,
                ),
            )
        if check == "quote":
            return self._capture_check(check, lambda: self.client.quote([symbol]))
        if check == "daily_quotes":
            return self._capture_check(check, lambda: self.client.daily_quotes(symbol))
        if check == "daily_price_ratios":
            return self._capture_check(
                check,
                lambda: self.client.daily_price_ratios(symbol, statement_type=self.statement_type),
            )
        if check == "annual_price_ratios":
            return self._capture_check(
                check,
                lambda: self.client.annual_price_ratios(symbol, statement_type=self.statement_type),
            )
        if check == "shareholding_pattern":
            return self._capture_check(
                check,
                lambda: self.client.shareholding_pattern(symbol, period=self.shareholding_period),
            )
        if check == "shareholding_summary":
            return self._capture_check(
                check,
                lambda: self.client.shareholding_summary(symbol, period=self.shareholding_period),
            )
        if check == "ownership_current":
            return self._capture_check(check, lambda: self.client.ownership_current(symbol))
        if check == "ownership_history":
            return self._capture_check(
                check,
                lambda: self.client.ownership_history(symbol, period=self.shareholding_period),
            )
        if check == "beneficial_owners":
            return self._capture_check(
                check,
                lambda: self.client.beneficial_owners(symbol, period=self.shareholding_period),
            )
        if check == "declarations":
            return self._capture_check(
                check,
                lambda: self.client.declarations(symbol, period=self.shareholding_period),
            )
        if check == "corporate_actions":
            return self._capture_check(
                check,
                lambda: self.client.corporate_actions(
                    symbol=symbol,
                    from_date=self.from_date,
                    to_date=self.to_date,
                ),
            )
        if check == "dividends":
            return self._capture_check(check, lambda: self.client.dividends(symbol))
        if check == "announcements":
            return self._capture_check(
                check,
                lambda: self.client.announcements(symbol=symbol, from_date=self.from_date, to_date=self.to_date),
            )
        if check == "credit_ratings":
            return self._capture_check(
                check,
                lambda: self.client.credit_ratings(symbol=symbol, from_date=self.from_date, to_date=self.to_date),
            )
        if check == "investor_presentations":
            return self._capture_check(
                check,
                lambda: self.client.investor_presentations(symbol=symbol, from_date=self.from_date, to_date=self.to_date),
            )
        if check == "investor_call_transcripts":
            return self._capture_check(
                check,
                lambda: self.client.investor_call_transcripts(symbol=symbol, from_date=self.from_date, to_date=self.to_date),
            )
        return {"ok": False, "error": f"unsupported FinEdge check '{check}'", "summary": {}}

    def _probe_market(self, checks: Sequence[str]) -> dict[str, Any]:
        report: dict[str, Any] = {"checks": {}}
        for check in checks:
            if self._halt_reason:
                report["checks"][check] = _skipped_result(self._halt_reason)
                continue
            LOGGER.info("FinEdge probe market: %s", check)
            if check == "stock_symbols":
                result = self._capture_check(check, self.client.stock_symbols)
            elif check == "results_calendar":
                result = self._capture_check(
                    check,
                    lambda: self.client.results_calendar(from_date=self.from_date, to_date=self.to_date),
                )
            elif check == "ipo_calendar":
                result = self._capture_check(
                    check,
                    lambda: self.client.ipo_calendar(from_date=self.from_date, to_date=self.to_date),
                )
            elif check == "index_master":
                result = self._capture_check(check, self.client.index_master)
            elif check == "index_market_history":
                result = self._capture_check(
                    check,
                    lambda: self.client.index_market_history(
                        index_symbol=self.index_symbol,
                        from_date=self.from_date or "2026-01-01",
                        to_date=self.to_date or "2026-05-28",
                    ),
                )
            elif check == "index_valuation_history":
                result = self._capture_check(
                    check,
                    lambda: self.client.index_valuation_history(
                        index_symbol=self.index_symbol,
                        from_date=self.from_date or "2026-01-01",
                        to_date=self.to_date or "2026-05-28",
                    ),
                )
            elif check == "health":
                result = self._capture_check(check, self.client.health)
            else:
                continue
            report["checks"][check] = result
        report["usable_sections"] = [
            name for name, result in report["checks"].items()
            if isinstance(result, dict) and result.get("ok")
        ]
        report["ok"] = bool(report["usable_sections"])
        return report

    def _capture_check(self, check: str, fetch: Callable[[], JSONPayload]) -> dict[str, Any]:
        result = _capture(fetch)
        _strip_payload(result)
        if not result["ok"]:
            error = str(result.get("error", ""))
            if _is_rate_limit_error(error):
                self._halt_reason = "FinEdge rate limit reached; stop this probe and retry later with fewer symbols/checks."
            elif _is_terminal_error(error):
                result["terminal_error"] = "auth_or_plan"
        return result


class FinEdgeSchemaInspector(FinEdgeProbe):
    """Inspect FinEdge payload shapes without retaining raw vendor records."""

    def __init__(
        self,
        client: FinEdgeClient,
        *,
        statement_type: str = "s",
        statement_code: str = "pl",
        period: str = "annual",
        ratio_type: str = "pr",
        metrics_ratio_type: str = "gr",
        shareholding_period: str = "quarterly",
        from_date: str | None = None,
        to_date: str | None = None,
        index_symbol: str = "NIFTY 50",
        max_depth: int = 4,
        max_fields: int = 80,
        max_list_items: int = 25,
    ) -> None:
        super().__init__(
            client,
            statement_type=statement_type,
            statement_code=statement_code,
            period=period,
            ratio_type=ratio_type,
            metrics_ratio_type=metrics_ratio_type,
            shareholding_period=shareholding_period,
            from_date=from_date,
            to_date=to_date,
            index_symbol=index_symbol,
        )
        self.max_depth = max(1, int(max_depth))
        self.max_fields = max(1, int(max_fields))
        self.max_list_items = max(1, int(max_list_items))

    def run(self, symbols: Sequence[str], checks: Sequence[str]) -> dict[str, Any]:
        report = super().run(symbols=symbols, checks=checks)
        report["pipeline"] = "finedge_schema_inspection"
        report["sanitization"] = {
            "raw_payload_persisted": False,
            "value_samples_persisted": False,
            "summary": "Only field names, types, counts, nesting, and error strings are retained.",
        }
        report["schema_index"] = _schema_index(report)
        report["recommendations"] = _schema_recommendations(report)
        report["schema_limits"] = {
            "max_depth": self.max_depth,
            "max_fields": self.max_fields,
            "max_list_items": self.max_list_items,
        }
        return report

    def _capture_check(self, check: str, fetch: Callable[[], JSONPayload]) -> dict[str, Any]:
        result = _capture_schema(
            fetch,
            max_depth=self.max_depth,
            max_fields=self.max_fields,
            max_list_items=self.max_list_items,
        )
        if not result["ok"]:
            error = str(result.get("error", ""))
            if _is_rate_limit_error(error):
                self._halt_reason = "FinEdge rate limit reached; stop this inspection and retry later with fewer symbols/checks."
            elif _is_terminal_error(error):
                result["terminal_error"] = "auth_or_plan"
        return result


def normalize_finedge_checks(checks: Sequence[str]) -> list[str]:
    out: list[str] = []
    for raw in checks:
        value = raw.strip().lower().replace("-", "_")
        if not value:
            continue
        if value == "all":
            values = list(DEFAULT_CHECKS)
        elif value == "smoke":
            values = ["stock_symbols", "company_profile"]
        elif value == "fundamentals":
            values = ["financials", "ratios", "financial_metrics", "basic_financials"]
        elif value == "ownership":
            values = ["shareholding_pattern", "shareholding_summary", "ownership_current", "ownership_history"]
        elif value == "events":
            values = ["corporate_actions", "announcements", "credit_ratings", "investor_presentations", "investor_call_transcripts"]
        elif value == "prices":
            values = ["quote", "daily_quotes", "daily_price_ratios", "annual_price_ratios"]
        else:
            if value not in VALID_CHECKS:
                raise ValueError(f"unsupported FinEdge check '{raw}'")
            values = [value]
        for item in values:
            if item not in out:
                out.append(item)
    return out or list(DEFAULT_CHECKS)


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
        return {"ok": False, "error": _safe_error(exc), "summary": {}}


def _capture_schema(
    fetch: Callable[[], JSONPayload],
    *,
    max_depth: int,
    max_fields: int,
    max_list_items: int,
) -> dict[str, Any]:
    try:
        payload = fetch()
        payload_error = _payload_error(payload)
        ok = _payload_has_data(payload)
        return {
            "ok": ok,
            "error": "" if ok else payload_error,
            "summary": _inspect_payload(
                payload,
                max_depth=max_depth,
                max_fields=max_fields,
                max_list_items=max_list_items,
            ),
        }
    except Exception as exc:  # noqa: BLE001 - inspection must continue across endpoints/symbols.
        return {"ok": False, "error": _safe_error(exc), "summary": {}}


def _strip_payload(result: dict[str, Any]) -> JSONPayload | None:
    summary = result.get("summary")
    if not isinstance(summary, dict):
        return None
    payload = summary.pop("_payload", None)
    return payload if isinstance(payload, (dict, list)) else None


def _summarize_payload(payload: JSONPayload) -> dict[str, Any]:
    if isinstance(payload, list):
        first = payload[0] if payload and isinstance(payload[0], dict) else {}
        return {
            "type": "list",
            "item_count": len(payload),
            "first_item_keys": sorted(first.keys())[:40] if isinstance(first, dict) else [],
        }
    keys = sorted(str(key) for key in payload.keys())
    summary: dict[str, Any] = {"type": "dict", "top_level_keys": keys[:50], "top_level_key_count": len(keys)}
    for list_key in ("symbols", "financials", "ratios", "price", "ownerships", "summary", "data"):
        value = payload.get(list_key)
        if isinstance(value, list):
            summary[f"{list_key}_count"] = len(value)
            if value and isinstance(value[0], dict):
                summary[f"{list_key}_first_item_keys"] = sorted(value[0].keys())[:30]
    return summary


def _inspect_payload(
    payload: JSONPayload,
    *,
    max_depth: int,
    max_fields: int,
    max_list_items: int,
) -> dict[str, Any]:
    record_sets = _find_record_sets(
        payload,
        path="$",
        max_depth=max_depth,
        max_fields=max_fields,
        max_list_items=max_list_items,
    )
    primary = _primary_record_set(record_sets)
    summary: dict[str, Any] = {
        "root_type": _schema_type(payload),
        "shape": _shape(
            payload,
            max_depth=max_depth,
            max_fields=max_fields,
            max_list_items=max_list_items,
        ),
        "record_set_count": len(record_sets),
        "record_sets": record_sets,
    }
    if isinstance(payload, dict):
        keys = sorted(str(key) for key in payload.keys())
        summary["top_level_key_count"] = len(keys)
        summary["top_level_keys"] = keys[:max_fields]
    elif isinstance(payload, list):
        summary["item_count"] = len(payload)
    if primary:
        summary["primary_record_set"] = {
            "path": primary.get("path"),
            "item_count": primary.get("item_count"),
            "field_count": primary.get("field_count"),
            "fields": primary.get("fields", [])[:max_fields],
            "date_like_fields": primary.get("date_like_fields", []),
            "numeric_like_fields": primary.get("numeric_like_fields", []),
        }
    return summary


def _shape(value: Any, *, max_depth: int, max_fields: int, max_list_items: int) -> dict[str, Any]:
    if max_depth <= 0:
        return {"type": _schema_type(value), "truncated": True}
    if isinstance(value, dict):
        keys = sorted(str(key) for key in value.keys())
        fields = {}
        for key in keys[:max_fields]:
            fields[key] = _shape(
                value.get(key),
                max_depth=max_depth - 1,
                max_fields=max_fields,
                max_list_items=max_list_items,
            )
        return {
            "type": "dict",
            "key_count": len(keys),
            "keys": keys[:max_fields],
            "fields": fields,
            "truncated": len(keys) > max_fields,
        }
    if isinstance(value, list):
        sampled = value[:max_list_items]
        item_types = sorted({_schema_type(item) for item in sampled})
        first_shapable = next((item for item in sampled if isinstance(item, (dict, list))), None)
        shape: dict[str, Any] = {
            "type": "list",
            "item_count": len(value),
            "sampled_item_count": len(sampled),
            "item_types": item_types,
            "truncated": len(value) > max_list_items,
        }
        if first_shapable is not None:
            shape["item_shape"] = _shape(
                first_shapable,
                max_depth=max_depth - 1,
                max_fields=max_fields,
                max_list_items=max_list_items,
            )
        return shape
    return {"type": _schema_type(value)}


def _find_record_sets(
    value: Any,
    *,
    path: str,
    max_depth: int,
    max_fields: int,
    max_list_items: int,
) -> list[dict[str, Any]]:
    if max_depth <= 0:
        return []
    if isinstance(value, list):
        record_sets = []
        if any(isinstance(item, dict) for item in value):
            record_sets.append(_summarize_records(path, value, max_fields=max_fields, max_list_items=max_list_items))
        for index, item in enumerate(value[:max_list_items]):
            if isinstance(item, (dict, list)):
                record_sets.extend(
                    _find_record_sets(
                        item,
                        path=f"{path}[{index}]",
                        max_depth=max_depth - 1,
                        max_fields=max_fields,
                        max_list_items=max_list_items,
                    )
                )
        return _dedupe_record_sets(record_sets)
    if isinstance(value, dict):
        record_sets = []
        for key in sorted(str(key) for key in value.keys())[:max_fields]:
            child = value.get(key)
            if isinstance(child, (dict, list)):
                record_sets.extend(
                    _find_record_sets(
                        child,
                        path=f"{path}.{key}",
                        max_depth=max_depth - 1,
                        max_fields=max_fields,
                        max_list_items=max_list_items,
                    )
                )
        return _dedupe_record_sets(record_sets)
    return []


def _summarize_records(
    path: str,
    records: Sequence[Any],
    *,
    max_fields: int,
    max_list_items: int,
) -> dict[str, Any]:
    sampled = [record for record in records[:max_list_items] if isinstance(record, dict)]
    field_names = sorted({str(key) for record in sampled for key in record.keys()})
    profiles = {
        field: _profile_field(field, sampled, total_records=len(records))
        for field in field_names[:max_fields]
    }
    numeric_like_fields = [
        field for field, profile in profiles.items()
        if set(profile.get("types", [])) <= {"int", "float", "null"}
    ]
    date_like_fields = [field for field in field_names[:max_fields] if _is_date_like_field(field)]
    return {
        "path": path,
        "item_count": len(records),
        "sampled_item_count": len(sampled),
        "field_count": len(field_names),
        "fields": field_names[:max_fields],
        "truncated": len(field_names) > max_fields,
        "field_profiles": profiles,
        "date_like_fields": date_like_fields,
        "numeric_like_fields": numeric_like_fields,
    }


def _profile_field(field: str, sampled: Sequence[Mapping[str, Any]], *, total_records: int) -> dict[str, Any]:
    present = 0
    nulls = 0
    types = set()
    for record in sampled:
        if field not in record:
            continue
        present += 1
        value = record[field]
        schema_type = _schema_type(value)
        types.add(schema_type)
        if value is None:
            nulls += 1
    return {
        "types": sorted(types),
        "present_count": present,
        "sampled_missing_count": max(0, len(sampled) - present),
        "total_records": total_records,
        "null_count": nulls,
        "date_like_name": _is_date_like_field(field),
    }


def _dedupe_record_sets(record_sets: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for record_set in record_sets:
        path = str(record_set.get("path", ""))
        if path in seen:
            continue
        seen.add(path)
        deduped.append(record_set)
    return deduped


def _primary_record_set(record_sets: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not record_sets:
        return None
    return max(
        record_sets,
        key=lambda record_set: (
            int(record_set.get("item_count", 0) or 0),
            int(record_set.get("field_count", 0) or 0),
        ),
    )


def _schema_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, list):
        return "list"
    return type(value).__name__


def _is_date_like_field(field: str) -> bool:
    normalized = field.lower().replace("-", "_").replace(" ", "_")
    tokens = ("date", "period", "year", "quarter", "fiscal", "fy", "as_of", "asof", "month")
    return any(token in normalized for token in tokens)


def _schema_index(report: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def visit(check: str, result: Any) -> None:
        if not isinstance(result, Mapping) or not result.get("ok"):
            return
        summary = result.get("summary")
        if not isinstance(summary, Mapping):
            return
        entry = checks.setdefault(
            check,
            {
                "observations": 0,
                "record_paths": set(),
                "field_names": set(),
                "date_like_fields": set(),
                "numeric_like_fields": set(),
            },
        )
        entry["observations"] += 1
        for record_set in summary.get("record_sets", []):
            if not isinstance(record_set, Mapping):
                continue
            if record_set.get("path"):
                entry["record_paths"].add(str(record_set["path"]))
            entry["field_names"].update(str(field) for field in record_set.get("fields", []))
            entry["date_like_fields"].update(str(field) for field in record_set.get("date_like_fields", []))
            entry["numeric_like_fields"].update(str(field) for field in record_set.get("numeric_like_fields", []))

    market_report = report.get("market_report")
    if isinstance(market_report, Mapping):
        market_checks = market_report.get("checks")
        if isinstance(market_checks, Mapping):
            for check, result in market_checks.items():
                visit(str(check), result)

    for symbol_report in report.get("symbol_reports", []):
        if not isinstance(symbol_report, Mapping):
            continue
        symbol_checks = symbol_report.get("checks")
        if not isinstance(symbol_checks, Mapping):
            continue
        for check, result in symbol_checks.items():
            visit(str(check), result)

    return {
        check: {
            "observations": entry["observations"],
            "record_paths": sorted(entry["record_paths"])[:40],
            "field_count": len(entry["field_names"]),
            "field_names": sorted(entry["field_names"])[:160],
            "date_like_fields": sorted(entry["date_like_fields"])[:80],
            "numeric_like_fields": sorted(entry["numeric_like_fields"])[:80],
        }
        for check, entry in sorted(checks.items())
    }


def _schema_recommendations(report: Mapping[str, Any]) -> list[str]:
    index = report.get("schema_index")
    if not isinstance(index, Mapping) or not index:
        return ["No usable FinEdge schema was captured; check token, endpoint parameters, plan, and symbol coverage."]
    recs = [
        "Use this schema report to build explicit FinEdge-to-factor field maps before canonical ingestion.",
        "Reconcile period/date fields and units against source filings before using values in scores.",
    ]
    if "financials" in index:
        recs.append("Financial statement schema is available; next map P&L, balance sheet, and cash-flow aliases separately.")
    if "shareholding_pattern" in index or "ownership_current" in index:
        recs.append("Ownership schema is available; next map promoter/FII/DII/public and pledge-related fields.")
    if "daily_quotes" in index:
        recs.append("Daily quote schema is available; compare adjustment conventions against canonical OHLCV bars.")
    return recs


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
    for key in ("error", "message", "detail", "Error Message"):
        value = payload.get(key)
        if value:
            return str(value)
    if set(payload.keys()) == {"info"}:
        return str(payload.get("info") or "info-only response")
    return ""


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
    checks_map = market_report.get("checks") if isinstance(market_report.get("checks"), Mapping) else {}
    for check in checks:
        check_result = checks_map.get(check) if isinstance(checks_map, Mapping) else None
        ok = 1 if isinstance(check_result, Mapping) and check_result.get("ok") else 0
        errors = []
        if isinstance(check_result, Mapping) and check_result.get("error"):
            errors.append(str(check_result["error"]))
        coverage[check] = {"ok": ok, "total": 1, "coverage": float(ok), "sample_errors": _dedupe(errors)[:5]}
    return coverage


def _recommendations(coverage: Mapping[str, Mapping[str, Any]], checks: Sequence[str]) -> list[str]:
    recs: list[str] = []
    if "financials" in checks and float(coverage.get("financials", {}).get("coverage", 0.0)) >= 0.8:
        recs.append("FinEdge financials look strong enough for a small point-in-time reconciliation test.")
    if "shareholding_pattern" in checks and float(coverage.get("shareholding_pattern", {}).get("coverage", 0.0)) > 0.0:
        recs.append("FinEdge shareholding payloads are promising for ownership/governance factors.")
    if "daily_quotes" in checks and float(coverage.get("daily_quotes", {}).get("coverage", 0.0)) > 0.0:
        recs.append("Compare FinEdge adjusted daily quotes against canonical broker/yfinance bars before ingesting.")
    if not recs:
        recs.append("FinEdge did not return enough usable payloads in this probe; check token, symbols, plan, and parameters.")
    return recs


def _skipped_symbol_report(symbol: str, checks: Sequence[str], reason: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "checks": {check: _skipped_result(reason) for check in checks},
        "usable_sections": [],
        "ok": False,
        "skipped": True,
        "skip_reason": reason,
    }


def _skipped_result(reason: str) -> dict[str, Any]:
    return {"ok": False, "error": reason, "summary": {}, "skipped": True}


def _is_rate_limit_error(error: str) -> bool:
    text = error.lower()
    return "http error 429" in text or "too many requests" in text or "rate limit" in text


def _is_terminal_error(error: str) -> bool:
    text = error.lower()
    return _is_rate_limit_error(error) or "http error 401" in text or "http error 403" in text


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
