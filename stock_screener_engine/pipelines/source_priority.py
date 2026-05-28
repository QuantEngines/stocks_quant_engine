"""Source-priority reporting for institutional data onboarding."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Mapping, Sequence

from stock_screener_engine.config.settings import DataSourceEntitlementSettings


@dataclass(frozen=True)
class SourcePriorityRule:
    domain: str
    primary: list[str]
    alternates: list[str] = field(default_factory=list)
    required_for: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_SOURCE_PRIORITY: tuple[SourcePriorityRule, ...] = (
    SourcePriorityRule(
        domain="security_master",
        primary=["nse_bse"],
        alternates=["finedge", "zerodha", "icici_breeze"],
        required_for=["all_scans", "research", "backtests"],
        notes="Exchange master is the canonical identity layer; brokers/vendors are reconciliation sources.",
    ),
    SourcePriorityRule(
        domain="daily_ohlcv",
        primary=["nse_bse", "zerodha"],
        alternates=["icici_breeze", "yfinance"],
        required_for=["swing_scan", "long_term_scan", "backtests"],
        notes="Prefer exchange/broker reconciled history; yfinance remains development fallback only.",
    ),
    SourcePriorityRule(
        domain="live_quotes",
        primary=["zerodha"],
        alternates=["icici_breeze"],
        required_for=["intraday_scan", "risk_alerts"],
        notes="Live broker data is useful for freshness and alerts; execution remains optional.",
    ),
    SourcePriorityRule(
        domain="delivery_turnover",
        primary=["nse_bse"],
        alternates=[],
        required_for=["swing_scan", "participation_filters", "volume_confirmation"],
        notes="Official delivery/turnover data should feed Indian-market participation and false-breakout filters.",
    ),
    SourcePriorityRule(
        domain="financials",
        primary=["finedge"],
        alternates=["nse_bse"],
        required_for=["long_term_scan", "deep_research", "factor_backtests"],
        notes="FinEdge becomes primary after paid Nifty coverage is confirmed; exchange filings are audit trail.",
    ),
    SourcePriorityRule(
        domain="valuations",
        primary=["finedge"],
        alternates=["nse_bse"],
        required_for=["long_term_scan", "sector_rotation", "deep_research"],
        notes="Valuation factors should be point-in-time and sector-normalized before production scoring.",
    ),
    SourcePriorityRule(
        domain="shareholding",
        primary=["finedge"],
        alternates=["nse_bse"],
        required_for=["governance", "ownership_trend", "deep_research"],
        notes="Ownership, pledge, FII/DII, and promoter trends feed quality and governance signals.",
    ),
    SourcePriorityRule(
        domain="banking_factors",
        primary=["finedge"],
        alternates=[],
        required_for=["financials_sector", "bank_deep_research"],
        notes="Banks/NBFCs need sector-specific factors such as NIM, GNPA, NNPA, CASA, PCR, and capital ratios.",
    ),
    SourcePriorityRule(
        domain="events_documents",
        primary=["nse_bse", "finedge"],
        alternates=["free_rss"],
        required_for=["event_signals", "deep_research", "document_intelligence"],
        notes="Announcements, ratings, presentations, transcripts, and filings should feed NLP/document features.",
    ),
    SourcePriorityRule(
        domain="corporate_actions",
        primary=["nse_bse"],
        alternates=["finedge"],
        required_for=["adjusted_history", "backtests", "portfolio_monitoring"],
        notes="Corporate actions must reconcile price history and long-horizon backtest labels.",
    ),
    SourcePriorityRule(
        domain="analyst_estimates",
        primary=[],
        alternates=["indianapi", "fmp"],
        required_for=["future_consensus_layer"],
        notes="Paused until a vendor proves reliable India coverage and licensing terms.",
    ),
)


def build_source_priority_report(
    *,
    entitlements: Sequence[DataSourceEntitlementSettings],
    rules: Sequence[SourcePriorityRule] = DEFAULT_SOURCE_PRIORITY,
) -> dict[str, object]:
    """Build a source-priority map enriched with current entitlement status."""
    entitlement_map = {item.source_id: item for item in entitlements}
    rows = [_rule_row(rule, entitlement_map) for rule in rules]
    report = {
        "pipeline": "data_source_priority",
        "run_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "rows": rows,
        "gaps": [row for row in rows if row["status"] in {"gap", "unproven"}],
    }
    report["markdown"] = render_source_priority_markdown(report)
    return report


def render_source_priority_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Data Source Priority Map",
        "",
        "| Domain | Primary | Alternates | Status | Required For | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {domain} | {primary} | {alternates} | {status} | {required_for} | {notes} |".format(
                domain=row.get("domain", ""),
                primary=", ".join(row.get("primary", [])) if isinstance(row.get("primary"), list) else "",
                alternates=", ".join(row.get("alternates", [])) if isinstance(row.get("alternates"), list) else "",
                status=row.get("status", ""),
                required_for=", ".join(row.get("required_for", [])) if isinstance(row.get("required_for"), list) else "",
                notes=row.get("notes", ""),
            )
        )

    gaps = report.get("gaps", [])
    if isinstance(gaps, list) and gaps:
        lines.extend(["", "## Gaps To Close", ""])
        for row in gaps:
            if isinstance(row, Mapping):
                lines.append(f"- {row.get('domain')}: {row.get('next_action') or row.get('notes')}")
    return "\n".join(lines) + "\n"


def _rule_row(
    rule: SourcePriorityRule,
    entitlement_map: Mapping[str, DataSourceEntitlementSettings],
) -> dict[str, object]:
    primary_statuses = [_source_status(source, entitlement_map) for source in rule.primary]
    alternate_statuses = [_source_status(source, entitlement_map) for source in rule.alternates]
    status = _domain_status(primary_statuses, alternate_statuses)
    next_action = _next_action(rule, primary_statuses, alternate_statuses)
    return {
        **rule.to_dict(),
        "status": status,
        "primary_statuses": primary_statuses,
        "alternate_statuses": alternate_statuses,
        "next_action": next_action,
    }


def _source_status(
    source_id: str,
    entitlement_map: Mapping[str, DataSourceEntitlementSettings],
) -> dict[str, object]:
    entitlement = entitlement_map.get(source_id)
    if entitlement is None:
        return {
            "source_id": source_id,
            "display_name": source_id,
            "enabled": False,
            "status": "missing_config",
            "plan_name": "",
            "license_status": "missing_config",
            "next_action": "Add source entitlement metadata.",
        }
    return {
        "source_id": source_id,
        "display_name": entitlement.display_name,
        "enabled": entitlement.enabled,
        "status": entitlement.status,
        "plan_name": entitlement.plan_name,
        "license_status": entitlement.license_status,
        "next_action": entitlement.next_action,
    }


def _domain_status(primary_statuses: Sequence[Mapping[str, object]], alternate_statuses: Sequence[Mapping[str, object]]) -> str:
    proven_statuses = {"primary_proven", "primary_partial", "basic_sandbox"}
    if any(bool(item.get("enabled")) and item.get("status") in proven_statuses for item in primary_statuses):
        if any(item.get("status") == "basic_sandbox" for item in primary_statuses):
            return "sandbox_ready"
        if any(item.get("status") == "primary_partial" for item in primary_statuses):
            return "partial"
        return "ready"
    if primary_statuses:
        return "unproven"
    if any(bool(item.get("enabled")) for item in alternate_statuses):
        return "fallback_only"
    return "gap"


def _next_action(
    rule: SourcePriorityRule,
    primary_statuses: Sequence[Mapping[str, object]],
    alternate_statuses: Sequence[Mapping[str, object]],
) -> str:
    for item in list(primary_statuses) + list(alternate_statuses):
        next_action = str(item.get("next_action", "")).strip()
        if next_action:
            return next_action
    return rule.notes
