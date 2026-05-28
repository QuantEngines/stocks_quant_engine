"""Coverage gates for production-grade research and signal workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Mapping, Sequence


DEFAULT_GATE_PROFILES: dict[str, dict[str, float]] = {
    "swing_scan": {
        "security_master": 0.95,
        "daily_ohlcv": 0.95,
    },
    "long_term_scan": {
        "security_master": 0.95,
        "daily_ohlcv": 0.90,
        "financials": 0.80,
        "valuations": 0.80,
        "shareholding": 0.75,
    },
    "deep_research": {
        "security_master": 0.95,
        "daily_ohlcv": 0.90,
        "financials": 0.80,
        "valuations": 0.80,
        "shareholding": 0.75,
        "events_documents": 0.50,
    },
    "backtest": {
        "security_master": 0.95,
        "daily_ohlcv": 0.95,
        "financials": 0.70,
        "valuations": 0.70,
        "shareholding": 0.70,
        "corporate_actions": 0.80,
        "historical_universe": 0.80,
    },
}


@dataclass(frozen=True)
class CoverageGateIssue:
    domain: str
    label: str
    required_coverage: float
    actual_coverage: float
    severity: str
    message: str
    missing_count: int = 0
    entitlement_note: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CoverageGateResult:
    mode: str
    decision: str
    passed: bool
    required_domains: dict[str, float]
    issues: list[CoverageGateIssue] = field(default_factory=list)
    warnings: list[CoverageGateIssue] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "decision": self.decision,
            "passed": self.passed,
            "required_domains": dict(self.required_domains),
            "issues": [issue.to_dict() for issue in self.issues],
            "warnings": [warning.to_dict() for warning in self.warnings],
            "recommendations": list(self.recommendations),
        }


class CoverageGateEvaluator:
    """Evaluate whether available data is strong enough for a workflow."""

    def __init__(self, profiles: Mapping[str, Mapping[str, float]] | None = None) -> None:
        self.profiles = {
            name: {str(domain): float(threshold) for domain, threshold in profile.items()}
            for name, profile in (profiles or DEFAULT_GATE_PROFILES).items()
        }

    def evaluate(self, coverage_report: Mapping[str, object], mode: str) -> CoverageGateResult:
        normalized_mode = mode.strip().lower().replace("-", "_")
        profile = self.profiles.get(normalized_mode)
        if profile is None:
            raise ValueError(f"Unknown coverage gate mode: {mode}")

        domain_rows = {
            str(row.get("domain")): row
            for row in coverage_report.get("domains", [])
            if isinstance(row, Mapping)
        }
        issues: list[CoverageGateIssue] = []
        warnings: list[CoverageGateIssue] = []
        recommendations: list[str] = []

        for domain, required in profile.items():
            row = domain_rows.get(domain, {})
            actual = _coverage(row)
            issue = _issue_for_domain(domain=domain, required=required, actual=actual, row=row)
            if actual + 1e-12 < required:
                issues.append(issue)
                recommendations.append(_recommendation_for_domain(domain, row))
            elif actual < 0.995:
                warnings.append(
                    CoverageGateIssue(
                        domain=domain,
                        label=str(row.get("label", domain)),
                        required_coverage=required,
                        actual_coverage=actual,
                        severity="warning",
                        message=f"{domain} passes the gate but is not complete.",
                        missing_count=int(row.get("missing_count", 0) or 0),
                        entitlement_note=str(row.get("entitlement_explanation", "")),
                    )
                )

        decision = "pass" if not issues else "block"
        if normalized_mode in {"swing_scan"} and not issues and warnings:
            decision = "pass_with_warnings"

        return CoverageGateResult(
            mode=normalized_mode,
            decision=decision,
            passed=not issues,
            required_domains=dict(profile),
            issues=issues,
            warnings=warnings,
            recommendations=_dedupe(recommendations),
        )


class FinEdgeOnboardingPlanner:
    """Build a repeatable paid-data onboarding plan without calling FinEdge."""

    def build(
        self,
        *,
        coverage_report: Mapping[str, object],
        gate_report: Mapping[str, object],
        universe_file: str | None,
        as_of: str,
        factor_root: str,
    ) -> dict[str, object]:
        finedge = _find_source(coverage_report, "finedge")
        finedge_entitlement = _find_entitlement(coverage_report, "finedge")
        domains = _domain_map(coverage_report)
        target_domains = ["financials", "valuations", "shareholding", "banking_factors", "events_documents"]

        command_universe = f'--universe-file "{universe_file}"' if universe_file else '--symbols "ITC,RELIANCE,HDFCBANK"'
        report = {
            "pipeline": "finedge_onboarding_plan",
            "run_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "as_of": as_of,
            "source": finedge,
            "entitlement": finedge_entitlement,
            "gate": gate_report,
            "target_domains": [
                {
                    "domain": domain,
                    "coverage": _coverage(domains.get(domain, {})),
                    "status": domains.get(domain, {}).get("status", "unknown"),
                    "missing_count": domains.get(domain, {}).get("missing_count", 0),
                    "entitlement": domains.get(domain, {}).get("entitlement", {}),
                }
                for domain in target_domains
            ],
            "paid_api_questions": [
                "Confirm Nifty 50 and Nifty 500 symbol coverage, including bank/NBFC names beyond HDFCBANK.",
                "Confirm annual and quarterly history depth, filing/result dates, and point-in-time availability.",
                "Confirm bank fields: NIM, GNPA, NNPA, CASA, PCR, credit cost, CET1, total CAR, advances, deposits, NII.",
                "Confirm bulk limits, rate limits, retry guidance, and whether response payloads can be stored internally.",
                "Confirm document endpoints for announcements, credit ratings, investor presentations, and call transcripts.",
                "Confirm redistribution and commercial-use rights for internal research reports and dashboards.",
            ],
            "commands_after_subscription": [
                f"stock-engine data-entitlements {command_universe} --format markdown",
                f"stock-engine finedge-probe {command_universe} --check smoke,fundamentals,prices,ownership,events --retries 2 --format table",
                f"stock-engine finedge-inspect {command_universe} --check financials,ownership --statement-type s --statement-code pl --period annual --format table",
                f"stock-engine finedge-factor-export {command_universe} --as-of {as_of} --output-root \"{factor_root}\" --sections financials,valuations,shareholding,banking --retries 2 --format table",
                f"stock-engine factor-ingest --root \"{factor_root}\" {command_universe} --as-of {as_of} --sections financials,valuations,shareholding,banking --min-coverage 0.95",
                f"stock-engine factor-qa {command_universe} --as-of {as_of} --format table",
                f"stock-engine data-source-coverage {command_universe} --end {as_of} --lookback-years 5 --format markdown",
                f"stock-engine data-readiness {command_universe} --mode long-term-scan --end {as_of} --lookback-years 5 --format markdown",
            ],
            "success_criteria": [
                "Financials, valuations, and shareholding coverage pass at or above 95% for the paid target universe.",
                "Bank/NBFC-specific coverage passes for Financial Services symbols, with explicit warnings for unavailable fields.",
                "Factor QA has zero errors and only explainable warnings.",
                "Long-term scan readiness changes from block to pass or pass_with_warnings.",
                "Coverage reports and onboarding artifacts remain under ignored local storage, not git.",
            ],
        }
        report["markdown"] = render_finedge_onboarding_markdown(report)
        return report


def build_data_readiness_report(
    *,
    coverage_report: Mapping[str, object],
    mode: str,
    profiles: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, object]:
    gate = CoverageGateEvaluator(profiles=profiles).evaluate(coverage_report, mode=mode)
    report = {
        "pipeline": "data_readiness",
        "run_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "mode": gate.mode,
        "coverage_as_of": coverage_report.get("as_of"),
        "coverage_start": coverage_report.get("start"),
        "decision": gate.decision,
        "passed": gate.passed,
        "gate": gate.to_dict(),
        "gross_coverage": coverage_report.get("gross_coverage", {}),
    }
    report["console_rows"] = _readiness_rows(report)
    report["markdown"] = render_data_readiness_markdown(report)
    return report


def render_data_readiness_markdown(report: Mapping[str, object]) -> str:
    gate = report.get("gate") if isinstance(report.get("gate"), Mapping) else {}
    lines = [
        "# Data Readiness Gate",
        "",
        f"- Mode: {report.get('mode')}",
        f"- Decision: {report.get('decision')}",
        f"- Passed: {report.get('passed')}",
        f"- Coverage as of: {report.get('coverage_as_of')}",
        "",
        "## Required Domains",
        "",
        "| Domain | Required | Actual | Severity | Message |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    rows = list(gate.get("issues", [])) + list(gate.get("warnings", [])) if isinstance(gate, Mapping) else []
    if not rows:
        lines.append("| All required domains | - | - | pass | All configured gate thresholds passed. |")
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {domain} | {required} | {actual} | {severity} | {message} |".format(
                domain=row.get("label", row.get("domain", "")),
                required=_pct(row.get("required_coverage")),
                actual=_pct(row.get("actual_coverage")),
                severity=row.get("severity", ""),
                message=row.get("message", ""),
            )
        )

    recommendations = gate.get("recommendations", []) if isinstance(gate, Mapping) else []
    if recommendations:
        lines.extend(["", "## Recommendations", ""])
        for item in recommendations:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def render_finedge_onboarding_markdown(report: Mapping[str, object]) -> str:
    source = report.get("source") if isinstance(report.get("source"), Mapping) else {}
    entitlement = report.get("entitlement") if isinstance(report.get("entitlement"), Mapping) else {}
    lines = [
        "# FinEdge Paid Onboarding Plan",
        "",
        f"- As of: {report.get('as_of')}",
        f"- Current plan: {entitlement.get('plan_name') or source.get('plan_name', '')}",
        f"- Current status: {entitlement.get('status') or source.get('status', '')}",
        "",
        "## Target Domain Coverage",
        "",
        "| Domain | Coverage | Status | Missing |",
        "| --- | ---: | --- | ---: |",
    ]
    for row in report.get("target_domains", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {domain} | {coverage} | {status} | {missing} |".format(
                domain=row.get("domain", ""),
                coverage=_pct(row.get("coverage")),
                status=row.get("status", ""),
                missing=row.get("missing_count", 0),
            )
        )

    lines.extend(["", "## Questions Before Paying", ""])
    for item in report.get("paid_api_questions", []):
        lines.append(f"- {item}")

    lines.extend(["", "## Commands After Subscription", ""])
    for command in report.get("commands_after_subscription", []):
        lines.append(f"```bash\n{command}\n```")

    lines.extend(["", "## Success Criteria", ""])
    for item in report.get("success_criteria", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _issue_for_domain(
    *,
    domain: str,
    required: float,
    actual: float,
    row: Mapping[str, object],
) -> CoverageGateIssue:
    label = str(row.get("label", domain))
    entitlement_note = str(row.get("entitlement_explanation", ""))
    message = f"{label} coverage {actual:.1%} is below required {required:.1%}."
    if entitlement_note:
        message = f"{message} {entitlement_note}"
    return CoverageGateIssue(
        domain=domain,
        label=label,
        required_coverage=required,
        actual_coverage=actual,
        severity="block",
        message=message,
        missing_count=int(row.get("missing_count", 0) or 0),
        entitlement_note=entitlement_note,
    )


def _recommendation_for_domain(domain: str, row: Mapping[str, object]) -> str:
    if domain in {"financials", "valuations", "shareholding", "banking_factors"}:
        return "Expand FinEdge/vendor factor coverage, export canonical factor CSVs, ingest them, and rerun factor QA."
    if domain == "daily_ohlcv":
        return "Refresh canonical OHLCV from Zerodha/exchange sources and rerun data-quality."
    if domain == "security_master":
        return "Ingest a clean universe/security master with sector and industry metadata."
    if domain == "corporate_actions":
        return "Add NSE/BSE/vendor corporate-action coverage before long-horizon backtests."
    if domain == "historical_universe":
        return "Acquire historical constituents, symbol changes, and delisting history before institutional backtests."
    if domain == "events_documents":
        return "Populate filings, announcements, presentations, transcripts, and document facts."
    return str(row.get("notes") or f"Improve {domain} coverage.")


def _readiness_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    gate = report.get("gate") if isinstance(report.get("gate"), Mapping) else {}
    rows = []
    for kind in ("issues", "warnings"):
        for row in gate.get(kind, []):
            if not isinstance(row, Mapping):
                continue
            rows.append(
                {
                    "mode": report.get("mode"),
                    "decision": report.get("decision"),
                    "domain": row.get("domain"),
                    "required": row.get("required_coverage"),
                    "actual": row.get("actual_coverage"),
                    "severity": row.get("severity"),
                    "missing": row.get("missing_count"),
                }
            )
    if not rows:
        rows.append(
            {
                "mode": report.get("mode"),
                "decision": report.get("decision"),
                "domain": "all_required",
                "required": "",
                "actual": "",
                "severity": "pass",
                "missing": 0,
            }
        )
    return rows


def _domain_map(report: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {
        str(row.get("domain")): row
        for row in report.get("domains", [])
        if isinstance(row, Mapping)
    }


def _find_source(report: Mapping[str, object], source_id: str) -> dict[str, object]:
    for row in report.get("sources", []):
        if isinstance(row, Mapping) and str(row.get("source_id", "")) == source_id:
            return dict(row)
    return {}


def _find_entitlement(report: Mapping[str, object], source_id: str) -> dict[str, object]:
    for row in report.get("entitlements", []):
        if isinstance(row, Mapping) and str(row.get("source_id", "")) == source_id:
            return dict(row)
    return {}


def _coverage(row: Mapping[str, object]) -> float:
    value = row.get("coverage", 0.0)
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _pct(value: object) -> str:
    if not isinstance(value, (int, float)):
        return ""
    return f"{float(value) * 100:.1f}%"
