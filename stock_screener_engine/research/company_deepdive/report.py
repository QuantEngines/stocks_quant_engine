"""Company deep-dive report assembly from structured engine outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Mapping


@dataclass(frozen=True)
class ResearchSection:
    title: str
    summary: str
    data_points: dict[str, object] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CompanyDeepDiveReport:
    symbol: str
    company_name: str
    sector: str
    as_of: str
    sections: list[ResearchSection]
    final_verdict: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "company_name": self.company_name,
            "sector": self.sector,
            "as_of": self.as_of,
            "sections": [asdict(section) for section in self.sections],
            "final_verdict": dict(self.final_verdict),
        }


class CompanyDeepDiveBuilder:
    """Build a data-backed report without inventing missing facts."""

    SECTION_TITLES = [
        "Executive Summary",
        "Business Overview",
        "Segment Analysis",
        "Financial Statement Analysis",
        "Ratio Analysis",
        "Growth Analysis",
        "Margin Analysis",
        "Cash Flow Quality",
        "Balance Sheet Strength",
        "Capital Allocation",
        "Valuation Analysis",
        "Technical Structure",
        "Peer Comparison",
        "Shareholding / Ownership",
        "Corporate Actions",
        "News and Event Timeline",
        "Annual Report / PDF Insights",
        "Management Commentary",
        "Governance and Risk Flags",
        "Bull Case",
        "Bear Case",
        "Key Monitorables",
        "Long-Term View",
        "Swing View",
        "Final Verdict",
    ]

    def build(
        self,
        single_stock_analysis: Mapping[str, object],
        document_insights: Mapping[str, object] | None = None,
        peer_insights: Mapping[str, object] | None = None,
    ) -> CompanyDeepDiveReport:
        symbol = str(single_stock_analysis.get("symbol", "")).upper()
        company_name = str(single_stock_analysis.get("company_name") or symbol)
        sector = str(single_stock_analysis.get("sector") or "Unknown")
        as_of = str(single_stock_analysis.get("as_of") or date.today().isoformat())
        docs = document_insights or {}
        peers = peer_insights or {}

        sections = [
            self._executive_summary(single_stock_analysis, docs),
            self._business_overview(single_stock_analysis),
            self._segment_analysis(docs),
            self._financial_statement(single_stock_analysis, docs),
            self._ratio_analysis(single_stock_analysis),
            self._growth_analysis(single_stock_analysis, docs),
            self._margin_analysis(single_stock_analysis, docs),
            self._cash_flow_quality(single_stock_analysis, docs),
            self._balance_sheet(single_stock_analysis, docs),
            self._capital_allocation(docs),
            self._valuation(single_stock_analysis),
            self._technical(single_stock_analysis),
            self._peer_comparison(peers),
            self._ownership(single_stock_analysis, docs),
            self._corporate_actions(docs),
            self._news_timeline(single_stock_analysis),
            self._document_insights(docs),
            self._management_commentary(docs),
            self._governance(single_stock_analysis, docs),
            self._bull_case(single_stock_analysis),
            self._bear_case(single_stock_analysis),
            self._monitorables(single_stock_analysis, docs),
            self._long_term_view(single_stock_analysis),
            self._swing_view(single_stock_analysis),
            self._final_verdict_section(single_stock_analysis),
        ]

        return CompanyDeepDiveReport(
            symbol=symbol,
            company_name=company_name,
            sector=sector,
            as_of=as_of,
            sections=sections,
            final_verdict=self._final_verdict(single_stock_analysis),
        )

    def render_markdown(self, report: CompanyDeepDiveReport) -> str:
        lines = [
            f"# {report.company_name} ({report.symbol}) Deep-Dive",
            "",
            f"Sector: {report.sector}",
            f"As of: {report.as_of}",
            "",
        ]
        for section in report.sections:
            lines.append(f"## {section.title}")
            lines.append(section.summary)
            if section.findings:
                lines.extend(["", "Findings:"])
                lines.extend(f"- {item}" for item in section.findings)
            if section.missing_data:
                lines.extend(["", "Unavailable:"])
                lines.extend(f"- {item}" for item in section.missing_data)
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def _executive_summary(self, analysis: Mapping[str, object], docs: Mapping[str, object]) -> ResearchSection:
        scores = _map(analysis.get("scores"))
        directional = _map(analysis.get("directional"))
        return ResearchSection(
            title="Executive Summary",
            summary=str(directional.get("interpretation", "No directional interpretation available.")),
            data_points={
                "long_term_score": scores.get("long_term_score"),
                "swing_score": scores.get("swing_score"),
                "risk_penalty": scores.get("risk_penalty"),
                "document_quality_score": docs.get("quality_score"),
            },
            findings=_drivers(analysis, limit=3),
            missing_data=_missing_if_absent(docs, "document_insights", "No document package supplied."),
        )

    def _business_overview(self, analysis: Mapping[str, object]) -> ResearchSection:
        return ResearchSection(
            title="Business Overview",
            summary="Business description is not yet sourced from a verified company master or annual report.",
            data_points={
                "sector": analysis.get("sector"),
                "industry": analysis.get("industry"),
                "exchange": analysis.get("exchange"),
            },
            missing_data=["Verified business model, products, and geographic revenue split"],
        )

    def _segment_analysis(self, docs: Mapping[str, object]) -> ResearchSection:
        facts = _facts_by_kind(docs, "segment")
        return ResearchSection(
            title="Segment Analysis",
            summary="Segment facts extracted from documents when available.",
            data_points={"segment_fact_count": len(facts)},
            findings=[_fact_sentence(f) for f in facts[:5]],
            missing_data=[] if facts else ["Revenue and EBIT by business segment"],
        )

    def _financial_statement(self, analysis: Mapping[str, object], docs: Mapping[str, object]) -> ResearchSection:
        facts = _facts_by_kind(docs, "financial")
        return ResearchSection(
            title="Financial Statement Analysis",
            summary="Combines available normalized fundamentals with document-extracted financial facts.",
            data_points=_map(analysis.get("fundamentals")),
            findings=[_fact_sentence(f) for f in facts[:5]],
            missing_data=[] if facts else ["Statement-level revenue, EBITDA, PAT, assets, liabilities, and cash-flow line items"],
        )

    def _ratio_analysis(self, analysis: Mapping[str, object]) -> ResearchSection:
        fundamentals = _map(analysis.get("fundamentals"))
        return ResearchSection(
            title="Ratio Analysis",
            summary="Ratio view from available normalized metrics.",
            data_points={
                "pe_ratio": fundamentals.get("pe_ratio"),
                "pb_ratio": fundamentals.get("pb_ratio"),
                "roe_pct": fundamentals.get("roe_pct"),
                "debt_to_equity": fundamentals.get("debt_to_equity"),
            },
            missing_data=_none_fields(fundamentals, ["pe_ratio", "pb_ratio", "roe_pct", "debt_to_equity"]),
        )

    def _growth_analysis(self, analysis: Mapping[str, object], docs: Mapping[str, object]) -> ResearchSection:
        fundamentals = _map(analysis.get("fundamentals"))
        growth_facts = _facts_by_kind(docs, "growth")
        return ResearchSection(
            title="Growth Analysis",
            summary="Growth view from normalized earnings growth and document facts.",
            data_points={"earnings_growth_pct": fundamentals.get("earnings_growth_pct")},
            findings=[_fact_sentence(f) for f in growth_facts[:5]],
            missing_data=[] if fundamentals.get("earnings_growth_pct") is not None else ["Revenue/PAT growth series"],
        )

    def _margin_analysis(self, analysis: Mapping[str, object], docs: Mapping[str, object]) -> ResearchSection:
        margin_facts = _facts_by_kind(docs, "margin")
        return ResearchSection(
            title="Margin Analysis",
            summary="Margin expansion or compression is document-driven until normalized statement history is available.",
            findings=[_fact_sentence(f) for f in margin_facts[:5]],
            missing_data=[] if margin_facts else ["EBITDA margin, PAT margin, gross margin trend"],
        )

    def _cash_flow_quality(self, analysis: Mapping[str, object], docs: Mapping[str, object]) -> ResearchSection:
        features = _map(analysis.get("all_features"))
        cash_facts = _facts_by_kind(docs, "cash_flow")
        return ResearchSection(
            title="Cash Flow Quality",
            summary="Cash conversion is assessed from engine features and any extracted cash-flow commentary.",
            data_points={
                "cash_flow_quality": features.get("cash_flow_quality"),
                "cfo_pat_ratio": features.get("cfo_pat_ratio"),
            },
            findings=[_fact_sentence(f) for f in cash_facts[:5]],
            missing_data=[] if cash_facts or features.get("cash_flow_quality") else ["CFO/PAT and FCF conversion history"],
        )

    def _balance_sheet(self, analysis: Mapping[str, object], docs: Mapping[str, object]) -> ResearchSection:
        features = _map(analysis.get("all_features"))
        debt_facts = _facts_by_kind(docs, "debt")
        return ResearchSection(
            title="Balance Sheet Strength",
            summary="Balance-sheet health is based on leverage features and debt-related document facts.",
            data_points={
                "balance_sheet_health": features.get("balance_sheet_health"),
                "debt_to_equity": features.get("debt_to_equity"),
                "leverage_trend": features.get("leverage_trend"),
            },
            findings=[_fact_sentence(f) for f in debt_facts[:5]],
            missing_data=[] if debt_facts or features.get("debt_to_equity") else ["Debt maturity, contingent liabilities, and liquidity schedule"],
        )

    def _capital_allocation(self, docs: Mapping[str, object]) -> ResearchSection:
        facts = _facts_by_kind(docs, "capital_allocation")
        return ResearchSection(
            title="Capital Allocation",
            summary="Capital allocation is extracted from capex, dividend, buyback, and acquisition commentary.",
            findings=[_fact_sentence(f) for f in facts[:5]],
            missing_data=[] if facts else ["Capex plan, dividend/buyback policy, M&A, and reinvestment returns"],
        )

    def _valuation(self, analysis: Mapping[str, object]) -> ResearchSection:
        fundamentals = _map(analysis.get("fundamentals"))
        features = _map(analysis.get("all_features"))
        return ResearchSection(
            title="Valuation Analysis",
            summary="Valuation view uses available PE/PB and relative valuation features.",
            data_points={
                "pe_ratio": fundamentals.get("pe_ratio"),
                "pb_ratio": fundamentals.get("pb_ratio"),
                "sector_pe_zscore": features.get("sector_pe_zscore"),
                "rolling_pe_zscore": features.get("rolling_pe_zscore"),
                "valuation_sanity": features.get("valuation_sanity"),
            },
            missing_data=["EV/EBITDA, FCF yield, own-history valuation bands"] if fundamentals.get("pe_ratio") is None else [],
        )

    def _technical(self, analysis: Mapping[str, object]) -> ResearchSection:
        price = _map(analysis.get("price"))
        technical = _map(analysis.get("technical_indicators"))
        return ResearchSection(
            title="Technical Structure",
            summary="Technical view from price, moving averages, momentum, RSI, ADX and ATR.",
            data_points={**price, **technical},
            missing_data=_none_fields(technical, ["rsi_14", "adx_14", "atr_14"]),
        )

    def _peer_comparison(self, peers: Mapping[str, object]) -> ResearchSection:
        target = _map(peers.get("target"))
        peer_rows = peers.get("peers")
        rows = peer_rows if isinstance(peer_rows, list) else []
        leader_items = peers.get("composite_leaders")
        warnings = peers.get("warnings")
        findings: list[str] = []
        if isinstance(leader_items, list) and leader_items:
            findings.append(f"Composite leaders: {', '.join(str(item) for item in leader_items[:5])}")
        if target:
            findings.append(
                "Target ranks - "
                f"composite: {target.get('composite_rank')}, "
                f"valuation: {target.get('valuation_rank')}, "
                f"quality: {target.get('quality_rank')}, "
                f"growth: {target.get('growth_rank')}, "
                f"risk: {target.get('risk_rank')}"
            )
        return ResearchSection(
            title="Peer Comparison",
            summary=(
                "Sector-relative peer ranks from canonical financial statements and valuations."
                if rows
                else "Peer comparison requires canonical security master, statements, and valuation facts."
            ),
            data_points={
                "peer_count": peers.get("peer_count"),
                "sector": peers.get("sector"),
                "target_composite_rank": target.get("composite_rank"),
                "target_valuation_rank": target.get("valuation_rank"),
                "target_quality_rank": target.get("quality_rank"),
                "target_growth_rank": target.get("growth_rank"),
            },
            findings=findings,
            missing_data=[str(item) for item in warnings] if isinstance(warnings, list) and warnings else ([] if rows else ["Peer universe, peer valuation, peer growth, peer quality ranks"]),
        )

    def _ownership(self, analysis: Mapping[str, object], docs: Mapping[str, object]) -> ResearchSection:
        facts = _facts_by_kind(docs, "ownership")
        features = _map(analysis.get("all_features"))
        return ResearchSection(
            title="Shareholding / Ownership",
            summary="Ownership view uses governance features and document-extracted shareholding commentary.",
            data_points={"governance_proxy": features.get("governance_proxy")},
            findings=[_fact_sentence(f) for f in facts[:5]],
            missing_data=[] if facts else ["Promoter/FII/DII trend, pledge trend, insider transactions"],
        )

    def _corporate_actions(self, docs: Mapping[str, object]) -> ResearchSection:
        facts = _facts_by_kind(docs, "corporate_action")
        return ResearchSection(
            title="Corporate Actions",
            summary="Corporate-action facts are included when documents or exchange feeds provide them.",
            findings=[_fact_sentence(f) for f in facts[:5]],
            missing_data=[] if facts else ["Dividend, split, bonus, buyback, merger/demerger timeline"],
        )

    def _news_timeline(self, analysis: Mapping[str, object]) -> ResearchSection:
        news = _map(analysis.get("news"))
        headlines_raw = news.get("headlines")
        headlines = headlines_raw if isinstance(headlines_raw, list) else []
        return ResearchSection(
            title="News and Event Timeline",
            summary=f"{len(headlines)} recent headlines/events available from configured text provider.",
            data_points={"headline_count": news.get("headline_count"), "lookback_days": news.get("lookback_days")},
            findings=[str(h) for h in headlines[:5]],
            missing_data=[] if headlines else ["Recent news/events from configured source"],
        )

    def _document_insights(self, docs: Mapping[str, object]) -> ResearchSection:
        sections = _map(docs.get("section_map"))
        return ResearchSection(
            title="Annual Report / PDF Insights",
            summary="Document insights are generated from local document ingestion when supplied.",
            data_points={"document_type": docs.get("document_type"), "quality_score": docs.get("quality_score"), "sections": list(sections.keys())[:10]},
            findings=[_fact_sentence(f) for f in _all_facts(docs)[:8]],
            missing_data=[] if docs else ["No document supplied"],
        )

    def _management_commentary(self, docs: Mapping[str, object]) -> ResearchSection:
        commentary = docs.get("management_commentary")
        rows = commentary if isinstance(commentary, list) else []
        return ResearchSection(
            title="Management Commentary",
            summary="Management commentary is extracted from outlook, demand, margin and capex language.",
            findings=[str(item) for item in rows[:8]],
            missing_data=[] if rows else ["Management outlook, demand commentary, margin commentary, capital allocation commentary"],
        )

    def _governance(self, analysis: Mapping[str, object], docs: Mapping[str, object]) -> ResearchSection:
        risk_flags = analysis.get("risk_flags")
        governance_facts = _facts_by_kind(docs, "governance")
        return ResearchSection(
            title="Governance and Risk Flags",
            summary="Governance combines engine risk flags and document-extracted governance facts.",
            data_points={"risk_flags": risk_flags if isinstance(risk_flags, list) else []},
            findings=[_fact_sentence(f) for f in governance_facts[:5]],
            missing_data=[] if governance_facts else ["Auditor remarks, related-party transactions, pledging and litigation details"],
        )

    def _bull_case(self, analysis: Mapping[str, object]) -> ResearchSection:
        return ResearchSection(
            title="Bull Case",
            summary="Bull case built from strongest positive drivers.",
            findings=_drivers(analysis, limit=5),
        )

    def _bear_case(self, analysis: Mapping[str, object]) -> ResearchSection:
        return ResearchSection(
            title="Bear Case",
            summary="Bear case built from risk and negative-driver stack.",
            findings=_negative_drivers(analysis, limit=5),
        )

    def _monitorables(self, analysis: Mapping[str, object], docs: Mapping[str, object]) -> ResearchSection:
        monitorables = [
            "Quarterly revenue growth and margin trend",
            "Price structure versus 50/200 DMA",
            "Volume confirmation around breakouts/breakdowns",
            "Governance, leverage, and event-risk updates",
        ]
        if docs:
            monitorables.append("Document-extracted capex, demand, margin, and management-tone changes")
        return ResearchSection(
            title="Key Monitorables",
            summary="Monitorables that would confirm, weaken, or invalidate the thesis.",
            findings=monitorables,
        )

    def _long_term_view(self, analysis: Mapping[str, object]) -> ResearchSection:
        horizons = _map(analysis.get("investment_horizons"))
        long_term = _map(horizons.get("long_term"))
        return ResearchSection(
            title="Long-Term View",
            summary=str(long_term.get("rationale", "No long-term view available.")),
            data_points=long_term,
        )

    def _swing_view(self, analysis: Mapping[str, object]) -> ResearchSection:
        horizons = _map(analysis.get("investment_horizons"))
        swing = _map(horizons.get("swing"))
        return ResearchSection(
            title="Swing View",
            summary=str(swing.get("rationale", "No swing view available.")),
            data_points=swing,
        )

    def _final_verdict_section(self, analysis: Mapping[str, object]) -> ResearchSection:
        verdict = self._final_verdict(analysis)
        return ResearchSection(
            title="Final Verdict",
            summary=str(verdict.get("summary", "No final verdict available.")),
            data_points=verdict,
        )

    def _final_verdict(self, analysis: Mapping[str, object]) -> dict[str, object]:
        directional = _map(analysis.get("directional"))
        horizons = _map(analysis.get("investment_horizons"))
        long_term = _map(horizons.get("long_term"))
        swing = _map(horizons.get("swing"))
        scores = _map(analysis.get("scores"))
        return {
            "directional_bias": directional.get("bias", "unknown"),
            "long_term_verdict": long_term.get("verdict", "unknown"),
            "swing_verdict": swing.get("verdict", "unknown"),
            "long_term_score": scores.get("long_term_score"),
            "swing_score": scores.get("swing_score"),
            "risk_penalty": scores.get("risk_penalty"),
            "summary": directional.get("interpretation", "Insufficient data for a final verdict."),
        }


def _map(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _none_fields(payload: Mapping[str, object], fields: list[str]) -> list[str]:
    return [field for field in fields if payload.get(field) is None]


def _missing_if_absent(payload: Mapping[str, object], key: str, message: str) -> list[str]:
    return [] if payload.get(key) else [message]


def _drivers(analysis: Mapping[str, object], limit: int) -> list[str]:
    kd = _map(analysis.get("key_drivers"))
    vals = kd.get("top_positive")
    return [str(v) for v in vals[:limit]] if isinstance(vals, list) else []


def _negative_drivers(analysis: Mapping[str, object], limit: int) -> list[str]:
    kd = _map(analysis.get("key_drivers"))
    vals = kd.get("top_negative")
    return [str(v) for v in vals[:limit]] if isinstance(vals, list) else []


def _all_facts(docs: Mapping[str, object]) -> list[dict]:
    facts = docs.get("facts")
    return [f for f in facts if isinstance(f, dict)] if isinstance(facts, list) else []


def _facts_by_kind(docs: Mapping[str, object], kind: str) -> list[dict]:
    return [fact for fact in _all_facts(docs) if str(fact.get("kind", "")).lower() == kind]


def _fact_sentence(fact: Mapping[str, object]) -> str:
    label = str(fact.get("label") or fact.get("kind") or "fact")
    value = fact.get("value")
    unit = str(fact.get("unit") or "").strip()
    confidence = fact.get("confidence")
    suffix = f" {unit}" if unit else ""
    conf = f" (confidence {float(confidence):.2f})" if isinstance(confidence, (int, float)) else ""
    return f"{label}: {value}{suffix}{conf}"
