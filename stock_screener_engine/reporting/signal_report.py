"""Professional signal report objects for analyst-grade output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping, cast

from stock_screener_engine.core.entities import FeatureVector, ScoreCard, SignalResult


@dataclass(frozen=True)
class SignalIdentity:
    symbol: str
    company_name: str | None
    sector: str
    industry: str | None = None
    market_cap_category: str = "unavailable"
    liquidity_classification: str = "unavailable"


@dataclass(frozen=True)
class SignalSummary:
    signal_type: str
    final_score: float
    long_term_score: float
    swing_score: float
    risk_penalty: float
    confidence: float
    horizon: str
    rank: int
    category: str


@dataclass(frozen=True)
class TechnicalMetrics:
    price_vs_20_dma_pct: float | None = None
    price_vs_50_dma_pct: float | None = None
    price_vs_200_dma_pct: float | None = None
    moving_average_alignment: str = "unavailable"
    relative_strength_vs_nifty: float | None = None
    relative_strength_vs_sector: float | None = None
    atr_pct: float | None = None
    volatility_regime: float | None = None
    volume_z_score: float | None = None
    breakout_status: str = "unavailable"
    trend_strength: float | None = None
    support_resistance: str = "unavailable"


@dataclass(frozen=True)
class FundamentalMetrics:
    revenue_growth: float | None = None
    ebitda_growth: float | None = None
    pat_growth: float | None = None
    eps_growth: float | None = None
    roe: float | None = None
    roce: float | None = None
    ebitda_margin: float | None = None
    pat_margin: float | None = None
    debt_to_equity: float | None = None
    interest_coverage: float | None = None
    cfo_pat: float | None = None
    free_cash_flow_conversion: float | None = None
    working_capital_quality: float | None = None


@dataclass(frozen=True)
class BankingMetrics:
    applicable: bool = False
    available: bool = False
    metric_coverage: float | None = None
    banking_quality_score: float | None = None
    asset_quality_score: float | None = None
    capital_strength_score: float | None = None
    franchise_strength_score: float | None = None
    profitability_score: float | None = None
    efficiency_score: float | None = None
    nim_pct: float | None = None
    gnpa_pct: float | None = None
    nnpa_pct: float | None = None
    casa_pct: float | None = None
    cet1_pct: float | None = None
    car_pct: float | None = None
    loan_to_deposit_pct: float | None = None


@dataclass(frozen=True)
class ValuationMetrics:
    pe_vs_own_history: float | None = None
    pe_vs_sector: float | None = None
    pb_vs_history: float | None = None
    ev_ebitda: float | None = None
    fcf_yield: float | None = None
    earnings_yield: float | None = None
    peg_like_metric: float | None = None
    valuation_risk_score: float | None = None
    pe_ratio: float | None = None
    pb_ratio: float | None = None


@dataclass(frozen=True)
class EventNlpMetrics:
    recent_positive_event_score: float | None = None
    recent_negative_event_score: float | None = None
    catalyst_strength: float | None = None
    management_tone: float | None = None
    governance_risk_flag: float | None = None
    sentiment_trend: float | None = None
    uncertainty_penalty: float | None = None


@dataclass(frozen=True)
class RiskMetrics:
    liquidity_risk: float | None = None
    volatility_risk: float | None = None
    leverage_risk: float | None = None
    valuation_risk: float | None = None
    earnings_instability_risk: float | None = None
    event_governance_risk: float | None = None
    missing_data_risk: float = 0.0


@dataclass(frozen=True)
class ConvictionMetrics:
    score_strength: float | None = None
    signal_agreement: float | None = None
    data_completeness: float | None = None
    source_confidence: float | None = None
    backtest_evidence: float | None = None
    sector_regime_confirmation: float | None = None
    risk_resilience: float | None = None
    support_score: float | None = None
    support_multiplier: float | None = None


@dataclass(frozen=True)
class PeerContextMetrics:
    sector_pe_zscore: float | None = None
    sector_pb_zscore: float | None = None
    valuation_position: str = "unavailable"
    valuation_note: str = "unavailable"


@dataclass(frozen=True)
class CrossSectionalMetrics:
    universe_momentum_rank: float | None = None
    sector_momentum_rank: float | None = None
    universe_quality_rank: float | None = None
    sector_quality_rank: float | None = None
    universe_value_rank: float | None = None
    quality_value_composite: float | None = None
    liquidity_percentile: float | None = None
    feature_coverage_score: float | None = None
    sector_feature_coverage_score: float | None = None
    research_readiness_score: float | None = None


@dataclass(frozen=True)
class SignalExplanationBlock:
    top_positive_drivers: list[str] = field(default_factory=list)
    top_negative_drivers: list[str] = field(default_factory=list)
    why_selected: str = ""
    why_rejected: str | None = None
    what_can_go_wrong: list[str] = field(default_factory=list)
    what_to_monitor: list[str] = field(default_factory=list)
    invalidation_logic: str = ""
    missing_data_warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProfessionalSignalReport:
    identity: SignalIdentity
    summary: SignalSummary
    technical: TechnicalMetrics
    fundamentals: FundamentalMetrics
    banking: BankingMetrics
    valuation: ValuationMetrics
    event_nlp: EventNlpMetrics
    risk: RiskMetrics
    conviction: ConvictionMetrics
    peer_context: PeerContextMetrics
    cross_sectional: CrossSectionalMetrics
    explanation: SignalExplanationBlock
    as_of: date

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["as_of"] = self.as_of.isoformat()
        return payload


def build_signal_reports(
    features: Iterable[FeatureVector],
    scores: Iterable[ScoreCard],
    signals: Iterable[SignalResult],
    signal_type: str,
    company_metadata: Mapping[str, Mapping[str, object]] | None = None,
    limit: int | None = None,
) -> list[ProfessionalSignalReport]:
    feature_by_symbol = {fv.symbol: fv for fv in features}
    score_by_symbol = {sc.symbol: sc for sc in scores}
    metadata = company_metadata or {}
    ordered = sorted(signals, key=lambda s: s.score, reverse=True)
    if limit is not None:
        ordered = ordered[:limit]

    reports: list[ProfessionalSignalReport] = []
    for rank, signal in enumerate(ordered, start=1):
        fv = feature_by_symbol.get(signal.symbol)
        sc = score_by_symbol.get(signal.symbol)
        if fv is None or sc is None:
            continue
        reports.append(
            build_signal_report(
                feature=fv,
                score=sc,
                signal=signal,
                signal_type=signal_type,
                rank=rank,
                company_metadata=metadata.get(signal.symbol, {}),
            )
        )
    return reports


def build_signal_report(
    feature: FeatureVector,
    score: ScoreCard,
    signal: SignalResult,
    signal_type: str,
    rank: int,
    company_metadata: Mapping[str, object] | None = None,
) -> ProfessionalSignalReport:
    values = dict(feature.values)
    components = dict(score.component_scores)
    meta = company_metadata or {}
    missing = _missing_data_warnings(values)
    liquidity = _liquidity_class(values)

    final_score = score.long_term_score if signal_type == "long_term" else score.swing_score
    if signal_type == "short":
        final_score = signal.score

    explanation = signal.explanation
    why_selected = _selection_summary(signal, final_score)
    what_can_go_wrong = _risk_monitorables(explanation.risk_flags, values)

    return ProfessionalSignalReport(
        identity=SignalIdentity(
            symbol=signal.symbol,
            company_name=_optional_str(meta.get("company_name")),
            sector=signal.sector or _optional_str(meta.get("sector")) or "Unknown",
            industry=_optional_str(meta.get("industry")),
            market_cap_category=_market_cap_category(meta.get("market_cap")),
            liquidity_classification=liquidity,
        ),
        summary=SignalSummary(
            signal_type=signal_type,
            final_score=round(final_score, 2),
            long_term_score=round(score.long_term_score, 2),
            swing_score=round(score.swing_score, 2),
            risk_penalty=round(score.risk_penalty, 2),
            confidence=round(explanation.confidence, 2),
            horizon=explanation.holding_horizon,
            rank=rank,
            category=signal.category,
        ),
        technical=TechnicalMetrics(
            relative_strength_vs_nifty=_get(values, "relative_strength_proxy"),
            relative_strength_vs_sector=_get(values, "sector_momentum"),
            volatility_regime=_get(values, "volatility_regime"),
            volume_z_score=_activity_to_zscore(_get(values, "activity_vs_avg")),
            breakout_status=_breakout_status(values),
            trend_strength=_get(values, "trend_strength"),
        ),
        fundamentals=FundamentalMetrics(
            revenue_growth=_get(values, "revenue_growth"),
            eps_growth=_get(values, "growth_quality"),
            roe=_get(values, "profitability_quality"),
            pat_margin=_get(values, "operating_margin"),
            debt_to_equity=_get(values, "debt_to_equity"),
            cfo_pat=_get(values, "cfo_pat_ratio"),
            free_cash_flow_conversion=_get(values, "cash_flow_quality"),
        ),
        banking=_banking_metrics(values),
        valuation=ValuationMetrics(
            pe_vs_own_history=_get(values, "rolling_pe_zscore"),
            pe_vs_sector=_get(values, "sector_pe_zscore"),
            pb_vs_history=_get(values, "rolling_pb_zscore"),
            earnings_yield=_earnings_yield(_get(values, "pe_ratio")),
            peg_like_metric=_peg_like(_get(values, "pe_ratio"), _get(values, "growth_quality")),
            valuation_risk_score=_valuation_risk(values),
            pe_ratio=_get(values, "pe_ratio"),
            pb_ratio=_get(values, "pb_ratio"),
        ),
        event_nlp=EventNlpMetrics(
            recent_positive_event_score=_get(values, "recent_positive_event_score"),
            recent_negative_event_score=_get(values, "recent_negative_event_score"),
            catalyst_strength=_get(values, "catalyst_strength_score"),
            management_tone=_get(values, "management_tone_score"),
            governance_risk_flag=_get(values, "governance_risk_score"),
            sentiment_trend=_get(values, "sentiment_trend"),
            uncertainty_penalty=_get(values, "uncertainty_penalty"),
        ),
        risk=RiskMetrics(
            liquidity_risk=_component(components, "risk_liquidity_risk"),
            volatility_risk=_component(components, "risk_volatility_risk"),
            leverage_risk=_component(components, "risk_leverage_risk"),
            valuation_risk=_valuation_risk(values),
            earnings_instability_risk=_component(components, "risk_earnings_instability_risk"),
            event_governance_risk=_event_governance_risk(components),
            missing_data_risk=round(min(1.0, len(missing) / 5.0), 3),
        ),
        conviction=ConvictionMetrics(
            score_strength=_component(components, "conviction_score_strength"),
            signal_agreement=_component(components, "conviction_signal_agreement"),
            data_completeness=_component(components, "conviction_data_completeness"),
            source_confidence=_component(components, "conviction_source_confidence"),
            backtest_evidence=_component(components, "conviction_backtest_evidence"),
            sector_regime_confirmation=_component(components, "conviction_sector_regime_confirmation"),
            risk_resilience=_component(components, "conviction_risk_resilience"),
            support_score=_component(components, "conviction_support_score"),
            support_multiplier=_component(components, "conviction_support_multiplier"),
        ),
        peer_context=PeerContextMetrics(
            sector_pe_zscore=_get(values, "sector_pe_zscore"),
            sector_pb_zscore=_get(values, "sector_pb_zscore"),
            valuation_position=_valuation_position(values),
            valuation_note=_valuation_note(values),
        ),
        cross_sectional=CrossSectionalMetrics(
            universe_momentum_rank=_get(values, "cross_sectional_momentum_rank"),
            sector_momentum_rank=_get(values, "sector_relative_momentum_rank"),
            universe_quality_rank=_get(values, "cross_sectional_quality_rank"),
            sector_quality_rank=_get(values, "sector_relative_quality_rank"),
            universe_value_rank=_get(values, "cross_sectional_value_rank"),
            quality_value_composite=_get(values, "quality_value_composite"),
            liquidity_percentile=_get(values, "liquidity_percentile"),
            feature_coverage_score=_get(values, "feature_coverage_score"),
            sector_feature_coverage_score=_get(values, "sector_feature_coverage_score"),
            research_readiness_score=_get(values, "research_readiness_score"),
        ),
        explanation=SignalExplanationBlock(
            top_positive_drivers=list(explanation.top_positive_drivers),
            top_negative_drivers=list(explanation.top_negative_drivers),
            why_selected=why_selected,
            why_rejected=explanation.rejection_reason,
            what_can_go_wrong=what_can_go_wrong,
            what_to_monitor=_monitorables(values, explanation.risk_flags),
            invalidation_logic=explanation.invalidation_logic,
            missing_data_warnings=missing,
        ),
        as_of=feature.as_of,
    )


def render_signal_markdown(report: ProfessionalSignalReport) -> str:
    ident = report.identity
    summary = report.summary
    conviction = report.conviction
    banking = report.banking
    peer_context = report.peer_context
    cross_sectional = report.cross_sectional
    explanation = report.explanation
    lines = [
        f"# {ident.symbol} Signal Report",
        "",
        f"Sector: {ident.sector}",
        f"Signal: {summary.signal_type} | Category: {summary.category} | Rank: {summary.rank}",
        f"Final Score: {summary.final_score} | Long: {summary.long_term_score} | Swing: {summary.swing_score} | Risk: {summary.risk_penalty}",
        f"Confidence: {summary.confidence} | Horizon: {summary.horizon}",
        "",
        "## Conviction",
        f"Score strength: {conviction.score_strength} | Agreement: {conviction.signal_agreement} | Data: {conviction.data_completeness}",
        f"Source: {conviction.source_confidence} | Backtest: {conviction.backtest_evidence} | Risk resilience: {conviction.risk_resilience}",
        "",
        "## Cross-Sectional Context",
        f"Momentum rank: {cross_sectional.universe_momentum_rank} | Sector momentum rank: {cross_sectional.sector_momentum_rank}",
        f"Quality rank: {cross_sectional.universe_quality_rank} | Value rank: {cross_sectional.universe_value_rank}",
        f"Research readiness: {cross_sectional.research_readiness_score} | Feature coverage: {cross_sectional.feature_coverage_score}",
        "",
    ]
    if banking.applicable:
        lines.extend(
            [
                "## Banking",
                f"Available: {banking.available} | Coverage: {banking.metric_coverage} | Quality: {banking.banking_quality_score}",
                f"NIM: {banking.nim_pct} | GNPA: {banking.gnpa_pct} | NNPA: {banking.nnpa_pct} | CET1: {banking.cet1_pct}",
                "",
            ]
        )
    lines.extend(
        [
            "## Drivers",
            *[f"- {item}" for item in explanation.top_positive_drivers],
            "",
            "## Risks",
            *[f"- {item}" for item in explanation.top_negative_drivers],
            "",
            "## Peer Context",
            f"Valuation position: {peer_context.valuation_position}",
            str(peer_context.valuation_note),
            "",
            "## Thesis",
            str(explanation.why_selected),
            "",
            "## Invalidation",
            str(explanation.invalidation_logic),
        ]
    )
    warnings = explanation.missing_data_warnings
    if warnings:
        lines.extend(["", "## Missing Data", *[f"- {item}" for item in warnings]])
    return "\n".join(lines)


def signal_reports_to_console_rows(reports: Iterable[ProfessionalSignalReport]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for report in reports:
        rows.append(
            {
                "rank": report.summary.rank,
                "symbol": report.identity.symbol,
                "sector": report.identity.sector,
                "signal": report.summary.signal_type,
                "category": report.summary.category,
                "score": report.summary.final_score,
                "risk": report.summary.risk_penalty,
                "confidence": report.summary.confidence,
                "liquidity": report.identity.liquidity_classification,
                "readiness": report.cross_sectional.research_readiness_score,
                "momentum_rank": report.cross_sectional.universe_momentum_rank,
                "sector_momentum_rank": report.cross_sectional.sector_momentum_rank,
                "quality_value": report.cross_sectional.quality_value_composite,
                "banking_coverage": report.banking.metric_coverage,
                "banking_quality": report.banking.banking_quality_score,
            }
        )
    return rows


def _get(values: Mapping[str, float], key: str) -> float | None:
    if key not in values:
        return None
    return round(float(values[key]), 4)


def _component(components: Mapping[str, float], key: str) -> float | None:
    if key not in components:
        return None
    return round(float(components[key]), 4)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _market_cap_category(value: object) -> str:
    try:
        market_cap = float(cast(Any, value))
    except (TypeError, ValueError):
        return "unavailable"
    if market_cap >= 1_000_000_000_000:
        return "large_cap"
    if market_cap >= 200_000_000_000:
        return "mid_cap"
    return "small_cap"


def _liquidity_class(values: Mapping[str, float]) -> str:
    vol = float(values.get("volume_confirmation", 0.0))
    if vol >= 0.30:
        return "high"
    if vol >= 0.08:
        return "moderate"
    if vol > 0.0:
        return "low"
    return "unavailable"


def _breakout_status(values: Mapping[str, float]) -> str:
    breakout = float(values.get("breakout_score", 0.0))
    compression = float(values.get("compression_score", 0.0))
    trend = float(values.get("trend_strength", 0.0))
    if breakout >= 0.75 and trend >= 0.55:
        return "breakout_or_near_high"
    if compression >= 0.70 and trend < 0.55:
        return "compression_setup"
    if trend >= 0.60:
        return "trending"
    if breakout <= 0.25:
        return "range_low_or_breakdown_risk"
    return "neutral"


def _activity_to_zscore(activity: float | None) -> float | None:
    if activity is None:
        return None
    return round(float(activity) - 1.0, 4)


def _valuation_risk(values: Mapping[str, float]) -> float | None:
    valuation = values.get("valuation_sanity")
    if valuation is None:
        return None
    return round(max(0.0, min(1.0, 1.0 - float(valuation))), 4)


def _valuation_position(values: Mapping[str, float]) -> str:
    if "sector_pe_zscore" not in values and "sector_pb_zscore" not in values:
        return "unavailable"
    pe_z = float(values.get("sector_pe_zscore", 0.0))
    pb_z = float(values.get("sector_pb_zscore", 0.0))
    combined = (pe_z + pb_z) / 2.0
    if combined <= -0.75:
        return "cheaper_than_peers"
    if combined >= 0.75:
        return "expensive_vs_peers"
    if abs(combined) <= 0.25:
        return "near_peer_median"
    return "mildly_expensive_vs_peers" if combined > 0.0 else "mildly_cheaper_than_peers"


def _valuation_note(values: Mapping[str, float]) -> str:
    position = _valuation_position(values)
    if position == "unavailable":
        return "Peer valuation context unavailable."
    pe_z = float(values.get("sector_pe_zscore", 0.0))
    pb_z = float(values.get("sector_pb_zscore", 0.0))
    return f"Sector PE z-score {pe_z:.2f}, PB z-score {pb_z:.2f}."


def _earnings_yield(pe: float | None) -> float | None:
    if pe is None or pe <= 0:
        return None
    return round(1.0 / pe, 4)


def _peg_like(pe: float | None, growth: float | None) -> float | None:
    if pe is None or growth is None or growth <= 0:
        return None
    return round(pe / (growth * 100.0), 4)


def _event_governance_risk(components: Mapping[str, float]) -> float | None:
    vals = [
        float(components[k])
        for k in ("risk_event_risk", "risk_governance_risk", "risk_text_uncertainty_risk")
        if k in components
    ]
    if not vals:
        return None
    return round(sum(vals), 4)


def _banking_metrics(values: Mapping[str, float]) -> BankingMetrics:
    applicable = float(values.get("banking_sector_applicable", 0.0)) >= 0.5
    if not applicable:
        return BankingMetrics(applicable=False)
    available = float(values.get("banking_factor_available", 0.0)) >= 0.5
    return BankingMetrics(
        applicable=True,
        available=available,
        metric_coverage=_display_score(values.get("banking_metric_coverage")),
        banking_quality_score=_display_score(values.get("banking_quality_score")),
        asset_quality_score=_display_score(values.get("bank_asset_quality_score")),
        capital_strength_score=_display_score(values.get("bank_capital_strength_score")),
        franchise_strength_score=_display_score(values.get("bank_franchise_strength_score")),
        profitability_score=_display_score(values.get("bank_profitability_score")),
        efficiency_score=_display_score(values.get("bank_efficiency_score")),
        nim_pct=_get(values, "bank_nim_pct"),
        gnpa_pct=_get(values, "bank_gnpa_pct"),
        nnpa_pct=_get(values, "bank_nnpa_pct"),
        casa_pct=_get(values, "bank_casa_pct"),
        cet1_pct=_get(values, "bank_cet1_pct"),
        car_pct=_get(values, "bank_capital_adequacy_pct"),
        loan_to_deposit_pct=_get(values, "bank_loan_to_deposit_pct"),
    )


def _display_score(value: object) -> float | None:
    if value is None:
        return None
    return round(float(value) * 100.0, 2)


def _missing_data_warnings(values: Mapping[str, float]) -> list[str]:
    warnings: list[str] = []
    fundamental_keys = [
        "growth_quality",
        "profitability_quality",
        "valuation_sanity",
        "debt_to_equity",
        "cfo_pat_ratio",
    ]
    event_keys = [
        "recent_positive_event_score",
        "recent_negative_event_score",
        "management_tone_score",
        "catalyst_strength_score",
    ]
    if all(abs(float(values.get(k, 0.0))) <= 1e-12 for k in fundamental_keys):
        warnings.append("Fundamental metrics unavailable or all-zero; long-term conviction should be discounted.")
    if all(abs(float(values.get(k, 0.0))) <= 1e-12 for k in event_keys):
        warnings.append("Document/NLP event metrics unavailable or all-zero.")
    if "trend_strength" not in values or "momentum_strength" not in values:
        warnings.append("Technical feature coverage incomplete.")
    if float(values.get("feature_coverage_score", 1.0)) < 0.50:
        warnings.append("Overall feature coverage is below institutional-readiness threshold.")
    if float(values.get("research_readiness_score", 1.0)) < 0.45:
        warnings.append("Research readiness is weak; treat signal as exploratory until data coverage improves.")
    if float(values.get("banking_sector_applicable", 0.0)) >= 0.5:
        if float(values.get("banking_factor_available", 0.0)) < 0.5:
            warnings.append("Bank/NBFC-specific factors unavailable; confidence is discounted.")
        elif float(values.get("banking_metric_coverage", 0.0)) < 0.5:
            warnings.append("Bank/NBFC-specific factor coverage is sparse; review vendor mapping.")
    return warnings


def _selection_summary(signal: SignalResult, score: float) -> str:
    if signal.category.endswith("candidate"):
        return (
            f"Selected as {signal.category} with score {score:.1f}. "
            f"Primary rationale: {signal.explanation.ranking_reason}"
        )
    return (
        f"Not selected as a candidate. Score {score:.1f}. "
        f"Reason: {signal.explanation.rejection_reason or signal.explanation.ranking_reason}"
    )


def _risk_monitorables(risk_flags: Iterable[str], values: Mapping[str, float]) -> list[str]:
    out: list[str] = []
    flags = set(risk_flags)
    if "liquidity_risk" in flags:
        out.append("Liquidity dries up or volume confirmation weakens further.")
    if "volatility_risk" in flags:
        out.append("ATR/volatility expands and invalidates position sizing assumptions.")
    if "leverage_risk" in flags:
        out.append("Debt metrics or interest coverage deteriorate.")
    if float(values.get("valuation_sanity", 0.5)) < 0.25:
        out.append("Valuation remains stretched versus history or sector.")
    if float(values.get("research_readiness_score", 0.5)) < 0.45:
        out.append("Insufficient research-readiness support from coverage, liquidity, and peer context.")
    if not out:
        out.append("Main risk is ordinary thesis drift: price trend, earnings, or event context weakens.")
    return out


def _monitorables(values: Mapping[str, float], risk_flags: Iterable[str]) -> list[str]:
    monitor = [
        "Quarterly revenue and margin trend",
        "Price behavior around major moving averages",
        "Volume participation on advances versus declines",
    ]
    if float(values.get("uncertainty_penalty", 0.0)) > 0.3:
        monitor.append("Uncertainty in management commentary and filings")
    if float(values.get("cross_sectional_momentum_rank", 0.5)) < 0.35:
        monitor.append("Relative momentum rank versus covered universe")
    if float(values.get("feature_coverage_score", 1.0)) < 0.65:
        monitor.append("Coverage gaps in financial, valuation, event, or sector context")
    if "leverage_risk" in set(risk_flags):
        monitor.append("Debt reduction, refinancing cost, and interest coverage")
    if float(values.get("banking_sector_applicable", 0.0)) >= 0.5:
        monitor.append("Bank asset quality, deposit franchise, CET1/CAR, and NIM trend")
    return monitor
