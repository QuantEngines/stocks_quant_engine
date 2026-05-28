"""Missing-data report with reusable cross-engine coverage discovery."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from stock_screener_engine.config.settings import DataSourceEntitlementSettings


@dataclass(frozen=True)
class DataRequirement:
    """One raw variable still relevant to the stock intelligence roadmap."""

    name: str
    domain: str
    priority: str
    preferred_sources: list[str] = field(default_factory=list)
    required_for: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CrossEngineVariable:
    requirement: str
    upstream_engine: str
    upstream_variable: str
    source_path: str
    status: str = "configured_upstream"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


CRITICAL = "critical"
HIGH = "high"
MEDIUM = "medium"
LOW = "low"


CROSS_ENGINE_REUSABLE: frozenset[str] = frozenset(
    {
        "usd_inr",
        "brent_crude",
        "gold_price",
        "india_10y_yield",
        "us_10y_yield",
        "repo_rate",
        "cpi_inflation",
        "wpi_inflation",
        "iip_growth",
        "pmi_manufacturing",
        "pmi_services",
        "credit_growth",
        "system_liquidity",
        "capacity_utilization",
        "forex_reserves",
        "trade_balance",
        "money_supply",
        "core_inflation",
        "global_pmi",
        "dxy",
        "india_vix",
        "vix_proxy",
        "nifty50_index",
        "nifty_bank_index",
        "sector_index_ohlcv",
        "sector_breadth",
        "sector_advance_decline",
        "sector_dispersion",
        "nse_index_option_pcr",
        "nse_index_oi_churn",
        "macro_event_calendar_basic",
        "rbi_mpc_dates",
        "union_budget_dates",
        "gdp_release_dates",
        "cpi_release_dates",
        "iip_release_dates",
        "wpi_release_dates",
        "pmi_release_dates",
    }
)


PARTIAL_OR_NEEDS_VALIDATION: frozenset[str] = frozenset(
    {
        "deposit_growth",
        "banking_system_credit_deposit_ratio",
        "gst_collection",
        "e_way_bill_count",
        "monsoon_rainfall_deviation",
        "rural_demand_proxy",
        "capex_cycle_proxy",
        "sector_fii_flow_value",
        "sector_dii_flow_value",
    }
)


VARIABLE_DEFINITIONS: dict[str, str] = {
    "point_in_time_as_of_date": "Date when data became usable.",
    "filing_date": "Official filing submission date.",
    "result_announcement_timestamp": "Timestamp of result announcement.",
    "exchange_announcement_timestamp": "Exchange disclosure publication time.",
    "data_vendor_timestamp": "Vendor data update timestamp.",
    "source_document_url": "Link to source document.",
    "source_document_id": "Unique source document identifier.",
    "source_confidence_score": "Reliability score for sourced data.",
    "data_revision_version": "Version of revised data point.",
    "restatement_flag": "Marks restated reported data.",
    "survivorship_status": "Listed, delisted, merged, or inactive.",
    "delisting_date": "Date security stopped trading.",
    "listing_date": "Date security started trading.",
    "symbol_change_history": "Historical ticker symbol changes.",
    "isin_change_history": "Historical ISIN changes.",
    "index_membership_start_date": "Date security entered an index.",
    "index_membership_end_date": "Date security exited an index.",
    "free_float_market_cap_history": "Historical free-float market value.",
    "shares_outstanding_history": "Historical total shares outstanding.",
    "float_shares_history": "Historical freely tradable shares.",
    "official_delivery_quantity": "Exchange-reported deliverable share count.",
    "official_delivery_percentage": "Deliverable volume as trade percentage.",
    "trade_count": "Number of trades executed.",
    "futures_open_interest_history": "Historical futures open interest.",
    "options_chain_history": "Historical option chain snapshots.",
    "earnings_calendar_date": "Expected or actual earnings date.",
    "earnings_surprise": "Earnings versus consensus expectation.",
    "revenue_surprise": "Revenue versus consensus expectation.",
    "eps_surprise": "EPS versus consensus expectation.",
    "estimate_revision_1m": "One-month consensus estimate change.",
    "estimate_revision_3m": "Three-month consensus estimate change.",
    "estimate_revision_6m": "Six-month consensus estimate change.",
    "analyst_count_history": "Historical number of covering analysts.",
    "consensus_dispersion": "Spread across analyst estimates.",
    "target_price_revision": "Change in consensus target price.",
    "management_guidance": "Forward guidance from management.",
    "guidance_revision": "Change in management guidance.",
    "auditor_name": "Company statutory auditor name.",
    "auditor_qualification_flag": "Flags qualified audit opinion.",
    "auditor_resignation_flag": "Flags auditor resignation event.",
    "related_party_transaction_value": "Value of related-party transactions.",
    "contingent_liabilities": "Potential liabilities not yet booked.",
    "pledged_shares_history": "Historical pledged promoter holdings.",
    "insider_trade_value": "Value of insider trades.",
    "bulk_deal_value": "Value of exchange bulk deals.",
    "block_deal_value": "Value of exchange block deals.",
    "mutual_fund_holding_pct": "Mutual fund ownership percentage.",
    "insurance_holding_pct": "Insurance company ownership percentage.",
    "fpi_holding_pct": "Foreign portfolio investor ownership.",
    "dii_flow_value": "Domestic institutional net flow.",
    "fii_flow_value": "Foreign institutional net flow.",
    "sector_fii_flow_value": "Sector-level foreign institutional flow.",
    "sector_dii_flow_value": "Sector-level domestic institutional flow.",
    "sector_constituents": "Stocks mapped to each sector.",
    "sector_policy_event_score": "Score for policy impact by sector.",
    "usd_inr": "USD/INR exchange rate.",
    "brent_crude": "Brent crude oil price.",
    "gold_price": "Gold spot or futures price.",
    "india_10y_yield": "Indian 10-year bond yield.",
    "us_10y_yield": "US 10-year treasury yield.",
    "repo_rate": "RBI policy repo rate.",
    "cpi_inflation": "Consumer price inflation rate.",
    "wpi_inflation": "Wholesale price inflation rate.",
    "iip_growth": "Industrial production growth rate.",
    "pmi_manufacturing": "Manufacturing PMI reading.",
    "pmi_services": "Services PMI reading.",
    "credit_growth": "Bank credit growth rate.",
    "system_liquidity": "Banking system liquidity condition.",
    "capacity_utilization": "Industrial capacity utilization rate.",
    "forex_reserves": "India foreign exchange reserves.",
    "trade_balance": "Exports minus imports balance.",
    "money_supply": "Broad money supply measure.",
    "core_inflation": "Inflation excluding volatile items.",
    "global_pmi": "Global purchasing managers index.",
    "dxy": "US dollar index.",
    "india_vix": "Indian equity volatility index.",
    "vix_proxy": "Global volatility proxy.",
    "nifty50_index": "Nifty 50 index level/history.",
    "nifty_bank_index": "Bank Nifty index level/history.",
    "sector_index_ohlcv": "Sector index OHLCV history.",
    "sector_breadth": "Sector participation breadth measure.",
    "sector_advance_decline": "Advancers versus decliners by sector.",
    "sector_dispersion": "Return dispersion across sectors.",
    "nse_index_option_pcr": "Index option put-call ratio.",
    "nse_index_oi_churn": "Index option open-interest churn.",
    "macro_event_calendar_basic": "Important macro event calendar.",
    "rbi_mpc_dates": "RBI policy meeting dates.",
    "union_budget_dates": "Union Budget event dates.",
    "gdp_release_dates": "GDP data release dates.",
    "cpi_release_dates": "CPI data release dates.",
    "iip_release_dates": "IIP data release dates.",
    "wpi_release_dates": "WPI data release dates.",
    "pmi_release_dates": "PMI data release dates.",
    "gst_collection": "Monthly GST tax collection.",
    "e_way_bill_count": "E-way bill activity count.",
    "monsoon_rainfall_deviation": "Rainfall deviation from normal.",
    "rural_demand_proxy": "Proxy for rural demand strength.",
    "capex_cycle_proxy": "Proxy for investment cycle strength.",
    "deposit_growth": "Bank deposit growth rate.",
    "banking_system_credit_deposit_ratio": "System credit-to-deposit ratio.",
    "raw_pdf_text": "Extracted text from PDFs.",
    "annual_report_sections": "Detected annual-report sections.",
    "concall_transcript_text": "Text of earnings call transcript.",
    "management_guidance_source_text": "Text supporting guidance extraction.",
    "risk_factor_source_text": "Text supporting risk extraction.",
    "litigation_event_flag": "Flags litigation-related event.",
    "regulatory_event_flag": "Flags regulatory event.",
    "news_full_text": "Full text of news article.",
    "news_source_reliability": "Reliability rating of news source.",
    "event_timestamp": "Timestamp of detected event.",
    "event_category": "Class of detected event.",
    "stt_cost": "Securities transaction tax cost.",
    "exchange_charges": "Exchange transaction charges.",
    "brokerage_cost": "Brokerage fee amount.",
    "tax_cost": "Applicable trading tax cost.",
}


MISSING_DATA_CSV_COLUMNS: list[str] = [
    "variable",
    "definition",
    "domain",
    "priority",
    "status",
    "preferred_sources",
    "required_for",
    "procurement_action",
    "notes",
    "upstream_coverage",
]


BASE_REQUIREMENTS: tuple[DataRequirement, ...] = (
    DataRequirement(
        "point_in_time_as_of_date",
        "auditability",
        CRITICAL,
        ["finedge", "nse_bse", "future_central_data_layer"],
        ["long_term_scan", "deep_research", "backtests"],
        "Every raw factor needs an explicit as-of date to avoid look-ahead bias.",
    ),
    DataRequirement("filing_date", "auditability", CRITICAL, ["nse_bse", "finedge"], ["deep_research", "backtests"]),
    DataRequirement(
        "result_announcement_timestamp",
        "auditability",
        CRITICAL,
        ["nse_bse", "finedge"],
        ["event_signals", "backtests"],
    ),
    DataRequirement(
        "exchange_announcement_timestamp",
        "auditability",
        CRITICAL,
        ["nse_bse", "finedge"],
        ["event_signals", "document_intelligence"],
    ),
    DataRequirement("data_vendor_timestamp", "auditability", HIGH, ["all_vendors"], ["quality_monitoring"]),
    DataRequirement("source_document_url", "auditability", HIGH, ["nse_bse", "finedge"], ["deep_research", "document_intelligence"]),
    DataRequirement("source_document_id", "auditability", HIGH, ["nse_bse", "finedge"], ["deep_research", "document_intelligence"]),
    DataRequirement("source_confidence_score", "auditability", HIGH, ["internal_qa"], ["all_research_outputs"]),
    DataRequirement("data_revision_version", "auditability", HIGH, ["future_central_data_layer"], ["backtests", "quality_monitoring"]),
    DataRequirement("restatement_flag", "auditability", HIGH, ["finedge", "nse_bse"], ["fundamental_research"]),
    DataRequirement("survivorship_status", "security_master", CRITICAL, ["nse_bse"], ["backtests"]),
    DataRequirement("delisting_date", "security_master", CRITICAL, ["nse_bse"], ["backtests"]),
    DataRequirement("listing_date", "security_master", HIGH, ["nse_bse", "finedge"], ["universe_filters", "backtests"]),
    DataRequirement("symbol_change_history", "security_master", CRITICAL, ["nse_bse"], ["backtests", "corporate_action_reconciliation"]),
    DataRequirement("isin_change_history", "security_master", HIGH, ["nse_bse"], ["identity_resolution"]),
    DataRequirement("index_membership_start_date", "security_master", CRITICAL, ["nse_bse"], ["index_backtests", "sector_rotation"]),
    DataRequirement("index_membership_end_date", "security_master", CRITICAL, ["nse_bse"], ["index_backtests", "sector_rotation"]),
    DataRequirement("free_float_market_cap_history", "security_master", HIGH, ["nse_bse", "finedge"], ["liquidity", "ranking"]),
    DataRequirement("shares_outstanding_history", "security_master", HIGH, ["nse_bse", "finedge"], ["valuation", "corporate_actions"]),
    DataRequirement("float_shares_history", "security_master", MEDIUM, ["nse_bse", "finedge"], ["liquidity", "ownership"]),
    DataRequirement(
        "official_delivery_quantity",
        "market_microstructure",
        HIGH,
        ["nse_bse"],
        ["swing_scan", "volume_confirmation"],
        "Stock engine has canonical ingestion support; official CSV/feed population is still required.",
    ),
    DataRequirement(
        "official_delivery_percentage",
        "market_microstructure",
        HIGH,
        ["nse_bse"],
        ["swing_scan", "participation_filters"],
        "Stock engine maps this into canonical snapshot delivery_ratio when data is present.",
    ),
    DataRequirement("trade_count", "market_microstructure", MEDIUM, ["nse_bse"], ["liquidity_quality", "false_breakout_filters"]),
    DataRequirement("futures_open_interest_history", "derivatives", HIGH, ["nse_bse", "future_options_engine_layer"], ["swing_scan", "regime"]),
    DataRequirement("options_chain_history", "derivatives", HIGH, ["nse_bse", "future_options_engine_layer"], ["sentiment", "regime"]),
    DataRequirement("earnings_calendar_date", "estimates_events", HIGH, ["finedge", "indianapi", "nse_bse"], ["event_signals"]),
    DataRequirement("earnings_surprise", "estimates_events", HIGH, ["paid_consensus_vendor"], ["long_term_scan", "event_studies"]),
    DataRequirement("revenue_surprise", "estimates_events", HIGH, ["paid_consensus_vendor"], ["long_term_scan", "event_studies"]),
    DataRequirement("eps_surprise", "estimates_events", HIGH, ["paid_consensus_vendor"], ["long_term_scan", "event_studies"]),
    DataRequirement("estimate_revision_1m", "estimates_events", HIGH, ["paid_consensus_vendor"], ["long_term_scan"]),
    DataRequirement("estimate_revision_3m", "estimates_events", HIGH, ["paid_consensus_vendor"], ["long_term_scan"]),
    DataRequirement("estimate_revision_6m", "estimates_events", MEDIUM, ["paid_consensus_vendor"], ["long_term_scan"]),
    DataRequirement("analyst_count_history", "estimates_events", MEDIUM, ["paid_consensus_vendor"], ["confidence"]),
    DataRequirement("consensus_dispersion", "estimates_events", MEDIUM, ["paid_consensus_vendor"], ["uncertainty"]),
    DataRequirement("target_price_revision", "estimates_events", LOW, ["paid_consensus_vendor"], ["research_context"]),
    DataRequirement("management_guidance", "estimates_events", HIGH, ["nse_bse", "finedge", "document_intelligence"], ["deep_research"]),
    DataRequirement("guidance_revision", "estimates_events", MEDIUM, ["document_intelligence"], ["event_signals"]),
    DataRequirement("auditor_name", "governance", MEDIUM, ["nse_bse", "document_intelligence"], ["governance"]),
    DataRequirement("auditor_qualification_flag", "governance", HIGH, ["nse_bse", "document_intelligence"], ["risk"]),
    DataRequirement("auditor_resignation_flag", "governance", HIGH, ["nse_bse", "document_intelligence"], ["risk_alerts"]),
    DataRequirement("related_party_transaction_value", "governance", HIGH, ["annual_reports", "document_intelligence"], ["governance"]),
    DataRequirement("contingent_liabilities", "governance", HIGH, ["annual_reports", "document_intelligence"], ["risk"]),
    DataRequirement("pledged_shares_history", "ownership", HIGH, ["finedge", "nse_bse"], ["governance"]),
    DataRequirement("insider_trade_value", "ownership", MEDIUM, ["nse_bse"], ["event_signals"]),
    DataRequirement("bulk_deal_value", "ownership_flow", MEDIUM, ["nse_bse"], ["flow_signals"]),
    DataRequirement("block_deal_value", "ownership_flow", MEDIUM, ["nse_bse"], ["flow_signals"]),
    DataRequirement("mutual_fund_holding_pct", "ownership_flow", MEDIUM, ["finedge", "amfi", "nse_bse"], ["ownership"]),
    DataRequirement("insurance_holding_pct", "ownership_flow", MEDIUM, ["finedge", "nse_bse"], ["ownership"]),
    DataRequirement("fpi_holding_pct", "ownership_flow", MEDIUM, ["finedge", "nse_bse"], ["ownership"]),
    DataRequirement("dii_flow_value", "ownership_flow", HIGH, ["nse_bse", "paid_flow_vendor"], ["sector_rotation"]),
    DataRequirement("fii_flow_value", "ownership_flow", HIGH, ["nse_bse", "paid_flow_vendor"], ["sector_rotation"]),
    DataRequirement("sector_fii_flow_value", "ownership_flow", MEDIUM, ["paid_flow_vendor"], ["sector_rotation"]),
    DataRequirement("sector_dii_flow_value", "ownership_flow", MEDIUM, ["paid_flow_vendor"], ["sector_rotation"]),
    DataRequirement("sector_constituents", "sector", HIGH, ["nse_bse", "finedge"], ["sector_rotation", "sector_breadth"]),
    DataRequirement("sector_policy_event_score", "sector", MEDIUM, ["document_intelligence", "macro_engine"], ["sector_research"]),
    DataRequirement("usd_inr", "cross_engine_macro", HIGH, ["fx_quant_engine", "macro_engine"], ["macro_regime", "export_sensitivity"]),
    DataRequirement("brent_crude", "cross_engine_macro", HIGH, ["macro_engine", "options_engine"], ["sector_research", "energy_sensitivity"]),
    DataRequirement("gold_price", "cross_engine_macro", MEDIUM, ["macro_engine", "options_engine"], ["macro_regime"]),
    DataRequirement("india_10y_yield", "cross_engine_rates", HIGH, ["rates_quant_engine", "macro_engine"], ["valuation_regime"]),
    DataRequirement("us_10y_yield", "cross_engine_rates", MEDIUM, ["rates_quant_engine", "macro_engine"], ["global_regime"]),
    DataRequirement("repo_rate", "cross_engine_rates", HIGH, ["rates_quant_engine", "macro_engine"], ["financials_sector"]),
    DataRequirement("cpi_inflation", "cross_engine_macro", HIGH, ["macro_engine"], ["sector_research", "margin_pressure"]),
    DataRequirement("wpi_inflation", "cross_engine_macro", MEDIUM, ["macro_engine"], ["input_cost_pressure"]),
    DataRequirement("iip_growth", "cross_engine_macro", MEDIUM, ["macro_engine"], ["growth_regime"]),
    DataRequirement("pmi_manufacturing", "cross_engine_macro", MEDIUM, ["macro_engine"], ["growth_regime"]),
    DataRequirement("pmi_services", "cross_engine_macro", MEDIUM, ["macro_engine"], ["growth_regime"]),
    DataRequirement("credit_growth", "cross_engine_macro", HIGH, ["macro_engine", "rates_quant_engine"], ["financials_sector"]),
    DataRequirement("system_liquidity", "cross_engine_macro", MEDIUM, ["rates_quant_engine", "macro_engine"], ["financials_sector"]),
    DataRequirement("capacity_utilization", "cross_engine_macro", MEDIUM, ["macro_engine"], ["capex_cycle"]),
    DataRequirement("forex_reserves", "cross_engine_macro", MEDIUM, ["macro_engine"], ["external_stability"]),
    DataRequirement("trade_balance", "cross_engine_macro", MEDIUM, ["macro_engine"], ["external_stability"]),
    DataRequirement("money_supply", "cross_engine_macro", MEDIUM, ["macro_engine"], ["liquidity_regime"]),
    DataRequirement("core_inflation", "cross_engine_macro", MEDIUM, ["macro_engine"], ["inflation_regime"]),
    DataRequirement("global_pmi", "cross_engine_macro", LOW, ["macro_engine"], ["export_demand"]),
    DataRequirement("dxy", "cross_engine_macro", MEDIUM, ["macro_engine", "options_engine"], ["currency_regime"]),
    DataRequirement("india_vix", "cross_engine_market", HIGH, ["macro_engine", "options_engine"], ["risk_regime"]),
    DataRequirement("vix_proxy", "cross_engine_market", MEDIUM, ["macro_engine", "options_engine"], ["global_risk_regime"]),
    DataRequirement("nifty50_index", "cross_engine_market", HIGH, ["macro_engine", "stock_engine"], ["relative_strength"]),
    DataRequirement("nifty_bank_index", "cross_engine_market", HIGH, ["macro_engine", "stock_engine"], ["financials_relative_strength"]),
    DataRequirement("sector_index_ohlcv", "cross_engine_sector", HIGH, ["macro_engine", "nse_bse"], ["sector_rotation"]),
    DataRequirement("sector_breadth", "cross_engine_sector", HIGH, ["macro_engine", "stock_engine"], ["sector_rotation"]),
    DataRequirement("sector_advance_decline", "cross_engine_sector", MEDIUM, ["macro_engine", "nse_bse"], ["sector_breadth"]),
    DataRequirement("sector_dispersion", "cross_engine_sector", MEDIUM, ["macro_engine"], ["sector_rotation"]),
    DataRequirement("nse_index_option_pcr", "cross_engine_derivatives", MEDIUM, ["macro_engine", "options_engine"], ["risk_regime"]),
    DataRequirement("nse_index_oi_churn", "cross_engine_derivatives", MEDIUM, ["macro_engine", "options_engine"], ["risk_regime"]),
    DataRequirement("macro_event_calendar_basic", "cross_engine_events", MEDIUM, ["macro_engine", "options_engine"], ["event_risk"]),
    DataRequirement("rbi_mpc_dates", "cross_engine_events", MEDIUM, ["macro_engine", "options_engine"], ["event_risk"]),
    DataRequirement("union_budget_dates", "cross_engine_events", MEDIUM, ["macro_engine", "options_engine"], ["event_risk"]),
    DataRequirement("gdp_release_dates", "cross_engine_events", LOW, ["macro_engine", "options_engine"], ["event_risk"]),
    DataRequirement("cpi_release_dates", "cross_engine_events", LOW, ["macro_engine", "options_engine"], ["event_risk"]),
    DataRequirement("iip_release_dates", "cross_engine_events", LOW, ["macro_engine", "options_engine"], ["event_risk"]),
    DataRequirement("wpi_release_dates", "cross_engine_events", LOW, ["macro_engine", "options_engine"], ["event_risk"]),
    DataRequirement("pmi_release_dates", "cross_engine_events", LOW, ["macro_engine", "options_engine"], ["event_risk"]),
    DataRequirement("gst_collection", "macro_activity", MEDIUM, ["government_sources", "macro_engine"], ["sector_research"]),
    DataRequirement("e_way_bill_count", "macro_activity", MEDIUM, ["government_sources", "macro_engine"], ["sector_research"]),
    DataRequirement("monsoon_rainfall_deviation", "macro_activity", MEDIUM, ["imd", "macro_engine"], ["rural_demand"]),
    DataRequirement("rural_demand_proxy", "macro_activity", MEDIUM, ["macro_engine"], ["sector_research"]),
    DataRequirement("capex_cycle_proxy", "macro_activity", MEDIUM, ["macro_engine"], ["sector_research"]),
    DataRequirement("deposit_growth", "banking_macro", MEDIUM, ["rbi", "rates_quant_engine"], ["banking_sector"]),
    DataRequirement("banking_system_credit_deposit_ratio", "banking_macro", MEDIUM, ["rbi", "rates_quant_engine"], ["banking_sector"]),
    DataRequirement("raw_pdf_text", "documents", HIGH, ["nse_bse", "finedge", "document_intelligence"], ["deep_research"]),
    DataRequirement("annual_report_sections", "documents", HIGH, ["document_intelligence"], ["deep_research"]),
    DataRequirement("concall_transcript_text", "documents", HIGH, ["finedge", "document_intelligence"], ["deep_research"]),
    DataRequirement("management_guidance_source_text", "documents", HIGH, ["document_intelligence"], ["deep_research"]),
    DataRequirement("risk_factor_source_text", "documents", HIGH, ["document_intelligence"], ["risk"]),
    DataRequirement("litigation_event_flag", "event_nlp", HIGH, ["nse_bse", "document_intelligence"], ["risk_alerts"]),
    DataRequirement("regulatory_event_flag", "event_nlp", HIGH, ["nse_bse", "document_intelligence"], ["risk_alerts"]),
    DataRequirement("news_full_text", "event_nlp", MEDIUM, ["paid_news_vendor"], ["event_signals"]),
    DataRequirement("news_source_reliability", "event_nlp", MEDIUM, ["paid_news_vendor", "internal_qa"], ["confidence"]),
    DataRequirement("event_timestamp", "event_nlp", HIGH, ["nse_bse", "news_vendor", "document_intelligence"], ["event_signals"]),
    DataRequirement("event_category", "event_nlp", HIGH, ["document_intelligence"], ["event_signals"]),
    DataRequirement("stt_cost", "transaction_costs", MEDIUM, ["nse_bse", "broker_terms"], ["backtests"]),
    DataRequirement("exchange_charges", "transaction_costs", MEDIUM, ["nse_bse", "broker_terms"], ["backtests"]),
    DataRequirement("brokerage_cost", "transaction_costs", MEDIUM, ["broker_terms"], ["backtests"]),
    DataRequirement("tax_cost", "transaction_costs", MEDIUM, ["nse_bse", "broker_terms"], ["backtests"]),
)


UPSTREAM_ALIASES: dict[str, set[str]] = {
    "usd_inr": {"usd_inr", "inr_usd", "inr_x", "fx_usdinr"},
    "brent_crude": {"brent", "brent_crude", "crude_oil", "oil", "cl_f"},
    "gold_price": {"gold", "gold_price", "gc_f"},
    "india_10y_yield": {"india_10y", "india_10y_yield", "in_10y", "indian_10y"},
    "us_10y_yield": {"us_10y", "us_10y_yield", "us10y", "tnx"},
    "repo_rate": {"repo_rate", "policy_rate", "rbi_repo_rate"},
    "cpi_inflation": {"india_cpi", "cpi", "cpi_inflation", "in_cpi_headline"},
    "core_inflation": {"core_inflation", "india_core_inflation", "in_cpi_core"},
    "wpi_inflation": {"wpi", "wpi_inflation", "in_wpi"},
    "iip_growth": {"india_iip", "iip", "iip_growth", "in_iip"},
    "pmi_manufacturing": {"india_pmi_manufacturing", "pmi_manufacturing", "in_pmi_manufacturing"},
    "pmi_services": {"india_pmi_services", "pmi_services", "in_pmi_services"},
    "credit_growth": {"bank_credit_growth", "credit_growth", "in_credit_growth"},
    "system_liquidity": {"system_liquidity", "india_system_liquidity", "in_system_liquidity", "liquidity_adjustment_facility"},
    "capacity_utilization": {"capacity_utilization", "india_capacity_utilization"},
    "forex_reserves": {"forex_reserves", "india_forex_reserves"},
    "trade_balance": {"trade_balance", "india_trade_balance"},
    "money_supply": {"money_supply", "m3", "india_money_supply"},
    "global_pmi": {"global_pmi"},
    "dxy": {"dxy", "dxy_proxy", "dx_y_nyb"},
    "india_vix": {"india_vix", "indiavix"},
    "vix_proxy": {"vix_proxy", "vix", "us_vix"},
    "nifty50_index": {"nifty50", "nifty_50", "nifty50_index"},
    "nifty_bank_index": {"nifty_bank", "banknifty", "nifty_bank_index"},
    "sector_breadth": {"sector_breadth", "nse_market_breadth", "nifty500_breadth", "nse_equity_breadth"},
    "sector_advance_decline": {"sector_advance_decline", "advance_decline", "nse_market_breadth"},
    "sector_dispersion": {"sector_dispersion", "nse_sector_dispersion"},
    "nse_index_option_pcr": {"nse_index_option_pcr", "index_option_pcr"},
    "nse_index_oi_churn": {"nse_index_oi_churn", "index_oi_churn"},
}


def build_missing_data_report(
    *,
    quant_root: Path | None = None,
    entitlements: Sequence[DataSourceEntitlementSettings] = (),
    include_cross_engine: bool = True,
) -> dict[str, object]:
    """Build the next-version data gap list for stock-engine sourcing decisions."""
    resolved_quant_root = _resolve_quant_root(quant_root)
    upstream = (
        discover_cross_engine_coverage(resolved_quant_root)
        if include_cross_engine
        else {}
    )
    rows = [_requirement_row(item, upstream.get(item.name, [])) for item in BASE_REQUIREMENTS]
    report = {
        "pipeline": "missing_data_list",
        "run_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "quant_root": str(resolved_quant_root),
        "central_data_layer_assumption": (
            "Sibling macro/rates/options variables are treated as reusable upstream coverage "
            "that should move into a future shared data layer. They are not yet wired into "
            "the stock engine."
        ),
        "source_summary": _source_summary(entitlements),
        "rows": rows,
        "removed_from_stock_specific_procurement": [row for row in rows if row["status"] == "covered_upstream_not_wired"],
        "partial_or_needs_validation": [row for row in rows if row["status"] in {"partial_needs_validation", "stock_ingestion_ready_needs_data"}],
        "still_missing": [row for row in rows if row["status"] == "still_required"],
        "counts": _status_counts(rows),
    }
    report["markdown"] = render_missing_data_markdown(report)
    return report


def discover_cross_engine_coverage(quant_root: Path) -> dict[str, list[dict[str, object]]]:
    """Discover sibling-engine macro/rates/options variables that stock can reuse later."""
    coverage: dict[str, list[CrossEngineVariable]] = defaultdict(list)
    _discover_macro_engine(quant_root, coverage)
    _discover_rates_engine(quant_root, coverage)
    _discover_options_engine(quant_root, coverage)
    return {key: [item.to_dict() for item in items] for key, items in sorted(coverage.items())}


def render_missing_data_markdown(report: Mapping[str, object]) -> str:
    counts = report.get("counts", {})
    lines = [
        "# Stock Engine Missing Data List",
        "",
        f"- Run at: {report.get('run_at', '')}",
        f"- Quant root: {report.get('quant_root', '')}",
        f"- Still required: {_mapping_value(counts, 'still_required')}",
        f"- Partial, ingestion-ready, or needs validation: {_mapping_value(counts, 'partial_or_needs_validation')}",
        f"- Covered upstream but not wired: {_mapping_value(counts, 'covered_upstream_not_wired')}",
        "",
        "## Operating Assumption",
        "",
        str(report.get("central_data_layer_assumption", "")),
        "",
        "## Current Source Position",
        "",
        "| Source | Status | Domains | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for item in report.get("source_summary", []):
        if isinstance(item, Mapping):
            lines.append(
                "| {source} | {status} | {domains} | {notes} |".format(
                    source=item.get("source_id", ""),
                    status=item.get("status", ""),
                    domains=", ".join(item.get("domains", [])) if isinstance(item.get("domains"), list) else "",
                    notes=item.get("notes", ""),
                )
            )

    lines.extend(["", "## Remove From Stock-Specific Procurement", ""])
    _append_requirement_table(
        lines,
        report.get("removed_from_stock_specific_procurement", []),
        include_upstream=True,
    )

    lines.extend(["", "## Partial Or Needs Validation", ""])
    _append_requirement_table(lines, report.get("partial_or_needs_validation", []), include_upstream=True)

    lines.extend(["", "## Still Required", ""])
    _append_requirement_table(lines, report.get("still_missing", []), include_upstream=False)

    lines.extend(
        [
            "",
            "## Near-Term Data Actions",
            "",
            "- Use FinEdge paid onboarding to close financials, valuations, shareholding, banking factors, and document/event coverage.",
            "- Populate NSE/BSE delivery, corporate-action, announcement/PDF, historical constituent, and symbol-change feeds as the exchange audit trail.",
            "- Define the future central data contract for macro/rates/options variables before duplicating those feeds inside stock engine.",
            "- Keep all generated coverage reports under ignored local storage, not inside git-tracked documentation.",
        ]
    )
    return "\n".join(lines) + "\n"


def missing_data_rows_for_csv(report: Mapping[str, object]) -> list[dict[str, object]]:
    """Flatten missing-data rows for a spreadsheet-friendly CSV artifact."""
    rows: list[dict[str, object]] = []
    for row in report.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        variable = str(row.get("name", ""))
        rows.append(
            {
                "variable": variable,
                "definition": VARIABLE_DEFINITIONS.get(variable, _fallback_definition(variable)),
                "domain": row.get("domain", ""),
                "priority": row.get("priority", ""),
                "status": row.get("status", ""),
                "preferred_sources": _join_list(row.get("preferred_sources", [])),
                "required_for": _join_list(row.get("required_for", [])),
                "procurement_action": row.get("procurement_action", ""),
                "notes": row.get("notes", ""),
                "upstream_coverage": _upstream_summary(row.get("upstream_coverage", [])),
            }
        )
    return rows


def _discover_macro_engine(quant_root: Path, coverage: dict[str, list[CrossEngineVariable]]) -> None:
    root = quant_root / "macros_quant_engine"
    sources = [
        root / "configs" / "macro_universe.yaml",
        root / "configs" / "data_sources.yaml",
    ]
    for source in sources:
        if source.exists():
            _discover_yaml(source, "macros_quant_engine", coverage)

    event_builder = root / "scripts" / "data_prep" / "build_historical_macro_events.py"
    if event_builder.exists():
        text = _safe_read_text(event_builder)
        event_map = {
            "macro_event_calendar_basic": "macro_event_calendar_basic",
            "rbi_mpc_dates": "RBI MPC",
            "union_budget_dates": "Union Budget",
            "gdp_release_dates": "India GDP",
            "cpi_release_dates": "CPI",
            "iip_release_dates": "IIP",
            "wpi_release_dates": "WPI",
            "pmi_release_dates": "PMI",
        }
        for requirement, marker in event_map.items():
            if marker in text:
                _add_coverage(coverage, requirement, "macros_quant_engine", marker, event_builder)


def _discover_rates_engine(quant_root: Path, coverage: dict[str, list[CrossEngineVariable]]) -> None:
    source = quant_root / "rates_quant_engine" / "configs" / "data_sources.yaml"
    if source.exists():
        _discover_yaml(source, "rates_quant_engine", coverage)


def _discover_options_engine(quant_root: Path, coverage: dict[str, list[CrossEngineVariable]]) -> None:
    root = quant_root / "options_quant_engine"
    policy = root / "config" / "market_data_policy.py"
    if policy.exists():
        text = _safe_read_text(policy)
        token_map = {
            "usd_inr": "INR=X",
            "gold_price": "GC=F",
            "dxy": "DX-Y.NYB",
            "india_vix": "INDIAVIX",
            "vix_proxy": "^VIX",
            "us_10y_yield": "^TNX",
        }
        for requirement, marker in token_map.items():
            if marker in text:
                _add_coverage(coverage, requirement, "options_quant_engine", marker, policy)

    event_builder = root / "scripts" / "data_prep" / "build_historical_macro_events.py"
    if event_builder.exists():
        text = _safe_read_text(event_builder)
        for requirement, marker in {
            "macro_event_calendar_basic": "macro event",
            "rbi_mpc_dates": "RBI MPC",
            "union_budget_dates": "Union Budget",
            "gdp_release_dates": "India GDP",
            "cpi_release_dates": "CPI",
            "iip_release_dates": "IIP",
            "wpi_release_dates": "WPI",
            "pmi_release_dates": "PMI",
        }.items():
            if marker in text:
                _add_coverage(coverage, requirement, "options_quant_engine", marker, event_builder)


def _discover_yaml(
    source: Path,
    engine: str,
    coverage: dict[str, list[CrossEngineVariable]],
) -> None:
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return
    for raw in _iter_yaml_tokens(payload):
        normalized = _normalize_token(raw)
        for requirement in _map_upstream_token(normalized):
            _add_coverage(coverage, requirement, engine, raw, source)


def _iter_yaml_tokens(payload: Any) -> list[str]:
    tokens: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            tokens.append(str(key))
            tokens.extend(_iter_yaml_tokens(value))
    elif isinstance(payload, list):
        for value in payload:
            tokens.extend(_iter_yaml_tokens(value))
    elif isinstance(payload, (str, int, float)):
        tokens.append(str(payload))
    return tokens


def _map_upstream_token(token: str) -> set[str]:
    matches = {requirement for requirement, aliases in UPSTREAM_ALIASES.items() if token in aliases}
    if token.startswith("nifty_") and token not in {"nifty_50", "nifty_bank"}:
        matches.add("sector_index_ohlcv")
    if token.startswith("nse_sector_"):
        matches.add("sector_index_ohlcv")
    return matches & CROSS_ENGINE_REUSABLE


def _add_coverage(
    coverage: dict[str, list[CrossEngineVariable]],
    requirement: str,
    engine: str,
    upstream_variable: str,
    source_path: Path,
) -> None:
    if requirement not in CROSS_ENGINE_REUSABLE:
        return
    item = CrossEngineVariable(
        requirement=requirement,
        upstream_engine=engine,
        upstream_variable=str(upstream_variable),
        source_path=str(source_path),
    )
    existing = {
        (entry.upstream_engine, entry.upstream_variable, entry.source_path)
        for entry in coverage[requirement]
    }
    key = (item.upstream_engine, item.upstream_variable, item.source_path)
    if key not in existing:
        coverage[requirement].append(item)


def _requirement_row(requirement: DataRequirement, upstream: list[dict[str, object]]) -> dict[str, object]:
    status = "still_required"
    if upstream:
        status = "covered_upstream_not_wired"
    elif requirement.name in {"official_delivery_quantity", "official_delivery_percentage"}:
        status = "stock_ingestion_ready_needs_data"
    elif requirement.name in PARTIAL_OR_NEEDS_VALIDATION:
        status = "partial_needs_validation"

    return {
        **requirement.to_dict(),
        "status": status,
        "upstream_coverage": upstream,
        "procurement_action": _procurement_action(requirement.name, status),
    }


def _procurement_action(name: str, status: str) -> str:
    if status == "covered_upstream_not_wired":
        return "Do not procure again for stock engine; wire through future central data layer."
    if status == "stock_ingestion_ready_needs_data":
        return "Populate official NSE/BSE files or licensed feed into the existing stock-engine table."
    if status == "partial_needs_validation":
        return "Validate existing sibling coverage or identify a small official source gap."
    if name.startswith(("estimate_", "earnings_", "revenue_", "eps_", "consensus_", "analyst_", "target_")):
        return "Evaluate paid consensus/vendor coverage after FinEdge subscription decision."
    if name in {"raw_pdf_text", "annual_report_sections", "concall_transcript_text"}:
        return "Prioritize FinEdge/NSE-BSE document endpoints and document-intelligence ingestion."
    return "Source from the listed primary/alternate providers."


def _source_summary(entitlements: Sequence[DataSourceEntitlementSettings]) -> list[dict[str, object]]:
    rows = []
    for item in entitlements:
        rows.append(
            {
                "source_id": item.source_id,
                "display_name": item.display_name,
                "status": item.status,
                "enabled": item.enabled,
                "domains": item.domains,
                "notes": item.next_action or item.notes,
            }
        )
    return rows


def _status_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = {
        "still_required": 0,
        "partial_or_needs_validation": 0,
        "covered_upstream_not_wired": 0,
        "stock_ingestion_ready_needs_data": 0,
        "total": len(rows),
    }
    for row in rows:
        status = str(row.get("status", ""))
        if status in {"partial_needs_validation", "stock_ingestion_ready_needs_data"}:
            counts["partial_or_needs_validation"] += 1
        if status in counts:
            counts[status] += 1
    return counts


def _append_requirement_table(lines: list[str], rows: object, *, include_upstream: bool) -> None:
    lines.extend(
        [
            "| Variable | Domain | Priority | Preferred Sources | Action | Upstream |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    if not isinstance(rows, list) or not rows:
        lines.append("| None |  |  |  |  |  |")
        return
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        upstream = ""
        if include_upstream:
            upstream = _upstream_summary(row.get("upstream_coverage", []))
        lines.append(
            "| {name} | {domain} | {priority} | {sources} | {action} | {upstream} |".format(
                name=row.get("name", ""),
                domain=row.get("domain", ""),
                priority=row.get("priority", ""),
                sources=", ".join(row.get("preferred_sources", [])) if isinstance(row.get("preferred_sources"), list) else "",
                action=str(row.get("procurement_action", "")).replace("|", "/"),
                upstream=upstream.replace("|", "/"),
            )
        )


def _upstream_summary(items: object) -> str:
    if not isinstance(items, list) or not items:
        return ""
    summaries = []
    for item in items[:3]:
        if isinstance(item, Mapping):
            summaries.append(f"{item.get('upstream_engine')}:{item.get('upstream_variable')}")
    suffix = "" if len(items) <= 3 else f" +{len(items) - 3}"
    return ", ".join(summaries) + suffix


def _join_list(items: object) -> str:
    if not isinstance(items, list):
        return ""
    return ", ".join(str(item) for item in items)


def _mapping_value(mapping: object, key: str) -> object:
    if isinstance(mapping, Mapping):
        return mapping.get(key, 0)
    return 0


def _normalize_token(value: str) -> str:
    token = value.strip().lower()
    token = token.replace("^", "")
    token = token.replace("&", "and")
    token = re.sub(r"[^a-z0-9]+", "_", token)
    token = token.strip("_")
    return token


def _fallback_definition(variable: str) -> str:
    return variable.replace("_", " ").strip().capitalize()


def _resolve_quant_root(quant_root: Path | None) -> Path:
    if quant_root is not None:
        return quant_root.expanduser().resolve()
    cwd = Path.cwd().resolve()
    if cwd.name == "stock_quant_engine":
        return cwd.parent
    package_root = Path(__file__).resolve().parents[2]
    return package_root.parent


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""
