"""Provider-agnostic LLM client interface and deterministic mock implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, task: str, prompt: str, payload: dict[str, object]) -> dict[str, object]:
        raise NotImplementedError


@dataclass
class HeuristicLLMClient:
    """Deterministic stand-in client; replace with real provider adapters in production."""

    model_name: str = "heuristic-finance-v1"

    def complete(self, task: str, prompt: str, payload: dict[str, object]) -> dict[str, object]:
        text = str(payload.get("text", "")).lower()
        title = str(payload.get("title", "")).lower()
        merged = f"{title} {text}"

        if task == "classification":
            category = "general_news"
            if any(t in merged for t in ("earnings", "results", "margin")):
                category = "earnings_result"
            elif any(t in merged for t in ("order", "contract", "deal")):
                category = "order_win"
            elif any(t in merged for t in ("guidance", "outlook")):
                category = "guidance_change"
            elif any(t in merged for t in ("litigation", "regulatory", "penalty")):
                category = "litigation_regulatory"
            return {"category": category, "confidence": 0.78}

        if task == "event_extraction":
            direction = "neutral"
            if any(t in merged for t in ("win", "raised", "upgrade", "strong")):
                direction = "positive"
            elif any(t in merged for t in ("cut", "downgrade", "litigation", "weak")):
                direction = "negative"
            return {
                "event_type": _infer_event_type(merged),
                "short_summary": str(payload.get("title", ""))[:180],
                "expected_direction": direction,
                "event_strength": 0.72 if direction != "neutral" else 0.45,
                "confidence": 0.74,
                "uncertainty_score": 0.22,
                "time_horizon": "near_term",
                "risk_flag": direction == "negative",
                "catalyst_flag": direction == "positive",
                "keywords": _keywords(merged),
            }

        if task == "sentiment":
            return {
                "earnings_sentiment": _score_terms(merged, ("beat", "margin expansion", "growth"), ("miss", "pressure", "decline")),
                "guidance_sentiment": _score_terms(merged, ("raised", "optimistic", "strong demand"), ("cut", "cautious", "weak demand")),
                "governance_sentiment": _score_terms(merged, ("promoter buying", "clean audit"), ("resigned", "audit issue", "litigation")),
                "balance_sheet_sentiment": _score_terms(merged, ("deleveraging", "cash flow strong"), ("debt increase", "liquidity stress")),
                "business_momentum_sentiment": _score_terms(merged, ("order book", "acceleration"), ("slowdown", "delay")),
                "confidence": 0.70,
            }

        if task == "management_tone":
            confidence_vs_caution = _score_terms(merged, ("confident", "optimistic", "visibility"), ("cautious", "uncertain", "headwind"))
            return {
                "management_tone_score": confidence_vs_caution,
                "expansion_intent": _score_terms(merged, ("capex", "expansion", "investment"), ("hold", "delay", "reduce")),
                "capital_allocation_stance": _score_terms(merged, ("buyback", "dividend", "discipline"), ("aggressive debt", "dilution")),
                "margin_commentary_score": _score_terms(merged, ("margin expansion", "cost control"), ("margin pressure", "cost inflation")),
                "demand_commentary_score": _score_terms(merged, ("strong demand", "order inflow"), ("weak demand", "soft volume")),
                "risk_commentary_score": _score_terms(merged, ("mitigated", "contained"), ("risk", "uncertain", "volatile")),
                "confidence": 0.66,
            }

        return {"confidence": 0.0}


def _infer_event_type(text: str) -> str:
    if any(t in text for t in ("earnings", "results", "margin")):
        return "earnings_result"
    if any(t in text for t in ("guidance", "outlook")):
        return "guidance_change"
    if any(t in text for t in ("order", "contract", "deal")):
        return "order_win"
    if any(t in text for t in ("acquisition", "merger")):
        return "merger_acquisition"
    if any(t in text for t in ("capex", "plant", "expansion")):
        return "capex_announcement"
    if any(t in text for t in ("ceo", "cfo", "resigned", "appointment")):
        return "management_change"
    if any(t in text for t in ("promoter", "insider")):
        return "insider_activity"
    if any(t in text for t in ("rating", "downgrade", "upgrade")):
        return "credit_rating_change"
    if any(t in text for t in ("litigation", "regulatory", "penalty")):
        return "litigation"
    return "unknown"


def _keywords(text: str) -> list[str]:
    toks = [w for w in text.split() if len(w) > 5]
    return sorted(set(toks[:10]))


def _score_terms(text: str, pos: tuple[str, ...], neg: tuple[str, ...]) -> float:
    p = sum(t in text for t in pos)
    n = sum(t in text for t in neg)
    if p + n == 0:
        return 0.0
    return max(-1.0, min(1.0, (p - n) / max(1, p + n)))
