"""Single-stock deep analysis pipeline.

Runs a full research pass on one symbol and returns a structured
investment report covering technicals, fundamentals, multi-horizon
assessments, risk flags, and NLP-driven event signals.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Sequence

import yfinance as yf

from stock_screener_engine.config.settings import AppSettings
from stock_screener_engine.core.engine import ResearchEngine
from stock_screener_engine.core.entities import StockSnapshot
from stock_screener_engine.core.technical_indicators import atr, adx, momentum
from stock_screener_engine.data_sources.base.interfaces import (
    MarketDataProvider,
    TextEventProvider,
)
from stock_screener_engine.data_sources.market.yfinance_market_data_provider import _to_yf_symbol
from stock_screener_engine.nlp.event_engine.pipeline import TextIntelligencePipeline

logger = logging.getLogger(__name__)

_DEEP_LOOKBACK_DAYS = 200


# ---------------------------------------------------------------------------
# Technical indicator helpers (RSI, SMA, MACD not in shared module)
# ---------------------------------------------------------------------------

def _rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(0.0, delta))
        losses.append(max(0.0, -delta))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss < 1e-12:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _sma(closes: list[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def _ema(data: list[float], period: int) -> list[float]:
    if not data:
        return []
    k = 2.0 / (period + 1)
    result = [data[0]]
    for x in data[1:]:
        result.append(x * k + result[-1] * (1.0 - k))
    return result


def _macd(closes: list[float]) -> dict[str, float | None]:
    if len(closes) < 35:
        return {"macd": None, "signal": None, "histogram": None}
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    n = min(len(ema12), len(ema26))
    macd_line = [ema12[len(ema12) - n + i] - ema26[len(ema26) - n + i] for i in range(n)]
    signal_line = _ema(macd_line, 9)
    if not signal_line:
        return {"macd": None, "signal": None, "histogram": None}
    macd_val = macd_line[-1]
    signal_val = signal_line[-1]
    return {
        "macd": round(macd_val, 4),
        "signal": round(signal_val, 4),
        "histogram": round(macd_val - signal_val, 4),
    }


# ---------------------------------------------------------------------------
# yfinance enrichment
# ---------------------------------------------------------------------------

def _fetch_yf_info(symbol: str) -> dict:
    """Fetch fundamental and identity metadata from Yahoo Finance."""
    try:
        info = yf.Ticker(_to_yf_symbol(symbol)).info
        roe_raw = info.get("returnOnEquity")
        de_raw = info.get("debtToEquity")
        eg_raw = info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth")
        dy_raw = info.get("dividendYield")
        fcf = info.get("freeCashflow")
        rev = info.get("totalRevenue") or 1
        fcf_margin = (float(fcf) / max(1, float(rev))) if fcf else None
        return {
            "company_name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "pe_ratio": info.get("trailingPE") or info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "roe": float(roe_raw) if roe_raw is not None else None,
            "debt_to_equity": float(de_raw) / 100.0 if de_raw is not None else None,
            "earnings_growth": float(eg_raw) if eg_raw is not None else None,
            "free_cash_flow": fcf,
            "free_cash_flow_margin": fcf_margin,
            "market_cap": info.get("marketCap"),
            "dividend_yield": float(dy_raw) if dy_raw is not None else None,
            "eps": info.get("trailingEps"),
            "beta": info.get("beta"),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch yfinance info for %s: %s", symbol, exc)
        return {}


def _build_enriched_snapshot(
    symbol: str,
    yf_info: dict,
    closes: list[float],
    vols: list[float],
    as_of: date,
) -> StockSnapshot:
    """Build a StockSnapshot using real yfinance fundamentals where available."""
    close = closes[-1] if closes else 0.0
    volume = vols[-1] if vols else 0.0
    avg_20 = sum(vols[-20:]) / max(1, min(20, len(vols))) if vols else 1.0
    mom = 0.0
    if len(closes) > 20 and closes[-21] > 0:
        mom = (closes[-1] - closes[-21]) / closes[-21]

    return StockSnapshot(
        symbol=symbol,
        as_of=as_of,
        sector=yf_info.get("sector") or "Unknown",
        close=close,
        volume=volume,
        delivery_ratio=0.5,
        pe_ratio=float(yf_info["pe_ratio"]) if yf_info.get("pe_ratio") else 15.0,
        roe=float(yf_info["roe"]) if yf_info.get("roe") else max(0.02, min(0.35, 0.1 + 0.4 * mom)),
        debt_to_equity=float(yf_info["debt_to_equity"]) if yf_info.get("debt_to_equity") else max(0.05, min(2.0, 1.0 - mom)),
        earnings_growth=float(yf_info["earnings_growth"]) if yf_info.get("earnings_growth") else max(-0.2, min(0.6, mom)),
        free_cash_flow_margin=float(yf_info["free_cash_flow_margin"]) if yf_info.get("free_cash_flow_margin") else max(-0.2, min(0.4, (volume / max(1.0, avg_20) - 1.0) * 0.05)),
        promoter_holding_change=0.0,
        insider_activity_score=0.0,
    )


# ---------------------------------------------------------------------------
# Provider wrapper that injects an enriched snapshot
# ---------------------------------------------------------------------------

class _EnrichedSnapshotProvider(MarketDataProvider):
    """Wraps a provider but overrides get_snapshots for the enriched symbol."""

    def __init__(self, base: MarketDataProvider, overrides: dict[str, StockSnapshot]) -> None:
        self._base = base
        self._overrides = overrides

    def get_universe(self) -> list[str]:
        return self._base.get_universe()

    def get_historical(self, symbol: str, interval: str, start: date, end: date) -> list[dict]:
        return self._base.get_historical(symbol, interval, start, end)

    def get_snapshots(self, symbols: Sequence[str]) -> list[StockSnapshot]:
        result: list[StockSnapshot] = []
        for s in symbols:
            if s in self._overrides:
                result.append(self._overrides[s])
            else:
                result.extend(self._base.get_snapshots([s]))
        return result


# ---------------------------------------------------------------------------
# Horizon assessment builders
# ---------------------------------------------------------------------------

def _swing_verdict(swing_score: float, swing_min: float, category: str, short_score: float = 0.0) -> str:
    if category == "swing_candidate":
        return "buy"
    if short_score >= 62.0:
        return "short"
    if swing_score >= swing_min * 0.92:
        return "watch"
    return "avoid"


def _long_verdict(long_score: float, long_min: float, category: str, short_score: float = 0.0) -> str:
    if short_score >= 62.0:
        return "sell"
    if category == "long_term_candidate":
        return "buy"
    if long_score >= long_min * 0.88:
        return "hold"
    if long_score >= long_min * 0.72:
        return "watch"
    return "avoid"


def _medium_verdict(
    long_score: float,
    long_min: float,
    swing_score: float,
    swing_min: float,
    mom_20: float,
    rsi: float | None,
    short_score: float = 0.0,
) -> str:
    long_ok  = long_score  >= long_min  * 0.85
    swing_ok = swing_score >= swing_min * 0.80
    oversold = rsi is not None and rsi < 38
    if short_score >= 65.0:
        return "sell"
    if short_score >= 55.0:
        return "reduce"
    if long_ok and swing_ok:
        return "accumulate"
    if long_ok and (oversold or mom_20 > 0):
        return "accumulate on dips"
    if long_ok:
        return "hold — await momentum"
    if swing_ok and not long_ok:
        return "short-term trade only"
    return "avoid"


def _build_horizons(
    long_signal: object | None,
    swing_signal: object | None,
    long_score: float,
    swing_score: float,
    conviction: float,
    long_min: float,
    swing_min: float,
    mom_20: float,
    rsi: float | None,
    adx_val: float | None,
    features: dict[str, float],
    short_score: float = 0.0,
) -> dict[str, object]:
    long_cat = getattr(long_signal, "category", "unknown") if long_signal else "unknown"
    swing_cat = getattr(swing_signal, "category", "unknown") if swing_signal else "unknown"
    long_exp = getattr(long_signal, "explanation", None) if long_signal else None
    swing_exp = getattr(swing_signal, "explanation", None) if swing_signal else None

    # Swing (3-15 trading days) -------------------------------------------------
    swing_verdict = _swing_verdict(swing_score, swing_min, swing_cat, short_score=short_score)
    swing_rationale = _build_swing_rationale(swing_verdict, swing_score, swing_min, mom_20, rsi, adx_val)
    swing_horizon: dict[str, object] = {
        "horizon": "3 – 15 trading days",
        "verdict": swing_verdict,
        "confidence": round(conviction, 2),
        "rationale": swing_rationale,
        "key_catalysts": getattr(swing_exp, "top_positive_drivers", [])[:3] if swing_exp else [],
        "key_risks": getattr(swing_exp, "top_negative_drivers", [])[:2] if swing_exp else [],
        "entry_logic": getattr(swing_exp, "entry_logic", "") if swing_exp else "",
        "invalidation_logic": getattr(swing_exp, "invalidation_logic", "") if swing_exp else "",
    }

    # Medium-term (1-3 months) --------------------------------------------------
    med_verdict = _medium_verdict(long_score, long_min, swing_score, swing_min, mom_20, rsi, short_score=short_score)
    med_rationale = _build_medium_rationale(med_verdict, long_score, long_min, swing_score, swing_min, mom_20, rsi)
    medium_horizon: dict[str, object] = {
        "horizon": "1 – 3 months",
        "verdict": med_verdict,
        "confidence": round(min(conviction, long_score / max(1.0, long_min) * conviction), 2),
        "rationale": med_rationale,
        "key_catalysts": (
            getattr(long_exp, "top_positive_drivers", [])[:3] if long_exp else []
        ),
        "key_risks": (
            getattr(long_exp, "top_negative_drivers", [])[:2] if long_exp else []
        ),
        "entry_logic": "Scale in on 2-5% pullbacks toward key moving averages.",
        "invalidation_logic": "Exit if price closes below 200-day SMA on high volume, or fundamental deterioration confirmed.",
    }

    # Long-term (6-24 months) ---------------------------------------------------
    lt_verdict = _long_verdict(long_score, long_min, long_cat, short_score=short_score)
    lt_rationale = _build_long_rationale(lt_verdict, long_score, long_min, conviction)
    long_horizon: dict[str, object] = {
        "horizon": "6 – 24 months",
        "verdict": lt_verdict,
        "confidence": round(conviction, 2),
        "rationale": lt_rationale,
        "key_catalysts": (
            getattr(long_exp, "top_positive_drivers", [])[:5] if long_exp else []
        ),
        "key_risks": (
            getattr(long_exp, "top_negative_drivers", [])[:3] if long_exp else []
        ),
        "entry_logic": getattr(long_exp, "entry_logic", "") if long_exp else "",
        "invalidation_logic": getattr(long_exp, "invalidation_logic", "") if long_exp else "",
    }

    return {
        "swing": swing_horizon,
        "medium_term": medium_horizon,
        "long_term": long_horizon,
    }


def _build_swing_rationale(
    verdict: str,
    score: float,
    threshold: float,
    mom_20: float,
    rsi: float | None,
    adx_val: float | None,
) -> str:
    parts: list[str] = [f"Swing score {score:.1f} vs threshold {threshold:.1f}."]
    if mom_20 > 0.05:
        parts.append(f"20-day momentum positive ({mom_20 * 100:.1f}%).")
    elif mom_20 < -0.05:
        parts.append(f"20-day momentum negative ({mom_20 * 100:.1f}%).")
    if rsi is not None:
        if rsi < 35:
            parts.append(f"RSI({rsi:.0f}) in oversold zone — potential mean reversion.")
        elif rsi > 70:
            parts.append(f"RSI({rsi:.0f}) in overbought zone — momentum extended.")
        else:
            parts.append(f"RSI({rsi:.0f}) in neutral range.")
    if adx_val is not None:
        if adx_val > 25:
            parts.append(f"ADX({adx_val:.0f}) signals strong trend in place.")
        else:
            parts.append(f"ADX({adx_val:.0f}) suggests ranging/weak trend.")
    if verdict == "short":
        parts.append("Short-sale signal: technical breakdown with bearish momentum confirmed.")
    return " ".join(parts)


def _build_medium_rationale(
    verdict: str,
    long_score: float,
    long_min: float,
    swing_score: float,
    swing_min: float,
    mom_20: float,
    rsi: float | None,
) -> str:
    parts: list[str] = [
        f"Long-term score {long_score:.1f} (threshold {long_min:.1f}), "
        f"swing score {swing_score:.1f} (threshold {swing_min:.1f})."
    ]
    if verdict == "accumulate":
        parts.append("Both quality and momentum are aligned — medium-term setup is constructive.")
    elif verdict == "accumulate on dips":
        parts.append("Quality is intact but momentum is mixed; use dips as entries.")
    elif verdict == "hold — await momentum":
        parts.append("Fundamentals are sound but price hasn't confirmed a move yet.")
    elif verdict == "short-term trade only":
        parts.append("Momentum is present but long-term quality does not yet support conviction sizing.")
    elif verdict == "reduce":
        parts.append("Bearish factors building — consider trimming existing long exposure.")
    elif verdict == "sell":
        parts.append("Short score is elevated; material fundamental or technical deterioration detected. Exit longs.")
    else:
        parts.append("Neither quality nor momentum supports a medium-term position at this time.")
    return " ".join(parts)


def _build_long_rationale(
    verdict: str,
    score: float,
    threshold: float,
    conviction: float,
) -> str:
    parts: list[str] = [
        f"Long-term score {score:.1f} vs threshold {threshold:.1f} (conviction {conviction:.1f})."
    ]
    if verdict == "buy":
        parts.append("Stock qualifies as a long-term candidate across quality, value, and governance dimensions.")
    elif verdict == "hold":
        parts.append("Score is close to the buy threshold — monitor for improvement in key fundamentals.")
    elif verdict == "watch":
        parts.append("Scores suggest partial quality; position sizing should reflect elevated uncertainty.")
    elif verdict == "sell":
        parts.append("Short score elevated — fundamental or technical deterioration warrants exiting long positions.")
    else:
        parts.append("Score is materially below the long-term threshold. Revisit if fundamentals improve.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Score-component categorisation
# ---------------------------------------------------------------------------

_FUNDAMENTAL_COMPONENTS = {
    "growth_quality", "profitability_quality", "balance_sheet_health",
    "cash_flow_quality", "valuation_sanity", "governance_proxy",
}
_TECHNICAL_COMPONENTS = {
    "trend_strength", "momentum_strength", "relative_strength_proxy",
    "volatility_regime", "volume_confirmation",
}
_EVENT_COMPONENTS = {"event_catalyst", "regime_tailwind", "sentiment_score"}


def _categorize_components(comp_scores: dict[str, float]) -> dict[str, dict[str, float]]:
    fundamental: dict[str, float] = {}
    technical: dict[str, float] = {}
    event_text: dict[str, float] = {}
    risk: dict[str, float] = {}
    other: dict[str, float] = {}

    for k, v in comp_scores.items():
        key_clean = k.removeprefix("long_").removeprefix("swing_").removeprefix("risk_")
        if "risk" in k:
            risk[k] = round(v, 3)
        elif key_clean in _FUNDAMENTAL_COMPONENTS:
            fundamental[k] = round(v, 3)
        elif key_clean in _TECHNICAL_COMPONENTS:
            technical[k] = round(v, 3)
        elif key_clean in _EVENT_COMPONENTS:
            event_text[k] = round(v, 3)
        else:
            other[k] = round(v, 3)

    return {
        "fundamental": fundamental,
        "technical": technical,
        "event_and_sentiment": event_text,
        "risk_penalties": risk,
        "other": other,
    }


# ---------------------------------------------------------------------------
# Directional summary
# ---------------------------------------------------------------------------


def _build_directional(
    long_score: float,
    swing_score: float,
    short_score: float,
    scoring_cfg: object,
) -> dict[str, object]:
    """Compute an overall directional bias from the three scores."""
    long_min  = getattr(scoring_cfg, "long_term_min_score", 24.0)
    swing_min = getattr(scoring_cfg, "swing_min_score", 28.0)
    short_min = getattr(scoring_cfg, "short_min_score", 58.0)

    bullish = long_score >= long_min or swing_score >= swing_min
    bearish = short_score >= short_min

    if bearish and short_score >= 70.0:
        bias = "strong_short"
    elif bearish and not bullish:
        bias = "bearish"
    elif bearish and bullish:
        bias = "conflicted"   # both signals present — treat with caution
    elif bullish:
        bias = "bullish"
    else:
        bias = "neutral"

    return {
        "bias": bias,
        "long_score": round(long_score, 2),
        "swing_score": round(swing_score, 2),
        "short_score": round(short_score, 2),
        "interpretation": {
            "strong_short": "Strong bearish signal — consider short-sell or avoid long positions.",
            "bearish": "Bearish tilt — reduce or avoid new long exposure.",
            "conflicted": "Mixed signals — elevated uncertainty; reduce position size.",
            "bullish": "Bullish bias — long-side opportunities are favoured.",
            "neutral": "No clear directional edge — wait for confirmation.",
        }[bias],
    }



# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class SingleStockPipeline:
    """Deep single-stock research: fundamentals, technicals, text intelligence,
    and multi-horizon investment assessment."""

    def __init__(
        self,
        settings: AppSettings,
        market_data: MarketDataProvider,
        text_data: TextEventProvider,
        text_pipeline: TextIntelligencePipeline | None = None,
    ) -> None:
        self.settings = settings
        self.market_data = market_data
        self.text_data = text_data
        self.text_pipeline = text_pipeline

    def run(self, symbol: str) -> dict[str, object]:
        symbol = symbol.strip().upper()
        logger.info("Single-stock deep analysis: %s", symbol)

        today = date.today()
        lookback = today - timedelta(days=_DEEP_LOOKBACK_DAYS)

        # ── 1. Price history ──────────────────────────────────────────────
        bars = self.market_data.get_historical(symbol, "1d", lookback, today)
        closes = [float(b["close"]) for b in bars]
        highs  = [float(b["high"])  for b in bars]
        lows   = [float(b["low"])   for b in bars]
        vols   = [float(b["volume"]) for b in bars]

        # ── 2. Fundamental enrichment from yfinance ───────────────────────
        yf_info = _fetch_yf_info(symbol)

        # ── 3. Build enriched snapshot ────────────────────────────────────
        enriched = _build_enriched_snapshot(symbol, yf_info, closes, vols, today)
        patched_provider = _EnrichedSnapshotProvider(
            base=self.market_data,
            overrides={symbol: enriched},
        )

        # ── 4. Run engine with enriched fundamentals ──────────────────────
        engine = ResearchEngine(
            settings=self.settings,
            market_data=patched_provider,
            text_data=self.text_data,
            text_pipeline=self.text_pipeline,
        )
        raw = engine.run(symbols=[symbol], regime_score=0.25)

        # ── 5. Extract core outputs ───────────────────────────────────────
        features_vec  = raw["features"][0]  if raw.get("features")      else None
        scorecard     = raw["scores"][0]    if raw.get("scores")         else None
        long_signal   = raw["long_signals"][0]  if raw.get("long_signals")  else None
        swing_signal  = raw["swing_signals"][0] if raw.get("swing_signals") else None
        short_signal  = raw["short_signals"][0] if raw.get("short_signals") else None

        features_dict: dict[str, float] = dict(features_vec.values) if features_vec else {}
        comp_scores:   dict[str, float] = dict(scorecard.component_scores) if scorecard else {}

        long_score  = scorecard.long_term_score if scorecard else 0.0
        swing_score = scorecard.swing_score     if scorecard else 0.0
        conviction  = scorecard.conviction      if scorecard else 0.0
        risk_pen    = scorecard.risk_penalty    if scorecard else 0.0
        short_score = getattr(short_signal, "score", 0.0) if short_signal else 0.0

        # ── 6. Technical indicators ───────────────────────────────────────
        current_price = closes[-1] if closes else 0.0
        rsi_val  = _rsi(closes)
        sma50    = _sma(closes, 50)
        sma200   = _sma(closes, 200)
        macd_val = _macd(closes)
        atr_val  = atr(highs, lows, closes) if len(closes) > 1 else None
        adx_val  = adx(highs, lows, closes) if len(closes) > 16 else None
        mom_20   = momentum(closes, 20)

        # ── 7. Multi-horizon assessments ──────────────────────────────────
        horizons = _build_horizons(
            long_signal=long_signal,
            swing_signal=swing_signal,
            long_score=long_score,
            swing_score=swing_score,
            conviction=conviction,
            long_min=self.settings.scoring.long_term_min_score,
            swing_min=self.settings.scoring.swing_min_score,
            mom_20=mom_20,
            rsi=rsi_val,
            adx_val=adx_val,
            features=features_dict,
            short_score=short_score,
        )

        # ── 8. Raw news headlines ────────────────────────────────────────
        lookback_news = self.settings.nlp.lookback_days
        try:
            raw_headlines: list[str] = self.text_data.get_recent_events(
                [symbol], lookback_days=lookback_news
            ).get(symbol, [])
        except Exception:  # noqa: BLE001
            raw_headlines = []

        # ── 9. NLP-derived text features ────────────────────────────────
        text_row: dict[str, object] = next(
            (r for r in raw.get("text_features", []) if r.get("symbol") == symbol),
            {},
        )
        text_features = {k: round(v, 4) for k, v in text_row.items() if k != "symbol" and isinstance(v, (int, float))}

        # ── 10. Score breakdown by category ──────────────────────────────
        breakdown = _categorize_components(comp_scores)

        # ── 11. Drivers & risk flags ──────────────────────────────────────
        long_exp  = getattr(long_signal,  "explanation", None) if long_signal  else None
        swing_exp = getattr(swing_signal, "explanation", None) if swing_signal else None

        top_positive = getattr(long_exp,  "top_positive_drivers", [])[:5] if long_exp else []
        top_negative = getattr(long_exp,  "top_negative_drivers", [])[:5] if long_exp else []
        risk_flags   = getattr(long_exp,  "risk_flags",            [])    if long_exp else []

        # ── 12. Assemble report ───────────────────────────────────────────
        return {
            "symbol": symbol,
            "company_name": yf_info.get("company_name") or symbol,
            "sector": yf_info.get("sector") or enriched.sector,
            "industry": yf_info.get("industry"),
            "exchange": "NSE",
            "as_of": today.isoformat(),

            "price": {
                "current": round(current_price, 2),
                "52w_high": yf_info.get("52w_high"),
                "52w_low": yf_info.get("52w_low"),
                "distance_from_52w_high_pct": (
                    round((current_price / yf_info["52w_high"] - 1.0) * 100.0, 2)
                    if yf_info.get("52w_high") and yf_info["52w_high"] > 0 else None
                ),
                "sma_50": round(sma50, 2) if sma50 is not None else None,
                "sma_200": round(sma200, 2) if sma200 is not None else None,
                "above_sma_50": (current_price > sma50) if sma50 is not None else None,
                "above_sma_200": (current_price > sma200) if sma200 is not None else None,
            },

            "technical_indicators": {
                "rsi_14": round(rsi_val, 2) if rsi_val is not None else None,
                "macd": macd_val,
                "adx_14": round(adx_val, 2) if adx_val is not None else None,
                "atr_14": round(atr_val, 2) if atr_val is not None else None,
                "momentum_20d_pct": round(mom_20 * 100.0, 2),
            },

            "fundamentals": {
                "pe_ratio": yf_info.get("pe_ratio"),
                "pb_ratio": yf_info.get("pb_ratio"),
                "roe_pct": round(float(yf_info["roe"]) * 100.0, 2) if yf_info.get("roe") is not None else None,
                "debt_to_equity": yf_info.get("debt_to_equity"),
                "earnings_growth_pct": round(float(yf_info["earnings_growth"]) * 100.0, 2) if yf_info.get("earnings_growth") is not None else None,
                "market_cap": yf_info.get("market_cap"),
                "dividend_yield_pct": round(float(yf_info["dividend_yield"]) * 100.0, 2) if yf_info.get("dividend_yield") is not None else None,
                "eps": yf_info.get("eps"),
                "beta": yf_info.get("beta"),
            },

            "scores": {
                "long_term_score": round(long_score, 2),
                "swing_score": round(swing_score, 2),
                "risk_penalty": round(risk_pen, 2),
                "conviction": round(conviction, 2),
                "long_category": getattr(long_signal,  "category", "unknown") if long_signal  else "unknown",
                "swing_category": getattr(swing_signal, "category", "unknown") if swing_signal else "unknown",
                "short_score": round(short_score, 2),
                "short_category": getattr(short_signal, "category", "unknown") if short_signal else "unknown",
            },

            "directional": _build_directional(long_score, swing_score, short_score, self.settings.scoring),

            "score_breakdown": breakdown,

            "investment_horizons": horizons,

            "key_drivers": {
                "top_positive": top_positive,
                "top_negative": top_negative,
            },

            "risk_flags": risk_flags,

            "entry_exit": {
                "long_entry_logic": getattr(long_exp, "entry_logic", "") if long_exp else "",
                "long_invalidation": getattr(long_exp, "invalidation_logic", "") if long_exp else "",
                "swing_entry_logic": getattr(swing_exp, "entry_logic", "") if swing_exp else "",
                "swing_invalidation": getattr(swing_exp, "invalidation_logic", "") if swing_exp else "",
            },

            "news": {
                "lookback_days": lookback_news,
                "headline_count": len(raw_headlines),
                "headlines": raw_headlines,
            },

            "nlp_signals": text_features if text_features else None,

            "all_features": {k: round(v, 4) for k, v in sorted(features_dict.items())},
        }
