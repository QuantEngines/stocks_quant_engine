from __future__ import annotations

from datetime import date, timedelta

from stock_screener_engine.monitoring.live_invalidation import ActiveSignal, LiveInvalidationMonitor


def test_live_invalidation_triggers_on_long_stop() -> None:
    monitor = LiveInvalidationMonitor()
    signal = ActiveSignal(
        symbol="ABC",
        side="long",
        entry_price=100.0,
        entered_on=date.today() - timedelta(days=3),
        stop_loss_pct=0.08,
    )

    decision = monitor.evaluate(signal=signal, as_of=date.today(), latest_price=91.0)
    assert decision.invalidated is True
    assert any("stop breached" in reason for reason in decision.reasons)


def test_live_invalidation_triggers_on_thesis_break_and_stale_holding() -> None:
    monitor = LiveInvalidationMonitor()
    signal = ActiveSignal(
        symbol="XYZ",
        side="short",
        entry_price=200.0,
        entered_on=date.today() - timedelta(days=45),
        max_holding_days=20,
        required_thesis_flags=["bearish_earnings_revision", "weak_guidance"],
    )

    decision = monitor.evaluate(
        signal=signal,
        as_of=date.today(),
        latest_price=180.0,
        active_thesis_flags=["weak_guidance"],
    )

    assert decision.invalidated is True
    assert any("holding period exceeded" in reason for reason in decision.reasons)
    assert any("thesis flags missing" in reason for reason in decision.reasons)
