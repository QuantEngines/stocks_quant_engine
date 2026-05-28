"""Canonical SQLite store for the data-foundation layer."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

from stock_screener_engine.data_sources.schemas import (
    BankingFactorRecord,
    CorporateActionRecord,
    DeliveryTurnoverRecord,
    EquityValuationRecord,
    FinancialStatementRecord,
    MarketSessionRecord,
    OHLCVBar,
    SecurityMasterRecord,
    ShareholdingRecord,
)


class MarketDataStore:
    """Canonical queryable store for securities, calendar, OHLCV, and actions."""

    def __init__(self, sqlite_path: str) -> None:
        self.path = Path(sqlite_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS security_master (
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                isin TEXT NOT NULL DEFAULT '',
                series TEXT NOT NULL DEFAULT 'EQ',
                company_name TEXT NOT NULL DEFAULT '',
                sector TEXT NOT NULL DEFAULT 'Unknown',
                industry TEXT NOT NULL DEFAULT 'Unknown',
                listing_date TEXT,
                delisting_date TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                lot_size INTEGER NOT NULL DEFAULT 1,
                tick_size REAL NOT NULL DEFAULT 0.05,
                source TEXT NOT NULL DEFAULT 'manual',
                PRIMARY KEY(symbol, exchange)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS market_calendar (
                venue TEXT NOT NULL,
                session_date TEXT NOT NULL,
                is_trading_day INTEGER NOT NULL,
                session_type TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                PRIMARY KEY(venue, session_date)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv_bars (
                venue TEXT NOT NULL,
                symbol TEXT NOT NULL,
                ts TEXT NOT NULL,
                interval TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(venue, symbol, ts, interval)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS corporate_actions (
                venue TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action_type TEXT NOT NULL,
                ex_date TEXT NOT NULL,
                record_date TEXT,
                ratio TEXT,
                cash_amount REAL,
                currency TEXT NOT NULL,
                source_id TEXT NOT NULL DEFAULT '',
                PRIMARY KEY(venue, symbol, action_type, ex_date, source_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS delivery_turnover (
                venue TEXT NOT NULL,
                symbol TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                traded_quantity REAL NOT NULL,
                delivery_quantity REAL NOT NULL,
                delivery_pct REAL NOT NULL,
                source_id TEXT NOT NULL DEFAULT '',
                ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(venue, symbol, trade_date, source_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS financial_statements (
                venue TEXT NOT NULL,
                symbol TEXT NOT NULL,
                period_end TEXT NOT NULL,
                filing_date TEXT NOT NULL,
                statement_type TEXT NOT NULL,
                revenue REAL NOT NULL,
                ebit REAL NOT NULL,
                net_income REAL NOT NULL,
                operating_cash_flow REAL NOT NULL,
                capex REAL NOT NULL,
                total_debt REAL NOT NULL,
                equity REAL NOT NULL,
                total_assets REAL NOT NULL,
                current_assets REAL NOT NULL,
                current_liabilities REAL NOT NULL,
                interest_expense REAL NOT NULL,
                source_id TEXT NOT NULL DEFAULT '',
                ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(venue, symbol, period_end, statement_type, source_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS equity_valuations (
                venue TEXT NOT NULL,
                symbol TEXT NOT NULL,
                as_of TEXT NOT NULL,
                market_cap REAL NOT NULL,
                shares_outstanding REAL NOT NULL DEFAULT 0,
                free_float_market_cap REAL NOT NULL DEFAULT 0,
                enterprise_value REAL NOT NULL DEFAULT 0,
                currency TEXT NOT NULL DEFAULT 'INR',
                source_id TEXT NOT NULL DEFAULT '',
                ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(venue, symbol, as_of, source_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS shareholding_patterns (
                venue TEXT NOT NULL,
                symbol TEXT NOT NULL,
                period_end TEXT NOT NULL,
                filing_date TEXT NOT NULL,
                promoter_pct REAL NOT NULL,
                fii_pct REAL NOT NULL,
                dii_pct REAL NOT NULL,
                public_pct REAL NOT NULL,
                source_id TEXT NOT NULL DEFAULT '',
                ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(venue, symbol, period_end, source_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS banking_factors (
                venue TEXT NOT NULL,
                symbol TEXT NOT NULL,
                period_end TEXT NOT NULL,
                filing_date TEXT NOT NULL,
                net_interest_income REAL NOT NULL DEFAULT 0,
                net_interest_margin_pct REAL NOT NULL DEFAULT 0,
                advances_growth_pct REAL NOT NULL DEFAULT 0,
                deposits_growth_pct REAL NOT NULL DEFAULT 0,
                casa_ratio_pct REAL NOT NULL DEFAULT 0,
                gnpa_ratio_pct REAL NOT NULL DEFAULT 0,
                nnpa_ratio_pct REAL NOT NULL DEFAULT 0,
                provision_coverage_ratio_pct REAL NOT NULL DEFAULT 0,
                credit_cost_pct REAL NOT NULL DEFAULT 0,
                capital_adequacy_ratio_pct REAL NOT NULL DEFAULT 0,
                cet1_ratio_pct REAL NOT NULL DEFAULT 0,
                cost_to_income_ratio_pct REAL NOT NULL DEFAULT 0,
                roa_pct REAL NOT NULL DEFAULT 0,
                roe_pct REAL NOT NULL DEFAULT 0,
                loan_to_deposit_ratio_pct REAL NOT NULL DEFAULT 0,
                source_id TEXT NOT NULL DEFAULT '',
                ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(venue, symbol, period_end, source_id)
            )
            """
        )
        self.conn.commit()

    def upsert_security_master(self, records: Iterable[SecurityMasterRecord]) -> int:
        rows = [self._merge_security_master_record(record) for record in records]
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO security_master(
                symbol, exchange, isin, series, company_name, sector, industry,
                listing_date, delisting_date, active, lot_size, tick_size, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.symbol.upper(),
                    record.exchange.upper(),
                    record.isin,
                    record.series,
                    record.company_name,
                    record.sector,
                    record.industry,
                    _date_or_none(record.listing_date),
                    _date_or_none(record.delisting_date),
                    1 if record.active else 0,
                    record.lot_size,
                    record.tick_size,
                    record.source,
                )
                for record in rows
            ],
        )
        self.conn.commit()
        return len(rows)

    def _merge_security_master_record(self, record: SecurityMasterRecord) -> SecurityMasterRecord:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM security_master WHERE symbol = ? AND exchange = ?",
            (record.symbol.strip().upper(), record.exchange.strip().upper()),
        )
        row = cur.fetchone()
        if row is None:
            return record

        existing = _security_from_row(row)
        return replace(
            record,
            isin=_prefer_rich_text(record.isin, existing.isin),
            series=_prefer_rich_text(record.series, existing.series),
            company_name=_prefer_company_name(record.company_name, existing.company_name, record.symbol),
            sector=_prefer_rich_text(record.sector, existing.sector),
            industry=_prefer_rich_text(record.industry, existing.industry),
            listing_date=record.listing_date or existing.listing_date,
            delisting_date=record.delisting_date or existing.delisting_date,
            lot_size=record.lot_size if record.lot_size != 1 else existing.lot_size,
            tick_size=record.tick_size if record.tick_size != 0.05 else existing.tick_size,
            source=_prefer_security_source(record.source, existing.source),
        )

    def get_security_master(self, symbols: Sequence[str] | None = None) -> list[SecurityMasterRecord]:
        cur = self.conn.cursor()
        requested = [s.strip().upper() for s in symbols or [] if s.strip()]
        if requested:
            placeholders = ",".join("?" for _ in requested)
            cur.execute(f"SELECT * FROM security_master WHERE symbol IN ({placeholders})", requested)
        elif symbols is not None:
            return []
        else:
            cur.execute("SELECT * FROM security_master ORDER BY exchange, symbol")
        return [_security_from_row(row) for row in cur.fetchall()]

    def list_active_securities(
        self,
        sectors: Sequence[str] | None = None,
        exchange: str | None = None,
    ) -> list[SecurityMasterRecord]:
        query = "SELECT * FROM security_master WHERE active = 1"
        params: list[object] = []
        if sectors:
            normalized = [sector.strip().lower() for sector in sectors if sector.strip()]
            if normalized:
                placeholders = ",".join("?" for _ in normalized)
                query += f" AND LOWER(sector) IN ({placeholders})"
                params.extend(normalized)
        if exchange:
            query += " AND exchange = ?"
            params.append(exchange.strip().upper())
        query += " ORDER BY exchange, sector, symbol"
        cur = self.conn.cursor()
        cur.execute(query, params)
        return [_security_from_row(row) for row in cur.fetchall()]

    def company_metadata(self, symbols: Sequence[str]) -> dict[str, dict[str, object]]:
        records = self.get_security_master(symbols)
        return {
            record.symbol: {
                "symbol": record.symbol,
                "company_name": record.company_name or None,
                "sector": record.sector,
                "industry": record.industry,
                "exchange": record.exchange,
                "isin": record.isin or None,
                "active": record.active,
            }
            for record in records
        }

    def list_ohlcv_symbols(self, interval: str = "1d") -> list[str]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT DISTINCT symbol FROM ohlcv_bars WHERE interval = ? ORDER BY symbol",
            (interval,),
        )
        return [str(row["symbol"]) for row in cur.fetchall()]

    def upsert_market_sessions(self, sessions: Iterable[MarketSessionRecord]) -> int:
        rows = list(sessions)
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO market_calendar(venue, session_date, is_trading_day, session_type, reason)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    session.venue.upper(),
                    session.session_date.isoformat(),
                    1 if session.is_trading_day else 0,
                    session.session_type,
                    session.reason,
                )
                for session in rows
            ],
        )
        self.conn.commit()
        return len(rows)

    def get_market_sessions(self, venue: str, start: date, end: date) -> list[MarketSessionRecord]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM market_calendar
            WHERE venue = ? AND session_date BETWEEN ? AND ?
            ORDER BY session_date
            """,
            (venue.upper(), start.isoformat(), end.isoformat()),
        )
        return [_session_from_row(row) for row in cur.fetchall()]

    def upsert_ohlcv(self, bars: Iterable[OHLCVBar], interval: str = "1d", source: str = "") -> int:
        rows = [_normalize_ohlcv_bar(bar, interval) for bar in bars]
        if interval == "1d" and rows:
            self.conn.executemany(
                """
                DELETE FROM ohlcv_bars
                WHERE venue = ? AND symbol = ? AND interval = ? AND substr(ts, 1, 10) = ?
                """,
                [
                    (
                        bar.venue.upper(),
                        bar.symbol.upper(),
                        interval,
                        _daily_trade_date(bar.ts),
                    )
                    for bar in rows
                ],
            )
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO ohlcv_bars(
                venue, symbol, ts, interval, open, high, low, close, volume, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    bar.venue.upper(),
                    bar.symbol.upper(),
                    bar.ts,
                    interval,
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                    source or bar.venue.upper(),
                )
                for bar in rows
            ],
        )
        self.conn.commit()
        return len(rows)

    def get_ohlcv(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        venue: str | None = None,
        interval: str = "1d",
        adjusted: bool = False,
    ) -> list[OHLCVBar]:
        query = "SELECT * FROM ohlcv_bars WHERE symbol = ? AND interval = ?"
        params: list[object] = [symbol.upper(), interval]
        if venue:
            query += " AND venue = ?"
            params.append(venue.upper())
        if start:
            query += " AND substr(ts, 1, 10) >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND substr(ts, 1, 10) <= ?"
            params.append(end.isoformat())
        query += " ORDER BY ts"
        cur = self.conn.cursor()
        cur.execute(query, params)
        bars = [_bar_from_row(row) for row in cur.fetchall()]
        if adjusted:
            actions = self.get_corporate_actions(symbol=symbol, venue=venue)
            return adjust_ohlcv_for_actions(bars, actions)
        return bars

    def normalize_daily_ohlcv(
        self,
        symbols: Sequence[str] | None = None,
        start: date | None = None,
        end: date | None = None,
        interval: str = "1d",
    ) -> dict[str, int]:
        """Collapse daily bars to one row per venue/symbol/local trade date."""
        if interval != "1d":
            return {"groups_seen": 0, "rows_deleted": 0, "rows_updated": 0}

        query = "SELECT rowid AS row_id, * FROM ohlcv_bars WHERE interval = ?"
        params: list[object] = [interval]
        requested = [symbol.strip().upper() for symbol in symbols or [] if symbol.strip()]
        if requested:
            placeholders = ",".join("?" for _ in requested)
            query += f" AND symbol IN ({placeholders})"
            params.extend(requested)
        if start:
            query += " AND substr(ts, 1, 10) >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND substr(ts, 1, 10) <= ?"
            params.append(end.isoformat())

        cur = self.conn.cursor()
        cur.execute(query, params)
        groups: dict[tuple[str, str, str, str], list[sqlite3.Row]] = {}
        for row in cur.fetchall():
            key = (
                str(row["venue"]).upper(),
                str(row["symbol"]).upper(),
                _daily_trade_date(str(row["ts"])),
                str(row["interval"]),
            )
            groups.setdefault(key, []).append(row)

        delete_ids: list[int] = []
        updates: list[tuple[str, int]] = []
        for (_, _, trade_date, _), rows in groups.items():
            keep = max(rows, key=lambda row: (str(row["ingested_at"]), int(row["row_id"])))
            for row in rows:
                row_id = int(row["row_id"])
                if row_id == int(keep["row_id"]):
                    continue
                delete_ids.append(row_id)
            if str(keep["ts"]) != trade_date:
                updates.append((trade_date, int(keep["row_id"])))

        if delete_ids:
            cur.executemany("DELETE FROM ohlcv_bars WHERE rowid = ?", [(row_id,) for row_id in delete_ids])
        if updates:
            cur.executemany("UPDATE ohlcv_bars SET ts = ? WHERE rowid = ?", updates)
        self.conn.commit()
        return {
            "groups_seen": len(groups),
            "rows_deleted": len(delete_ids),
            "rows_updated": len(updates),
        }

    def upsert_corporate_actions(self, actions: Iterable[CorporateActionRecord]) -> int:
        rows = list(actions)
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO corporate_actions(
                venue, symbol, action_type, ex_date, record_date, ratio, cash_amount, currency, source_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    action.venue.upper(),
                    action.symbol.upper(),
                    action.action_type,
                    action.ex_date,
                    action.record_date,
                    action.ratio,
                    action.cash_amount,
                    action.currency,
                    action.source_id,
                )
                for action in rows
            ],
        )
        self.conn.commit()
        return len(rows)

    def get_corporate_actions(
        self,
        symbol: str,
        venue: str | None = None,
    ) -> list[CorporateActionRecord]:
        query = "SELECT * FROM corporate_actions WHERE symbol = ?"
        params: list[object] = [symbol.upper()]
        if venue:
            query += " AND venue = ?"
            params.append(venue.upper())
        query += " ORDER BY ex_date"
        cur = self.conn.cursor()
        cur.execute(query, params)
        return [_action_from_row(row) for row in cur.fetchall()]

    def upsert_delivery_turnover(self, records: Iterable[DeliveryTurnoverRecord]) -> int:
        rows = list(records)
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO delivery_turnover(
                venue, symbol, trade_date, traded_quantity, delivery_quantity, delivery_pct, source_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.venue.upper(),
                    record.symbol.upper(),
                    record.trade_date.isoformat(),
                    record.traded_quantity,
                    record.delivery_quantity,
                    record.delivery_pct,
                    record.source_id,
                )
                for record in rows
            ],
        )
        self.conn.commit()
        return len(rows)

    def get_delivery_turnover(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        venue: str | None = None,
    ) -> list[DeliveryTurnoverRecord]:
        query = "SELECT * FROM delivery_turnover WHERE symbol = ?"
        params: list[object] = [symbol.upper()]
        if venue:
            query += " AND venue = ?"
            params.append(venue.upper())
        if start:
            query += " AND trade_date >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND trade_date <= ?"
            params.append(end.isoformat())
        query += " ORDER BY trade_date"
        cur = self.conn.cursor()
        cur.execute(query, params)
        return [_delivery_turnover_from_row(row) for row in cur.fetchall()]

    def latest_delivery_turnover(
        self,
        symbol: str,
        as_of: date,
        venue: str | None = None,
    ) -> DeliveryTurnoverRecord | None:
        query = """
            SELECT * FROM delivery_turnover
            WHERE symbol = ? AND trade_date <= ?
        """
        params: list[object] = [symbol.upper(), as_of.isoformat()]
        if venue:
            query += " AND venue = ?"
            params.append(venue.upper())
        query += " ORDER BY trade_date DESC LIMIT 1"
        cur = self.conn.cursor()
        cur.execute(query, params)
        row = cur.fetchone()
        return _delivery_turnover_from_row(row) if row is not None else None

    def delivery_turnover_coverage(
        self,
        symbols: Sequence[str],
        as_of: date,
        venue: str | None = None,
        max_age_days: int = 10,
    ) -> dict[str, object]:
        requested = {s.strip().upper() for s in symbols if s.strip()}
        available: dict[str, str] = {}
        stale: list[str] = []
        for symbol in requested:
            latest = self.latest_delivery_turnover(symbol=symbol, as_of=as_of, venue=venue)
            if latest is None:
                continue
            available[symbol] = latest.trade_date.isoformat()
            if (as_of - latest.trade_date).days > max_age_days:
                stale.append(symbol)
        missing = sorted(requested - set(available))
        return {
            "symbols_requested": len(requested),
            "symbols_with_delivery": len(available),
            "missing_symbols": missing,
            "stale_symbols": sorted(stale),
            "coverage": round(len(available) / len(requested), 4) if requested else 0.0,
            "latest_delivery_by_symbol": available,
        }

    def upsert_financial_statements(self, records: Iterable[FinancialStatementRecord]) -> int:
        rows = list(records)
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO financial_statements(
                venue, symbol, period_end, filing_date, statement_type,
                revenue, ebit, net_income, operating_cash_flow, capex,
                total_debt, equity, total_assets, current_assets,
                current_liabilities, interest_expense, source_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.venue.upper(),
                    record.symbol.upper(),
                    record.period_end.isoformat(),
                    record.filing_date.isoformat(),
                    record.statement_type.lower(),
                    record.revenue,
                    record.ebit,
                    record.net_income,
                    record.operating_cash_flow,
                    record.capex,
                    record.total_debt,
                    record.equity,
                    record.total_assets,
                    record.current_assets,
                    record.current_liabilities,
                    record.interest_expense,
                    record.source_id,
                )
                for record in rows
            ],
        )
        self.conn.commit()
        return len(rows)

    def get_financial_statements(
        self,
        symbol: str,
        venue: str | None = None,
        statement_type: str | None = None,
        start: date | None = None,
        end: date | None = None,
        as_of: date | None = None,
    ) -> list[FinancialStatementRecord]:
        query = "SELECT * FROM financial_statements WHERE symbol = ?"
        params: list[object] = [symbol.strip().upper()]
        if venue:
            query += " AND venue = ?"
            params.append(venue.strip().upper())
        if statement_type:
            query += " AND statement_type = ?"
            params.append(statement_type.strip().lower())
        if start:
            query += " AND period_end >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND period_end <= ?"
            params.append(end.isoformat())
        if as_of:
            query += " AND period_end <= ? AND filing_date <= ?"
            params.extend([as_of.isoformat(), as_of.isoformat()])
        query += " ORDER BY period_end DESC, filing_date DESC"
        cur = self.conn.cursor()
        cur.execute(query, params)
        return [_statement_from_row(row) for row in cur.fetchall()]

    def latest_financial_statement_as_of(
        self,
        symbol: str,
        as_of: date,
        venue: str | None = None,
        statement_type: str | None = None,
    ) -> FinancialStatementRecord | None:
        rows = self.get_financial_statements(
            symbol=symbol,
            venue=venue,
            statement_type=statement_type,
            as_of=as_of,
        )
        return rows[0] if rows else None

    def financial_statement_coverage(
        self,
        symbols: Sequence[str],
        as_of: date,
        venue: str | None = None,
    ) -> dict[str, object]:
        requested = {s.strip().upper() for s in symbols if s.strip()}
        available: dict[str, str] = {}
        for symbol in requested:
            latest = self.latest_financial_statement_as_of(symbol=symbol, as_of=as_of, venue=venue)
            if latest is not None:
                available[symbol] = latest.period_end.isoformat()
        missing = sorted(requested - set(available))
        return {
            "symbols_requested": len(requested),
            "symbols_with_statements": len(available),
            "missing_symbols": missing,
            "coverage": round(len(available) / len(requested), 4) if requested else 0.0,
            "latest_period_by_symbol": available,
        }

    def upsert_equity_valuations(self, records: Iterable[EquityValuationRecord]) -> int:
        rows = list(records)
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO equity_valuations(
                venue, symbol, as_of, market_cap, shares_outstanding,
                free_float_market_cap, enterprise_value, currency, source_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.venue.upper(),
                    record.symbol.upper(),
                    record.as_of.isoformat(),
                    record.market_cap,
                    record.shares_outstanding,
                    record.free_float_market_cap,
                    record.enterprise_value,
                    record.currency,
                    record.source_id,
                )
                for record in rows
            ],
        )
        self.conn.commit()
        return len(rows)

    def get_equity_valuations(
        self,
        symbol: str,
        venue: str | None = None,
        start: date | None = None,
        end: date | None = None,
        as_of: date | None = None,
    ) -> list[EquityValuationRecord]:
        query = "SELECT * FROM equity_valuations WHERE symbol = ?"
        params: list[object] = [symbol.strip().upper()]
        if venue:
            query += " AND venue = ?"
            params.append(venue.strip().upper())
        if start:
            query += " AND as_of >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND as_of <= ?"
            params.append(end.isoformat())
        if as_of:
            query += " AND as_of <= ?"
            params.append(as_of.isoformat())
        query += " ORDER BY as_of DESC"
        cur = self.conn.cursor()
        cur.execute(query, params)
        return [_valuation_from_row(row) for row in cur.fetchall()]

    def latest_equity_valuation_as_of(
        self,
        symbol: str,
        as_of: date,
        venue: str | None = None,
    ) -> EquityValuationRecord | None:
        rows = self.get_equity_valuations(symbol=symbol, venue=venue, as_of=as_of)
        return rows[0] if rows else None

    def equity_valuation_coverage(
        self,
        symbols: Sequence[str],
        as_of: date,
        venue: str | None = None,
    ) -> dict[str, object]:
        requested = {s.strip().upper() for s in symbols if s.strip()}
        available: dict[str, str] = {}
        for symbol in requested:
            latest = self.latest_equity_valuation_as_of(symbol=symbol, as_of=as_of, venue=venue)
            if latest is not None:
                available[symbol] = latest.as_of.isoformat()
        missing = sorted(requested - set(available))
        return {
            "symbols_requested": len(requested),
            "symbols_with_valuations": len(available),
            "missing_symbols": missing,
            "coverage": round(len(available) / len(requested), 4) if requested else 0.0,
            "latest_valuation_by_symbol": available,
        }

    def upsert_shareholding(self, records: Iterable[ShareholdingRecord]) -> int:
        rows = list(records)
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO shareholding_patterns(
                venue, symbol, period_end, filing_date, promoter_pct,
                fii_pct, dii_pct, public_pct, source_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.venue.upper(),
                    record.symbol.upper(),
                    record.period_end.isoformat(),
                    record.filing_date.isoformat(),
                    record.promoter_pct,
                    record.fii_pct,
                    record.dii_pct,
                    record.public_pct,
                    record.source_id,
                )
                for record in rows
            ],
        )
        self.conn.commit()
        return len(rows)

    def get_shareholding(
        self,
        symbol: str,
        venue: str | None = None,
        start: date | None = None,
        end: date | None = None,
        as_of: date | None = None,
    ) -> list[ShareholdingRecord]:
        query = "SELECT * FROM shareholding_patterns WHERE symbol = ?"
        params: list[object] = [symbol.strip().upper()]
        if venue:
            query += " AND venue = ?"
            params.append(venue.strip().upper())
        if start:
            query += " AND period_end >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND period_end <= ?"
            params.append(end.isoformat())
        if as_of:
            query += " AND period_end <= ? AND filing_date <= ?"
            params.extend([as_of.isoformat(), as_of.isoformat()])
        query += " ORDER BY period_end DESC, filing_date DESC"
        cur = self.conn.cursor()
        cur.execute(query, params)
        return [_shareholding_from_row(row) for row in cur.fetchall()]

    def latest_shareholding_as_of(
        self,
        symbol: str,
        as_of: date,
        venue: str | None = None,
    ) -> ShareholdingRecord | None:
        rows = self.get_shareholding(symbol=symbol, venue=venue, as_of=as_of)
        return rows[0] if rows else None

    def shareholding_coverage(
        self,
        symbols: Sequence[str],
        as_of: date,
        venue: str | None = None,
    ) -> dict[str, object]:
        requested = {s.strip().upper() for s in symbols if s.strip()}
        available: dict[str, str] = {}
        for symbol in requested:
            latest = self.latest_shareholding_as_of(symbol=symbol, as_of=as_of, venue=venue)
            if latest is not None:
                available[symbol] = latest.period_end.isoformat()
        missing = sorted(requested - set(available))
        return {
            "symbols_requested": len(requested),
            "symbols_with_shareholding": len(available),
            "missing_symbols": missing,
            "coverage": round(len(available) / len(requested), 4) if requested else 0.0,
            "latest_period_by_symbol": available,
        }

    def upsert_banking_factors(self, records: Iterable[BankingFactorRecord]) -> int:
        rows = list(records)
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO banking_factors(
                venue, symbol, period_end, filing_date, net_interest_income,
                net_interest_margin_pct, advances_growth_pct, deposits_growth_pct,
                casa_ratio_pct, gnpa_ratio_pct, nnpa_ratio_pct,
                provision_coverage_ratio_pct, credit_cost_pct,
                capital_adequacy_ratio_pct, cet1_ratio_pct,
                cost_to_income_ratio_pct, roa_pct, roe_pct,
                loan_to_deposit_ratio_pct, source_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.venue.upper(),
                    record.symbol.upper(),
                    record.period_end.isoformat(),
                    record.filing_date.isoformat(),
                    record.net_interest_income,
                    record.net_interest_margin_pct,
                    record.advances_growth_pct,
                    record.deposits_growth_pct,
                    record.casa_ratio_pct,
                    record.gnpa_ratio_pct,
                    record.nnpa_ratio_pct,
                    record.provision_coverage_ratio_pct,
                    record.credit_cost_pct,
                    record.capital_adequacy_ratio_pct,
                    record.cet1_ratio_pct,
                    record.cost_to_income_ratio_pct,
                    record.roa_pct,
                    record.roe_pct,
                    record.loan_to_deposit_ratio_pct,
                    record.source_id,
                )
                for record in rows
            ],
        )
        self.conn.commit()
        return len(rows)

    def get_banking_factors(
        self,
        symbol: str,
        venue: str | None = None,
        start: date | None = None,
        end: date | None = None,
        as_of: date | None = None,
    ) -> list[BankingFactorRecord]:
        query = "SELECT * FROM banking_factors WHERE symbol = ?"
        params: list[object] = [symbol.strip().upper()]
        if venue:
            query += " AND venue = ?"
            params.append(venue.strip().upper())
        if start:
            query += " AND period_end >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND period_end <= ?"
            params.append(end.isoformat())
        if as_of:
            query += " AND period_end <= ? AND filing_date <= ?"
            params.extend([as_of.isoformat(), as_of.isoformat()])
        query += " ORDER BY period_end DESC, filing_date DESC"
        cur = self.conn.cursor()
        cur.execute(query, params)
        return [_banking_factor_from_row(row) for row in cur.fetchall()]

    def latest_banking_factor_as_of(
        self,
        symbol: str,
        as_of: date,
        venue: str | None = None,
    ) -> BankingFactorRecord | None:
        rows = self.get_banking_factors(symbol=symbol, venue=venue, as_of=as_of)
        return rows[0] if rows else None

    def banking_factor_coverage(
        self,
        symbols: Sequence[str],
        as_of: date,
        venue: str | None = None,
    ) -> dict[str, object]:
        requested = {s.strip().upper() for s in symbols if s.strip()}
        available: dict[str, str] = {}
        for symbol in requested:
            latest = self.latest_banking_factor_as_of(symbol=symbol, as_of=as_of, venue=venue)
            if latest is not None:
                available[symbol] = latest.period_end.isoformat()
        missing = sorted(requested - set(available))
        return {
            "symbols_requested": len(requested),
            "symbols_with_banking_factors": len(available),
            "missing_symbols": missing,
            "coverage": round(len(available) / len(requested), 4) if requested else 0.0,
            "latest_period_by_symbol": available,
        }

    def coverage_summary(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        interval: str = "1d",
    ) -> dict[str, object]:
        requested = {s.strip().upper() for s in symbols if s.strip()}
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT symbol, COUNT(*) AS n, MIN(ts) AS min_ts, MAX(ts) AS max_ts
            FROM ohlcv_bars
            WHERE interval = ? AND substr(ts, 1, 10) BETWEEN ? AND ?
            GROUP BY symbol
            """,
            (interval, start.isoformat(), end.isoformat()),
        )
        rows = cur.fetchall()
        by_symbol = {row["symbol"]: dict(row) for row in rows}
        missing = sorted(requested - set(by_symbol))
        return {
            "symbols_requested": len(requested),
            "symbols_with_bars": len(requested - set(missing)),
            "missing_symbols": missing,
            "coverage": round((len(requested) - len(missing)) / len(requested), 4) if requested else 0.0,
            "rows_by_symbol": by_symbol,
        }

    def close(self) -> None:
        self.conn.close()


def adjust_ohlcv_for_actions(
    bars: list[OHLCVBar],
    actions: list[CorporateActionRecord],
) -> list[OHLCVBar]:
    """Return price-adjusted bars using split/bonus action ratios.

    Cash dividends are retained as action metadata for now; total-return
    adjustment should be added once dividend amount quality is production-grade.
    """
    if not bars or not actions:
        return bars[:]
    adjusted: list[OHLCVBar] = []
    for bar in bars:
        factor = 1.0
        for action in actions:
            if action.ex_date and bar.ts < action.ex_date:
                factor *= _price_adjustment_factor(action)
        if factor == 1.0:
            adjusted.append(bar)
        else:
            adjusted.append(
                replace(
                    bar,
                    open=bar.open * factor,
                    high=bar.high * factor,
                    low=bar.low * factor,
                    close=bar.close * factor,
                    volume=bar.volume / factor if factor > 0 else bar.volume,
                )
            )
    return adjusted


def _normalize_ohlcv_bar(bar: OHLCVBar, interval: str) -> OHLCVBar:
    if interval != "1d":
        return bar
    normalized_ts = _daily_trade_date(bar.ts)
    return replace(bar, ts=normalized_ts) if normalized_ts != bar.ts else bar


def _daily_trade_date(ts: str) -> str:
    return str(ts or "").strip()[:10]


def _price_adjustment_factor(action: CorporateActionRecord) -> float:
    ratio = _parse_ratio(action.ratio)
    if ratio is None:
        return 1.0
    numerator, denominator = ratio
    if numerator <= 0 or denominator <= 0:
        return 1.0
    kind = action.action_type.lower()
    if "split" in kind:
        return denominator / numerator
    if "bonus" in kind:
        return denominator / (numerator + denominator)
    return 1.0


def _parse_ratio(value: str | None) -> tuple[float, float] | None:
    if not value:
        return None
    text = value.strip().lower().replace(" ", "")
    for sep in (":", "/", "-"):
        if sep in text:
            left, right = text.split(sep, maxsplit=1)
            try:
                return float(left), float(right)
            except ValueError:
                return None
    return None


def _date_or_none(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def _prefer_rich_text(incoming: str, existing: str) -> str:
    return existing if _is_placeholder_text(incoming) and not _is_placeholder_text(existing) else incoming


def _prefer_company_name(incoming: str, existing: str, symbol: str) -> str:
    if incoming.strip().upper() == symbol.strip().upper() and not _is_placeholder_text(existing):
        return existing
    return _prefer_rich_text(incoming, existing)


def _is_placeholder_text(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().upper() in {"", "UNKNOWN", "NA", "N/A", "NONE", "NULL"}


def _prefer_security_source(incoming: str, existing: str) -> str:
    if incoming == "runtime_universe" and existing and existing != "runtime_universe":
        return existing
    return incoming


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _security_from_row(row: sqlite3.Row) -> SecurityMasterRecord:
    return SecurityMasterRecord(
        symbol=row["symbol"],
        isin=row["isin"],
        exchange=row["exchange"],
        series=row["series"],
        company_name=row["company_name"],
        sector=row["sector"],
        industry=row["industry"],
        listing_date=_parse_date(row["listing_date"]),
        delisting_date=_parse_date(row["delisting_date"]),
        active=bool(row["active"]),
        lot_size=int(row["lot_size"]),
        tick_size=float(row["tick_size"]),
        source=row["source"],
    )


def _session_from_row(row: sqlite3.Row) -> MarketSessionRecord:
    return MarketSessionRecord(
        venue=row["venue"],
        session_date=date.fromisoformat(row["session_date"]),
        is_trading_day=bool(row["is_trading_day"]),
        session_type=row["session_type"],
        reason=row["reason"],
    )


def _bar_from_row(row: sqlite3.Row) -> OHLCVBar:
    return OHLCVBar(
        venue=row["venue"],
        symbol=row["symbol"],
        ts=row["ts"],
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
    )


def _action_from_row(row: sqlite3.Row) -> CorporateActionRecord:
    return CorporateActionRecord(
        venue=row["venue"],
        symbol=row["symbol"],
        action_type=row["action_type"],
        ex_date=row["ex_date"],
        record_date=row["record_date"],
        ratio=row["ratio"],
        cash_amount=row["cash_amount"],
        currency=row["currency"],
        source_id=row["source_id"],
    )


def _delivery_turnover_from_row(row: sqlite3.Row) -> DeliveryTurnoverRecord:
    return DeliveryTurnoverRecord(
        venue=row["venue"],
        symbol=row["symbol"],
        trade_date=date.fromisoformat(row["trade_date"]),
        traded_quantity=float(row["traded_quantity"]),
        delivery_quantity=float(row["delivery_quantity"]),
        delivery_pct=float(row["delivery_pct"]),
        source_id=row["source_id"],
    )


def _statement_from_row(row: sqlite3.Row) -> FinancialStatementRecord:
    return FinancialStatementRecord(
        venue=row["venue"],
        symbol=row["symbol"],
        period_end=date.fromisoformat(row["period_end"]),
        filing_date=date.fromisoformat(row["filing_date"]),
        statement_type=row["statement_type"],
        revenue=float(row["revenue"]),
        ebit=float(row["ebit"]),
        net_income=float(row["net_income"]),
        operating_cash_flow=float(row["operating_cash_flow"]),
        capex=float(row["capex"]),
        total_debt=float(row["total_debt"]),
        equity=float(row["equity"]),
        total_assets=float(row["total_assets"]),
        current_assets=float(row["current_assets"]),
        current_liabilities=float(row["current_liabilities"]),
        interest_expense=float(row["interest_expense"]),
        source_id=row["source_id"],
    )


def _valuation_from_row(row: sqlite3.Row) -> EquityValuationRecord:
    return EquityValuationRecord(
        venue=row["venue"],
        symbol=row["symbol"],
        as_of=date.fromisoformat(row["as_of"]),
        market_cap=float(row["market_cap"]),
        shares_outstanding=float(row["shares_outstanding"]),
        free_float_market_cap=float(row["free_float_market_cap"]),
        enterprise_value=float(row["enterprise_value"]),
        currency=row["currency"],
        source_id=row["source_id"],
    )


def _shareholding_from_row(row: sqlite3.Row) -> ShareholdingRecord:
    return ShareholdingRecord(
        venue=row["venue"],
        symbol=row["symbol"],
        period_end=date.fromisoformat(row["period_end"]),
        filing_date=date.fromisoformat(row["filing_date"]),
        promoter_pct=float(row["promoter_pct"]),
        fii_pct=float(row["fii_pct"]),
        dii_pct=float(row["dii_pct"]),
        public_pct=float(row["public_pct"]),
        source_id=row["source_id"],
    )


def _banking_factor_from_row(row: sqlite3.Row) -> BankingFactorRecord:
    return BankingFactorRecord(
        venue=row["venue"],
        symbol=row["symbol"],
        period_end=date.fromisoformat(row["period_end"]),
        filing_date=date.fromisoformat(row["filing_date"]),
        net_interest_income=float(row["net_interest_income"]),
        net_interest_margin_pct=float(row["net_interest_margin_pct"]),
        advances_growth_pct=float(row["advances_growth_pct"]),
        deposits_growth_pct=float(row["deposits_growth_pct"]),
        casa_ratio_pct=float(row["casa_ratio_pct"]),
        gnpa_ratio_pct=float(row["gnpa_ratio_pct"]),
        nnpa_ratio_pct=float(row["nnpa_ratio_pct"]),
        provision_coverage_ratio_pct=float(row["provision_coverage_ratio_pct"]),
        credit_cost_pct=float(row["credit_cost_pct"]),
        capital_adequacy_ratio_pct=float(row["capital_adequacy_ratio_pct"]),
        cet1_ratio_pct=float(row["cet1_ratio_pct"]),
        cost_to_income_ratio_pct=float(row["cost_to_income_ratio_pct"]),
        roa_pct=float(row["roa_pct"]),
        roe_pct=float(row["roe_pct"]),
        loan_to_deposit_ratio_pct=float(row["loan_to_deposit_ratio_pct"]),
        source_id=row["source_id"],
    )
