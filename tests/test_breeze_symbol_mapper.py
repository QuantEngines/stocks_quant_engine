from __future__ import annotations

from datetime import date

from stock_screener_engine.config.settings import BrokerIntegrationSettings
from stock_screener_engine.data_sources.broker.breeze_adapter import BreezeAdapter
from stock_screener_engine.data_sources.broker.breeze_symbol_mapper import BreezeSymbolMapper


class _BreezeClient:
    def __init__(self) -> None:
        self.quote_codes: list[str] = []
        self.history_codes: list[str] = []
        self.name_calls: list[str] = []

    def get_names(self, exchange_code, stock_code):
        self.name_calls.append(stock_code)
        return {
            "exchange_code": exchange_code,
            "exchange_stock_code": stock_code,
            "isec_stock_code": "RELIND",
            "isec_token": "2885",
            "company name": "RELIANCE INDUSTRIES",
        }

    def get_quotes(self, stock_code="", exchange_code="", expiry_date="", product_type="", right="", strike_price=""):
        self.quote_codes.append(stock_code)
        return {"Success": [{"ltp": "2500.5", "total_quantity_traded": "1000"}]}

    def get_historical_data_v2(
        self,
        interval="",
        from_date="",
        to_date="",
        stock_code="",
        exchange_code="",
        product_type="",
        expiry_date="",
        right="",
        strike_price="",
    ):
        self.history_codes.append(stock_code)
        return {"Success": [{"datetime": "2026-05-27", "open": "2490", "high": "2510", "low": "2480", "close": "2500"}]}

    def place_order(self, **payload):
        return "order-1"


def test_breeze_symbol_mapper_learns_and_persists_get_names_mapping(tmp_path) -> None:
    client = _BreezeClient()
    path = tmp_path / "breeze_symbol_map.csv"
    mapper = BreezeSymbolMapper(map_path=path)

    resolved = mapper.resolve("RELIANCE", client=client)
    reloaded = BreezeSymbolMapper(map_path=path)

    assert resolved.stock_code == "RELIND"
    assert resolved.source == "breeze_get_names"
    assert path.exists()
    assert reloaded.resolve("RELIANCE").stock_code == "RELIND"
    assert client.name_calls == ["RELIANCE"]


def test_breeze_symbol_mapper_uses_manual_map_file_before_client(tmp_path) -> None:
    path = tmp_path / "breeze_symbol_map.csv"
    path.write_text("symbol,stock_code,source\nBAJAJ-AUTO,BAAUTO,manual\n", encoding="utf-8")
    client = _BreezeClient()
    mapper = BreezeSymbolMapper(map_path=path)

    resolved = mapper.resolve("BAJAJ-AUTO", client=client)

    assert resolved.stock_code == "BAAUTO"
    assert resolved.source == "manual"
    assert client.name_calls == []


def test_breeze_adapter_uses_mapped_stock_code_for_quote_and_history(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BREEZE_KEY", "key")
    monkeypatch.setenv("BREEZE_SECRET", "secret")
    monkeypatch.setenv("BREEZE_TOKEN", "token")
    client = _BreezeClient()
    mapper = BreezeSymbolMapper(map_path=tmp_path / "breeze_symbol_map.csv")
    adapter = BreezeAdapter(
        BrokerIntegrationSettings(
            enabled=True,
            api_key_env="BREEZE_KEY",
            api_secret_env="BREEZE_SECRET",
            token_env="BREEZE_TOKEN",
        ),
        client=client,
        symbol_mapper=mapper,
    )

    quote = adapter.get_quote(["RELIANCE"])["RELIANCE"]
    history = adapter.get_historical("RELIANCE", "1d", date(2026, 5, 26), date(2026, 5, 27))

    assert client.quote_codes == ["RELIND"]
    assert client.history_codes == ["RELIND"]
    assert quote["broker_symbol"] == "RELIND"
    assert quote["mapping_source"] == "breeze_get_names"
    assert history[0]["broker_symbol"] == "RELIND"
