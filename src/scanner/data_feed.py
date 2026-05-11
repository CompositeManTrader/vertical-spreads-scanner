"""
Data feed abstraido: interfaz comun para multiple providers.
- YFinanceFeed: gratis, sin auth (default).
- SchwabFeed: oficial, requiere OAuth (schwab-py).

Para usar Schwab, crear .env con:
    SCHWAB_APP_KEY=...
    SCHWAB_APP_SECRET=...
    SCHWAB_CALLBACK_URL=https://127.0.0.1:8182
    SCHWAB_TOKEN_PATH=./schwab_token.json
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Interfaz comun
# ---------------------------------------------------------------------------

class DataFeed(ABC):
    name: str = "base"

    @abstractmethod
    def fetch_history(self, ticker: str, years: int = 2) -> pd.DataFrame:
        """OHLCV diario. Cols: Date, Open, High, Low, Close, Volume."""

    @abstractmethod
    def fetch_atm_iv_30d(self, ticker: str) -> tuple[float | None, dict]:
        """ATM IV ~30D constant maturity (interpolada del chain)."""

    @abstractmethod
    def fetch_vix_close(self) -> tuple[float | None, dict]:
        """Cierre del VIX."""


# ---------------------------------------------------------------------------
# YFinanceFeed (default, sin auth)
# ---------------------------------------------------------------------------

class YFinanceFeed(DataFeed):
    name = "yfinance"

    def fetch_history(self, ticker: str, years: int = 2) -> pd.DataFrame:
        import yfinance as yf
        end = datetime.today()
        start = end - timedelta(days=int(years * 365.25) + 30)
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"),
                         progress=False, auto_adjust=False)
        if df.empty:
            raise RuntimeError(f"yfinance no devolvio datos para {ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        return df.sort_values("Date").reset_index(drop=True)

    def fetch_atm_iv_30d(self, ticker: str) -> tuple[float | None, dict]:
        import yfinance as yf
        yt = yf.Ticker(ticker)
        expirations = yt.options
        if not expirations:
            return None, {"error": "no expirations"}
        info = yt.history(period="1d")
        if info.empty:
            return None, {"error": "no quote"}
        spot = float(info["Close"].iloc[-1])
        today = datetime.today().date()
        expiry_dte = []
        for e_str in expirations:
            e_date = datetime.strptime(e_str, "%Y-%m-%d").date()
            dte = (e_date - today).days
            if dte > 0:
                expiry_dte.append((e_str, dte))
        if not expiry_dte:
            return None, {"error": "sin expirations futuras"}
        expiry_dte.sort(key=lambda x: abs(x[1] - 30))
        ivs = []
        for e_str, dte in expiry_dte[:2]:
            try:
                chain = yt.option_chain(e_str)
            except Exception:
                continue
            calls = chain.calls
            if calls.empty:
                continue
            idx = (calls["strike"] - spot).abs().idxmin()
            iv = calls.loc[idx, "impliedVolatility"]
            if pd.notna(iv) and iv > 0:
                ivs.append((dte, float(iv)))
        if not ivs:
            return None, {"error": "no IVs validas"}
        if len(ivs) == 1:
            return ivs[0][1], {"spot": spot, "method": "single", "dte": ivs[0][0]}
        (dte1, iv1), (dte2, iv2) = ivs[0], ivs[1]
        if dte1 == dte2:
            return iv1, {"spot": spot, "method": "duplicate"}
        iv30 = iv1 + (iv2 - iv1) * (30 - dte1) / (dte2 - dte1)
        return float(iv30), {
            "spot": spot, "method": "interpolated",
            "anchors": [(dte1, iv1), (dte2, iv2)],
        }

    def fetch_vix_close(self) -> tuple[float | None, dict]:
        import yfinance as yf
        yt = yf.Ticker("^VIX")
        h = yt.history(period="5d")
        if h.empty:
            return None, {"error": "no VIX"}
        return float(h["Close"].iloc[-1]), {"date": str(h.index[-1].date())}


# ---------------------------------------------------------------------------
# SchwabFeed (oficial, OAuth)
# ---------------------------------------------------------------------------

class SchwabFeed(DataFeed):
    """
    Wrapper de schwab-py.

    Setup primer uso (interactivo, en terminal):
        from schwab.auth import easy_client
        c = easy_client(api_key=APP_KEY, app_secret=APP_SECRET,
                        callback_url=CALLBACK_URL, token_path=TOKEN_PATH)
        # Abre browser, autorizas, copia URL de redirect.
    Despues queda guardado en TOKEN_PATH y refresca solo.
    """
    name = "schwab"

    def __init__(self):
        # Soporta tres fuentes para credenciales (en orden de prioridad):
        #   1) st.secrets (Streamlit Cloud / .streamlit/secrets.toml local)
        #   2) os.environ (vars de entorno)
        #   3) .env file (python-dotenv, dev local)
        creds = self._load_credentials()
        self.app_key = creds.get("SCHWAB_APP_KEY")
        self.app_secret = creds.get("SCHWAB_APP_SECRET")
        self.callback_url = creds.get("SCHWAB_CALLBACK_URL", "https://127.0.0.1:8182")
        self.token_path = creds.get("SCHWAB_TOKEN_PATH", "./schwab_token.json")
        if not (self.app_key and self.app_secret):
            raise RuntimeError(
                "Faltan SCHWAB_APP_KEY o SCHWAB_APP_SECRET. "
                "Configurar en .env (local), variables de entorno, o "
                "st.secrets (Streamlit Cloud)."
            )
        self._client = None

    @staticmethod
    def _load_credentials() -> dict:
        out = {}
        # 1) Streamlit secrets si esta corriendo dentro de Streamlit
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                for k in ("SCHWAB_APP_KEY", "SCHWAB_APP_SECRET",
                          "SCHWAB_CALLBACK_URL", "SCHWAB_TOKEN_PATH"):
                    try:
                        v = st.secrets.get(k) if hasattr(st.secrets, "get") else st.secrets[k]
                    except Exception:
                        v = None
                    if v:
                        out[k] = v
        except Exception:
            pass
        # 2) .env file (no sobrescribe streamlit)
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass
        # 3) Environ (no sobrescribe streamlit)
        for k in ("SCHWAB_APP_KEY", "SCHWAB_APP_SECRET",
                  "SCHWAB_CALLBACK_URL", "SCHWAB_TOKEN_PATH"):
            if k not in out and os.getenv(k):
                out[k] = os.getenv(k)
        return out

    @property
    def client(self):
        if self._client is None:
            from schwab.auth import easy_client
            self._client = easy_client(
                api_key=self.app_key, app_secret=self.app_secret,
                callback_url=self.callback_url, token_path=self.token_path,
            )
        return self._client

    def fetch_history(self, ticker: str, years: int = 2) -> pd.DataFrame:
        from schwab.client import Client
        end = datetime.today()
        start = end - timedelta(days=int(years * 365.25) + 30)
        resp = self.client.get_price_history_every_day(
            symbol=ticker,
            start_datetime=start, end_datetime=end,
            need_extended_hours_data=False,
        )
        data = resp.json()
        candles = data.get("candles", [])
        if not candles:
            raise RuntimeError(f"Schwab sin candles para {ticker}")
        df = pd.DataFrame(candles)
        df["Date"] = pd.to_datetime(df["datetime"], unit="ms")
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                "close": "Close", "volume": "Volume"})
        return df[["Date", "Open", "High", "Low", "Close", "Volume"]] \
                  .sort_values("Date").reset_index(drop=True)

    def fetch_atm_iv_30d(self, ticker: str) -> tuple[float | None, dict]:
        """
        Schwab option chain endpoint da volatility por strike. Buscamos las dos
        expirations mas cercanas a 30 DTE y interpolamos la ATM IV.
        """
        from schwab.client import Client
        resp = self.client.get_option_chain(
            symbol=ticker,
            contract_type=Client.Options.ContractType.PUT,
            include_underlying_quote=True,
            strategy=Client.Options.Strategy.SINGLE,
            range_=Client.Options.Range.NEAR_THE_MONEY,
            from_date=datetime.today(),
            to_date=datetime.today() + timedelta(days=70),
        )
        data = resp.json()
        underlying = data.get("underlying", {})
        spot = float(underlying.get("last", underlying.get("mark", 0))) or None
        if spot is None:
            return None, {"error": "sin spot en chain"}

        put_map = data.get("putExpDateMap", {})
        ivs = []  # list of (dte, iv_atm)
        for exp_str, strikes_dict in put_map.items():
            # exp_str fmt: '2026-06-13:30'
            try:
                date_part, dte_str = exp_str.split(":")
                dte = int(dte_str)
            except Exception:
                continue
            if dte <= 0:
                continue
            # ATM = strike mas cercano al spot
            best_strike = None
            best_diff = 1e18
            best_iv = None
            for strike_str, opts in strikes_dict.items():
                strike = float(strike_str)
                diff = abs(strike - spot)
                if diff < best_diff and opts:
                    iv = opts[0].get("volatility")
                    if iv is not None and iv > 0:
                        best_diff = diff
                        best_strike = strike
                        best_iv = iv / 100.0  # Schwab da en % anualizado
            if best_iv:
                ivs.append((dte, best_iv))

        if not ivs:
            return None, {"error": "sin IVs en chain"}
        ivs.sort(key=lambda x: abs(x[0] - 30))
        if len(ivs) == 1:
            return ivs[0][1], {"spot": spot, "method": "single", "dte": ivs[0][0]}
        (dte1, iv1), (dte2, iv2) = ivs[0], ivs[1]
        if dte1 == dte2:
            return iv1, {"spot": spot, "method": "duplicate"}
        iv30 = iv1 + (iv2 - iv1) * (30 - dte1) / (dte2 - dte1)
        return float(iv30), {
            "spot": spot, "method": "interpolated",
            "anchors": [(dte1, iv1), (dte2, iv2)],
        }

    def fetch_vix_close(self) -> tuple[float | None, dict]:
        # Schwab puede o no exponer ^VIX. Fallback a yfinance para indices CBOE.
        return YFinanceFeed().fetch_vix_close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_feed(provider: str = "yfinance") -> DataFeed:
    if provider == "schwab":
        return SchwabFeed()
    return YFinanceFeed()


# ---------------------------------------------------------------------------
# Helpers para retro-compatibilidad con scanner/main.py
# ---------------------------------------------------------------------------

_default_feed = YFinanceFeed()

def fetch_history(ticker: str, years: int = 3) -> pd.DataFrame:
    return _default_feed.fetch_history(ticker, years)

def fetch_atm_iv_30d(ticker: str) -> tuple[float | None, dict]:
    return _default_feed.fetch_atm_iv_30d(ticker)

def fetch_vix_close() -> tuple[float | None, dict]:
    return _default_feed.fetch_vix_close()
