"""
Construye el panel enriquecido por ticker con TODAS las features de regimen
necesarias para Fase 3.

ANTI-LOOK-AHEAD: cada feature usa SOLO info <=t. Calculadas con rolling/
expanding cerradas en t.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import DATA_CLEAN, DATA_EXTERNAL
from src.analysis.indicators import (
    days_since_ath, drawdown_from_high, log_return_back, realized_vol,
    sma, sma_slope,
)


def build_regime_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Toma el panel limpio del ticker (con OHLC, IV, VIX, etc.) y agrega columnas
    de regimen. Devuelve nuevo DataFrame con todas las features point-in-time.
    """
    df = panel.copy()
    closes = df["Close"]
    rets = log_return_back(closes, 1)
    df["ret_back_1d"] = rets

    # ----- Volatilidad realizada (RV) -----
    df["rv_20d"] = realized_vol(rets, window=20)
    df["rv_60d"] = realized_vol(rets, window=60)

    # ----- VRP (variance risk premium): IV - RV -----
    # IV ya en fraccion (0.17), RV idem (anualizada en fraccion).
    df["vrp"] = df["iv_atm_barchart"] - df["rv_20d"]

    # ----- Vol of vol: rolling std de RV -----
    df["vol_of_vol_60d"] = df["rv_20d"].rolling(60, min_periods=60).std()

    # ----- Tendencia: precio vs SMA -----
    df["sma_20"] = sma(closes, 20)
    df["sma_50"] = sma(closes, 50)
    df["sma_200"] = sma(closes, 200)
    df["price_to_sma50"] = closes / df["sma_50"] - 1.0
    df["price_to_sma200"] = closes / df["sma_200"] - 1.0
    df["sma200_slope_60d"] = sma_slope(closes, 200, slope_lookback=60)
    df["above_sma50"] = (closes > df["sma_50"]).astype(int)
    df["above_sma200"] = (closes > df["sma_200"]).astype(int)

    # ----- Drawdown desde maximos point-in-time -----
    df["dd_60d"] = drawdown_from_high(closes, lookback=60)
    df["dd_252d"] = drawdown_from_high(closes, lookback=252)
    df["dd_alltime"] = drawdown_from_high(closes, lookback=None)
    df["days_since_ath"] = days_since_ath(closes)

    # ----- VIX features (del master vol) -----
    if "VIX_Close" in df.columns:
        df["vix_change_5d"] = df["VIX_Close"].pct_change(5)
        df["vix_change_20d"] = df["VIX_Close"].pct_change(20)
        # Ratio VIX/VIX3M ya viene como columna VIX_VIX3M_Ratio del master
        # >1 = backwardation, <1 = contango
        if "VIX_VIX3M_Ratio" in df.columns:
            df["term_structure_ratio"] = df["VIX_VIX3M_Ratio"]

    # ----- Calendarios macro (flags) -----
    df = _add_calendar_flags(df)

    # ----- Stress flag: ret diario <= -2% (proxy de panic day) -----
    df["panic_day"] = (rets <= -0.02).astype(int)
    df["panic_5d_window"] = df["panic_day"].rolling(5, min_periods=1).max()

    return df


def _add_calendar_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Anade flags binarias para FOMC, CPI, NFP, earnings season."""
    out = df.copy()
    for fname, flag in [("calendar_fomc.parquet", "is_FOMC"),
                        ("calendar_cpi.parquet", "is_CPI"),
                        ("calendar_nfp.parquet", "is_NFP"),
                        ("calendar_earnings_season.parquet", "is_earnings_season")]:
        cal = pd.read_parquet(DATA_EXTERNAL / fname)
        cal["Date"] = pd.to_datetime(cal["Date"])
        out = out.merge(cal[["Date", flag]], on="Date", how="left")
        out[flag] = out[flag].fillna(False).astype(bool)
        # Tambien: dia ANTES del evento
        out[f"{flag}_t1"] = out[flag].shift(-1).fillna(False).astype(bool)
        # Dia de la semana / mes (para seasonality)
    out["dow"] = out["Date"].dt.dayofweek  # 0=lunes
    out["month"] = out["Date"].dt.month
    return out


def main():
    """Construye paneles enriquecidos para SPY/QQQ/IWM (train)."""
    out_dir = DATA_CLEAN / "train_enriched"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ticker in ["SPY", "QQQ", "IWM"]:
        df = pd.read_parquet(DATA_CLEAN / "train" / f"{ticker}.parquet")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        enriched = build_regime_features(df)
        out = out_dir / f"{ticker}.parquet"
        enriched.to_parquet(out, index=False)
        print(f"{ticker}: {enriched.shape[0]} filas x {enriched.shape[1]} cols -> {out.name}")


if __name__ == "__main__":
    main()
