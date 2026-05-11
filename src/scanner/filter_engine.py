"""
Filter engine: aplica el filtro 'above_sma200 AND vrp_high' point-in-time.

Reusa los indicadores de src.analysis.indicators (point-in-time, sin look-ahead).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.indicators import (
    log_return_back, realized_vol, sma,
)


def compute_features_for_today(history: pd.DataFrame, today_iv: float) -> dict:
    """
    Toma la historia diaria y la IV actual, devuelve un dict con todas las
    features point-in-time evaluadas en el ULTIMO dia de history.
    """
    df = history.copy().sort_values("Date").reset_index(drop=True)
    closes = df["Close"]

    # SMAs
    sma_50 = sma(closes, 50)
    sma_200 = sma(closes, 200)

    # Returns y RV
    rets = log_return_back(closes, 1)
    rv_20d = realized_vol(rets, window=20)

    # VRP threshold expanding (al menos 252 obs)
    # iv historico: lo aproximamos con rv_20d como base, pero la IV actual del
    # ultimo dia se introduce manualmente (el iv real)
    # Para el threshold de VRP usamos serie de RV historica de la cual no
    # tenemos IV. Solucion practica: definimos vrp = today_iv - rv_20d_today
    # y comparamos contra distribucion historica de (vix_history - rv_20d) o
    # mas simple: vrp threshold = 0 (IV > RV).
    #
    # Para esta version, simplificamos:
    #   vrp_high si IV(today) - RV20d(today) > X
    # X razonable = percentil 80 historico de (RV30d.shift(-30) - RV30d). No
    # tenemos IV historica facil. Usamos una regla simple: VRP > 0.03 (3pp)
    # como threshold absoluto, calibrable.
    last_rv = float(rv_20d.iloc[-1])
    last_close = float(closes.iloc[-1])
    last_sma50 = float(sma_50.iloc[-1])
    last_sma200 = float(sma_200.iloc[-1])

    vrp_today = today_iv - last_rv  # ambos en fraccion anual

    return {
        "date": df["Date"].iloc[-1],
        "close": last_close,
        "iv_atm_today": float(today_iv),
        "rv_20d": last_rv,
        "vrp": float(vrp_today),
        "sma_50": last_sma50,
        "sma_200": last_sma200,
        "above_sma50": int(last_close > last_sma50),
        "above_sma200": int(last_close > last_sma200),
        "price_to_sma200_pct": float(last_close / last_sma200 - 1.0),
    }


def evaluate_filter(features: dict, vrp_threshold: float = 0.03) -> dict:
    """
    Aplica el filtro 'above_sma200 AND vrp >= threshold'.
    Threshold default 0.03 (3 vol points): IV ATM 3pp por sobre RV20d.
    Calibrado aprox a quintil 80 historico en research.
    """
    cond1 = features["above_sma200"] == 1
    cond2 = features["vrp"] >= vrp_threshold

    return {
        "above_sma200_pass": bool(cond1),
        "vrp_pass": bool(cond2),
        "vrp_threshold_used": vrp_threshold,
        "filter_pass": bool(cond1 and cond2),
        "vrp_value": features["vrp"],
        "vrp_distance_to_threshold": features["vrp"] - vrp_threshold,
    }
