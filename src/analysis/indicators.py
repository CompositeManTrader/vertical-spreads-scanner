"""
Indicadores tecnicos y derivados, calculados con ANTI-LOOK-AHEAD estricto.

Convencion:
- Toda ventana es CERRADA en t (incluye t, no t+1).
- Funciones que devuelven una serie alineada con el input.
- Ningun .shift(-N).
- Tests unitarios verifican que f(s[:t]) == f(s)[:t] (point-in-time consistency).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# IV Rank y IV Percentile (recalculados nosotros, sin look-ahead)
# ---------------------------------------------------------------------------

def iv_rank(iv: pd.Series, lookback: int = 252) -> pd.Series:
    """
    IV Rank canonico (Tastytrade / Barchart):

        IV_Rank(t) = (IV(t) - min(IV[t-lookback+1:t+1])) /
                     (max(IV[t-lookback+1:t+1]) - min(IV[t-lookback+1:t+1]))

    Devuelve valor en [0, 1]. NaN para los primeros lookback-1 puntos.

    ANTI-LOOK-AHEAD: ventana cerrada en t (incluye t, no usa t+1 ni mas alla).
    """
    iv_min = iv.rolling(window=lookback, min_periods=lookback).min()
    iv_max = iv.rolling(window=lookback, min_periods=lookback).max()
    rng = iv_max - iv_min
    rank = (iv - iv_min) / rng
    # Cuando rng == 0 (toda la ventana es igual), rank queda indefinido -> NaN.
    rank = rank.where(rng > 0)
    return rank


def iv_percentile(iv: pd.Series, lookback: int = 252) -> pd.Series:
    """
    IV Percentile: fraccion de dias en la ventana donde IV < IV(t).

        IV_Pct(t) = (#{i in [t-lookback+1, t] : IV(i) < IV(t)}) / lookback

    Devuelve valor en [0, 1]. NaN para los primeros lookback-1 puntos.

    ANTI-LOOK-AHEAD: ventana cerrada en t.
    """
    def pct(window: np.ndarray) -> float:
        if len(window) < lookback:
            return np.nan
        current = window[-1]
        return float(np.sum(window[:-1] < current)) / (lookback - 1)

    return iv.rolling(window=lookback, min_periods=lookback).apply(pct, raw=True)


# ---------------------------------------------------------------------------
# Realized volatility
# ---------------------------------------------------------------------------

def realized_vol(returns: pd.Series, window: int = 20, annualize: int = 252) -> pd.Series:
    """
    Volatilidad realizada anualizada de log-returns diarios.

    ANTI-LOOK-AHEAD: rolling cerrado en t.
    """
    return returns.rolling(window=window, min_periods=window).std() * np.sqrt(annualize)


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def log_return_back(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    Log-return MIRANDO HACIA ATRAS: ret_back(t) = ln(P(t) / P(t-periods)).
    Es un FEATURE legitimo en t. Anti-look-ahead OK.
    """
    return np.log(prices / prices.shift(periods))


def log_return_forward(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    Log-return MIRANDO HACIA ADELANTE: ret_fwd(t) = ln(P(t+periods) / P(t)).
    Es un LABEL (outcome). NO debe usarse como feature en t.
    Convencion explicita en el nombre para evitar confusion.
    """
    return np.log(prices.shift(-periods) / prices)


# ---------------------------------------------------------------------------
# Drawdown desde maximo expandiendo (NO usa max global - eso seria look-ahead)
# ---------------------------------------------------------------------------

def drawdown_from_high(prices: pd.Series, lookback: int | None = None) -> pd.Series:
    """
    Drawdown actual desde el maximo de los ultimos `lookback` dias (cerrado en t).
    Si lookback=None, usa expanding max desde el inicio (max histórico hasta t).

    ANTI-LOOK-AHEAD: si fuera prices.max() global, seria leak.
    """
    if lookback is None:
        roll_max = prices.expanding(min_periods=1).max()
    else:
        roll_max = prices.rolling(window=lookback, min_periods=1).max()
    return 1.0 - prices / roll_max


# ---------------------------------------------------------------------------
# SMA y slope
# ---------------------------------------------------------------------------

def sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window=window, min_periods=window).mean()


def sma_slope(prices: pd.Series, window: int, slope_lookback: int = 20) -> pd.Series:
    """
    Slope de la SMA: pendiente lineal sobre los ultimos `slope_lookback` puntos
    de la SMA. Devuelve change_pct (sma(t) / sma(t-slope_lookback) - 1).
    """
    s = sma(prices, window)
    return s / s.shift(slope_lookback) - 1.0


# ---------------------------------------------------------------------------
# Days since all-time high (point-in-time)
# ---------------------------------------------------------------------------

def days_since_ath(prices: pd.Series) -> pd.Series:
    """
    Cuenta dias desde el ultimo all-time high observado <= t.

    ANTI-LOOK-AHEAD: usa expanding max, no max global.
    """
    expanding_max = prices.expanding(min_periods=1).max()
    is_ath = prices >= expanding_max  # tolerante a ties
    # Para cada t, dias desde ultimo True
    last_true_idx = np.where(is_ath, np.arange(len(prices)), np.nan)
    last_true_idx = pd.Series(last_true_idx, index=prices.index).ffill()
    days = np.arange(len(prices)) - last_true_idx.values
    return pd.Series(days, index=prices.index)
