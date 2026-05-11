"""
Fase 1: dependencia temporal de retornos.

- Autocorrelacion de log returns diarios (test de random walk).
- Autocorrelacion de log returns al cuadrado (volatility clustering).
- Test de Ljung-Box.
- Test de mean reversion / momentum a 5, 10, 20 dias.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf


def autocorrelation_table(returns: pd.Series, lags: list[int] | None = None) -> pd.DataFrame:
    """
    Autocorrelacion de retornos diarios y de retornos al cuadrado.
    """
    if lags is None:
        lags = [1, 2, 5, 10, 20, 60]
    r = returns.dropna()
    r2 = (r ** 2).dropna()

    nlags = max(lags)
    acf_r = acf(r, nlags=nlags, fft=False)
    acf_r2 = acf(r2, nlags=nlags, fft=False)

    rows = []
    for lag in lags:
        rows.append({
            "lag": lag,
            "acf_returns": float(acf_r[lag]),
            "acf_squared_returns": float(acf_r2[lag]),
        })
    return pd.DataFrame(rows)


def ljung_box(returns: pd.Series, lags: list[int] | None = None) -> pd.DataFrame:
    """
    Test Ljung-Box. H0: no autocorrelacion hasta el lag k.
    p < 0.05 -> rechazar H0 -> hay autocorrelacion significativa.
    """
    if lags is None:
        lags = [5, 10, 20]
    r = returns.dropna()
    r2 = (r ** 2).dropna()

    lb_r = acorr_ljungbox(r, lags=lags, return_df=True)
    lb_r2 = acorr_ljungbox(r2, lags=lags, return_df=True)

    out = pd.DataFrame({
        "lag": lags,
        "lb_stat_returns": lb_r["lb_stat"].values,
        "lb_pvalue_returns": lb_r["lb_pvalue"].values,
        "lb_stat_squared": lb_r2["lb_stat"].values,
        "lb_pvalue_squared": lb_r2["lb_pvalue"].values,
    })
    return out


def momentum_reversion_test(panel: pd.DataFrame, lookback: int, fwd: int) -> dict:
    """
    Mide la correlacion entre ret_back(t, lookback) y ret_fwd(t, fwd).
    >0 -> momentum (returns pasados predicen futuros del mismo signo).
    <0 -> reversion.
    Usa Spearman (no parametrico, robusto a outliers).
    """
    from src.analysis.indicators import log_return_back, log_return_forward
    rb = log_return_back(panel["Close"], lookback)
    rf = log_return_forward(panel["Close"], fwd)
    df = pd.DataFrame({"rb": rb, "rf": rf}).dropna()
    if len(df) < 50:
        return {"n": 0}
    rho, p = stats.spearmanr(df["rb"], df["rf"])
    return {
        "lookback_days": lookback,
        "fwd_days": fwd,
        "n": int(len(df)),
        "spearman_rho": float(rho),
        "pvalue": float(p),
        "interpretation": "momentum" if rho > 0 else "reversion",
    }
