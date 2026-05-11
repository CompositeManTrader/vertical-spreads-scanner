"""
Fase 1: distribucion empirica de retornos a T dias.

Convencion:
- log_return_forward(prices, T) en t devuelve ln(P(t+T)/P(t)).
  Es un LABEL/outcome (solo se usa en analisis, NO como feature).
- Se usa para caracterizar la distribucion empirica.
- Solo se usa el panel TRAIN (2018-10-03 a 2024-10-03).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.indicators import log_return_back, log_return_forward


# ---------------------------------------------------------------------------
# Distribucion descriptiva
# ---------------------------------------------------------------------------

def distribution_stats(returns: pd.Series) -> dict:
    """
    Stats descriptivos de una serie de retornos.
    Devuelve dict con metricas en porcentaje (mas legibles).
    """
    r = returns.dropna()
    if len(r) == 0:
        return {}

    # Stats basicos (en log-return units)
    mean = r.mean()
    median = r.median()
    std = r.std(ddof=1)
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=True)  # excess kurtosis (normal = 0)

    # Test Jarque-Bera de normalidad
    jb_stat, jb_pvalue = stats.jarque_bera(r)

    return {
        "n": int(len(r)),
        "mean_pct": float(mean * 100),
        "median_pct": float(median * 100),
        "std_pct": float(std * 100),
        "skew": float(skew),
        "excess_kurt": float(kurt),
        "min_pct": float(r.min() * 100),
        "max_pct": float(r.max() * 100),
        "jb_stat": float(jb_stat),
        "jb_pvalue": float(jb_pvalue),
    }


def percentile_stats(returns: pd.Series, percentiles: list[float] | None = None) -> dict:
    """Percentiles clave de la distribucion (en %)."""
    if percentiles is None:
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    r = returns.dropna()
    if len(r) == 0:
        return {}
    return {f"p{p}_pct": float(np.percentile(r, p) * 100) for p in percentiles}


def value_at_risk(returns: pd.Series, alphas: list[float] | None = None) -> dict:
    """
    VaR empirico y Expected Shortfall (ES).
    VaR_a = -percentil(a) -> perdida a la que se llega o supera con prob a.
    ES_a = -mean(returns | returns < VaR_a)
    """
    if alphas is None:
        alphas = [0.01, 0.05]
    r = returns.dropna()
    if len(r) == 0:
        return {}
    out = {}
    for a in alphas:
        var_quantile = np.percentile(r, a * 100)  # ej. p5
        var = -var_quantile  # como perdida positiva
        es = -r[r <= var_quantile].mean() if (r <= var_quantile).any() else np.nan
        out[f"VaR_{int(a*100)}pct"] = float(var * 100)
        out[f"ES_{int(a*100)}pct"] = float(es * 100)
    return out


# ---------------------------------------------------------------------------
# Comparacion con normal (BSM assumption)
# ---------------------------------------------------------------------------

def lognormal_comparison(returns: pd.Series) -> dict:
    """
    Compara stats empiricos vs los teoricos bajo distribucion normal con
    misma media y desvio.
    Reporta el VaR_5% empirico y el VaR_5% normal teorico.
    """
    r = returns.dropna()
    if len(r) == 0:
        return {}
    mu = r.mean()
    sigma = r.std(ddof=1)

    # VaR teorico bajo normal
    var_5_emp = -np.percentile(r, 5)
    var_5_theo = -(mu + sigma * stats.norm.ppf(0.05))
    var_1_emp = -np.percentile(r, 1)
    var_1_theo = -(mu + sigma * stats.norm.ppf(0.01))

    # Tail ratio: cuanto mas grande es el extremo izquierdo vs derecho
    tail_ratio = abs(np.percentile(r, 1)) / abs(np.percentile(r, 99))

    return {
        "VaR_5_emp_pct": float(var_5_emp * 100),
        "VaR_5_theo_pct": float(var_5_theo * 100),
        "VaR_5_excess_pct": float((var_5_emp - var_5_theo) * 100),
        "VaR_1_emp_pct": float(var_1_emp * 100),
        "VaR_1_theo_pct": float(var_1_theo * 100),
        "VaR_1_excess_pct": float((var_1_emp - var_1_theo) * 100),
        "tail_ratio_p1_p99": float(tail_ratio),
    }


# ---------------------------------------------------------------------------
# Peores ventanas y drawdown intra-ventana
# ---------------------------------------------------------------------------

def worst_windows(panel: pd.DataFrame, T: int, n: int = 10) -> pd.DataFrame:
    """
    Devuelve las n ventanas T-day con peor retorno fwd.
    """
    r = log_return_forward(panel["Close"], T)
    df = pd.DataFrame({"Date": panel["Date"], "ret_fwd": r}).dropna()
    df["ret_fwd_pct"] = df["ret_fwd"] * 100
    df = df.nsmallest(n, "ret_fwd_pct")
    df["expiry_date"] = df.apply(lambda row: panel.loc[panel["Date"] == row["Date"], :].index[0] + T, axis=1)
    df["expiry_date"] = df["expiry_date"].apply(
        lambda i: panel["Date"].iloc[i] if i < len(panel) else pd.NaT
    )
    df = df[["Date", "expiry_date", "ret_fwd_pct"]]
    df.columns = ["Open Date", "Close Date", "Return % (log)"]
    return df.reset_index(drop=True)


def intra_window_max_drawdown(panel: pd.DataFrame, T: int) -> pd.Series:
    """
    Para cada t, calcula el max drawdown (positivo, fraccional) ocurrido entre
    t y t+T (inclusive de close del dia t y close de t+T).

    drawdown(t) = 1 - min(Close[t..t+T]) / Close[t]

    Anti-look-ahead: este es un calculo POST-FACTUM (con info t+T) usado solo
    para caracterizar el comportamiento historico, NO como feature en t.
    """
    closes = panel["Close"].values
    n = len(closes)
    out = np.full(n, np.nan)
    for t in range(n - T):
        window = closes[t:t + T + 1]
        out[t] = 1.0 - window.min() / window[0]
    return pd.Series(out, index=panel.index)


def intra_window_min_below_pct(panel: pd.DataFrame, T: int, pct_grid: list[float]) -> dict:
    """
    Para cada x in pct_grid: % de ventanas donde el min intra-ventana cayo
    al menos x% (touch probability cruda en %).
    """
    dd = intra_window_max_drawdown(panel, T).dropna()
    return {f"touch_{x:.0%}": float((dd >= x).mean()) for x in pct_grid}


# ---------------------------------------------------------------------------
# Best/worst por anio
# ---------------------------------------------------------------------------

def yearly_extremes(panel: pd.DataFrame, T: int) -> pd.DataFrame:
    """Para cada anio en train, peor ret_fwd T-dias del anio."""
    r = log_return_forward(panel["Close"], T)
    df = pd.DataFrame({"Date": panel["Date"], "ret_fwd": r}).dropna()
    df["year"] = df["Date"].dt.year
    g = df.groupby("year")["ret_fwd"]
    out = pd.DataFrame({
        "year": g.min().index,
        f"worst_ret_pct_T{T}": (g.min() * 100).values,
        f"best_ret_pct_T{T}": (g.max() * 100).values,
        f"frac_neg_5pct_T{T}": (df.assign(neg=df["ret_fwd"] < -0.05).groupby("year")["neg"].mean()).values,
        "n_obs": g.count().values,
    })
    return out.reset_index(drop=True)
