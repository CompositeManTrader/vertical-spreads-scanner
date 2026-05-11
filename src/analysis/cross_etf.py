"""
Fase 1: correlacion cross-ETF.

- Matriz de correlacion de log returns diarios y T-day fwd.
- Correlacion condicional en stress (SPY ret < -2%).
"""

import numpy as np
import pandas as pd

from src.analysis.indicators import log_return_back, log_return_forward


def correlation_matrix(panels: dict[str, pd.DataFrame], horizon: int = 1) -> pd.DataFrame:
    """
    Matriz de correlacion (Pearson) de log returns.
    horizon=1 -> daily; horizon=30 -> ret a 30 dias (overlapping).
    """
    rets = {}
    for ticker, panel in panels.items():
        if horizon == 1:
            rets[ticker] = log_return_back(panel["Close"], 1)
        else:
            rets[ticker] = log_return_forward(panel["Close"], horizon)
        rets[ticker].index = panel["Date"]

    df = pd.DataFrame(rets).dropna()
    return df.corr()


def stress_correlation(panels: dict[str, pd.DataFrame],
                       trigger_ticker: str = "SPY",
                       trigger_threshold: float = -0.02) -> dict:
    """
    Correlacion entre los ETFs en dias donde el trigger_ticker tuvo ret_back_1d
    <= trigger_threshold (default: SPY < -2%).

    Devuelve:
      - corr matrix unconditional
      - corr matrix en stress
      - n dias en stress
    """
    rets = {}
    for ticker, panel in panels.items():
        rets[ticker] = log_return_back(panel["Close"], 1)
        rets[ticker].index = panel["Date"]
    df = pd.DataFrame(rets).dropna()

    stress_mask = df[trigger_ticker] <= trigger_threshold
    n_stress = int(stress_mask.sum())

    return {
        "n_total": int(len(df)),
        "n_stress": n_stress,
        "trigger": f"{trigger_ticker} <= {trigger_threshold:.0%}",
        "corr_unconditional": df.corr(),
        "corr_stress": df[stress_mask].corr() if n_stress > 5 else None,
    }
