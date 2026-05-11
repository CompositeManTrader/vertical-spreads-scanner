"""Metricas de performance para una serie de trades."""

import numpy as np
import pandas as pd


def equity_curve(trades: pd.DataFrame, pnl_col: str = "pnl_net_after_costs") -> pd.DataFrame:
    """Construye equity curve por fecha de cierre del trade."""
    if trades.empty:
        return pd.DataFrame(columns=["date", "cumulative_pnl"])
    df = trades.sort_values("exit_date").copy()
    df["cumulative_pnl"] = df[pnl_col].cumsum()
    return df[["exit_date", "cumulative_pnl", pnl_col]].rename(columns={"exit_date": "date"})


def perf_metrics(trades: pd.DataFrame, pnl_col: str = "pnl_net_after_costs") -> dict:
    """
    Calcula metricas standard. PnL en USD por contrato.
    Sharpe se calcula sobre los retornos por trade (no anualizado por dia).
    """
    if trades.empty:
        return {"n_trades": 0}
    p = trades[pnl_col].values
    n = len(p)
    wins = p > 0
    losses = p < 0
    avg_win = p[wins].mean() if wins.any() else 0
    avg_loss = -p[losses].mean() if losses.any() else 0
    win_rate = wins.mean()
    expectancy = p.mean()
    std_p = p.std(ddof=1) if n > 1 else 0
    sharpe_per_trade = expectancy / std_p if std_p > 0 else np.nan
    profit_factor = (p[wins].sum() / -p[losses].sum()) if losses.any() and p[losses].sum() < 0 else np.nan

    # Drawdown sobre equity curve
    eq = equity_curve(trades, pnl_col)
    if not eq.empty:
        eq_vals = eq["cumulative_pnl"].values
        peaks = np.maximum.accumulate(eq_vals)
        dd = peaks - eq_vals
        max_dd = dd.max()
        # tiempo bajo agua
        underwater = (dd > 0).mean()
    else:
        max_dd = 0
        underwater = 0

    # Max consecutive losses
    streak = 0
    max_streak = 0
    for x in p:
        if x < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # Sortino: Sharpe usando solo desvio negativo
    neg_returns = p[p < 0]
    downside_std = np.sqrt((neg_returns ** 2).mean()) if len(neg_returns) else 0
    sortino = expectancy / downside_std if downside_std > 0 else np.nan

    # ES: expected shortfall en peor 5%
    p_sorted = np.sort(p)
    es_5 = p_sorted[:max(1, int(n * 0.05))].mean()

    return {
        "n_trades": n,
        "win_rate": float(win_rate),
        "avg_win_per_contract": float(avg_win),
        "avg_loss_per_contract": float(avg_loss),
        "expectancy_per_contract": float(expectancy),
        "total_pnl": float(p.sum()),
        "profit_factor": float(profit_factor) if not np.isnan(profit_factor) else None,
        "sharpe_per_trade": float(sharpe_per_trade),
        "sortino": float(sortino) if not np.isnan(sortino) else None,
        "max_drawdown_usd": float(max_dd),
        "max_consecutive_losses": int(max_streak),
        "pct_time_underwater": float(underwater),
        "ES_5pct": float(es_5),
        "min_pnl": float(p.min()),
        "max_pnl": float(p.max()),
    }
