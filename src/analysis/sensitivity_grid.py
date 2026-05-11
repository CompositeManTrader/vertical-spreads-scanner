"""
Fase 5: sensibilidad a delta, DTE y ancho del spread.

Para cada combinacion (ticker, delta_short, T_days, ancho_$):
  - Calcular K_short = solve_strike_for_delta(delta_short)
  - K_long = K_short - ancho
  - Credito BSM = put_price(K_short) - put_price(K_long)  (sigma = IV ATM)
  - Max loss = (K_short - K_long) - credito  (en USD por contrato sin x100)
  - Outcome empirico:
      pnl(t) = credito - max(0, min(K_short - Close[t+T], K_short - K_long))
  - Metricas: win rate, mean pnl, sharpe (mean/std), credito/max_loss, etc.

Anti-look-ahead: cada calculo en t usa solo info <=t. Outcome usa Close[t+T].
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import (
    DATA_CLEAN, DELTA_GRID, DIVIDEND_YIELDS, REPORTS, WIDTH_GRID,
)
from src.pricing.black_scholes import put_price, solve_put_strike_for_delta

TICKERS = ["SPY", "QQQ", "IWM"]
RESULTS_DIR = REPORTS / "_phase5_results"

DTE_GRID_EXT = [21, 30, 35, 40, 45, 60]


def annual_T(days: int) -> float:
    return days / 365.0


def simulate_pcs_grid(panel: pd.DataFrame, ticker: str,
                       delta_short: float, T_days: int, width: float) -> pd.DataFrame:
    """
    Simula PCS dia por dia. Devuelve DataFrame con outcomes y P&L por trade.
    """
    q = DIVIDEND_YIELDS[ticker]
    closes = panel["Close"].values
    iv_atm = panel["iv_atm_barchart"].values
    rfr_pct = panel["rfr_pct"].values
    n = len(panel)
    T_years = annual_T(T_days)

    rows = []
    for t in range(n - T_days):
        S = closes[t]
        iv = iv_atm[t]
        r = rfr_pct[t] / 100.0 if not np.isnan(rfr_pct[t]) else 0.04
        if np.isnan(iv) or iv <= 0:
            continue
        try:
            K_short = solve_put_strike_for_delta(S, T_years, r, q, iv, delta_short)
        except Exception:
            continue
        K_long = K_short - width
        if K_long <= 0:
            continue
        # Precio BSM de cada leg
        p_short = put_price(S, K_short, T_years, r, q, iv)
        p_long = put_price(S, K_long, T_years, r, q, iv)
        credit = p_short - p_long
        if credit <= 0:
            continue
        max_loss = width - credit  # en $ por share

        # P&L al expiry
        end_close = closes[t + T_days]
        intrinsic_loss = max(0.0, K_short - max(K_long, end_close))
        # equivalente a: si end >= K_short -> 0; si end <= K_long -> K_short - K_long; en medio -> K_short - end
        pnl = credit - intrinsic_loss

        rows.append({
            "Date": panel["Date"].iloc[t],
            "ticker": ticker,
            "delta_short": delta_short,
            "T_days": T_days,
            "width": width,
            "S": S,
            "K_short": K_short,
            "K_long": K_long,
            "credit_per_share": credit,
            "max_loss_per_share": max_loss,
            "credit_to_maxloss_ratio": credit / max_loss if max_loss > 0 else np.nan,
            "S_at_expiry": end_close,
            "intrinsic_loss": intrinsic_loss,
            "pnl_per_share": pnl,
            "is_full_loss": int(end_close <= K_long),
            "is_partial_loss": int((end_close < K_short) and (end_close > K_long)),
            "is_win": int(end_close >= K_short),
        })
    return pd.DataFrame(rows)


def aggregate_grid(records: pd.DataFrame) -> pd.DataFrame:
    """Resumen por (ticker, delta, T, width)."""
    rows = []
    for (ticker, delta, T, width), g in records.groupby(
            ["ticker", "delta_short", "T_days", "width"]):
        n = len(g)
        if n == 0:
            continue
        win_rate = g["is_win"].mean()
        full_loss_rate = g["is_full_loss"].mean()
        partial_loss_rate = g["is_partial_loss"].mean()
        mean_pnl = g["pnl_per_share"].mean()
        std_pnl = g["pnl_per_share"].std(ddof=1)
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else np.nan
        # Bruto (sin costos)
        avg_credit = g["credit_per_share"].mean()
        avg_maxloss = g["max_loss_per_share"].mean()
        avg_ratio = (g["credit_per_share"] / g["max_loss_per_share"]).mean()
        rows.append({
            "ticker": ticker, "delta_short": delta, "T_days": T, "width": width,
            "n_trades": n,
            "win_rate": win_rate,
            "full_loss_rate": full_loss_rate,
            "partial_loss_rate": partial_loss_rate,
            "expectancy_per_share": mean_pnl,
            "expectancy_pct_of_maxloss": mean_pnl / avg_maxloss if avg_maxloss > 0 else np.nan,
            "sharpe": sharpe,
            "avg_credit": avg_credit,
            "avg_maxloss": avg_maxloss,
            "avg_credit_to_maxloss": avg_ratio,
            "min_pnl": g["pnl_per_share"].min(),
            "max_pnl": g["pnl_per_share"].max(),
        })
    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []
    for ticker in TICKERS:
        panel = pd.read_parquet(DATA_CLEAN / "train" / f"{ticker}.parquet")
        panel["Date"] = pd.to_datetime(panel["Date"])
        panel = panel.sort_values("Date").reset_index(drop=True)
        for delta in DELTA_GRID:
            for T in DTE_GRID_EXT:
                for width in WIDTH_GRID:
                    d = simulate_pcs_grid(panel, ticker, -delta, T, width)
                    if not d.empty:
                        all_records.append(d)
        print(f"  {ticker} OK")

    full = pd.concat(all_records, ignore_index=True)
    full.to_parquet(RESULTS_DIR / "pcs_grid_records.parquet")
    print(f"Records totales: {len(full):,}")

    summary = aggregate_grid(full)
    summary.to_parquet(RESULTS_DIR / "pcs_grid_summary.parquet")
    print(f"Summary: {len(summary)} filas (ticker x delta x T x width)")


if __name__ == "__main__":
    main()
