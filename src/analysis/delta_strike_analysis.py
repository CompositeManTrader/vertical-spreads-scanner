"""
Fase 4: bridge del % below al strike delta-20.

Para cada dia t con IV disponible y cada T en {30,35,40,45}:
  1. Calcular K = strike delta-20 con BSM (sin skew, IV ATM).
  2. Distancia % below = 1 - K/S.
  3. Outcomes empiricos: was_ITM (S(t+T) < K), was_touched (min < K).

Comparar con teoria:
  - P(ITM) bajo BSM con IV ATM (la 'teoria nominal').
  - |delta nominal| (deberia ser ~0.20 por construccion).

Cuantificar el sesgo por SKEW: ajustar IV con bumps {1.0, 1.05, 1.10, 1.15}
y ver con cual la P(ITM) empirica iguala el delta nominal.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import DATA_CLEAN, DIVIDEND_YIELDS, REPORTS
from src.analysis.regime_conditional import (
    two_proportion_z_test, wilson_ci,
)
from src.pricing.black_scholes import (
    prob_below_riskneutral, put_delta, solve_put_strike_for_delta,
)

TICKERS = ["SPY", "QQQ", "IWM"]
RESULTS_DIR = REPORTS / "_phase4_results"


def annual_T(days: int) -> float:
    """Convertimos dias calendario a fraccion de anio (365)."""
    return days / 365.0


def compute_delta_strikes(panel: pd.DataFrame, ticker: str,
                          T_days: int, target_delta: float,
                          iv_bumps: list[float] | None = None) -> pd.DataFrame:
    """
    Para cada dia t con IV no nulo:
      - Calcular K = solve_put_strike_for_delta(S, T, r, q, sigma_bumped, target).
      - Calcular distancia % below = 1 - K/S.
      - Outcomes was_ITM, was_touched.
      - Para cada IV bump: K_bumped, dist_bumped, was_ITM_bumped.

    Anti-look-ahead:
      - S, sigma, r en t -> usados como input.
      - was_ITM usa Close[t+T] (label, NO feature).
      - El strike calculado en t es la 'decision' que se hubiera tomado en t.
    """
    if iv_bumps is None:
        iv_bumps = [1.00, 1.05, 1.10, 1.15, 1.20]

    q = DIVIDEND_YIELDS[ticker]
    closes = panel["Close"].values
    lows = panel["Low"].values
    iv_atm = panel["iv_atm_barchart"].values
    rfr_pct = panel["rfr_pct"].values  # en %
    n = len(panel)
    T_years = annual_T(T_days)

    rows = []
    for t in range(n - T_days):
        S = closes[t]
        iv = iv_atm[t]
        r = rfr_pct[t] / 100.0 if not np.isnan(rfr_pct[t]) else 0.04
        if np.isnan(iv) or iv <= 0:
            continue

        end_close = closes[t + T_days]
        min_low = lows[t:t + T_days + 1].min()

        for bump in iv_bumps:
            sigma = iv * bump
            try:
                K = solve_put_strike_for_delta(S, T_years, r, q, sigma, target_delta)
            except Exception:
                continue
            dist_pct = 1.0 - K / S
            was_itm = end_close < K
            was_touch = min_low < K
            # P(ITM) teorica bajo BSM con IV bumpeada (medida RN)
            p_itm_theo = prob_below_riskneutral(S, K, T_years, r, q, sigma)

            rows.append({
                "Date": panel["Date"].iloc[t],
                "ticker": ticker,
                "T_days": T_days,
                "iv_bump": bump,
                "iv_atm": iv,
                "iv_used": sigma,
                "r": r,
                "S": S,
                "K_delta20": K,
                "dist_pct_below": dist_pct,
                "delta_check": put_delta(S, K, T_years, r, q, sigma),
                "p_itm_theo_RN": p_itm_theo,
                "S_at_expiry": end_close,
                "min_low_in_window": min_low,
                "was_ITM": int(was_itm),
                "was_touched": int(was_touch),
            })
    return pd.DataFrame(rows)


def aggregate_summary(df: pd.DataFrame, target_delta: float) -> pd.DataFrame:
    """Agrega P(ITM) empirica vs teorica para cada (ticker, T_days, iv_bump)."""
    rows = []
    for (ticker, T, bump), g in df.groupby(["ticker", "T_days", "iv_bump"]):
        n = len(g)
        if n == 0:
            continue
        # Empirico
        p_itm_emp = g["was_ITM"].mean()
        p_touch_emp = g["was_touched"].mean()
        # CI Wilson
        ci_lo, ci_hi = wilson_ci(int(g["was_ITM"].sum()), n)
        # Teorico promedio
        p_itm_theo_mean = g["p_itm_theo_RN"].mean()
        # Distancia mean
        dist_mean = g["dist_pct_below"].mean()
        dist_p10 = np.percentile(g["dist_pct_below"], 10)
        dist_p90 = np.percentile(g["dist_pct_below"], 90)
        # z-test: P(ITM) emp vs |target_delta|
        z, pval = two_proportion_z_test(
            int(g["was_ITM"].sum()), n,
            int(round(abs(target_delta) * n)), n
        )
        rows.append({
            "ticker": ticker, "T_days": T, "iv_bump": bump, "n": n,
            "p_itm_emp": p_itm_emp,
            "p_itm_emp_ci_lo": ci_lo,
            "p_itm_emp_ci_hi": ci_hi,
            "p_itm_theo_RN_mean": p_itm_theo_mean,
            "p_touch_emp": p_touch_emp,
            "ratio_touch_itm_emp": p_touch_emp / p_itm_emp if p_itm_emp > 0 else np.nan,
            "delta_nominal_target": abs(target_delta),
            "dist_pct_mean": dist_mean,
            "dist_pct_p10": dist_p10,
            "dist_pct_p90": dist_p90,
            "z_emp_vs_delta_nominal": z,
            "pvalue_emp_vs_delta_nominal": pval,
        })
    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    target_delta = -0.20
    iv_bumps = [1.00, 1.05, 1.10, 1.15, 1.20]
    T_days_grid = [30, 35, 40, 45]

    all_records = []
    for ticker in TICKERS:
        panel = pd.read_parquet(DATA_CLEAN / "train" / f"{ticker}.parquet")
        panel["Date"] = pd.to_datetime(panel["Date"])
        panel = panel.sort_values("Date").reset_index(drop=True)
        for T in T_days_grid:
            print(f"  {ticker} T={T}d ...")
            d = compute_delta_strikes(panel, ticker, T, target_delta, iv_bumps)
            all_records.append(d)
    full = pd.concat(all_records, ignore_index=True)
    full.to_parquet(RESULTS_DIR / "delta20_full_records.parquet")
    print(f"\nFull records: {len(full):,} filas")

    summary = aggregate_summary(full, target_delta)
    summary.to_parquet(RESULTS_DIR / "delta20_summary.parquet")
    print(f"Summary: {len(summary)} filas")


if __name__ == "__main__":
    main()
