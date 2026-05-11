"""
Fase 2: probabilidades empiricas no-parametricas.

Para cada (ticker, T, x%):
  - P(close < S(t) * (1-x))   en t+T            -> "ITM at expiry"
  - P(min(S, t..t+T) < S(t) * (1-x))            -> "touch durante la vida"
  - Path analysis: en cuantos casos el strike fue tocado pero el cierre OK
  - Gap risk: cuantas veces el precio cayo > x% en 1 dia

ANTI-LOOK-AHEAD: usa solo TRAIN. Las metricas son post-factum y se usan para
caracterizar comportamiento, NO como features en una decision en t.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import DATA_CLEAN, DTE_GRID, PCT_BELOW_GRID, REPORTS

TICKERS = ["SPY", "QQQ", "IWM"]
RESULTS_DIR = REPORTS / "_phase2_results"
CHARTS_DIR = REPORTS / "_phase2_charts"


def load_train(ticker: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_CLEAN / "train" / f"{ticker}.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Wilson confidence interval (mejor que Wald para proporciones extremas)
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Devuelve (lower, upper) del CI Wilson para una proporcion."""
    if n == 0:
        return (np.nan, np.nan)
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


# ---------------------------------------------------------------------------
# Calculo de las probabilidades (vectorizado)
# ---------------------------------------------------------------------------

def grid_probabilities(panel: pd.DataFrame, T: int, pct_grid: list[float]) -> pd.DataFrame:
    """
    Para cada x in pct_grid:
      itm  = mean(  Close[t+T] < Close[t] * (1-x)  )
      touch = mean( min(Low[t..t+T]) < Close[t] * (1-x) )

    Anti-look-ahead: la metrica es post-factum, descriptiva. No se usa como
    feature en t.
    """
    closes = panel["Close"].values
    lows = panel["Low"].values
    n = len(closes)

    # Pre-calcular min(Low) en cada ventana [t..t+T]
    min_in_window = np.full(n, np.nan)
    for t in range(n - T):
        min_in_window[t] = lows[t:t + T + 1].min()
    end_close = np.concatenate([closes[T:], np.full(T, np.nan)])

    s0 = closes
    rows = []
    for x in pct_grid:
        target = s0 * (1.0 - x)
        valid = ~np.isnan(end_close)
        n_valid = int(valid.sum())
        # ITM at expiry
        itm = (end_close < target) & valid
        n_itm = int(itm.sum())
        p_itm = n_itm / n_valid if n_valid > 0 else np.nan
        ci_itm = wilson_ci(n_itm, n_valid)
        # Touch
        touch = (min_in_window < target) & valid
        n_touch = int(touch.sum())
        p_touch = n_touch / n_valid if n_valid > 0 else np.nan
        ci_touch = wilson_ci(n_touch, n_valid)
        # Touch but recovered (no ITM at expiry)
        touch_recover = touch & ~itm
        rows.append({
            "T": T, "x_pct_below": x,
            "n": n_valid,
            "p_itm": p_itm, "p_itm_ci_lo": ci_itm[0], "p_itm_ci_hi": ci_itm[1],
            "p_touch": p_touch, "p_touch_ci_lo": ci_touch[0], "p_touch_ci_hi": ci_touch[1],
            "p_touch_recovered": float(touch_recover.sum() / n_valid) if n_valid > 0 else np.nan,
            "ratio_touch_itm": float(p_touch / p_itm) if p_itm and p_itm > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Path analysis: cuando dentro de la ventana ocurrio el primer touch
# ---------------------------------------------------------------------------

def first_touch_timing(panel: pd.DataFrame, T: int, x: float) -> pd.DataFrame:
    """
    Para cada t donde hubo touch en (t, t+T]:
      first_touch_day = primer dia (1..T) donde Low[t+i] < S(t)*(1-x)
      itm_at_expiry  = Close[t+T] < S(t)*(1-x)

    Devuelve distribucion del first_touch_day y % de touched-recovered.
    """
    closes = panel["Close"].values
    lows = panel["Low"].values
    n = len(closes)

    timings = []
    n_total = 0
    n_touched = 0
    n_touched_recovered = 0
    n_itm = 0

    for t in range(n - T):
        s0 = closes[t]
        target = s0 * (1.0 - x)
        end_close = closes[t + T]
        n_total += 1

        # Primer dia de touch
        first_day = None
        for i in range(1, T + 1):
            if lows[t + i] < target:
                first_day = i
                break

        if first_day is not None:
            n_touched += 1
            timings.append(first_day)
            if end_close < target:
                n_itm += 1
            else:
                n_touched_recovered += 1

    if not timings:
        return pd.DataFrame()

    timings = np.array(timings)
    return pd.DataFrame([{
        "T": T, "x_pct_below": x,
        "n_total_windows": n_total,
        "n_touched": n_touched,
        "n_touched_recovered": n_touched_recovered,
        "n_touched_and_itm": n_itm,
        "pct_touched_recovered_of_touched": float(n_touched_recovered / n_touched),
        "first_touch_mean_day": float(timings.mean()),
        "first_touch_median_day": float(np.median(timings)),
        "first_touch_p25_day": float(np.percentile(timings, 25)),
        "first_touch_p75_day": float(np.percentile(timings, 75)),
        "first_touch_in_first_third_pct": float(np.mean(timings <= T / 3)),
        "first_touch_in_last_third_pct": float(np.mean(timings >= 2 * T / 3)),
    }])


# ---------------------------------------------------------------------------
# Gap risk: cuantos dias el precio cayo > x% en 1 sola sesion
# ---------------------------------------------------------------------------

def daily_gap_risk(panel: pd.DataFrame, pct_grid: list[float]) -> pd.DataFrame:
    """
    Para cada x: % de dias con (Open - Close[t-1])/Close[t-1] < -x  (gap-down overnight)
    y % con (Close - Close[t-1])/Close[t-1] < -x (caida total intra-dia).
    """
    closes = panel["Close"].values
    opens = panel["Open"].values
    n = len(closes)
    gap_open = np.concatenate([[np.nan], (opens[1:] - closes[:-1]) / closes[:-1]])
    daily_ret = np.concatenate([[np.nan], (closes[1:] - closes[:-1]) / closes[:-1]])

    rows = []
    for x in pct_grid:
        rows.append({
            "x_pct": x,
            "n_total": int(np.isfinite(daily_ret).sum()),
            "p_overnight_gap_down": float(np.nanmean(gap_open < -x)),
            "p_intraday_drop": float(np.nanmean(daily_ret < -x)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    panels = {t: load_train(t) for t in TICKERS}

    # 2.1 Grilla de probabilidades P(ITM) y P(touch)
    print("[2.1] Grilla P(ITM) y P(touch)...")
    grid_rows = []
    for ticker in TICKERS:
        for T in DTE_GRID:
            df = grid_probabilities(panels[ticker], T, PCT_BELOW_GRID)
            df["ticker"] = ticker
            grid_rows.append(df)
    grid = pd.concat(grid_rows, ignore_index=True)
    grid.to_parquet(RESULTS_DIR / "grid_probabilities.parquet")
    print(f"  Filas: {len(grid)}")

    # 2.2 Path analysis: timing del primer touch
    print("\n[2.2] Path analysis (timing del primer touch)...")
    path_rows = []
    # Para cada T y un sub-grid de x relevante
    sub_x = [0.03, 0.05, 0.07, 0.10]
    for ticker in TICKERS:
        for T in DTE_GRID:
            for x in sub_x:
                d = first_touch_timing(panels[ticker], T, x)
                if not d.empty:
                    d["ticker"] = ticker
                    path_rows.append(d)
    path = pd.concat(path_rows, ignore_index=True) if path_rows else pd.DataFrame()
    path.to_parquet(RESULTS_DIR / "first_touch_timing.parquet")
    print(f"  Filas: {len(path)}")

    # 2.3 Gap risk
    print("\n[2.3] Gap risk diario...")
    gap_rows = []
    for ticker in TICKERS:
        d = daily_gap_risk(panels[ticker], [0.01, 0.02, 0.03, 0.05, 0.07])
        d["ticker"] = ticker
        gap_rows.append(d)
    gaps = pd.concat(gap_rows, ignore_index=True)
    gaps.to_parquet(RESULTS_DIR / "gap_risk.parquet")

    print("\nFase 2 outputs en:", RESULTS_DIR)


if __name__ == "__main__":
    main()
