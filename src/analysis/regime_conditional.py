"""
Fase 3: probabilidades condicionales por regimen.

Para cada feature de regimen, segmenta el train en buckets y reporta:
- P(ITM) y P(touch) condicionales
- CI Wilson 95%
- z-test de proporciones vs unconditional
- p-value y p-value Bonferroni-adjusted

ANTI-LOOK-AHEAD: las features ya estan calculadas point-in-time. Solo se hace
agregacion descriptiva sobre train.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import DATA_CLEAN, REPORTS

TICKERS = ["SPY", "QQQ", "IWM"]
RESULTS_DIR = REPORTS / "_phase3_results"
CHARTS_DIR = REPORTS / "_phase3_charts"


# ---------------------------------------------------------------------------
# Outcomes: ITM y touch para strikes 5% y 7% below
# ---------------------------------------------------------------------------

def add_outcomes(df: pd.DataFrame, T: int, x_pcts: list[float]) -> pd.DataFrame:
    """Anade columnas was_itm_x.x_T y was_touch_x.x_T por cada x."""
    df = df.copy()
    closes = df["Close"].values
    lows = df["Low"].values
    n = len(closes)

    end_close = np.concatenate([closes[T:], np.full(T, np.nan)])
    min_in_window = np.full(n, np.nan)
    for t in range(n - T):
        min_in_window[t] = lows[t:t + T + 1].min()

    for x in x_pcts:
        target = closes * (1.0 - x)
        x_str = f"{x:.0%}".replace("%", "p")
        df[f"itm_{x_str}_T{T}"] = (end_close < target).astype(float)
        df.loc[np.isnan(end_close), f"itm_{x_str}_T{T}"] = np.nan
        df[f"touch_{x_str}_T{T}"] = (min_in_window < target).astype(float)
        df.loc[np.isnan(end_close), f"touch_{x_str}_T{T}"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Wilson CI y z-test
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


def two_proportion_z_test(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Test bilateral H0: p1 == p2. Devuelve (z, p_value)."""
    if n1 == 0 or n2 == 0:
        return (np.nan, np.nan)
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (np.nan, np.nan)
    z = (p1 - p2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def bucket_by_quantile(s: pd.Series, n_buckets: int = 5,
                       labels: list[str] | None = None) -> pd.Series:
    """Quintiles globales. CAVEAT look-ahead: usa stats globales del train.
    Para uso descriptivo en train OK; para reglas de la estrategia, usar
    buckets absolutos.
    """
    if labels is None:
        labels = [f"Q{i+1}" for i in range(n_buckets)]
    return pd.qcut(s, n_buckets, labels=labels, duplicates="drop")


def bucket_by_thresholds(s: pd.Series, thresholds: list[float],
                         labels: list[str]) -> pd.Series:
    """Buckets absolutos definidos por thresholds. NO usa stats del train."""
    bins = [-np.inf] + thresholds + [np.inf]
    return pd.cut(s, bins=bins, labels=labels)


# ---------------------------------------------------------------------------
# Calculo principal: P(ITM) y P(touch) por bucket
# ---------------------------------------------------------------------------

def conditional_probs(df: pd.DataFrame, regime_col: str, outcome_col: str,
                      bucket_method: str = "quantile",
                      n_buckets: int = 5,
                      thresholds: list[float] | None = None,
                      labels: list[str] | None = None) -> pd.DataFrame:
    """
    Devuelve: para cada bucket de regime_col, la probabilidad empirica del
    outcome_col, su CI Wilson, y p-value vs unconditional.
    """
    work = df[[regime_col, outcome_col]].dropna().copy()
    n_total = len(work)
    if n_total == 0:
        return pd.DataFrame()
    k_total = int(work[outcome_col].sum())
    p_uncond = k_total / n_total

    if bucket_method == "quantile":
        work["bucket"] = bucket_by_quantile(work[regime_col], n_buckets, labels)
    elif bucket_method == "thresholds":
        work["bucket"] = bucket_by_thresholds(work[regime_col], thresholds, labels)
    elif bucket_method == "binary":
        # Asume regime_col ya es 0/1
        work["bucket"] = work[regime_col].map({0: "off", 1: "on"})
    else:
        raise ValueError(f"bucket_method desconocido: {bucket_method}")

    rows = []
    for bucket, sub in work.groupby("bucket", observed=True):
        n = len(sub)
        k = int(sub[outcome_col].sum())
        p = k / n if n > 0 else np.nan
        ci_lo, ci_hi = wilson_ci(k, n)
        # z-test vs unconditional
        z, pval = two_proportion_z_test(k, n, k_total, n_total)
        # Lift
        lift = p / p_uncond if p_uncond > 0 else np.nan
        rows.append({
            "regime": regime_col,
            "bucket": str(bucket),
            "n": n,
            "p_outcome": p,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_uncond": p_uncond,
            "lift": lift,
            "z": z,
            "pvalue": pval,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main: corre todo el analisis univariado
# ---------------------------------------------------------------------------

# Definicion de regimenes a testear:
# (col_name, method, params, labels)
REGIMES = [
    # IV-related
    ("iv_rank_252", "thresholds", [0.20, 0.40, 0.60, 0.80], ["VeryLow", "Low", "Mid", "High", "VeryHigh"]),
    ("iv_percentile_252", "thresholds", [0.20, 0.40, 0.60, 0.80], ["VeryLow", "Low", "Mid", "High", "VeryHigh"]),
    ("iv_atm_barchart", "thresholds", [0.10, 0.15, 0.20, 0.30], ["<10", "10-15", "15-20", "20-30", ">30"]),

    # VIX-related
    ("VIX_Close", "thresholds", [13, 17, 22, 30], ["<13", "13-17", "17-22", "22-30", ">30"]),
    ("vix_change_5d", "quantile", None, ["Q1", "Q2", "Q3", "Q4", "Q5"]),
    ("vix_change_20d", "quantile", None, ["Q1", "Q2", "Q3", "Q4", "Q5"]),
    ("term_structure_ratio", "thresholds", [0.85, 0.95, 1.00, 1.05], ["DeepContango", "Contango", "Flat", "Mild_Bw", "Strong_Bw"]),
    ("VVIX_Close", "thresholds", [85, 95, 110], ["<85", "85-95", "95-110", ">110"]),

    # VRP and RV
    ("vrp", "quantile", None, ["Q1", "Q2", "Q3", "Q4", "Q5"]),
    ("rv_20d", "thresholds", [0.10, 0.15, 0.20, 0.30], ["<10", "10-15", "15-20", "20-30", ">30"]),
    ("vol_of_vol_60d", "quantile", None, ["Q1", "Q2", "Q3", "Q4", "Q5"]),

    # Trend
    ("price_to_sma50", "thresholds", [-0.05, -0.01, 0.01, 0.05], ["<-5%", "-5/-1%", "-1/1%", "1/5%", ">5%"]),
    ("price_to_sma200", "thresholds", [-0.10, -0.02, 0.02, 0.10], ["<-10%", "-10/-2%", "-2/2%", "2/10%", ">10%"]),
    ("sma200_slope_60d", "thresholds", [-0.02, 0.0, 0.02], ["Down", "Flat-", "Flat+", "Up"]),
    ("above_sma50", "binary", None, None),
    ("above_sma200", "binary", None, None),

    # Drawdown
    ("dd_60d", "thresholds", [0.02, 0.05, 0.10], ["<2%", "2-5%", "5-10%", ">10%"]),
    ("dd_252d", "thresholds", [0.05, 0.10, 0.20], ["<5%", "5-10%", "10-20%", ">20%"]),
    ("days_since_ath", "thresholds", [10, 30, 90, 180], ["<10", "10-30", "30-90", "90-180", ">180"]),

    # P/C ratios
    ("pc_volume_ratio", "quantile", None, ["Q1", "Q2", "Q3", "Q4", "Q5"]),
    ("pc_oi_ratio", "quantile", None, ["Q1", "Q2", "Q3", "Q4", "Q5"]),

    # Calendar
    ("is_FOMC", "binary", None, None),
    ("is_FOMC_t1", "binary", None, None),
    ("is_CPI", "binary", None, None),
    ("is_CPI_t1", "binary", None, None),
    ("is_NFP", "binary", None, None),
    ("is_earnings_season", "binary", None, None),
    ("dow", "thresholds", [0.5, 1.5, 2.5, 3.5], ["Mon", "Tue", "Wed", "Thu", "Fri"]),
    ("month", "thresholds", [3.5, 6.5, 9.5], ["Q1", "Q2", "Q3", "Q4"]),
    ("panic_5d_window", "binary", None, None),
]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for ticker in TICKERS:
        print(f"\n--- {ticker} ---")
        df = pd.read_parquet(DATA_CLEAN / "train_enriched" / f"{ticker}.parquet")
        df["Date"] = pd.to_datetime(df["Date"])

        # Anadir outcomes para T=30 y T=45 con x=5% y 7%
        for T in [30, 45]:
            df = add_outcomes(df, T, [0.05, 0.07])

        # Para cada outcome y cada regimen
        for T in [30, 45]:
            for x in [0.05, 0.07]:
                x_str = f"{x:.0%}".replace("%", "p")
                outcome = f"itm_{x_str}_T{T}"
                if outcome not in df.columns:
                    continue
                for col, method, params, labels in REGIMES:
                    if col not in df.columns:
                        continue
                    try:
                        cp = conditional_probs(df, col, outcome,
                                               bucket_method=method,
                                               thresholds=params,
                                               labels=labels)
                        cp["ticker"] = ticker
                        cp["T"] = T
                        cp["x_pct_below"] = x
                        cp["outcome"] = "itm"
                        rows.append(cp)
                    except Exception as e:
                        print(f"  WARN regime={col} outcome={outcome}: {e}")

                # Tambien P(touch)
                outcome_t = f"touch_{x_str}_T{T}"
                if outcome_t not in df.columns:
                    continue
                for col, method, params, labels in REGIMES:
                    if col not in df.columns:
                        continue
                    try:
                        cp = conditional_probs(df, col, outcome_t,
                                               bucket_method=method,
                                               thresholds=params,
                                               labels=labels)
                        cp["ticker"] = ticker
                        cp["T"] = T
                        cp["x_pct_below"] = x
                        cp["outcome"] = "touch"
                        rows.append(cp)
                    except Exception as e:
                        print(f"  WARN regime={col} outcome={outcome_t}: {e}")
        print(f"  outcomes calculados.")

    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(RESULTS_DIR / "regime_univariate.parquet")
    print(f"\nTotal filas regimen-univariado: {len(out)}")
    print(f"Guardado: {RESULTS_DIR}/regime_univariate.parquet")


if __name__ == "__main__":
    main()
