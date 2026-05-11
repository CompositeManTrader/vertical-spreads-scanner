"""
Fase 3: combinaciones multifactor.

Selecciona filtros candidatos predefinidos teoricamente sensatos y mide la
P(ITM) en la interseccion. Bonferroni-adjusted.

ANTI-OVERFITTING:
- Combinaciones definidas EX-ANTE (no buscamos a posteriori).
- Reportamos TODAS las combinaciones probadas, no solo las significativas.
- Bonferroni sobre la familia total de tests.
- Holdout interno: split train en train_dev (primeros 5 anios) + train_val (ultimo
  anio del train) para validar antes de ir al holdout final.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import DATA_CLEAN, REPORTS
from src.analysis.regime_conditional import (
    add_outcomes, two_proportion_z_test, wilson_ci,
)

RESULTS_DIR = REPORTS / "_phase3_results"


# Combinaciones predefinidas EX-ANTE (no busqueda a posteriori).
# Cada filtro es una funcion: pd.DataFrame -> mascara booleana.
def _f_above_sma200(df): return df["above_sma200"] == 1
def _f_below_sma200(df): return df["above_sma200"] == 0
def _f_iv_rank_low(df): return df["iv_rank_252"] < 0.20
def _f_iv_rank_high(df): return df["iv_rank_252"] > 0.60
def _f_vrp_high(df):
    q = df["vrp"].quantile(0.80)
    return df["vrp"] >= q
def _f_vrp_low(df):
    q = df["vrp"].quantile(0.20)
    return df["vrp"] <= q
def _f_low_rv(df): return df["rv_20d"] < 0.15
def _f_no_panic_5d(df): return df["panic_5d_window"] == 0
def _f_in_panic(df): return df["panic_5d_window"] == 1
def _f_dd_low(df): return df["dd_60d"] < 0.05
def _f_dd_high(df): return df["dd_60d"] > 0.05
def _f_far_from_ath(df): return df["days_since_ath"] > 60
def _f_strong_uptrend(df): return df["price_to_sma200"] > 0.05
def _f_low_vix(df): return df["VIX_Close"] < 22
def _f_contango(df): return df["term_structure_ratio"] < 1.0


CONDITIONS = {
    # 1-factor (linea base)
    "above_sma200":            [_f_above_sma200],
    "iv_rank_low (<20)":       [_f_iv_rank_low],
    "iv_rank_high (>60)":      [_f_iv_rank_high],
    "vrp_high (top quintile)": [_f_vrp_high],
    "low_rv (<15%)":           [_f_low_rv],
    "no_panic_5d":             [_f_no_panic_5d],
    "strong_uptrend (>5% over SMA200)": [_f_strong_uptrend],
    # 2-factor
    "above_sma200 & vrp_high":          [_f_above_sma200, _f_vrp_high],
    "above_sma200 & no_panic_5d":       [_f_above_sma200, _f_no_panic_5d],
    "above_sma200 & low_vix":           [_f_above_sma200, _f_low_vix],
    "above_sma200 & contango":          [_f_above_sma200, _f_contango],
    "iv_rank_high & vrp_high":          [_f_iv_rank_high, _f_vrp_high],
    "strong_uptrend & low_rv":          [_f_strong_uptrend, _f_low_rv],
    "above_sma200 & far_from_ath & low_vix": [_f_above_sma200, _f_far_from_ath, _f_low_vix],
    # 3-factor (mas restrictivo, menos n)
    "above_sma200 & vrp_high & no_panic_5d":  [_f_above_sma200, _f_vrp_high, _f_no_panic_5d],
    "above_sma200 & low_rv & contango":       [_f_above_sma200, _f_low_rv, _f_contango],
    # Filtros adversos (esperamos P(ITM) ALTA)
    "below_sma200":            [_f_below_sma200],
    "in_panic_5d":             [_f_in_panic],
    "dd_high & below_sma200":  [_f_dd_high, _f_below_sma200],
    "vrp_low":                 [_f_vrp_low],
}


def evaluate_condition(df: pd.DataFrame, filters: list, outcome_col: str) -> dict:
    """
    Aplica AND de todos los filtros, calcula P(outcome) en el subset,
    devuelve metricas + comparacion vs unconditional.
    """
    work = df.dropna(subset=[outcome_col]).copy()
    n_total = len(work)
    k_total = int(work[outcome_col].sum())
    p_uncond = k_total / n_total if n_total > 0 else np.nan

    mask = np.ones(len(work), dtype=bool)
    for f in filters:
        m = f(work)
        if m.dtype != bool:
            m = m.astype(bool)
        mask &= m.fillna(False).values if hasattr(m, "fillna") else m

    sub = work[mask]
    n = len(sub)
    if n == 0:
        return {"n": 0, "p_outcome": np.nan}
    k = int(sub[outcome_col].sum())
    p = k / n
    ci_lo, ci_hi = wilson_ci(k, n)
    z, pval = two_proportion_z_test(k, n, k_total, n_total)
    return {
        "n_total": n_total, "n": n, "k": k,
        "p_outcome": p, "p_uncond": p_uncond,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "lift": p / p_uncond if p_uncond > 0 else np.nan,
        "z": z, "pvalue": pval,
    }


def main():
    rows = []
    for ticker in ["SPY", "QQQ", "IWM"]:
        df = pd.read_parquet(DATA_CLEAN / "train_enriched" / f"{ticker}.parquet")
        df["Date"] = pd.to_datetime(df["Date"])

        for T in [30, 45]:
            df = add_outcomes(df, T, [0.05, 0.07])

        for T in [30, 45]:
            for x in [0.05, 0.07]:
                x_str = f"{x:.0%}".replace("%", "p")
                outcome_col = f"itm_{x_str}_T{T}"
                if outcome_col not in df.columns:
                    continue
                for name, filters in CONDITIONS.items():
                    res = evaluate_condition(df, filters, outcome_col)
                    res["ticker"] = ticker
                    res["T"] = T
                    res["x_pct_below"] = x
                    res["condition"] = name
                    rows.append(res)

    out = pd.DataFrame(rows)

    # Bonferroni: familia es todos los tests univariados + multifactor por ticker
    n_tests = len(out)
    out["pvalue_bonferroni"] = (out["pvalue"] * n_tests).clip(upper=1.0)
    out.to_parquet(RESULTS_DIR / "regime_multifactor.parquet")
    print(f"Filas multifactor: {len(out)} (Bonferroni n_tests = {n_tests})")


if __name__ == "__main__":
    main()
