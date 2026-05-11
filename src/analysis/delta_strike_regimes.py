"""
Fase 4: re-analisis condicional con strike DINAMICO delta-20.

Mismas conditions multifactor que Fase 3, pero con strike delta-20 real
(no 5% below fijo). Verifica que los filtros encontrados se mantienen
cuando el strike se ajusta con la IV.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import DATA_CLEAN, REPORTS
from src.analysis.regime_conditional import (
    two_proportion_z_test, wilson_ci,
)
from src.analysis.regime_multifactor import CONDITIONS

RESULTS_DIR = REPORTS / "_phase4_results"


def main():
    rows = []
    # Cargar records delta-20 (full): tiene Date, ticker, T_days, iv_bump, was_ITM
    full = pd.read_parquet(RESULTS_DIR / "delta20_full_records.parquet")
    full["Date"] = pd.to_datetime(full["Date"])

    # Solo iv_bump=1.0 (sin skew adj) para la comparacion directa con Fase 3.
    full = full[full["iv_bump"] == 1.0].copy()

    for ticker in ["SPY", "QQQ", "IWM"]:
        enriched = pd.read_parquet(DATA_CLEAN / "train_enriched" / f"{ticker}.parquet")
        enriched["Date"] = pd.to_datetime(enriched["Date"])

        sub_full = full[full["ticker"] == ticker]
        for T_days in [30, 45]:
            d = sub_full[sub_full["T_days"] == T_days][["Date", "was_ITM"]].copy()
            merged = enriched.merge(d, on="Date", how="inner")
            n_total = len(merged.dropna(subset=["was_ITM"]))
            k_total = int(merged["was_ITM"].sum())
            p_uncond = k_total / n_total if n_total > 0 else 0
            for name, filters in CONDITIONS.items():
                work = merged.dropna(subset=["was_ITM"]).copy()
                import numpy as np
                mask = np.ones(len(work), dtype=bool)
                for f in filters:
                    m = f(work)
                    if hasattr(m, "fillna"):
                        m = m.fillna(False)
                    mask &= m.values if hasattr(m, "values") else m
                sub = work[mask]
                n = len(sub)
                k = int(sub["was_ITM"].sum())
                if n == 0:
                    continue
                p = k / n
                ci_lo, ci_hi = wilson_ci(k, n)
                z, pval = two_proportion_z_test(k, n, k_total, n_total)
                rows.append({
                    "ticker": ticker, "T_days": T_days,
                    "condition": name, "n_total": n_total, "n": n,
                    "k": k, "p_itm": p, "p_uncond": p_uncond,
                    "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "lift": p / p_uncond if p_uncond > 0 else None,
                    "pvalue": pval,
                })
    out = pd.DataFrame(rows)
    out["pvalue_bonferroni"] = (out["pvalue"] * len(out)).clip(upper=1.0)
    out.to_parquet(RESULTS_DIR / "delta20_regimes.parquet")
    print(f"Guardado: {len(out)} filas. Bonferroni n_tests={len(out)}.")


if __name__ == "__main__":
    main()
