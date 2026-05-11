"""
Orquestador Fase 1: corre todos los analisis y guarda resultados a parquet
para que el reporte Word los consuma.

ANTI-LOOK-AHEAD: usa SOLO data/clean/train/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import DATA_CLEAN, DTE_GRID, REPORTS
from src.analysis.cross_etf import correlation_matrix, stress_correlation
from src.analysis.dependence import (
    autocorrelation_table, ljung_box, momentum_reversion_test,
)
from src.analysis.indicators import log_return_back, log_return_forward
from src.analysis.returns_analysis import (
    distribution_stats, intra_window_max_drawdown,
    lognormal_comparison, percentile_stats, value_at_risk,
    worst_windows, yearly_extremes,
)
from src.reporting.charts import (
    autocorrelation_plot, correlation_heatmap,
    histogram_with_normal, qq_plot, yearly_extremes_bar,
)


TICKERS = ["SPY", "QQQ", "IWM"]
RESULTS_DIR = REPORTS / "_phase1_results"
CHARTS_DIR = REPORTS / "_phase1_charts"


def load_train(ticker: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_CLEAN / "train" / f"{ticker}.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.reset_index(drop=True)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    panels = {t: load_train(t) for t in TICKERS}
    print(f"Cargados {len(panels)} paneles train.")
    for t, p in panels.items():
        print(f"  {t}: {len(p):,} dias  ({p['Date'].min().date()} -> {p['Date'].max().date()})")

    # ---------------- Distribucion de retornos ---------------------------
    print("\n[1.1] Distribucion empirica de retornos T-dias...")
    rows = []
    for t in TICKERS:
        for T in DTE_GRID:
            r = log_return_forward(panels[t]["Close"], T)
            d = {"ticker": t, "T": T}
            d.update(distribution_stats(r))
            d.update(percentile_stats(r))
            d.update(value_at_risk(r))
            d.update(lognormal_comparison(r))
            rows.append(d)
    dist_df = pd.DataFrame(rows)
    dist_df.to_parquet(RESULTS_DIR / "distribution_stats.parquet")
    print(f"  Guardado: {len(dist_df)} filas (ticker x T)")

    # ---------------- Charts: histograma + QQ ----------------------------
    print("\n[1.2] Generando histogramas y QQ-plots...")
    for t in TICKERS:
        for T in DTE_GRID:
            r = log_return_forward(panels[t]["Close"], T)
            histogram_with_normal(
                r, f"{t} - Distribucion log-return {T}d",
                CHARTS_DIR / f"hist_{t}_T{T}.png"
            )
            qq_plot(r, f"{t} - QQ vs Normal ({T}d)",
                    CHARTS_DIR / f"qq_{t}_T{T}.png")

    # ---------------- Peores ventanas ------------------------------------
    print("\n[1.3] Peores ventanas historicas...")
    worst_rows = []
    for t in TICKERS:
        for T in DTE_GRID:
            ww = worst_windows(panels[t], T, n=10)
            ww["ticker"] = t
            ww["T"] = T
            worst_rows.append(ww)
    pd.concat(worst_rows, ignore_index=True).to_parquet(
        RESULTS_DIR / "worst_windows.parquet")
    print(f"  Guardadas peores 10 ventanas para {len(TICKERS) * len(DTE_GRID)} combinaciones.")

    # ---------------- Drawdown intra-ventana -----------------------------
    print("\n[1.4] Drawdown intra-ventana...")
    rows = []
    for t in TICKERS:
        for T in DTE_GRID:
            dd = intra_window_max_drawdown(panels[t], T).dropna()
            rows.append({
                "ticker": t, "T": T, "n": int(len(dd)),
                "mean_dd_pct": float(dd.mean() * 100),
                "median_dd_pct": float(dd.median() * 100),
                "p75_dd_pct": float(np.percentile(dd, 75) * 100),
                "p90_dd_pct": float(np.percentile(dd, 90) * 100),
                "p95_dd_pct": float(np.percentile(dd, 95) * 100),
                "p99_dd_pct": float(np.percentile(dd, 99) * 100),
                "max_dd_pct": float(dd.max() * 100),
            })
    pd.DataFrame(rows).to_parquet(RESULTS_DIR / "intra_drawdown.parquet")

    # ---------------- Yearly extremes ------------------------------------
    print("\n[1.5] Mejor / peor por anio...")
    for t in TICKERS:
        # Solo T=30 para grafico
        ye = yearly_extremes(panels[t], 30)
        ye.to_parquet(RESULTS_DIR / f"yearly_extremes_{t}.parquet")
        yearly_extremes_bar(ye, t, 30, CHARTS_DIR / f"yearly_{t}_T30.png")

    # ---------------- Dependencia temporal -------------------------------
    print("\n[1.6] Autocorrelacion y Ljung-Box...")
    autocorr_collected = {}
    lb_collected = {}
    for t in TICKERS:
        rets = log_return_back(panels[t]["Close"], 1)
        autocorr_collected[t] = autocorrelation_table(rets)
        lb_collected[t] = ljung_box(rets)
        # Chart
        autocorrelation_plot(rets, t, CHARTS_DIR / f"acf_{t}.png", max_lag=60)
    autocorr_rows = []
    for t, df in autocorr_collected.items():
        df = df.copy(); df["ticker"] = t
        autocorr_rows.append(df)
    pd.concat(autocorr_rows, ignore_index=True).to_parquet(
        RESULTS_DIR / "autocorrelation.parquet")
    lb_rows = []
    for t, df in lb_collected.items():
        df = df.copy(); df["ticker"] = t
        lb_rows.append(df)
    pd.concat(lb_rows, ignore_index=True).to_parquet(
        RESULTS_DIR / "ljung_box.parquet")

    # ---------------- Momentum / reversion -------------------------------
    print("\n[1.7] Momentum / reversion tests...")
    rows = []
    for t in TICKERS:
        for lookback in [5, 20, 60]:
            for fwd in [5, 20, 30]:
                r = momentum_reversion_test(panels[t], lookback, fwd)
                r["ticker"] = t
                rows.append(r)
    mr_df = pd.DataFrame(rows)
    mr_df.to_parquet(RESULTS_DIR / "momentum_reversion.parquet")

    # ---------------- Cross-ETF correlation ------------------------------
    print("\n[1.8] Correlacion cross-ETF...")
    corr_daily = correlation_matrix(panels, horizon=1)
    corr_30d = correlation_matrix(panels, horizon=30)
    corr_daily.to_parquet(RESULTS_DIR / "corr_daily.parquet")
    corr_30d.to_parquet(RESULTS_DIR / "corr_30d.parquet")

    correlation_heatmap(corr_daily, "Correlacion daily log returns",
                        CHARTS_DIR / "corr_daily.png")
    correlation_heatmap(corr_30d, "Correlacion 30d fwd log returns",
                        CHARTS_DIR / "corr_30d.png")

    stress = stress_correlation(panels, "SPY", -0.02)
    if stress["corr_stress"] is not None:
        stress["corr_stress"].to_parquet(RESULTS_DIR / "corr_stress.parquet")
        correlation_heatmap(stress["corr_stress"],
                            f"Correlacion en stress (SPY <= -2%, n={stress['n_stress']})",
                            CHARTS_DIR / "corr_stress.png")
    pd.DataFrame([{"n_total": stress["n_total"],
                   "n_stress": stress["n_stress"],
                   "trigger": stress["trigger"]}]).to_parquet(
        RESULTS_DIR / "stress_meta.parquet")

    print("\nFase 1: analisis completos. Output en", RESULTS_DIR)


if __name__ == "__main__":
    main()
