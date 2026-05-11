"""
Fase 6: simular y comparar configuraciones de estrategia.

Configuraciones a comparar (en train, sin holdout):
  - VANILLA-baseline: delta-30, T=45, w=$5, TP50/SL2x/timestop14, sin filtros.
  - VANILLA-conservative: delta-20, T=30, w=$5, TP50/SL2x.
  - VANILLA-aggressive: delta-40, T=60, w=$5, TP50/SL2x.
  - FILTRADA: misma config baseline + filtro 'above_sma200 & vrp_high'.
  - HOLD-TO-EXPIRY: baseline sin gestion (validacion).
  - PORTFOLIO: SPY + QQQ con FILTRADA, equal weight.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import DATA_CLEAN, REPORTS
from src.simulation.pcs_simulator import (
    ManagementRules, TradeConfig, simulate_strategy,
)
from src.simulation.strategy_metrics import equity_curve, perf_metrics

RESULTS_DIR = REPORTS / "_phase6_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_enriched(ticker: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_CLEAN / "train_enriched" / f"{ticker}.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def filter_above_sma200_and_vrp_high(df: pd.DataFrame) -> pd.Series:
    vrp_top = df["vrp"].quantile(0.80)
    return (df["above_sma200"] == 1) & (df["vrp"] >= vrp_top)


def main():
    panels = {t: load_enriched(t) for t in ["SPY", "QQQ", "IWM"]}

    # Configs a probar
    configs = {
        "VANILLA-baseline":     (TradeConfig(-0.30, 45, 5),  ManagementRules(0.50, 2.0, 14, False, False)),
        "VANILLA-conservative": (TradeConfig(-0.20, 30, 5),  ManagementRules(0.50, 2.0, 14, False, False)),
        "VANILLA-aggressive":   (TradeConfig(-0.40, 60, 5),  ManagementRules(0.50, 2.0, 14, False, False)),
        "FILTRADA":             (TradeConfig(-0.30, 45, 5),  ManagementRules(0.50, 2.0, 14, False, False)),
        "HOLD-TO-EXPIRY":       (TradeConfig(-0.30, 45, 5),  ManagementRules(None, None, None, False, True)),
    }

    all_trades = []
    summary_rows = []
    for cfg_name, (cfg, rules) in configs.items():
        for ticker in ["SPY", "QQQ", "IWM"]:
            mask = filter_above_sma200_and_vrp_high(panels[ticker]) if cfg_name == "FILTRADA" else None
            trades = simulate_strategy(panels[ticker], ticker, cfg, rules, filter_mask=mask)
            if trades.empty:
                continue
            trades["config"] = cfg_name
            all_trades.append(trades)
            metrics = perf_metrics(trades)
            metrics["ticker"] = ticker
            metrics["config"] = cfg_name
            summary_rows.append(metrics)
            print(f"  {cfg_name:24s} {ticker}: n={metrics['n_trades']:4d} winR={metrics['win_rate']:.3f} "
                  f"exp={metrics['expectancy_per_contract']:.2f} sharpe={metrics['sharpe_per_trade']:.3f} "
                  f"maxDD={metrics['max_drawdown_usd']:.0f}")

    all_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    all_trades_df.to_parquet(RESULTS_DIR / "all_trades.parquet")
    pd.DataFrame(summary_rows).to_parquet(RESULTS_DIR / "summary.parquet")

    # Portfolio: combinar SPY + QQQ filtradas
    portfolio_trades = all_trades_df[
        (all_trades_df["config"] == "FILTRADA") &
        (all_trades_df["ticker"].isin(["SPY", "QQQ"]))
    ].sort_values("exit_date").copy()
    if not portfolio_trades.empty:
        portfolio_trades.to_parquet(RESULTS_DIR / "portfolio_trades.parquet")
        port_metrics = perf_metrics(portfolio_trades)
        port_metrics["ticker"] = "PORTFOLIO_SPY_QQQ"
        port_metrics["config"] = "FILTRADA"
        port_df = pd.DataFrame([port_metrics])
        port_df.to_parquet(RESULTS_DIR / "portfolio_summary.parquet")
        print(f"\nPortfolio SPY+QQQ filtrado: n={port_metrics['n_trades']} "
              f"winR={port_metrics['win_rate']:.3f} exp={port_metrics['expectancy_per_contract']:.2f} "
              f"sharpe={port_metrics['sharpe_per_trade']:.3f} maxDD={port_metrics['max_drawdown_usd']:.0f}")

    print(f"\nGuardado en: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
