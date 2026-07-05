"""
Comparativa de reglas de gestion: hold-to-expiry vs TP solo vs SL solo vs ambos.
Sin filtros. Por ticker x delta x DTE.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import DATA_CLEAN, TOTAL_COST_PER_TRADE
from src.simulation.pcs_simulator import (
    ManagementRules, TradeConfig, simulate_strategy,
)
from src.simulation.strategy_metrics import perf_metrics

OUT = Path("reports/_phase8_results")
OUT.mkdir(parents=True, exist_ok=True)


def load(ticker):
    df = pd.read_parquet(DATA_CLEAN / "train" / f"{ticker}.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


# 4 reglas a comparar
CONFIGS = {
    "HOLD-TO-EXPIRY":     ManagementRules(None, None, None, False, True),
    "TP50 only":          ManagementRules(0.50, None, None, False, False),
    "SL2x only":          ManagementRules(None, 2.0, None, False, False),
    "TP50 + SL2x":        ManagementRules(0.50, 2.0, None, False, False),
    "TP50 + SL2x + 14DTE":ManagementRules(0.50, 2.0, 14, False, False),
}

DELTAS = [-0.20, -0.30]
DTES = [21, 30, 45]
WIDTH = 5.0

rows = []
for ticker in ["SPY", "QQQ", "IWM"]:
    panel = load(ticker)
    for delta in DELTAS:
        for dte in DTES:
            for cfg_name, rules in CONFIGS.items():
                cfg = TradeConfig(delta, dte, WIDTH)
                trades = simulate_strategy(panel, ticker, cfg, rules)
                if trades.empty:
                    continue
                m = perf_metrics(trades)
                m["ticker"] = ticker
                m["delta"] = abs(delta)
                m["dte"] = dte
                m["rule"] = cfg_name

                # Adicional: razones de salida
                exit_counts = trades["exit_reason"].value_counts(normalize=True).to_dict()
                m["pct_TP"] = round(exit_counts.get("take_profit", 0)*100, 1)
                m["pct_SL"] = round(exit_counts.get("stop_loss", 0)*100, 1)
                m["pct_time"] = round(exit_counts.get("time_stop", 0)*100, 1)
                m["pct_expiry"] = round(exit_counts.get("expiry", 0)*100, 1)
                rows.append(m)

df = pd.DataFrame(rows)
df.to_parquet(OUT / "management_comparison.parquet")
print(f"OK: {len(df)} filas")
