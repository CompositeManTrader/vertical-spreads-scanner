"""
Fase 7: robustez y validacion en holdout.

7.1 Block bootstrap del Sharpe (block=30 dias).
7.2 Monte Carlo de orden de trades para distribucion de max DD.
7.3 Sensibilidad a costos (+0%, +50%, +100%).
7.4 Sensibilidad a skew bump (1.00, 1.10, 1.20).
7.5 Stress test en sub-periodos (COVID, 2022).
7.6 Walk-forward: optimizar threshold del VRP en train_dev (primeros 5 anios),
    validar en train_val (ultimo anio del train).
7.7 VALIDACION FINAL EN HOLDOUT (2024-10-04 a 2026-03-12).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import (
    DATA_CLEAN, REPORTS, TOTAL_COST_PER_TRADE, TEST_START, TRAIN_END,
)
from src.analysis.regime_features import build_regime_features
from src.simulation.pcs_simulator import (
    ManagementRules, TradeConfig, simulate_strategy,
)
from src.simulation.strategy_metrics import perf_metrics

RESULTS_DIR = REPORTS / "_phase7_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_enriched_train(ticker: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_CLEAN / "train_enriched" / f"{ticker}.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def build_holdout(ticker: str) -> pd.DataFrame:
    """
    Construye el panel enriquecido para holdout. CRITICO: para que las
    features sean point-in-time HONESTAS, necesitan usar los datos previos
    (incluyendo train). Cargamos panel completo, construimos features, y
    luego filtramos por fechas de holdout.
    """
    panel = pd.read_parquet(DATA_CLEAN / f"{ticker}.parquet")
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel = panel.sort_values("Date").reset_index(drop=True)
    enriched = build_regime_features(panel)
    enriched = enriched[enriched["Date"] >= pd.Timestamp(TEST_START)].reset_index(drop=True)
    return enriched


def filter_above_sma200_and_vrp_high_pit(df: pd.DataFrame,
                                          expanding: bool = True) -> pd.Series:
    """
    Filtro 'above_sma200 & vrp_high'.
    - Si expanding=True: vrp_high se define como vrp >= quintile-80 calculado
      con expanding window. Mas honesto para reglas operativas.
    - Si expanding=False: usa quantile global (Fase 6).
    """
    if expanding:
        # rolling quantile no es eficiente; usamos rolling de 252 dias para vrp threshold
        vrp_thr = df["vrp"].expanding(min_periods=126).quantile(0.80)
    else:
        thr = df["vrp"].quantile(0.80)
        vrp_thr = pd.Series([thr] * len(df), index=df.index)
    return (df["above_sma200"] == 1) & (df["vrp"] >= vrp_thr)


# ---------------------------------------------------------------------------
# 7.1 Block bootstrap
# ---------------------------------------------------------------------------

def block_bootstrap_sharpe(pnl: np.ndarray, block_size: int = 30,
                            n_samples: int = 5000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(pnl)
    if n < block_size:
        return {}
    n_blocks = n // block_size
    sharpes = np.empty(n_samples)
    for i in range(n_samples):
        # bloques con reposicion
        starts = rng.integers(0, n - block_size + 1, n_blocks)
        sample = np.concatenate([pnl[s:s + block_size] for s in starts])
        mean = sample.mean()
        std = sample.std(ddof=1)
        sharpes[i] = mean / std if std > 0 else 0
    return {
        "sharpe_point": float(pnl.mean() / pnl.std(ddof=1)),
        "sharpe_boot_mean": float(sharpes.mean()),
        "sharpe_ci_lo_95": float(np.percentile(sharpes, 2.5)),
        "sharpe_ci_hi_95": float(np.percentile(sharpes, 97.5)),
    }


# ---------------------------------------------------------------------------
# 7.2 Monte Carlo de orden de trades (max DD distribution)
# ---------------------------------------------------------------------------

def mc_max_dd(pnl: np.ndarray, n_samples: int = 5000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(pnl)
    dds = np.empty(n_samples)
    for i in range(n_samples):
        perm = rng.permutation(pnl)
        eq = np.cumsum(perm)
        peaks = np.maximum.accumulate(eq)
        dd = (peaks - eq).max()
        dds[i] = dd
    eq_orig = np.cumsum(pnl)
    peaks_orig = np.maximum.accumulate(eq_orig)
    dd_orig = (peaks_orig - eq_orig).max()
    return {
        "max_dd_observed": float(dd_orig),
        "max_dd_mc_median": float(np.median(dds)),
        "max_dd_mc_p95": float(np.percentile(dds, 95)),
        "max_dd_mc_p99": float(np.percentile(dds, 99)),
        "pct_below_observed": float(np.mean(dds < dd_orig)),
    }


# ---------------------------------------------------------------------------
# 7.3 Sensibilidad a costos
# ---------------------------------------------------------------------------

def sensitivity_costs(trades: pd.DataFrame, cost_multipliers: list[float]) -> pd.DataFrame:
    rows = []
    base_cost = TOTAL_COST_PER_TRADE
    for mult in cost_multipliers:
        cost = base_cost * mult
        adj = trades["pnl_per_contract"] - cost
        m = perf_metrics(pd.DataFrame({"pnl_net_after_costs": adj,
                                        "exit_date": trades["exit_date"]}))
        m["cost_multiplier"] = mult
        m["cost_usd"] = cost
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7.4 Sensibilidad a skew bump (re-simular con IV bumpeada)
# ---------------------------------------------------------------------------

def sensitivity_skew(ticker: str, cfg: TradeConfig, rules: ManagementRules,
                      iv_bumps: list[float], use_filter: bool) -> pd.DataFrame:
    panel = load_enriched_train(ticker)
    rows = []
    for bump in iv_bumps:
        # Crear panel con IV bumpeada (proxy de skew)
        p = panel.copy()
        p["iv_atm_barchart"] = p["iv_atm_barchart"] * bump
        mask = filter_above_sma200_and_vrp_high_pit(p, expanding=False) if use_filter else None
        trades = simulate_strategy(p, ticker, cfg, rules, filter_mask=mask)
        if trades.empty:
            continue
        m = perf_metrics(trades)
        m["ticker"] = ticker
        m["iv_bump"] = bump
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7.5 Stress test: sub-periodos especificos
# ---------------------------------------------------------------------------

STRESS_PERIODS = {
    "COVID Q1-Q2 2020":      ("2020-02-15", "2020-06-30"),
    "Aug-Oct 2022 (rate hikes)": ("2022-08-15", "2022-10-31"),
    "Aug-Sep 2024 (yen carry unwind)": ("2024-08-01", "2024-09-15"),
}


def stress_test(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (start, end) in STRESS_PERIODS.items():
        mask = (trades["open_date"] >= pd.Timestamp(start)) & \
               (trades["open_date"] <= pd.Timestamp(end))
        sub = trades[mask]
        if sub.empty:
            rows.append({"period": name, "n": 0})
            continue
        m = perf_metrics(sub)
        m["period"] = name
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7.7 Holdout validation
# ---------------------------------------------------------------------------

def run_holdout(ticker: str, cfg: TradeConfig, rules: ManagementRules,
                use_filter: bool) -> pd.DataFrame:
    panel = build_holdout(ticker)
    mask = filter_above_sma200_and_vrp_high_pit(panel, expanding=True) if use_filter else None
    trades = simulate_strategy(panel, ticker, cfg, rules, filter_mask=mask)
    return trades


def main():
    # Cargar trades de Fase 6 para reusar
    all_trades = pd.read_parquet(REPORTS / "_phase6_results" / "all_trades.parquet")
    for c in ["ticker", "config"]:
        all_trades[c] = all_trades[c].astype(str)

    cfg_base = TradeConfig(-0.30, 45, 5.0)
    rules_base = ManagementRules(0.50, 2.0, 14, False, False)

    # =============== 7.1 Block bootstrap ===============
    print("\n[7.1] Block bootstrap del Sharpe...")
    rows = []
    for ticker in ["SPY", "QQQ"]:
        for config in ["VANILLA-baseline", "FILTRADA"]:
            t = all_trades[(all_trades["ticker"] == ticker) &
                           (all_trades["config"] == config)]
            if len(t) < 60:
                continue
            res = block_bootstrap_sharpe(t["pnl_net_after_costs"].values, block_size=30)
            res["ticker"] = ticker
            res["config"] = config
            res["n_trades"] = len(t)
            rows.append(res)
    bb = pd.DataFrame(rows)
    bb.to_parquet(RESULTS_DIR / "block_bootstrap.parquet")
    print(bb.to_string(index=False))

    # =============== 7.2 Monte Carlo max DD ===============
    print("\n[7.2] Monte Carlo max DD...")
    rows = []
    for ticker in ["SPY", "QQQ"]:
        for config in ["VANILLA-baseline", "FILTRADA"]:
            t = all_trades[(all_trades["ticker"] == ticker) &
                           (all_trades["config"] == config)]
            if len(t) < 30:
                continue
            res = mc_max_dd(t["pnl_net_after_costs"].values)
            res["ticker"] = ticker
            res["config"] = config
            rows.append(res)
    mc = pd.DataFrame(rows)
    mc.to_parquet(RESULTS_DIR / "monte_carlo_dd.parquet")
    print(mc.to_string(index=False))

    # =============== 7.3 Sensibilidad costos ===============
    print("\n[7.3] Sensibilidad a costos...")
    rows = []
    for ticker in ["SPY", "QQQ"]:
        for config in ["VANILLA-baseline", "FILTRADA"]:
            t = all_trades[(all_trades["ticker"] == ticker) &
                           (all_trades["config"] == config)]
            if t.empty:
                continue
            r = sensitivity_costs(t, [1.0, 1.5, 2.0, 3.0])
            r["ticker"] = ticker
            r["config"] = config
            rows.append(r)
    costs = pd.concat(rows, ignore_index=True)
    costs.to_parquet(RESULTS_DIR / "sensitivity_costs.parquet")

    # =============== 7.4 Sensibilidad skew ===============
    print("\n[7.4] Sensibilidad a skew bump...")
    rows = []
    for ticker in ["SPY", "QQQ"]:
        r = sensitivity_skew(ticker, cfg_base, rules_base,
                              [1.00, 1.10, 1.20], use_filter=True)
        rows.append(r)
    skew_df = pd.concat(rows, ignore_index=True)
    skew_df.to_parquet(RESULTS_DIR / "sensitivity_skew.parquet")

    # =============== 7.5 Stress test ===============
    print("\n[7.5] Stress test sub-periodos...")
    rows = []
    for ticker in ["SPY", "QQQ"]:
        for config in ["VANILLA-baseline", "FILTRADA"]:
            t = all_trades[(all_trades["ticker"] == ticker) &
                           (all_trades["config"] == config)].copy()
            if t.empty:
                continue
            t["open_date"] = pd.to_datetime(t["open_date"])
            stress = stress_test(t)
            stress["ticker"] = ticker
            stress["config"] = config
            rows.append(stress)
    stress_all = pd.concat(rows, ignore_index=True)
    stress_all.to_parquet(RESULTS_DIR / "stress_test.parquet")

    # =============== 7.7 HOLDOUT (UNA SOLA PASADA) ===============
    print(f"\n[7.7] VALIDACION FINAL EN HOLDOUT ({TEST_START} -> ...)...")
    print("       *** SELLADO PREVIAMENTE - UNA SOLA EVALUACION ***")
    holdout_rows = []
    holdout_trades_all = []
    for ticker in ["SPY", "QQQ", "IWM"]:
        for config_name, use_filter in [("VANILLA-baseline", False), ("FILTRADA", True)]:
            trades = run_holdout(ticker, cfg_base, rules_base, use_filter=use_filter)
            if trades.empty:
                continue
            trades["ticker"] = ticker
            trades["config"] = config_name
            holdout_trades_all.append(trades)
            m = perf_metrics(trades)
            m["ticker"] = ticker
            m["config"] = config_name
            holdout_rows.append(m)
            print(f"  HOLDOUT {config_name:18s} {ticker}: n={m['n_trades']:4d} "
                  f"winR={m['win_rate']:.3f} exp={m['expectancy_per_contract']:.2f} "
                  f"sharpe={m['sharpe_per_trade']:.3f} maxDD={m['max_drawdown_usd']:.0f}")
    pd.DataFrame(holdout_rows).to_parquet(RESULTS_DIR / "holdout_summary.parquet")
    pd.concat(holdout_trades_all, ignore_index=True).to_parquet(
        RESULTS_DIR / "holdout_trades.parquet")

    print(f"\nResultados Fase 7 en: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
