"""
Backtest HOLD-TO-EXPIRY de Put Credit Spreads — fidelidad exact_terminal.

PRINCIPIO DE CONFIABILIDAD:
El payoff al vencimiento es EXACTO dado el strike y el credito:
    pnl = credito_neto - 100 * max(0, min(K_short - S_expiry, width))
Lo UNICO modelado es el credito de entrada: BSM x haircut (0.87 calibrado
contra quote real de Schwab; sensibilidad 0.80-0.90). No hay marks diarios,
no hay proxy de path. Si la estrategia es rentable con haircut 0.80, el
resultado no depende del modelo de pricing.

REGLAS PRE-REGISTRADAS (definidas en conversacion ANTES de correr esto):

COMUN a las 3 estrategias:
  - Universo: SPY y QQQ (IWM excluido: sin VRP, research Fase 4).
  - Estructura: PCS. Short strike = strike de grilla $1 mas cercano al que
    resuelve delta BSM = -0.30 (IV ATM, r de FRED, q dividendo).
  - Width $5. DTE 45 calendario; expiry = primer dia habil >= open + 45d.
  - Costos: $2.80 por spread round-trip (Schwab).
  - Credito neto = (BSM_short - BSM_long) * 100 * haircut - 2.80.
  - Filtro de credito minimo: saltear si credito neto < $55 (11% del width).
  - Hold to expiry: SIN take profit, SIN stop loss, SIN touch stop.

E1 BASELINE (incondicional):
  - Entrada: primer dia habil de cada semana (ladder semanal), por ticker.
  - Sin ningun filtro de regimen. Es el control metodologico: cualquier
    filtro debe ganarle a esto para justificar sus parametros.

E2 VRP HARVEST:
  - Entrada: primer dia habil de la semana SI:
      (a) VRP = IV_ATM - RV20d >= percentil 60 de su rolling 252d (cerrado en t)
      (b) VIX3M >= VIX (termino no invertido)

E3 PULLBACK PREMIUM HARVEST:
  - Entrada: cualquier dia habil SI:
      (a) Close > SMA200 (uptrend intacto)
      (b) drawdown desde max 60d en [5%, 10%] (el susto)
      (c) IV_ATM >= 1.10 * SMA(IV_ATM, 60) (prima inflada)
      (d) sin otra posicion E3 abierta en el ticker (cooldown natural)

ANTI-LOOK-AHEAD:
  - Todas las features usan rolling/expanding CERRADO en t.
  - La decision del dia t usa el Close de t (entrada EOD del mismo dia).
  - El outcome usa SOLO S del dia de expiry.

Periodo: oct-2019 (necesita 252d de historia de VRP) -> mar-2026.
Se reporta ademas el split train (<= 2024-10-03) vs holdout (>= 2024-10-04):
los parametros vienen pre-registrados, no hubo re-tuneo sobre el holdout.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import (
    DATA_CLEAN, DIVIDEND_YIELDS_BY_YEAR, REPORTS, TRAIN_END,
)
from src.pricing.black_scholes import put_price, solve_put_strike_for_delta

RESULTS_DIR = REPORTS / "_backtest_holdexpiry"

# ---- Parametros pre-registrados (NO tunear tras ver resultados) ----
TICKERS = ["SPY", "QQQ"]
DELTA_SHORT = -0.30
DTE_CAL = 45                  # dias calendario
WIDTH = 5.0
COST_PER_TRADE = 2.80
MIN_NET_CREDIT = 55.0         # USD por contrato
HAIRCUTS = [0.80, 0.85, 0.87, 0.90, 1.00]
BASE_HAIRCUT = 0.87           # calibrado contra quote real Schwab
# Estres de skew: pata larga priceada con IV mas alta (skew put real reduce
# el credito neto del spread). 0.01 = 1 punto de vol entre patas.
SKEW_BUMPS = [0.0, 0.005, 0.01]
VRP_LOOKBACK = 252
VRP_PCTL = 0.60
SMA_TREND = 200
PB_DD_WINDOW = 60
PB_DD_MIN, PB_DD_MAX = 0.05, 0.10
PB_IV_SMA = 60
PB_IV_MULT = 1.10
WARMUP = 252                  # dias antes de la primera senal posible


def load_panel(ticker: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_CLEAN / f"{ticker}.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # ---- Features point-in-time (rolling cerrado en t) ----
    logret = np.log(df["Close"] / df["Close"].shift(1))
    df["rv_20d"] = logret.rolling(20, min_periods=20).std() * np.sqrt(252)
    df["vrp"] = df["iv_atm_barchart"] - df["rv_20d"]
    df["vrp_p60_252"] = df["vrp"].rolling(VRP_LOOKBACK, min_periods=VRP_LOOKBACK).quantile(VRP_PCTL)
    df["sma200"] = df["Close"].rolling(SMA_TREND, min_periods=SMA_TREND).mean()
    df["max60"] = df["Close"].rolling(PB_DD_WINDOW, min_periods=PB_DD_WINDOW).max()
    df["dd60"] = 1.0 - df["Close"] / df["max60"]
    df["iv_sma60"] = df["iv_atm_barchart"].rolling(PB_IV_SMA, min_periods=PB_IV_SMA).mean()

    # Primer dia habil de cada semana ISO (para cadencia semanal)
    iso = df["Date"].dt.isocalendar()
    df["iso_week"] = iso["year"].astype(str) + "-" + iso["week"].astype(str)
    df["is_week_first_day"] = ~df["iso_week"].duplicated()

    return df


def build_trade(df: pd.DataFrame, t: int, ticker: str, haircut: float,
                skew_bump: float = 0.0) -> dict | None:
    """
    Construye el trade abierto en la fila t (EOD). Devuelve None si invalido
    (sin IV, sin expiry en datos, strike invalido, credito < minimo).

    ANTI-CONTAMINACION DE SENSIBILIDAD (fix del code review adversarial):
    la DECISION de entrada (filtro de credito minimo) se toma SIEMPRE con
    BASE_HAIRCUT y sin skew bump — asi todos los escenarios de sensibilidad
    operan el MISMO conjunto de trades y el haircut/skew del escenario solo
    revalua el credito. Con esto, haircut 0.80 es una cota inferior genuina.
    """
    S = df["Close"].iloc[t]
    iv = df["iv_atm_barchart"].iloc[t]
    rfr = df["rfr_pct"].iloc[t]
    if pd.isna(iv) or iv <= 0 or pd.isna(S):
        return None
    r = rfr / 100.0 if pd.notna(rfr) else 0.04
    open_date = df["Date"].iloc[t]
    # Dividend yield point-in-time por anio (no fin de muestra retroactivo)
    q = DIVIDEND_YIELDS_BY_YEAR[ticker].get(open_date.year, 0.013)

    target_expiry = open_date + pd.Timedelta(days=DTE_CAL)
    # Primer dia habil >= target (busqueda en el indice de fechas del panel)
    future = df[df["Date"] >= target_expiry]
    if future.empty:
        return None  # expiry cae fuera del dataset -> trade no evaluable
    t_exp = future.index[0]
    expiry_date = df["Date"].iloc[t_exp]
    dte_effective = (expiry_date - open_date).days
    T = dte_effective / 365.0

    try:
        K_short_theo = solve_put_strike_for_delta(S, T, r, q, iv, DELTA_SHORT)
    except Exception:
        return None
    K_short = round(K_short_theo)          # grilla $1
    K_long = K_short - WIDTH
    if K_long <= 0 or K_short >= S:
        return None

    # Credito de referencia (decision de entrada): BASE_HAIRCUT, sin skew.
    credit_ref_share = put_price(S, K_short, T, r, q, iv) - put_price(S, K_long, T, r, q, iv)
    if credit_ref_share <= 0:
        return None
    credit_ref_net = credit_ref_share * 100.0 * BASE_HAIRCUT - COST_PER_TRADE
    if credit_ref_net < MIN_NET_CREDIT:
        return None

    # Credito del ESCENARIO: haircut del escenario + skew bump en pata larga.
    credit_gross_share = (put_price(S, K_short, T, r, q, iv)
                          - put_price(S, K_long, T, r, q, iv + skew_bump))
    credit_net = credit_gross_share * 100.0 * haircut - COST_PER_TRADE

    S_exp = df["Close"].iloc[t_exp]
    intrinsic = max(0.0, min(K_short - S_exp, WIDTH))
    pnl = credit_net - 100.0 * intrinsic

    return {
        "ticker": ticker,
        "open_date": open_date,
        "expiry_date": expiry_date,
        "dte": dte_effective,
        "S_open": S,
        "iv_open": iv,
        "K_short": K_short,
        "K_long": K_long,
        "credit_net": credit_net,
        "max_loss": WIDTH * 100.0 - credit_net,
        "S_expiry": S_exp,
        "pnl": pnl,
        "is_win": int(pnl > 0),
        "is_full_loss": int(np.isclose(intrinsic, WIDTH)),
    }


# ---------------------------------------------------------------------------
# Senales de entrada por estrategia (point-in-time, evaluadas en fila t)
# ---------------------------------------------------------------------------

def signal_baseline(df: pd.DataFrame, t: int) -> bool:
    return bool(df["is_week_first_day"].iloc[t])


def signal_vrp_harvest(df: pd.DataFrame, t: int) -> bool:
    if not df["is_week_first_day"].iloc[t]:
        return False
    vrp = df["vrp"].iloc[t]
    thr = df["vrp_p60_252"].iloc[t]
    vix = df["VIX_Close"].iloc[t]
    vix3m = df["VIX3M_Close"].iloc[t]
    if pd.isna(vrp) or pd.isna(thr) or pd.isna(vix) or pd.isna(vix3m):
        return False
    return bool(vrp >= thr and vix3m >= vix)


def signal_pullback(df: pd.DataFrame, t: int) -> bool:
    c = df["Close"].iloc[t]
    sma = df["sma200"].iloc[t]
    dd = df["dd60"].iloc[t]
    iv = df["iv_atm_barchart"].iloc[t]
    iv_sma = df["iv_sma60"].iloc[t]
    if any(pd.isna(x) for x in (sma, dd, iv, iv_sma)):
        return False
    return bool(c > sma and PB_DD_MIN <= dd <= PB_DD_MAX and iv >= PB_IV_MULT * iv_sma)


# ---------------------------------------------------------------------------
# Motor de backtest
# ---------------------------------------------------------------------------

def run_strategy(name: str, signal_fn, panels: dict[str, pd.DataFrame],
                 haircut: float, cooldown_open_position: bool = False,
                 skew_bump: float = 0.0) -> pd.DataFrame:
    """
    Recorre cada panel dia a dia; si la senal dispara, abre trade EOD.
    cooldown_open_position: si True, no abre mientras haya una posicion de
    esta estrategia abierta en el ticker (usado por E3 Pullback).
    """
    trades = []
    for ticker, df in panels.items():
        open_until = None  # expiry_date de la posicion abierta (cooldown)
        for t in range(WARMUP, len(df)):
            if cooldown_open_position and open_until is not None \
               and df["Date"].iloc[t] <= open_until:
                continue
            if not signal_fn(df, t):
                continue
            tr = build_trade(df, t, ticker, haircut, skew_bump)
            if tr is None:
                continue
            tr["strategy"] = name
            trades.append(tr)
            if cooldown_open_position:
                open_until = tr["expiry_date"]
    return pd.DataFrame(trades)


def metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0}
    p = trades["pnl"].values
    wins = p > 0
    n = len(p)
    gross_win = p[wins].sum() if wins.any() else 0.0
    gross_loss = -p[~wins].sum() if (~wins).any() else 0.0
    eq = np.cumsum(p)  # orden por fecha de apertura
    peaks = np.maximum.accumulate(eq)
    max_dd = float((peaks - eq).max()) if n else 0.0
    streak = max_streak = 0
    for x in p:
        streak = streak + 1 if x < 0 else 0
        max_streak = max(max_streak, streak)
    return {
        "n": n,
        "win_rate": float(wins.mean()),
        "full_loss_rate": float(trades["is_full_loss"].mean()),
        "avg_credit": float(trades["credit_net"].mean()),
        "expectancy": float(p.mean()),
        "total_pnl": float(p.sum()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else np.inf,
        "sharpe_trade": float(p.mean() / p.std(ddof=1)) if n > 1 and p.std(ddof=1) > 0 else np.nan,
        "max_dd_realized": max_dd,
        "max_consec_losses": int(max_streak),
        "worst_trade": float(p.min()),
        "best_trade": float(p.max()),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    panels = {tk: load_panel(tk) for tk in TICKERS}
    for tk, df in panels.items():
        n_iv = df["iv_atm_barchart"].notna().sum()
        print(f"{tk}: {len(df)} filas ({df['Date'].min().date()} -> {df['Date'].max().date()}), IV en {n_iv}")

    strategies = {
        "E1_BASELINE": (signal_baseline, False),
        "E2_VRP_HARVEST": (signal_vrp_harvest, False),
        "E3_PULLBACK": (signal_pullback, True),
    }

    all_trades = []
    summary_rows = []
    # Sensibilidad: (haircut, skew_bump). Todos los escenarios operan el
    # MISMO conjunto de trades (decision con BASE_HAIRCUT); solo se revalua
    # el credito. Escenario estres maximo: haircut 0.80 + skew 0.01.
    scenarios = [(h, 0.0) for h in HAIRCUTS] + \
                [(0.87, 0.005), (0.87, 0.01), (0.80, 0.01)]
    for haircut, skew in scenarios:
        for name, (fn, cooldown) in strategies.items():
            tr = run_strategy(name, fn, panels, haircut, cooldown, skew)
            if tr.empty:
                continue
            tr = tr.sort_values("open_date").reset_index(drop=True)
            tr["haircut"] = haircut
            tr["skew_bump"] = skew
            if haircut == BASE_HAIRCUT and skew == 0.0:
                all_trades.append(tr)

            # Split PURGADO (fix code review): TRAIN = ciclo completo antes
            # del corte (expiry <= TRAIN_END); HOLDOUT = abierto despues del
            # corte. Trades que cruzan la frontera solo cuentan en FULL.
            m = metrics(tr); m.update(strategy=name, haircut=haircut, skew_bump=skew, segment="FULL")
            summary_rows.append(m)
            tr_train = tr[tr["expiry_date"] <= pd.Timestamp(TRAIN_END)]
            tr_hold = tr[tr["open_date"] > pd.Timestamp(TRAIN_END)]
            m = metrics(tr_train); m.update(strategy=name, haircut=haircut, skew_bump=skew, segment="TRAIN")
            summary_rows.append(m)
            m = metrics(tr_hold); m.update(strategy=name, haircut=haircut, skew_bump=skew, segment="HOLDOUT")
            summary_rows.append(m)

    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(RESULTS_DIR / "summary.parquet")
    trades_df = pd.concat(all_trades, ignore_index=True)
    trades_df.to_parquet(RESULTS_DIR / "trades_haircut087.parquet")

    print(f"\nGuardado: {RESULTS_DIR}")
    print(f"Trades (haircut {BASE_HAIRCUT}, sin skew): {len(trades_df)}")


if __name__ == "__main__":
    main()
