"""
Fase 6: simulador P&L completo de PCS con mark-to-market diario y reglas de
gestion activa.

Para cada dia t de entrada:
  1. Calcular K_short, K_long, credito teorico al abrir.
  2. Para cada dia hasta el expiry, recalcular el spread value con BSM
     usando S(t_actual), IV ATM(t_actual), r(t_actual), T_restante.
  3. Aplicar reglas de salida: TP, SL, time stop, touch (intra-day).
  4. Registrar resultado: dia de cierre, razon, P&L bruto y neto de costos.

ANTI-LOOK-AHEAD: la decision en cada dia usa solo info >=t_apertura y <=dia
actual. La IV usada para revaluar es la del dia actual (no se mira adelante).

Caveats:
- Pricing intra-trade asume IV ATM constante por strike (sin skew). En la
  realidad, la IV del put corto sera mayor (skew put), por lo que el spread
  real vale MAS que el modelado -> P&L mark-to-market real sera PEOR cuando
  va en contra. Documentado.
- Touch stop usa el Low del dia (proxy de minimo intra-day).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import DATA_CLEAN, DIVIDEND_YIELDS, TOTAL_COST_PER_TRADE
from src.pricing.black_scholes import (
    put_delta, put_price, solve_put_strike_for_delta,
)


# ---------------------------------------------------------------------------
# Reglas de gestion (configurables)
# ---------------------------------------------------------------------------

@dataclass
class ManagementRules:
    take_profit_pct: float | None = 0.50    # cerrar si pnl >= TP * credit
    stop_loss_mult: float | None = 2.0      # cerrar si pnl <= -SL * credit
    time_stop_dte: int | None = 14          # cerrar si DTE_restante <= time_stop
    touch_stop: bool = False                # cerrar si Low <= K_short en el dia
    hold_to_expiry: bool = False            # ignora otras reglas


@dataclass
class TradeConfig:
    delta_short: float = -0.30   # NEGATIVO
    T_days: int = 45
    width: float = 5.0


# ---------------------------------------------------------------------------
# Simulador
# ---------------------------------------------------------------------------

def simulate_one_trade(panel: pd.DataFrame, t_open: int,
                       cfg: TradeConfig, rules: ManagementRules,
                       q: float) -> dict:
    """Simula un trade abierto en t_open. Devuelve dict con metricas."""
    closes = panel["Close"].values
    lows = panel["Low"].values
    iv_atm = panel["iv_atm_barchart"].values
    rfr = panel["rfr_pct"].values

    n = len(panel)
    if t_open + cfg.T_days >= n:
        return {"valid": False}

    S0 = closes[t_open]
    iv0 = iv_atm[t_open]
    r0 = rfr[t_open] / 100.0 if not np.isnan(rfr[t_open]) else 0.04
    if np.isnan(iv0) or iv0 <= 0:
        return {"valid": False}

    T0 = cfg.T_days / 365.0
    try:
        K_short = solve_put_strike_for_delta(S0, T0, r0, q, iv0, cfg.delta_short)
    except Exception:
        return {"valid": False}
    K_long = K_short - cfg.width
    if K_long <= 0:
        return {"valid": False}

    p_short_open = put_price(S0, K_short, T0, r0, q, iv0)
    p_long_open = put_price(S0, K_long, T0, r0, q, iv0)
    credit_open = p_short_open - p_long_open
    if credit_open <= 0:
        return {"valid": False}
    max_loss = cfg.width - credit_open

    # Loop dia por dia
    exit_day = None
    exit_reason = None
    exit_pnl = None

    for d in range(1, cfg.T_days + 1):
        t_now = t_open + d
        S = closes[t_now]
        iv_now = iv_atm[t_now]
        r_now = rfr[t_now] / 100.0 if not np.isnan(rfr[t_now]) else r0
        if np.isnan(iv_now) or iv_now <= 0:
            iv_now = iv0  # fallback

        T_remain = max((cfg.T_days - d) / 365.0, 1e-6)

        # Touch stop (intra-day)
        if rules.touch_stop and lows[t_now] <= K_short:
            # Si tocó: al toque del strike, valor del spread se aproxima a Width-eps
            # En la practica el cierre seria cerca pero no exacto; aproximamos
            # asumiendo que cerramos en EOD del dia con S = K_short (peor estim).
            S_assume = K_short
            p_s = put_price(S_assume, K_short, T_remain, r_now, q, iv_now)
            p_l = put_price(S_assume, K_long, T_remain, r_now, q, iv_now)
            spread_val = p_s - p_l
            pnl = credit_open - spread_val
            exit_day, exit_reason, exit_pnl = d, "touch_stop", pnl
            break

        # Mark-to-market EOD
        p_s = put_price(S, K_short, T_remain, r_now, q, iv_now)
        p_l = put_price(S, K_long, T_remain, r_now, q, iv_now)
        spread_val = p_s - p_l
        pnl = credit_open - spread_val

        # Reglas
        if not rules.hold_to_expiry:
            if rules.take_profit_pct is not None and pnl >= rules.take_profit_pct * credit_open:
                exit_day, exit_reason, exit_pnl = d, "take_profit", pnl
                break
            if rules.stop_loss_mult is not None and pnl <= -rules.stop_loss_mult * credit_open:
                exit_day, exit_reason, exit_pnl = d, "stop_loss", pnl
                break
            if rules.time_stop_dte is not None and (cfg.T_days - d) <= rules.time_stop_dte:
                exit_day, exit_reason, exit_pnl = d, "time_stop", pnl
                break

    # Si no salio, hold to expiry
    if exit_day is None:
        S_end = closes[t_open + cfg.T_days]
        intrinsic_loss = max(0.0, K_short - max(K_long, S_end))
        pnl = credit_open - intrinsic_loss
        exit_day, exit_reason, exit_pnl = cfg.T_days, "expiry", pnl

    pnl_per_share = exit_pnl
    pnl_per_contract = pnl_per_share * 100  # standard option contract = 100 shares
    pnl_net = pnl_per_contract - TOTAL_COST_PER_TRADE

    return {
        "valid": True,
        "open_date": panel["Date"].iloc[t_open],
        "exit_date": panel["Date"].iloc[t_open + exit_day],
        "exit_day": exit_day,
        "exit_reason": exit_reason,
        "S_open": S0,
        "iv_open": iv0,
        "K_short": K_short,
        "K_long": K_long,
        "credit_open": credit_open,
        "max_loss": max_loss,
        "pnl_per_share": pnl_per_share,
        "pnl_per_contract": pnl_per_contract,
        "pnl_net_after_costs": pnl_net,
        "is_win": int(pnl_per_share > 0),
        "is_full_loss": int(np.isclose(pnl_per_share, -max_loss, atol=0.05)),
    }


def simulate_strategy(panel: pd.DataFrame, ticker: str,
                      cfg: TradeConfig, rules: ManagementRules,
                      filter_mask: pd.Series | None = None) -> pd.DataFrame:
    """Para cada dia t valido, abrir trade. filter_mask aplica filtros de regimen."""
    q = DIVIDEND_YIELDS[ticker]
    n = len(panel)
    rows = []
    for t in range(n - cfg.T_days):
        if filter_mask is not None:
            mfit = filter_mask.iloc[t]
            if not mfit:
                continue
        res = simulate_one_trade(panel, t, cfg, rules, q)
        if res.get("valid"):
            rows.append(res)
    df = pd.DataFrame(rows)
    if len(df):
        df["ticker"] = ticker
    return df
