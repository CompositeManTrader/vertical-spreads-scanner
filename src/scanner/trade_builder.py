"""
Trade builder: dado el subyacente y la IV, calcula el setup teorico del PCS:
- K_short delta-30
- K_long = K_short - $5
- Credito BSM teorico
- Max loss
- TP / SL niveles
- Time stop
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import DIVIDEND_YIELDS, TOTAL_COST_PER_TRADE
from src.pricing.black_scholes import (
    put_delta, put_price, solve_put_strike_for_delta,
)


def build_pcs_setup(ticker: str, spot: float, iv: float, rfr_pct: float,
                    delta_short: float = -0.30, dte: int = 45,
                    width: float = 5.0, tp_pct: float = 0.50,
                    sl_mult: float = 2.0, time_stop_dte: int = 14) -> dict:
    """Devuelve dict con todos los detalles del trade sugerido."""
    q = DIVIDEND_YIELDS.get(ticker, 0.0)
    r = rfr_pct / 100.0
    T = dte / 365.0

    K_short = solve_put_strike_for_delta(spot, T, r, q, iv, delta_short)
    K_long = K_short - width

    # Round strikes a la convencion del subyacente
    if ticker in ("SPY", "QQQ"):
        # SPY/QQQ tienen strikes cada $1 (algunos a $0.50)
        K_short_rounded = round(K_short)
        K_long_rounded = round(K_long)
    elif ticker == "IWM":
        K_short_rounded = round(K_short * 2) / 2  # $0.50 increments
        K_long_rounded = round(K_long * 2) / 2
    else:
        K_short_rounded = round(K_short)
        K_long_rounded = round(K_long)

    p_short = put_price(spot, K_short_rounded, T, r, q, iv)
    p_long = put_price(spot, K_long_rounded, T, r, q, iv)
    credit = p_short - p_long
    max_loss = (K_short_rounded - K_long_rounded) - credit

    # Greeks orientativos
    delta_short_actual = put_delta(spot, K_short_rounded, T, r, q, iv)
    delta_long_actual = put_delta(spot, K_long_rounded, T, r, q, iv)

    # Fechas
    open_date = datetime.today().date()
    expiry_date = open_date + timedelta(days=dte)
    time_stop_date = expiry_date - timedelta(days=time_stop_dte)

    return {
        "ticker": ticker,
        "open_date": open_date.isoformat(),
        "expiry_date": expiry_date.isoformat(),
        "time_stop_date": time_stop_date.isoformat(),
        "dte_at_open": dte,
        "spot": round(spot, 2),
        "iv_atm_used": round(iv, 4),
        "rfr_pct": round(rfr_pct, 2),
        "delta_short_target": delta_short,
        "K_short_theoretical": round(K_short, 2),
        "K_short_recommended": K_short_rounded,
        "K_short_delta_actual": round(delta_short_actual, 4),
        "K_long_theoretical": round(K_long, 2),
        "K_long_recommended": K_long_rounded,
        "K_long_delta_actual": round(delta_long_actual, 4),
        "width_recommended": K_short_rounded - K_long_rounded,
        "credit_per_share_BSM": round(credit, 3),
        "credit_per_contract_BSM": round(credit * 100, 2),
        "max_loss_per_share": round(max_loss, 3),
        "max_loss_per_contract": round(max_loss * 100, 2),
        "tp_per_contract": round(credit * tp_pct * 100, 2),
        "sl_per_contract": round(-credit * sl_mult * 100, 2),
        "credit_to_maxloss_ratio": round(credit / max_loss, 3) if max_loss > 0 else None,
        "estimated_costs_total": TOTAL_COST_PER_TRADE,
        "tp_pct": tp_pct,
        "sl_mult": sl_mult,
        "time_stop_dte": time_stop_dte,
        "rules": (
            f"TP @ {int(tp_pct*100)}% del credito (~${round(credit*tp_pct*100, 2)}/contract). "
            f"SL @ -{sl_mult}x credito (~${round(-credit*sl_mult*100, 2)}/contract). "
            f"Time stop @ {time_stop_date.isoformat()} ({time_stop_dte} DTE remanentes)."
        ),
    }
