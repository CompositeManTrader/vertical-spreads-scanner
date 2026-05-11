"""
Tests del modulo de pricing.

- Sanity: put_call parity.
- Sanity: delta de put en ATM ~ -0.5 (con dividend yield correction).
- Sanity: solve_put_strike_for_delta es inverso de put_delta.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pricing.black_scholes import (
    call_delta, call_price, prob_below_riskneutral, prob_below_with_drift,
    put_delta, put_price, solve_put_strike_for_delta,
)


def test_put_call_parity():
    """C - P = S*e^(-qT) - K*e^(-rT)."""
    S, K, T, r, q, sigma = 100, 100, 30 / 365, 0.04, 0.013, 0.18
    c = call_price(S, K, T, r, q, sigma)
    p = put_price(S, K, T, r, q, sigma)
    lhs = c - p
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert np.isclose(lhs, rhs, atol=1e-8), f"Parity violada: {lhs} vs {rhs}"


def test_atm_put_delta_near_minus_half():
    """Delta de put ATM con q=0 debe estar cerca de -0.5."""
    S, K, T, r, q, sigma = 100, 100, 30 / 365, 0.04, 0.0, 0.18
    d = put_delta(S, K, T, r, q, sigma)
    # Con r>0 esta levemente arriba de -0.5
    assert -0.55 < d < -0.45, f"Delta ATM fuera de rango razonable: {d}"


def test_atm_call_delta_near_half():
    S, K, T, r, q, sigma = 100, 100, 30 / 365, 0.04, 0.0, 0.18
    d = call_delta(S, K, T, r, q, sigma)
    assert 0.45 < d < 0.60


def test_solve_strike_inverse():
    """solve_put_strike_for_delta es inverso de put_delta."""
    S, T, r, q, sigma = 100, 30 / 365, 0.04, 0.013, 0.18
    for target in [-0.10, -0.20, -0.30, -0.40]:
        K = solve_put_strike_for_delta(S, T, r, q, sigma, target)
        d = put_delta(S, K, T, r, q, sigma)
        assert np.isclose(d, target, atol=1e-6), f"target={target}, recovered delta={d}, K={K}"


def test_solve_strike_below_spot_for_otm_put():
    """Para target_delta = -0.20, el strike debe estar BAJO el spot."""
    S, T, r, q, sigma = 100, 30 / 365, 0.04, 0.013, 0.18
    K = solve_put_strike_for_delta(S, T, r, q, sigma, -0.20)
    assert K < S, f"K debe estar bajo spot. K={K}, S={S}"


def test_put_price_greater_for_higher_iv():
    S, K, T, r, q = 100, 95, 30 / 365, 0.04, 0.013
    p_low = put_price(S, K, T, r, q, 0.10)
    p_high = put_price(S, K, T, r, q, 0.30)
    assert p_high > p_low


def test_prob_below_with_higher_drift_is_lower():
    """Si mu (drift real) > r, P_real(ITM) < P_RN(ITM) (drift positivo aleja del strike bajo)."""
    S, K, T, sigma, q = 100, 95, 30 / 365, 0.18, 0.013
    p_rn = prob_below_riskneutral(S, K, T, 0.04, q, sigma)
    p_real_high_drift = prob_below_with_drift(S, K, T, mu=0.10, q=q, sigma=sigma)
    p_real_low_drift = prob_below_with_drift(S, K, T, mu=-0.05, q=q, sigma=sigma)
    assert p_real_high_drift < p_rn < p_real_low_drift


def test_delta_relation_to_prob_itm():
    """Para puts: |delta| ~ e^(-qT) * N(-d1); prob ITM (RN) = N(-d2). Misma escala."""
    S, T, r, q, sigma = 100, 30 / 365, 0.04, 0.013, 0.18
    K = solve_put_strike_for_delta(S, T, r, q, sigma, -0.20)
    p_rn_itm = prob_below_riskneutral(S, K, T, r, q, sigma)
    assert 0.15 < p_rn_itm < 0.25, f"P RN ITM lejos del delta: {p_rn_itm}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
