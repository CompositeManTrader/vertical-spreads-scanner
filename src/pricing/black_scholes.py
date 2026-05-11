"""
Black-Scholes-Merton para opciones europeas sobre acciones con dividend yield
continuo q. Convencion:
- T en anios (T_dias / 365 o /252 segun lo que defina trading_days).
- sigma anualizada (fraccion).
- r anualizada (fraccion).
- q dividend yield anualizado (fraccion).
- delta_put devuelve negativo (segun convencion estandar).

ANTI-LOOK-AHEAD: estas son funciones puras matematicas, no acceden a datos
historicos. El cuidado anti-look-ahead se hace al PROVEER (S, sigma, r) en t.
"""

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# d1, d2 helpers
# ---------------------------------------------------------------------------

def _d1_d2(S, K, T, r, q, sigma):
    sigma_sqrtT = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sigma_sqrtT
    d2 = d1 - sigma_sqrtT
    return d1, d2


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def call_price(S, K, T, r, q, sigma):
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S, K, T, r, q, sigma):
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def call_delta(S, K, T, r, q, sigma):
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return np.exp(-q * T) * norm.cdf(d1)


def put_delta(S, K, T, r, q, sigma):
    """Devuelve negativo (convencion estandar)."""
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return -np.exp(-q * T) * norm.cdf(-d1)


def vega(S, K, T, r, q, sigma):
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


# ---------------------------------------------------------------------------
# Resolver strike para delta target
# ---------------------------------------------------------------------------

def solve_put_strike_for_delta(S, T, r, q, sigma, target_delta):
    """
    Encuentra K tal que put_delta(K) = target_delta (target_delta NEGATIVO).

    Bajo BSM, la solucion analitica para put OTM es:

        put_delta = -e^{-qT} * N(-d1) = target_delta
        => N(-d1) = -target_delta * e^{qT}
        => -d1 = N^{-1}(-target_delta * e^{qT})
        => d1 = -N^{-1}(-target_delta * e^{qT})

    Y luego:
        ln(S/K) + (r - q + 0.5*sigma^2) * T = d1 * sigma * sqrt(T)
        K = S * exp(-(d1 * sigma * sqrt(T)) + (r - q + 0.5*sigma^2) * T)

    Este es un calculo cerrado, sin Newton-Raphson.
    """
    target_delta = float(target_delta)
    if target_delta >= 0 or target_delta <= -1:
        raise ValueError(f"target_delta debe estar en (-1, 0). Recibido: {target_delta}")
    arg = -target_delta * np.exp(q * T)
    if arg <= 0 or arg >= 1:
        raise ValueError(f"target_delta * exp(qT) fuera de (0,1): arg={arg}")
    d1 = -norm.ppf(arg)
    K = S * np.exp(-(d1 * sigma * np.sqrt(T)) + (r - q + 0.5 * sigma**2) * T)
    return K


## ---------------------------------------------------------------------------
# Probabilidad bajo BSM
# ---------------------------------------------------------------------------

def prob_below_riskneutral(S, K, T, r, q, sigma):
    """
    P(S(T) < K) bajo medida risk-neutral (Q).
    P_Q(S(T) < K) = N(-d2)

    NOTA: la probabilidad "real" (bajo medida P) requiere conocer el drift
    real mu (esperanza de retorno), que no se observa. BSM no asume nada
    sobre mu para pricing. Por eso aqui solo usamos la version risk-neutral.
    """
    _, d2 = _d1_d2(S, K, T, r, q, sigma)
    return norm.cdf(-d2)


def prob_below_with_drift(S, K, T, mu, q, sigma):
    """
    P(S(T) < K) bajo proceso real con drift esperado mu (CAPM o similar).
    Util para comparar la teoria con la frecuencia empirica observada.

    P_REAL(S(T) < K) = N(-d2_real) donde d2_real usa (mu - q) en vez de r.
    """
    sigma_sqrtT = sigma * np.sqrt(T)
    d2_real = (np.log(S / K) + (mu - q - 0.5 * sigma**2) * T) / sigma_sqrtT
    return norm.cdf(-d2_real)
