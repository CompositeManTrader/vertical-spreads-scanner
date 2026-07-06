# -*- coding: utf-8 -*-
"""Construye el notebook de Google Colab (JSON ipynb) del backtest VRP Harvest."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "vrp_harvest_backtest.ipynb"
OUT.parent.mkdir(parents=True, exist_ok=True)


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


cells = []

cells.append(md("""# Backtest: Put Credit Spreads con VRP Harvest (SPY/QQQ)

**Reproduccion del backtest hold-to-expiry del research de Vertical Spreads.**

Este notebook es auto-contenido: descarga precios de Yahoo Finance y corre el backtest completo
de las 3 estrategias (E1 Baseline, E2 VRP Harvest, E3 Pullback) con los tests de estres de pricing.

## Como usarlo
1. Menu: **Entorno de ejecucion → Ejecutar todas** (Runtime → Run all)
2. Esperar ~2-3 minutos
3. Ver tablas y graficos al final

## Principio de confiabilidad
- **Hold-to-expiry**: el payoff se calcula EXACTO con el precio de cierre del dia de vencimiento.
  `pnl = credito_neto - 100 * max(0, min(K_short - S_expiry, ancho))`
- Lo UNICO modelado es el credito de entrada: Black-Scholes x haircut 0.87 (calibrado contra
  una orden real de Schwab), estresado en banda 0.80-1.00 + skew de pata larga.
- **IV proxy**: este notebook usa `IV_ATM ≈ 0.85 x VIX` (SPY) y `0.85 x VXN` (QQQ), relacion
  medida en el research (mediana IV_ATM/VIX = 0.846 en 7 anos de datos Barchart). La seccion
  opcional del final permite subir los xlsx de Barchart para reproduccion exacta.

**Parametros PRE-REGISTRADOS** (no ajustados despues de ver resultados). Si cambias parametros
y encontras "mejores" resultados, eso es overfitting: cualquier cambio debe re-validarse en el holdout.

*Este notebook no constituye asesoramiento financiero.*"""))

cells.append(md("## 1. Setup"))
cells.append(code("""!pip install -q yfinance
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
pd.set_option('display.width', 200)
plt.rcParams.update({'figure.dpi': 100, 'axes.grid': True, 'grid.alpha': 0.3,
                     'axes.spines.top': False, 'axes.spines.right': False})
print('Setup OK')"""))

cells.append(md("## 2. Parametros pre-registrados"))
cells.append(code("""# ---- Trade (comun a las 3 estrategias) ----
TICKERS       = ["SPY", "QQQ"]     # IWM excluido: sin VRP estructural (research Fase 4)
DELTA_SHORT   = -0.30              # delta-30 supera a delta-20 con costos reales
DTE_CAL       = 45                 # dias calendario al vencimiento
WIDTH         = 5.0                # ancho del spread en $
COST_PER_TRADE = 2.80              # Schwab: 4 legs x $0.65 + slippage
MIN_NET_CREDIT = 55.0              # crédito neto minimo por contrato (11% del ancho)

# ---- Pricing ----
BASE_HAIRCUT  = 0.87               # BSM x 0.87 = credito real (calibrado con quote Schwab)
HAIRCUTS      = [0.80, 0.85, 0.87, 0.90, 1.00]
SKEW_BUMPS_EXTRA = [(0.87, 0.005), (0.87, 0.01), (0.80, 0.01)]  # estres skew pata larga

# ---- Senales ----
VRP_LOOKBACK  = 252                # ventana del percentil del VRP
VRP_PCTL      = 0.60               # umbral: percentil 60
SMA_TREND     = 200
PB_DD_MIN, PB_DD_MAX = 0.05, 0.10  # pullback: drawdown 5-10% desde max 60d
PB_IV_MULT    = 1.10               # pullback: IV >= 1.1x su SMA60
WARMUP        = 252

# ---- IV proxy (relacion medida en el research: IV_ATM/VIX mediana = 0.846) ----
IV_PROXY_RATIO = 0.85
VOL_INDEX = {"SPY": "^VIX", "QQQ": "^VXN"}

# ---- Otros ----
RFR_ANNUAL    = 0.03               # tasa libre de riesgo aproximada (sensibilidad baja a 45 DTE)
DIV_YIELD     = {"SPY": 0.014, "QQQ": 0.006}
TRAIN_END     = pd.Timestamp("2024-10-03")   # split purgado train/holdout
START         = "2017-06-01"       # margen para warmup de features"""))

cells.append(md("## 3. Datos (Yahoo Finance)"))
cells.append(code("""def download(symbol):
    df = yf.download(symbol, start=START, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()[["Date", "Close"]].dropna()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df.sort_values("Date").reset_index(drop=True)

vix   = download("^VIX").rename(columns={"Close": "VIX"})
vix3m = download("^VIX3M").rename(columns={"Close": "VIX3M"})

panels = {}
for tk in TICKERS:
    px = download(tk)
    vi = download(VOL_INDEX[tk]).rename(columns={"Close": "VOLIDX"})
    df = px.merge(vi, on="Date", how="left").merge(vix, on="Date", how="left") \\
           .merge(vix3m, on="Date", how="left")
    # IV ATM proxy: 0.85 x indice de vol (en fraccion)
    df["iv_atm"] = df["VOLIDX"] / 100.0 * IV_PROXY_RATIO
    panels[tk] = df
    print(f"{tk}: {len(df)} filas  ({df['Date'].min().date()} -> {df['Date'].max().date()})")"""))

cells.append(md("## 4. Black-Scholes (pricing del credito de entrada)"))
cells.append(code("""def _d1(S, K, T, r, q, sigma):
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def put_price(S, K, T, r, q, sigma):
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def solve_put_strike_for_delta(S, T, r, q, sigma, target_delta):
    \"\"\"K tal que delta_put BSM = target_delta (cerrado, sin iteracion).\"\"\"
    arg = -target_delta * np.exp(q * T)
    d1 = -norm.ppf(arg)
    return S * np.exp(-(d1 * sigma * np.sqrt(T)) + (r - q + 0.5 * sigma**2) * T)

# Sanity check: el solver es inverso del delta
S0, T0, r0, q0, iv0 = 600.0, 45/365, 0.03, 0.014, 0.16
K0 = solve_put_strike_for_delta(S0, T0, r0, q0, iv0, -0.30)
print(f"Strike delta-30 para spot {S0}: {K0:.2f}  ({(1-K0/S0)*100:.1f}% OTM)")"""))

cells.append(md("## 5. Features point-in-time (sin look-ahead)\n\nTodas las ventanas estan CERRADAS en t: la senal del dia t solo usa datos hasta t."))
cells.append(code("""for tk, df in panels.items():
    logret = np.log(df["Close"] / df["Close"].shift(1))
    df["rv_20d"]   = logret.rolling(20, min_periods=20).std() * np.sqrt(252)
    df["vrp"]      = df["iv_atm"] - df["rv_20d"]
    df["vrp_thr"]  = df["vrp"].rolling(VRP_LOOKBACK, min_periods=VRP_LOOKBACK).quantile(VRP_PCTL)
    df["sma200"]   = df["Close"].rolling(SMA_TREND, min_periods=SMA_TREND).mean()
    df["max60"]    = df["Close"].rolling(60, min_periods=60).max()
    df["dd60"]     = 1 - df["Close"] / df["max60"]
    df["iv_sma60"] = df["iv_atm"].rolling(60, min_periods=60).mean()
    iso = df["Date"].dt.isocalendar()
    df["iso_week"] = iso["year"].astype(str) + "-" + iso["week"].astype(str)
    df["week_first"] = ~df["iso_week"].duplicated()
print("Features OK")"""))

cells.append(md("""## 6. Motor del backtest

- `build_trade`: arma el spread en t (EOD), calcula el credito modelado y el payoff EXACTO al vencimiento.
- **Fix anti-contaminacion** (del code review adversarial): la decision de entrada usa SIEMPRE el
  haircut base; los escenarios de estres solo revaluan el credito de los MISMOS trades."""))
cells.append(code("""def build_trade(df, t, tk, haircut, skew_bump=0.0):
    S, iv = df["Close"].iloc[t], df["iv_atm"].iloc[t]
    if pd.isna(iv) or iv <= 0:
        return None
    r, q = RFR_ANNUAL, DIV_YIELD[tk]
    open_date = df["Date"].iloc[t]
    future = df[df["Date"] >= open_date + pd.Timedelta(days=DTE_CAL)]
    if future.empty:
        return None
    t_exp = future.index[0]
    T = (df["Date"].iloc[t_exp] - open_date).days / 365.0
    try:
        K_short = round(solve_put_strike_for_delta(S, T, r, q, iv, DELTA_SHORT))
    except Exception:
        return None
    K_long = K_short - WIDTH
    if K_long <= 0 or K_short >= S:
        return None
    # Decision de entrada: haircut BASE, sin skew (mismos trades en todos los escenarios)
    ref = (put_price(S, K_short, T, r, q, iv) - put_price(S, K_long, T, r, q, iv))
    if ref <= 0 or ref * 100 * BASE_HAIRCUT - COST_PER_TRADE < MIN_NET_CREDIT:
        return None
    # Credito del escenario (skew bump: pata larga con IV mas alta reduce el credito)
    gross = put_price(S, K_short, T, r, q, iv) - put_price(S, K_long, T, r, q, iv + skew_bump)
    credit = gross * 100 * haircut - COST_PER_TRADE
    S_exp = df["Close"].iloc[t_exp]
    pnl = credit - 100 * max(0.0, min(K_short - S_exp, WIDTH))
    return {"ticker": tk, "open_date": open_date, "expiry_date": df["Date"].iloc[t_exp],
            "S_open": S, "K_short": K_short, "K_long": K_long, "credit_net": credit,
            "S_expiry": S_exp, "pnl": pnl, "is_win": int(pnl > 0)}

def sig_baseline(df, t):
    return bool(df["week_first"].iloc[t])

def sig_vrp(df, t):
    if not df["week_first"].iloc[t]:
        return False
    v, thr = df["vrp"].iloc[t], df["vrp_thr"].iloc[t]
    vix, v3m = df["VIX"].iloc[t], df["VIX3M"].iloc[t]
    if any(pd.isna(x) for x in (v, thr, vix, v3m)):
        return False
    return bool(v >= thr and v3m >= vix)

def sig_pullback(df, t):
    c, sma, dd = df["Close"].iloc[t], df["sma200"].iloc[t], df["dd60"].iloc[t]
    iv, ivs = df["iv_atm"].iloc[t], df["iv_sma60"].iloc[t]
    if any(pd.isna(x) for x in (sma, dd, iv, ivs)):
        return False
    return bool(c > sma and PB_DD_MIN <= dd <= PB_DD_MAX and iv >= PB_IV_MULT * ivs)

def run(name, sig_fn, haircut, skew=0.0, cooldown=False):
    rows = []
    for tk, df in panels.items():
        open_until = None
        for t in range(WARMUP, len(df)):
            if cooldown and open_until is not None and df["Date"].iloc[t] <= open_until:
                continue
            if not sig_fn(df, t):
                continue
            tr = build_trade(df, t, tk, haircut, skew)
            if tr is None:
                continue
            tr["strategy"] = name
            rows.append(tr)
            if cooldown:
                open_until = tr["expiry_date"]
    return pd.DataFrame(rows)

def metrics(tr):
    if tr.empty:
        return {"n": 0}
    p = tr["pnl"].values
    wins = p > 0
    eq = np.cumsum(p)
    dd = (np.maximum.accumulate(eq) - eq).max()
    gl = -p[~wins].sum()
    return {"n": len(p), "win_rate": round(wins.mean(), 3),
            "exp_$": round(p.mean(), 1), "total_$": round(p.sum(), 0),
            "PF": round(p[wins].sum() / gl, 2) if gl > 0 else np.inf,
            "maxDD_$": round(dd, 0), "worst_$": round(p.min(), 0)}

print("Motor OK")"""))

cells.append(md("## 7. Correr el backtest (caso base + estres)"))
cells.append(code("""STRATS = {"E1_BASELINE": (sig_baseline, False),
          "E2_VRP_HARVEST": (sig_vrp, False),
          "E3_PULLBACK": (sig_pullback, True)}

# ---- Caso base con splits ----
base_trades = {}
rows = []
for name, (fn, cd) in STRATS.items():
    tr = run(name, fn, BASE_HAIRCUT, 0.0, cd).sort_values("open_date").reset_index(drop=True)
    base_trades[name] = tr
    for seg, sub in [("FULL", tr),
                     ("TRAIN(purgado)", tr[tr["expiry_date"] <= TRAIN_END]),
                     ("HOLDOUT", tr[tr["open_date"] > TRAIN_END])]:
        m = metrics(sub); m.update(strategy=name, segment=seg)
        rows.append(m)
print("=== CASO BASE (haircut 0.87) — split purgado ===")
display(pd.DataFrame(rows)[["strategy","segment","n","win_rate","exp_$","total_$","PF","maxDD_$","worst_$"]])"""))

cells.append(code("""# ---- Sensibilidad / estres (mismos trades, credito revaluado) ----
rows = []
scen = [(h, 0.0) for h in HAIRCUTS] + SKEW_BUMPS_EXTRA
for h, sk in scen:
    for name, (fn, cd) in STRATS.items():
        tr = run(name, fn, h, sk, cd)
        m = metrics(tr); m.update(strategy=name, haircut=h, skew=sk)
        rows.append(m)
sens = pd.DataFrame(rows)[["strategy","haircut","skew","n","win_rate","exp_$","total_$","PF"]]
print("=== ESTRES DE PRICING ===")
print("Peor caso = haircut 0.80 + skew 0.01. Si exp_$ > 0 ahi, el edge NO depende del modelo.")
display(sens.sort_values(["strategy","haircut","skew"]).reset_index(drop=True))"""))

cells.append(md("## 8. Graficos"))
cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

# Equity curves
for name, color in [("E1_BASELINE", "#888888"), ("E2_VRP_HARVEST", "#1f6f3d")]:
    tr = base_trades[name].sort_values("expiry_date")
    axes[0].plot(tr["expiry_date"], tr["pnl"].cumsum(), lw=1.5, color=color, label=name)
axes[0].axhline(0, color="black", lw=0.5)
axes[0].set_title("Equity curves (PnL realizado al vencimiento)")
axes[0].set_ylabel("USD por contrato")
axes[0].legend()

# PnL anual
allt = pd.concat(base_trades.values())
allt["year"] = allt["open_date"].dt.year
piv = allt.groupby(["year", "strategy"])["pnl"].sum().unstack()
piv.plot(kind="bar", ax=axes[1], color=["#888888", "#1f6f3d", "#c9a227"], width=0.8)
axes[1].axhline(0, color="black", lw=0.5)
axes[1].set_title("PnL por año de apertura")
axes[1].set_xlabel("")
plt.setp(axes[1].get_xticklabels(), rotation=0)
plt.tight_layout()
plt.show()"""))

cells.append(md("""## 9. Interpretacion de resultados

**Que buscar en las tablas:**
1. **E2 vs E1 en el caso base**: E2 debe superar a E1 en $/trade y PF (el filtro debe pagarse).
2. **HOLDOUT de E2 positivo**: la estrategia generaliza a datos nunca vistos (post oct-2024).
3. **El peor caso (haircut 0.80 + skew 0.01)**: si E2 sigue positiva ahi, el edge sobrevive
   incluso si el modelo de credito es un 30%+ optimista. Esa es la prueba acida.
4. **E1 en el peor caso**: tipicamente cae a ~$0 o negativa — vender sin filtros NO tiene
   edge robusto al error de pricing.

**Nota sobre el proxy de IV**: los numeros de este notebook difieren algo de los del research
(que usa IV ATM real de Barchart), porque aqui IV = 0.85 x VIX/VXN. La ESTRUCTURA del resultado
(E2 >> E1 >> E3, supervivencia al estres) debe mantenerse; los valores exactos varian.

## Caveats
- Trades overlapping: n no son observaciones independientes.
- maxDD es sobre PnL realizado, no mark-to-market: el drawdown intra-trade real sera mayor.
- Haircut calibrado con 1 quote real; recolectar mas quotes de Schwab lo mejora.
- Historia IV: 2018+ (COVID y 2022 son los unicos stress reales dentro de muestra)."""))

cells.append(md("""## 10. (Opcional) Reproduccion exacta con tus datos de Barchart

Si tenes los archivos `SPY DAILY.xlsx` y `QQQ DAILY.xlsx` (formato Barchart con columna
`Imp Vol`), ejecuta esta celda y subilos cuando lo pida. El backtest se re-corre con la
IV ATM real en vez del proxy VIX/VXN."""))
cells.append(code("""from google.colab import files
print("Subir SPY DAILY.xlsx y QQQ DAILY.xlsx (formato Barchart)...")
up = files.upload()

for tk in TICKERS:
    fname = next((f for f in up if tk in f.upper()), None)
    if fname is None:
        print(f"AVISO: no se subio archivo para {tk}, se mantiene proxy VIX/VXN")
        continue
    raw = pd.read_excel(fname, header=1)
    raw["Date"] = pd.to_datetime(raw["Date"], origin="1899-12-30", unit="D") \\
        if pd.api.types.is_numeric_dtype(raw["Date"]) else pd.to_datetime(raw["Date"])
    raw = raw[["Date", "Imp Vol"]].rename(columns={"Imp Vol": "iv_real"})
    raw["iv_real"] = pd.to_numeric(raw["iv_real"], errors="coerce")
    raw.loc[raw["iv_real"] <= 0, "iv_real"] = np.nan   # limpiar IV=0 (bug Barchart)
    df = panels[tk].merge(raw, on="Date", how="left")
    df["iv_atm"] = df["iv_real"].where(df["iv_real"].notna(), df["iv_atm"])
    # Recalcular features que dependen de la IV
    df["vrp"] = df["iv_atm"] - df["rv_20d"]
    df["vrp_thr"] = df["vrp"].rolling(VRP_LOOKBACK, min_periods=VRP_LOOKBACK).quantile(VRP_PCTL)
    df["iv_sma60"] = df["iv_atm"].rolling(60, min_periods=60).mean()
    panels[tk] = df
    print(f"{tk}: IV real cargada ({raw['iv_real'].notna().sum()} dias)")

print("\\nRe-ejecutar las secciones 7 y 8 para ver los resultados con IV real.")"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "name": "vrp_harvest_backtest.ipynb"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook generado: {OUT}")
