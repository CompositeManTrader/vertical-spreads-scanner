"""
Reporte de Fase 7 y reporte MASTER consolidado del proyecto.

Genera 2 docx:
  - Phase7_Robustness_and_Holdout.docx
  - MASTER_Final_Report.docx
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import (
    DELTA_GRID, DTE_GRID, REPORTS, TEST_START, TOTAL_COST_PER_TRADE, TRAIN_END,
    WIDTH_GRID,
)
from src.reporting.word_builder import (
    add_bullet, add_callout, add_code_block, add_heading, add_image,
    add_paragraph, add_table_from_df, new_document, save,
)

R7 = REPORTS / "_phase7_results"
CHARTS = REPORTS / "_phase7_charts"
CHARTS.mkdir(parents=True, exist_ok=True)


def setup_plt():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 130,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "font.size": 9,
    })


def chart_holdout_equity(holdout_trades: pd.DataFrame, out_path: Path) -> Path:
    setup_plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (ticker, config), g in holdout_trades.groupby(["ticker", "config"]):
        if len(g) < 2:
            continue
        g = g.sort_values("exit_date").copy()
        g["cum"] = g["pnl_net_after_costs"].cumsum()
        ax.plot(g["exit_date"], g["cum"], label=f"{config} {ticker} (n={len(g)})", lw=1.4)
    ax.set_title(f"HOLDOUT (sellado) Equity Curves - desde {TEST_START}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("P&L acumulado USD por contrato")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def chart_bootstrap_distribution(bb: pd.DataFrame, out_path: Path) -> Path:
    setup_plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(bb))
    labels = [f"{r['config']} {r['ticker']}" for _, r in bb.iterrows()]
    ax.errorbar(bb["sharpe_point"], y,
                xerr=[bb["sharpe_point"] - bb["sharpe_ci_lo_95"],
                      bb["sharpe_ci_hi_95"] - bb["sharpe_point"]],
                fmt="o", capsize=5, color="#3470b8", ecolor="gray", lw=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Sharpe per-trade")
    ax.set_title("Block Bootstrap CI 95% del Sharpe (block=30 dias, 5,000 samples)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main_phase7():
    bb = pd.read_parquet(R7 / "block_bootstrap.parquet")
    mc = pd.read_parquet(R7 / "monte_carlo_dd.parquet")
    sc = pd.read_parquet(R7 / "sensitivity_costs.parquet")
    sk = pd.read_parquet(R7 / "sensitivity_skew.parquet")
    st = pd.read_parquet(R7 / "stress_test.parquet")
    ho = pd.read_parquet(R7 / "holdout_summary.parquet")
    ht = pd.read_parquet(R7 / "holdout_trades.parquet")
    for df in (bb, mc, sc, sk, st, ho, ht):
        for c in ("ticker", "config", "period"):
            if c in df.columns:
                df[c] = df[c].astype(str)

    chart_holdout_equity(ht, CHARTS / "holdout_equity.png")
    chart_bootstrap_distribution(bb, CHARTS / "bootstrap_ci.png")

    doc = new_document(
        "Fase 7 — Robustez y Validacion en Holdout",
        "Vertical Spreads Edge Research — Tests de robustez + holdout final",
    )
    add_heading(doc, "Resumen ejecutivo", 1)
    add_callout(doc,
        "🎯 La estrategia FILTRADA pasa TODOS los tests de robustez: bootstrap "
        "Sharpe CI ancho pero positivo (1.4-5.0 SPY, 1.1-4.5 QQQ); robusta a "
        "costos triplicados (Sharpe -8%); MEJORA con skew bumpeada; en stress "
        "(2022 hike) NO ABRE TRADES (safety off-switch); en COVID FILTRADA "
        "QQQ tuvo Sharpe 7.07 con n=12.",
        color="success",
    )
    add_callout(doc,
        "🎯 HOLDOUT (2024-10-04 a 2026-03-04, ~17 meses): SPY/QQQ FILTRADA "
        "winRate 100%, Sharpe 8.7-9.3, max DD $0. n=35-48. CAVEAT: n chico "
        "para celebrar, pero confirma robustez direccional.",
        color="success",
    )

    add_heading(doc, "1. Block Bootstrap del Sharpe", 1)
    add_paragraph(doc,
        "Test: 5,000 muestras con block size = 30 dias (compensa overlapping "
        "intra-trade). CI 95% del Sharpe per-trade.")
    add_table_from_df(doc, bb.round(4), float_format="{:.4f}")
    add_image(doc, CHARTS / "bootstrap_ci.png", width_inches=6.8)
    add_callout(doc,
        "VANILLA SPY: Sharpe punto 0.38, CI [0.19, 0.71]. Edge real pero "
        "modesto. FILTRADA SPY: 1.89 punto, CI [1.44, 5.03]. Edge robusto, "
        "incluso el limite inferior es muy bueno.",
        color="info",
    )

    add_heading(doc, "2. Monte Carlo de orden de trades (max DD)", 1)
    add_paragraph(doc,
        "Para cada estrategia, permutamos el orden de los trades 5,000 veces "
        "y calculamos el max DD. Compara el max DD observado vs distribucion "
        "de permutaciones.")
    add_table_from_df(doc, mc.round(2), float_format="{:.2f}")
    add_paragraph(doc,
        "Lectura: VANILLA SPY tiene max DD observado de $8,988 vs MC mediano "
        "$1,144 (p99 $2,006). El observado esta MUY arriba: el orden temporal "
        "real concentro perdidas (clusters). FILTRADA: el observado esta en "
        "linea con MC -> sin clustering anomalo.",
        italic=True,
    )

    add_heading(doc, "3. Sensibilidad a costos (Schwab approx)", 1)
    add_paragraph(doc, "Multiplicadores 1x, 1.5x, 2x, 3x del costo base ($2.80/trade):")
    add_table_from_df(doc, sc.round(3), float_format="{:.3f}")
    add_paragraph(doc, "Conclusion: edge es ROBUSTO a costos. Incluso 3x costo, FILTRADA mantiene Sharpe 1.74-1.80 en SPY/QQQ.", bold=True)

    add_heading(doc, "4. Sensibilidad a skew (IV bump)", 1)
    add_paragraph(doc, "Bumpeamos IV ATM por {1.0, 1.10, 1.20} (proxy de skew put).")
    add_table_from_df(doc, sk.round(3), float_format="{:.3f}")
    add_callout(doc,
        "HALLAZGO INESPERADO: Sharpe MEJORA con bump (1.89 -> 2.30 SPY; 1.95 "
        "-> 2.60 QQQ). Razon: el filtro selecciona dias con vrp_high (top "
        "quintile). Bumpear IV no afecta el filtro (las features se calculan "
        "antes), pero al re-pricear los trades el credit es mayor -> P&L mayor. "
        "El edge se sostiene incluso bajo asunciones conservadoras de skew.",
        color="success",
    )

    add_heading(doc, "5. Stress test sub-periodos", 1)
    add_table_from_df(doc, st.round(2), float_format="{:.2f}")
    add_callout(doc,
        "Validacion clave: el filtro 'above_sma200 & vrp_high' SE APAGA "
        "automaticamente en stress. En Aug-Oct 2022 (rate hikes) FILTRADA "
        "abrio CERO trades. VANILLA en ese periodo perdio (-$23 SPY, -$51 "
        "QQQ por trade). El filtro evita los regimenes peligrosos.",
        color="success",
    )

    add_heading(doc, "6. VALIDACION HOLDOUT (2024-10-04 -> 2026-03-04)", 1)
    add_paragraph(doc,
        "Holdout sellado pre-research. Una sola pasada, sin re-tuneo. "
        "Filtros y parametros fijados en train (Fase 6). Resultado:"
    )
    add_table_from_df(doc, ho.round(3), float_format="{:.3f}")
    add_image(doc, CHARTS / "holdout_equity.png", width_inches=6.8)

    add_callout(doc,
        "VEREDICTO HOLDOUT:\n"
        "✅ SPY FILTRADA: 100% winRate, n=35, Sharpe 8.69, max DD 0.\n"
        "✅ QQQ FILTRADA: 100% winRate, n=48, Sharpe 9.32, max DD 0.\n"
        "✅ IWM FILTRADA: 92% winRate, n=12, Sharpe 1.20, max DD 106. (sample chico)\n"
        "✅ VANILLA-baseline confirma edge moderado (Sharpe 0.45 SPY).\n"
        "n FILTRADA chico pero direccion CONFIRMADA. Cero stop loss en holdout.",
        color="success",
    )

    save(doc, REPORTS / "Phase7_Robustness_and_Holdout.docx")
    print("Phase 7 report OK")


# ===========================================================================
# REPORTE MASTER CONSOLIDADO
# ===========================================================================

def main_master():
    doc = new_document(
        "REPORTE MASTER — Vertical Spreads Edge Research",
        "Resumen ejecutivo + estrategia operativa final + caveats",
    )

    add_heading(doc, "Veredicto final: GO / NO GO", 1)
    add_callout(doc,
        "✅ GO con la siguiente estrategia: PCS sobre SPY y QQQ, delta-30, "
        "T=45 DTE, ancho $5, TP 50%, SL 2x, time stop 14 DTE, filtro "
        "'above_sma200 AND vrp_high (expanding top quintile)'. NO operar IWM. "
        "Sizing max 2% del capital por trade. Maximo 2-3 trades concurrentes.",
        color="success",
    )
    add_callout(doc,
        "Performance esperada (basada en train + holdout):\n"
        "  - Win rate: 96-100%\n"
        "  - Sharpe per-trade: 1.5-3 (CI bootstrap 1.4-5)\n"
        "  - Max drawdown: $300-450 USD por contrato sobre 6.5 anios\n"
        "  - Frecuencia: ~30-40 trades por anio (filtro restrictivo)\n"
        "  - Profit por contrato: $70-80 USD promedio (neto de costos Schwab)",
        color="info",
    )

    add_heading(doc, "1. Pregunta original y respuesta", 1)
    add_paragraph(doc,
        "PREGUNTA: 'Si abro un PCS con strike a 30-45 DTE y delta -0.20, "
        "cuantas veces el activo termina ITM o lo toca?'"
    )
    add_paragraph(doc, "RESPUESTAS EMPIRICAS (train 2018-10 a 2024-10):", bold=True)
    add_bullet(doc, "Strike delta-20 a 30 DTE: P(ITM at expiry) SPY=14.8%, QQQ=16.1%, IWM=20.9%. P(touch) ~3x. Ratio touch/ITM 2.5-3.0 vs teoria 2.0.")
    add_bullet(doc, "P(touch & recovered): 60-70% de los touches recuperan. Stops por touch destruyen expectancy.")
    add_bullet(doc, "Strike empirico delta-20: ~3-5% below en SPY (varia con IV), ~5% en QQQ, ~5-6% en IWM.")
    add_bullet(doc, "VRP (variance risk premium): SPY/QQQ tienen P(ITM) MENOR que el |delta| nominal (14-16% vs 20%) -> EDGE ESTRUCTURAL. IWM no.")

    add_heading(doc, "2. Hallazgos contraintuitivos clave", 1)
    add_callout(doc,
        "HALLAZGO 1: Delta-20 NO es el sweet spot. En SPY/QQQ, delta-30 a "
        "delta-40 tienen mejor Sharpe por mejor relacion credit/maxloss "
        "(~30-65% vs ~14% a delta-20). Costos Schwab ahogan delta-20 "
        "(credit chico vs costos fijos).",
        color="info",
    )
    add_callout(doc,
        "HALLAZGO 2: 'Vender prima cuando IV es alto' es UN MITO. Lo que "
        "importa NO es IV absoluto sino VRP (IV - RV). Cuando IV alta es "
        "porque RV es alta, no hay edge. Cuando IV alta esta sobre la RV "
        "(VRP top quintile), hay edge fuerte.",
        color="info",
    )
    add_callout(doc,
        "HALLAZGO 3: 'Diversificar entre SPY/QQQ/IWM reduce riesgo'. FALSO. "
        "Correlaciones suben en stress (Phase 1: SPY-IWM 0.87 -> 0.91 en "
        "panic days). Diversificar agrega trades, NO seguridad.",
        color="info",
    )
    add_callout(doc,
        "HALLAZGO 4: T=60 DTE supera a T=30 en Sharpe. Captura mas theta "
        "efectivo aunque tenga mas exposicion. T=45 es compromiso razonable.",
        color="info",
    )
    add_callout(doc,
        "HALLAZGO 5: IV Rank de Barchart NO es canonico (correlacion 0.84 "
        "con calculo standard de 252 dias). Recalcular siempre.",
        color="warning",
    )
    add_callout(doc,
        "HALLAZGO 6: IWM es estructuralmente distinto. NO tiene VRP. NO usar "
        "la misma estrategia. Investigar setup propio (Phase 8 futura).",
        color="warning",
    )

    add_heading(doc, "3. Recorrido del research (resumen por fase)", 1)
    fases = [
        ("Fase 0 — Setup", "Infraestructura, ETL, validacion IV Rank Barchart vs propio (CRITICO: difieren), train/test sellado, 20 tests anti-look-ahead pasados."),
        ("Fase 1 — Distribucion retornos", "Normalidad rechazada (JB p~0). Skew negativa, kurtosis alta. Reversion a 30d (Spearman p<1e-12). Vol clustering fuerte. Correlacion sube en stress."),
        ("Fase 2 — P(ITM) y P(touch) no-parametrico", "Tabla maestra por (ticker, T, x%). 60-70% de touches recuperan. Mediana del primer touch dia 8-15. Gap risk: 4 gaps >5% en 1,509 dias."),
        ("Fase 3 — Regimenes", "VRP_high y above_sma200 reducen P(ITM) ~70%. iv_rank_low (<20) tambien. Mitos derribados (vender en IV alto, panic_5d). 30 regimenes testeados con Bonferroni."),
        ("Fase 4 — Bridge a delta-20", "Strike delta-20 tipico ~4% SPY, ~5% QQQ, ~5-6% IWM. P(ITM) emp 14.8% SPY vs 20% nominal -> VRP EXISTE en SPY/QQQ. NO en IWM. Robusto a skew bump."),
        ("Fase 5 — Sensibilidad", "Surface delta x DTE x ancho. Delta-40 + T=60 + w=$5 max Sharpe SIN costos. delta-20 IWM Sharpe ~0 (no rentable). Width chico mejor risk-adj."),
        ("Fase 6 — Simulacion P&L", "Vanilla baseline Sharpe 0.30-0.38. FILTRADA: Sharpe 1.89-1.95 SPY/QQQ. Hold-to-expiry maxDD $28-37k. Time stop critico para reducir DD."),
        ("Fase 7 — Robustez + Holdout", "Bootstrap CI Sharpe robusto. Resiliente a costos 3x. Mejora con skew. SE APAGA en stress 2022. HOLDOUT: SPY/QQQ FILTRADA winRate 100%, Sharpe 8.7-9.3."),
    ]
    for n, d in fases:
        add_paragraph(doc, n, bold=True)
        add_paragraph(doc, d)

    add_heading(doc, "4. Estrategia operativa recomendada", 1)
    add_heading(doc, "4.1 Setup", 2)
    add_table_from_df(doc, pd.DataFrame([
        {"Parametro": "Subyacentes", "Valor": "SPY y QQQ (NO IWM)"},
        {"Parametro": "Estructura", "Valor": "Put Credit Spread (PCS) corto"},
        {"Parametro": "Delta short put", "Valor": "-0.30"},
        {"Parametro": "DTE al abrir", "Valor": "45 dias"},
        {"Parametro": "Ancho del spread", "Valor": "$5"},
        {"Parametro": "Take profit", "Valor": "50% del credito recibido"},
        {"Parametro": "Stop loss", "Valor": "2x el credito recibido"},
        {"Parametro": "Time stop", "Valor": "Cerrar a 14 DTE remanentes"},
        {"Parametro": "Touch stop", "Valor": "NO (60-70% recuperan)"},
    ]))

    add_heading(doc, "4.2 Filtros de entrada (TODOS deben cumplirse)", 2)
    add_bullet(doc, "Precio del subyacente > SMA200 (cerrado en t)")
    add_bullet(doc, "VRP(t) >= percentil 80 del expanding window (vrp = IV_ATM - RV20d)")
    add_bullet(doc, "(Validacion holdout: aprox 30-40 trades/anio por ticker)")

    add_heading(doc, "4.3 Position sizing", 2)
    add_bullet(doc, "Max loss por trade <= 2% del capital total")
    add_bullet(doc, "Maximo 2-3 trades concurrentes (cuenta correlacion en stress)")
    add_bullet(doc, "Si SPY y QQQ ambos disparan filtro: priorizar el que tenga VRP mayor")

    add_heading(doc, "4.4 Operacion en Schwab", 2)
    add_bullet(doc, "Cada trade implica 4 legs: 2 al abrir (sell put short, buy put long) + 2 al cerrar")
    add_bullet(doc, "Comision: ~$0.65/contrato/leg = $2.60 + $0.20 slippage = $2.80/trade")
    add_bullet(doc, "Operar EOD: chequear filtros despues del cierre, abrir si aplican")
    add_bullet(doc, "Time stop: anotar fecha al abrir, programar cierre EOD a 14 DTE remanente")
    add_bullet(doc, "TP/SL: orden GTC con triggers automaticos")

    add_heading(doc, "5. Riesgos y caveats QUE DEBES TENER PRESENTES", 1)
    add_callout(doc,
        "1. Sample 6 anios train + 1.5 anios holdout. Si el regimen cambia "
        "(crash 2008-style, japonificacion), filtros pueden romperse.",
        color="warning",
    )
    add_bullet(doc, "Sin chain real, los creditos teoricos sobreestiman levemente el real (skew put). Sharpe real probable: 30-40% menor que el reportado.")
    add_bullet(doc, "n holdout chico: SPY filtrada n=35, QQQ n=48. Sharpe 8-9 es OPTIMISTA. Bootstrap train CI [1.4, 5.0] es realista.")
    add_bullet(doc, "Trades overlapping: 'n efectiva' menor. Block bootstrap compensa parcialmente.")
    add_bullet(doc, "Diversificar SPY+QQQ NO reduce max DD proporcionalmente. Cuenta correlacion ~0.93.")
    add_bullet(doc, "Filtro restrictivo: dias sin operar son normales. Paciencia.")
    add_bullet(doc, "Costos asumidos Schwab. Otros brokers (~$0.50/contrato) similar. Brokers caros (>$1) erosionan edge.")
    add_bullet(doc, "Sin gestion de tail risk catastrofico (gap >10% overnight). Position sizing es la unica defensa.")
    add_bullet(doc, "Sin ajuste por dividend dates: en SPY/QQQ son trimestrales, riesgo de assignment muy bajo lejos del strike.")
    add_bullet(doc, "El holdout cubre Oct 2024 - Mar 2026: bull market post-eleccion + IA boom. Otro tipo de mercado puede no replicar.")

    add_heading(doc, "6. Que NO hacer", 1)
    add_bullet(doc, "NO operar IWM con esta estrategia. Sin VRP estructural. Sharpe ~0 sin filtro, debil con filtro.")
    add_bullet(doc, "NO usar IV Rank de Barchart como filtro. Usar el calculado canonicamente o directo IV ATM thresholds.")
    add_bullet(doc, "NO 'vender en VIX alto' sin medir VRP. Es el VRP que importa, no el VIX.")
    add_bullet(doc, "NO cerrar por touch del strike. 60-70% de los trades que tocan recuperan.")
    add_bullet(doc, "NO meter mas de 2-3 trades concurrentes. Correlacion en stress = riesgo concentrado.")
    add_bullet(doc, "NO bajar a delta-20 'porque es lo que dice todo el mundo'. Costos Schwab ahogan el credit chico.")
    add_bullet(doc, "NO usar holdout para tunear nada mas. Solo se evaluo UNA vez. Re-tuneo es overfit.")
    add_bullet(doc, "NO operar antes de paper trade 1-2 meses para validar ejecucion.")

    add_heading(doc, "7. Roadmap de implementacion", 1)
    add_bullet(doc, "PASO 1 (paper trading 1-2 meses): implementar el scanner que calcule el filtro EOD y notifique. Tomar setups en paper para validar workflow.")
    add_bullet(doc, "PASO 2 (capital chico, 1-2 contratos): operar real con 1 contrato, validar fills y costos reales vs modelados.")
    add_bullet(doc, "PASO 3 (escalado gradual): aumentar size en funcion de track record, manteniendo max loss <= 2% del capital.")
    add_bullet(doc, "PASO 4 (re-validacion anual): re-ejecutar el research con datos actualizados cada 12 meses para validar que filtros siguen vigentes.")

    add_heading(doc, "8. Apendice: lista completa de archivos del proyecto", 1)
    add_bullet(doc, "Phase0_Setup_and_Validation.docx — Setup + tests anti-look-ahead.")
    add_bullet(doc, "Phase1_Price_Distribution.docx — Distribucion empirica de retornos.")
    add_bullet(doc, "Phase2_NonParametric_Probabilities.docx — P(ITM) y P(touch) no-parametricos.")
    add_bullet(doc, "Phase3_Regime_Analysis.docx — 30 regimenes + multifactor + Bonferroni.")
    add_bullet(doc, "Phase4_Delta20_Bridge.docx — Bridge a delta-20 con BSM, validacion VRP.")
    add_bullet(doc, "Phase5_Sensitivity.docx — Sensibilidad delta x DTE x ancho.")
    add_bullet(doc, "Phase6_Strategy_Simulation.docx — Simulacion completa con gestion y filtros.")
    add_bullet(doc, "Phase7_Robustness_and_Holdout.docx — Robustez + validacion holdout.")
    add_bullet(doc, "MASTER_Final_Report.docx — Este documento.")

    save(doc, REPORTS / "MASTER_Final_Report.docx")
    print("Master report OK")


if __name__ == "__main__":
    main_phase7()
    main_master()
