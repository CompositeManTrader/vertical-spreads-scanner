"""Reporte Word de Fase 5."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import REPORTS
from src.reporting.charts_phase5 import (
    credit_loss_scatter, curve_metric_by_delta, heatmap_metric,
)
from src.reporting.word_builder import (
    add_bullet, add_callout, add_heading, add_image, add_paragraph,
    add_table_from_df, new_document, save,
)

RESULTS = REPORTS / "_phase5_results"
CHARTS = REPORTS / "_phase5_charts"


def main():
    CHARTS.mkdir(parents=True, exist_ok=True)
    s = pd.read_parquet(RESULTS / "pcs_grid_summary.parquet")
    s["ticker"] = s["ticker"].astype(str)

    # Charts
    for ticker in ["SPY", "QQQ", "IWM"]:
        for w in [5, 10]:
            heatmap_metric(s, ticker, w, "sharpe", "Sharpe (sin costos)",
                           CHARTS / f"heat_sharpe_{ticker}_w{w}.png")
            heatmap_metric(s, ticker, w, "expectancy_per_share",
                           "Expectancy $/share",
                           CHARTS / f"heat_exp_{ticker}_w{w}.png")
        for T in [30, 45]:
            curve_metric_by_delta(s, ticker, T, "sharpe",
                                  CHARTS / f"curve_sharpe_{ticker}_T{T}.png")
            curve_metric_by_delta(s, ticker, T, "expectancy_pct_of_maxloss",
                                  CHARTS / f"curve_eff_{ticker}_T{T}.png")
        credit_loss_scatter(s, ticker, CHARTS / f"frontier_{ticker}.png")

    doc = new_document(
        "Fase 5 — Sensibilidad a delta, DTE y ancho del spread",
        "Vertical Spreads Edge Research — Optimizacion de parametros",
    )

    add_heading(doc, "Resumen ejecutivo", 1)
    add_paragraph(doc,
        "Simulamos 432 combinaciones de (ticker, delta, DTE, ancho) sobre el "
        "train, generando 634,536 trades simulados. Para cada trade calculamos "
        "P&L bajo BSM (sin costos en esta fase; los costos se introducen en "
        "Fase 6)."
    )
    add_callout(doc,
        "🎯 HALLAZGO QUE CAMBIA LA ESTRATEGIA: la sabiduria 'delta-20' NO "
        "es optima. Delta-40 con T=60d y ancho $3-5 da Sharpe 0.47-0.49 en "
        "SPY/QQQ, mientras que delta-20 con T=30d y ancho $5 da Sharpe 0.14 "
        "(3x menor).",
        color="success",
    )
    add_callout(doc,
        "🎯 IWM con delta-20 NO es rentable. Sharpe ~0.01-0.02. Solo a "
        "delta-30/40 con T=60 logra Sharpe positivo modesto (0.22-0.23). "
        "Confirma lo visto en Fase 4: IWM no participa del VRP estructural.",
        color="warning",
    )
    add_callout(doc,
        "🎯 Tradeoff width: anchos chicos ($3) maximizan Sharpe pero crédito "
        "absoluto es bajo (poca eficiencia capital). Anchos grandes ($15) "
        "maximizan expectancy pero peor risk-adjusted. Decision depende de "
        "constraint de capital del trader.",
        color="info",
    )

    # ---- 1. Tabla maestra: top combinaciones por ticker ----
    add_heading(doc, "1. Top 15 combinaciones por Sharpe (por ticker)", 1)
    for ticker in ["SPY", "QQQ", "IWM"]:
        add_paragraph(doc, f"{ticker}:", bold=True)
        sub = s[s["ticker"] == ticker].copy()
        sub["delta_short"] = sub["delta_short"].abs()
        sub_show = sub[["delta_short", "T_days", "width", "n_trades", "win_rate",
                        "full_loss_rate", "expectancy_per_share",
                        "expectancy_pct_of_maxloss", "sharpe",
                        "avg_credit_to_maxloss"]].copy()
        for c in ["win_rate", "full_loss_rate", "expectancy_per_share",
                  "expectancy_pct_of_maxloss", "sharpe", "avg_credit_to_maxloss"]:
            sub_show[c] = sub_show[c].round(4)
        sub_show["n_trades"] = sub_show["n_trades"].astype(int)
        add_table_from_df(doc, sub_show.nlargest(15, "sharpe"))

    # ---- 2. Heatmaps Sharpe ----
    add_heading(doc, "2. Heatmaps de Sharpe (ticker x delta x DTE)", 1)
    add_paragraph(doc, "Sharpe = mean(P&L) / std(P&L). Color verde = mejor.", italic=True)
    for ticker in ["SPY", "QQQ", "IWM"]:
        add_paragraph(doc, f"{ticker} (width=$5):", bold=True)
        add_image(doc, CHARTS / f"heat_sharpe_{ticker}_w5.png", width_inches=6.8)
        add_paragraph(doc, f"{ticker} (width=$10):", bold=True)
        add_image(doc, CHARTS / f"heat_sharpe_{ticker}_w10.png", width_inches=6.8)

    # ---- 3. Heatmaps Expectancy ----
    add_heading(doc, "3. Heatmaps de Expectancy ($/share)", 1)
    for ticker in ["SPY", "QQQ", "IWM"]:
        add_paragraph(doc, f"{ticker} (width=$5):", bold=True)
        add_image(doc, CHARTS / f"heat_exp_{ticker}_w5.png", width_inches=6.8)
        add_paragraph(doc, f"{ticker} (width=$10):", bold=True)
        add_image(doc, CHARTS / f"heat_exp_{ticker}_w10.png", width_inches=6.8)

    # ---- 4. Curvas por delta ----
    add_heading(doc, "4. Sharpe vs Delta (curvas por width)", 1)
    for ticker in ["SPY", "QQQ", "IWM"]:
        add_paragraph(doc, f"{ticker} - T=30d:", bold=True)
        add_image(doc, CHARTS / f"curve_sharpe_{ticker}_T30.png", width_inches=6.5)
        add_paragraph(doc, f"{ticker} - T=45d:", bold=True)
        add_image(doc, CHARTS / f"curve_sharpe_{ticker}_T45.png", width_inches=6.5)

    add_paragraph(doc,
        "Lectura: Sharpe es MONOTONICA creciente con delta en SPY/QQQ "
        "(hasta delta-40, no testeamos mas alla). En IWM, mas mixto: delta "
        "30-40 supera a delta 10-20. Width chico (3-5) supera consistentemente "
        "a width grande (10-15) en Sharpe."
    )

    # ---- 5. Frontera riesgo-retorno ----
    add_heading(doc, "5. Frontera riesgo-retorno", 1)
    add_paragraph(doc,
        "Cada punto es una combinacion (delta, T, width). Eje X: full loss "
        "rate (% de trades con max loss). Eje Y: expectancy / max_loss "
        "(retorno sobre capital comprometido). Color: delta short."
    )
    for ticker in ["SPY", "QQQ", "IWM"]:
        add_paragraph(doc, f"{ticker}:", bold=True)
        add_image(doc, CHARTS / f"frontier_{ticker}.png", width_inches=6.5)

    # ---- 6. Comparacion delta-20 vs delta-40 ----
    add_heading(doc, "6. Comparacion baseline (delta-20) vs optimo (delta-40)", 1)
    rows = []
    for ticker in ["SPY", "QQQ", "IWM"]:
        for delta in [0.20, 0.30, 0.40]:
            for T in [30, 45, 60]:
                for w in [5, 10]:
                    sub = s[(s["ticker"] == ticker) & (s["delta_short"].abs() == delta) &
                            (s["T_days"] == T) & (s["width"] == w)]
                    if not sub.empty:
                        r = sub.iloc[0]
                        rows.append({
                            "Ticker": ticker, "|Δ|": delta, "T": T, "Width": w,
                            "WinRate": round(r["win_rate"], 3),
                            "FullLoss": round(r["full_loss_rate"], 3),
                            "Exp/share $": round(r["expectancy_per_share"], 3),
                            "Exp/MaxLoss": round(r["expectancy_pct_of_maxloss"], 3),
                            "Sharpe": round(r["sharpe"], 3),
                        })
    add_table_from_df(doc, pd.DataFrame(rows))

    # ---- 7. Recomendaciones por ticker ----
    add_heading(doc, "7. Recomendaciones operativas por ticker", 1)

    add_heading(doc, "7.1 SPY", 2)
    add_paragraph(doc,
        "Configuracion optima por Sharpe: delta-40, T=60d, width=$3-5. "
        "Sharpe 0.47-0.49, win rate 79%, full loss 19-20%."
    )
    add_bullet(doc, "Conservador (delta-20, T=30, width $5): Sharpe 0.14, win rate 85%, full loss 11%, exp $0.23/share.")
    add_bullet(doc, "Balanceado (delta-30, T=45, width $5): Sharpe ~0.25, mejor risk-adjusted que delta-20 sin sumar mucho riesgo de full loss.")
    add_bullet(doc, "Agresivo (delta-40, T=60, width $5): Sharpe 0.49 pero full loss 19% (1 de 5 trades es max loss).")

    add_heading(doc, "7.2 QQQ", 2)
    add_bullet(doc, "Mismo patron que SPY. Optimo: delta-40, T=60, width $3-5. Sharpe 0.47-0.48.")
    add_bullet(doc, "Conservador: delta-20, T=30, width $5: Sharpe 0.15, similar a SPY.")

    add_heading(doc, "7.3 IWM", 2)
    add_callout(doc,
        "IWM es el outlier negativo. Delta-20 no funciona (Sharpe ~0). "
        "Solo delta-30/40 con T=60 son rentables, pero Sharpe modesto "
        "(0.20-0.23). Recomendacion: NO operar IWM hasta encontrar setup "
        "especifico (Fase 6+ con filtros de regimen puede revivir IWM).",
        color="warning",
    )

    # ---- 8. Caveats ----
    add_heading(doc, "8. Caveats criticos", 1)
    add_bullet(doc, "Sharpe SIN COSTOS. Comisiones Schwab ~$2.80/trade reducen expectancy proporcionalmente al credit. Con credit de $0.23 (delta-20 SPY), $2.80 = 12% del credit -> impacto significativo. Fase 6 incorpora costos.")
    add_bullet(doc, "Sin skew: usamos IV ATM constante. Pricing del credit puede sobreestimar (los puts OTM cotizan a IV mayor por skew).")
    add_bullet(doc, "Sin gestion: hold to expiry. Fase 6 testea TP 50%, SL 2x, time stops.")
    add_bullet(doc, "Sin filtros: usamos TODOS los dias del train. Fase 6 aplica los filtros encontrados en Fase 3.")
    add_bullet(doc, "Sample 1,510 dias. Trades overlapping (n efectiva menor). Fase 7 hace block bootstrap.")

    # ---- 9. Hallazgos contraintuitivos ----
    add_heading(doc, "9. Hallazgos contraintuitivos resumidos", 1)
    add_callout(doc,
        "MITO: 'delta-20 es el sweet spot'. REALIDAD: en SPY/QQQ, delta-40 "
        "tiene 3x mejor Sharpe. La razon es que el credito al delta-40 es "
        "mucho mas grande relativo al max loss (~65% vs ~14%), y el aumento "
        "en P(loss) no compensa en risk-adjusted.",
        color="info",
    )
    add_callout(doc,
        "MITO: 'mas DTE = mas decadencia capturada en trade'. PARCIALMENTE "
        "VERDADERO: Sharpe MAYOR a T=60 que a T=30 en SPY/QQQ. T=60 captura "
        "mas theta efectivo, pese a tener mas tiempo de exposicion al "
        "movimiento direccional.",
        color="info",
    )
    add_callout(doc,
        "VALIDADO: 'ancho chico = mejor Sharpe'. Width $3-5 supera $10-15 "
        "consistentemente. Pero el credit absoluto es chico, lo que importara "
        "para comisiones (Fase 6).",
        color="success",
    )

    # ---- 10. Proximos pasos ----
    add_heading(doc, "10. Proximos pasos (Fase 6)", 1)
    add_paragraph(doc,
        "Construir simulacion P&L COMPLETA con: (a) costos Schwab realistas, "
        "(b) reglas de gestion (TP, SL, time stop, touch stop, delta stop), "
        "(c) aplicacion de filtros encontrados en Fase 3 ('above_sma200 + "
        "vrp_high'), (d) position sizing, (e) portfolio multi-ticker. Eje "
        "comparativo: vanilla (sin filtros) vs filtrada (con filtros)."
    )

    out = REPORTS / "Phase5_Sensitivity.docx"
    save(doc, out)
    print(f"Reporte guardado: {out}")


if __name__ == "__main__":
    main()
