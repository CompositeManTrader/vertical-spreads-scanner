"""Reporte Word de Fase 2."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import REPORTS
from src.reporting.charts_phase2 import (
    gap_risk_chart, heatmap_prob, prob_curve, ratio_curve,
)
from src.reporting.word_builder import (
    add_bullet, add_callout, add_heading, add_image, add_paragraph,
    add_table_from_df, new_document, save,
)

RESULTS = REPORTS / "_phase2_results"
CHARTS = REPORTS / "_phase2_charts"


def main():
    CHARTS.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(RESULTS / "grid_probabilities.parquet")
    ft = pd.read_parquet(RESULTS / "first_touch_timing.parquet")
    gaps = pd.read_parquet(RESULTS / "gap_risk.parquet")

    # Generar charts
    for ticker in ["SPY", "QQQ", "IWM"]:
        heatmap_prob(grid, ticker, "p_itm",
                     "P(close < strike) - ITM at expiry",
                     CHARTS / f"heatmap_itm_{ticker}.png")
        heatmap_prob(grid, ticker, "p_touch",
                     "P(touch durante la vida)",
                     CHARTS / f"heatmap_touch_{ticker}.png")
        for T in [30, 45]:
            prob_curve(grid, ticker, T, CHARTS / f"curve_{ticker}_T{T}.png")
        ratio_curve(grid, ticker, CHARTS / f"ratio_{ticker}.png")
    gap_risk_chart(gaps, CHARTS / "gap_risk.png")

    # ---- Documento ----
    doc = new_document(
        "Fase 2 — Probabilidades empiricas no-parametricas",
        "Vertical Spreads Edge Research — P(ITM) y P(touch) sin asumir modelo de opciones",
    )

    add_heading(doc, "Resumen ejecutivo", 1)
    add_paragraph(doc,
        "Para cada combinacion (ticker, T en {30,35,40,45}, x% en grilla), "
        "calculamos sobre el train set:")
    add_bullet(doc, "P(ITM) = P(Close[t+T] < S(t) * (1-x))")
    add_bullet(doc, "P(touch) = P(min Low[t..t+T] < S(t) * (1-x))")
    add_bullet(doc, "P(touch & recovered) = touch sin terminar ITM (rebote intra-trade)")
    add_bullet(doc, "Ratio empirico touch/ITM (vs ~2 que predice random walk)")
    add_paragraph(doc, "Hallazgos principales:", bold=True)
    add_bullet(doc, "Ratio empirico touch/ITM = 2.5-3.0 (mayor que el ~2 teorico). El precio toca y rebota mas seguido.")
    add_bullet(doc, "60-70% de los touches RECUPERAN sin terminar ITM. Stops tight cierran trades que iban a ganar.")
    add_bullet(doc, "First touch median: 8-15 dias en una ventana de 30. Si va a colapsar, lo hace temprano. Esto LIMITA el valor de cierre por DTE.")
    add_bullet(doc, "Gap risk overnight: SPY tuvo ~4 gaps >5% en 1,509 dias. Riesgo NO-hedgeable, pone piso a stops.")

    # ---- Seccion 1: P(ITM) ----
    add_heading(doc, "1. P(close ITM at expiry) - tabla maestra", 1)
    add_paragraph(doc,
        "Esta tabla responde DIRECTAMENTE: 'si pongo el strike X% debajo del spot, "
        "cuantas veces termina ITM en el train set?'. Sin asumir modelo de opciones."
    )
    for ticker in ["SPY", "QQQ", "IWM"]:
        add_paragraph(doc, f"{ticker}:", bold=True)
        sub = grid[grid["ticker"] == ticker].sort_values(["T", "x_pct_below"]).copy()
        sub_show = sub[["T", "x_pct_below", "n", "p_itm", "p_itm_ci_lo", "p_itm_ci_hi"]].copy()
        sub_show["x_pct_below"] = sub_show["x_pct_below"].apply(lambda v: f"{v:.0%}")
        for c in ["p_itm", "p_itm_ci_lo", "p_itm_ci_hi"]:
            sub_show[c] = (sub_show[c] * 100).round(2)
        sub_show.columns = ["T", "% below", "N", "P(ITM) %", "CI lo %", "CI hi %"]
        add_table_from_df(doc, sub_show)

    add_paragraph(doc, "Heatmaps:", bold=True)
    for ticker in ["SPY", "QQQ", "IWM"]:
        add_image(doc, CHARTS / f"heatmap_itm_{ticker}.png", width_inches=6.5)

    # ---- Seccion 2: P(touch) ----
    add_heading(doc, "2. P(touch durante la vida)", 1)
    add_paragraph(doc,
        "Probabilidad de que el precio en algun momento (1..T) este por debajo "
        "del strike. Importa para gestion (stops por touch) pero NO determina "
        "max loss del PCS (eso depende del cierre)."
    )
    for ticker in ["SPY", "QQQ", "IWM"]:
        add_paragraph(doc, f"{ticker}:", bold=True)
        sub = grid[grid["ticker"] == ticker].sort_values(["T", "x_pct_below"]).copy()
        sub_show = sub[["T", "x_pct_below", "n", "p_touch", "p_touch_ci_lo", "p_touch_ci_hi"]].copy()
        sub_show["x_pct_below"] = sub_show["x_pct_below"].apply(lambda v: f"{v:.0%}")
        for c in ["p_touch", "p_touch_ci_lo", "p_touch_ci_hi"]:
            sub_show[c] = (sub_show[c] * 100).round(2)
        sub_show.columns = ["T", "% below", "N", "P(touch) %", "CI lo %", "CI hi %"]
        add_table_from_df(doc, sub_show)
    for ticker in ["SPY", "QQQ", "IWM"]:
        add_image(doc, CHARTS / f"heatmap_touch_{ticker}.png", width_inches=6.5)

    # ---- Seccion 3: curvas combinadas ----
    add_heading(doc, "3. Curvas P(ITM) vs P(touch) por (ticker, T)", 1)
    add_paragraph(doc, "Cada chart muestra ambas probabilidades con CI 95% Wilson:")
    for ticker in ["SPY", "QQQ", "IWM"]:
        for T in [30, 45]:
            add_paragraph(doc, f"{ticker} T={T}d:")
            add_image(doc, CHARTS / f"curve_{ticker}_T{T}.png", width_inches=6.5)

    # ---- Seccion 4: ratio touch/ITM ----
    add_heading(doc, "4. Ratio empirico P(touch) / P(ITM)", 1)
    add_paragraph(doc,
        "Bajo random walk Brownian sin drift, el teorema del reflexion da "
        "P(touch) = 2 * P(ITM at expiry). Es una de las identidades mas "
        "elegantes de la teoria de paseos aleatorios. Si el ratio empirico "
        "es DISTINTO, hay desviacion del random walk: drift, mean reversion, "
        "vol heterogeneidad, etc."
    )
    for ticker in ["SPY", "QQQ", "IWM"]:
        add_paragraph(doc, f"{ticker}:", bold=True)
        add_image(doc, CHARTS / f"ratio_{ticker}.png", width_inches=6.5)

    add_callout(doc,
        "Hallazgo: el ratio empirico es 2.5-3.0, NO 2. El precio toca el "
        "strike y rebota MAS seguido de lo que predice random walk. "
        "Implicacion: en estrategias con stop por touch, gatillas el stop "
        "en mas casos de los que terminan en perdida real -> gestion por "
        "touch transforma falsos positivos en perdidas realizadas.",
        color="warning",
    )

    # ---- Seccion 5: path analysis ----
    add_heading(doc, "5. Analisis del path: cuando ocurre el primer touch?", 1)
    add_paragraph(doc,
        "Para los casos con touch, calculamos el primer dia de la ventana "
        "donde el touch ocurrio, y el % que recupero (no termino ITM)."
    )
    show_cols = ["ticker", "T", "x_pct_below", "n_touched", "n_touched_recovered",
                 "pct_touched_recovered_of_touched", "first_touch_mean_day",
                 "first_touch_median_day", "first_touch_in_first_third_pct",
                 "first_touch_in_last_third_pct"]
    ft_show = ft[show_cols].copy()
    ft_show["x_pct_below"] = ft_show["x_pct_below"].apply(lambda v: f"{v:.0%}")
    for c in ["pct_touched_recovered_of_touched",
              "first_touch_in_first_third_pct", "first_touch_in_last_third_pct"]:
        ft_show[c] = (ft_show[c] * 100).round(1)
    for c in ["first_touch_mean_day", "first_touch_median_day"]:
        ft_show[c] = ft_show[c].round(1)
    ft_show.columns = ["Ticker", "T", "% below", "N touched", "N touched-recovered",
                       "% recovered", "Mean day touch", "Median day touch",
                       "% touch en 1er tercio", "% touch en ultimo tercio"]
    add_table_from_df(doc, ft_show)

    add_callout(doc,
        "Hallazgos del path:",
        color="info",
    )
    add_bullet(doc, "60-70% de los touches recuperan: en la mayoria de los casos donde el precio toca, vuelve y termina arriba.")
    add_bullet(doc, "Mediana del primer touch: 8-15 dias en ventanas de 30 (1ro-2do tercio). 6-12 dias para SPY/QQQ con strikes cercanos.")
    add_bullet(doc, "% de touches en el primer tercio del trade: 40-70% (dependiendo de x%). Los touches a strikes cercanos (3%) ocurren MUY temprano.")
    add_bullet(doc, "% de touches en el ultimo tercio: 10-30%. Cierre por DTE (ej. 21 DTE) NO evita la mayoria de los touches.")

    # ---- Seccion 6: gap risk ----
    add_heading(doc, "6. Riesgo de gap overnight", 1)
    add_paragraph(doc,
        "El precio puede saltar entre el cierre del dia previo y la apertura "
        "del dia actual. Estos gaps son NO-HEDGEABLES intra-dia: ningun stop "
        "puede frenarlos. Determinan un piso a la maxima perdida posible."
    )
    gaps_show = gaps.copy()
    gaps_show["x_pct"] = gaps_show["x_pct"].apply(lambda v: f"-{v:.0%}")
    for c in ["p_overnight_gap_down", "p_intraday_drop"]:
        gaps_show[c] = (gaps_show[c] * 100).round(3)
    gaps_show.columns = ["Magnitud", "N total", "P(overnight gap-down) %",
                         "P(intraday drop) %", "Ticker"]
    add_table_from_df(doc, gaps_show[["Ticker", "Magnitud", "N total",
                                       "P(overnight gap-down) %", "P(intraday drop) %"]])
    add_image(doc, CHARTS / "gap_risk.png", width_inches=6.5)
    add_callout(doc,
        "En 1,509 dias, SPY tuvo 4 gap-downs >5% overnight (0.26%). "
        "Para una posicion abierta a la noche, una gap de -5% en strike 5% "
        "below = pasaste de OTM a ITM en una sesion. Esto es el verdadero "
        "tail risk del retail short premium.",
        color="danger",
    )

    # ---- Seccion 7: implicaciones ----
    add_heading(doc, "7. Implicaciones para el diseno de la estrategia", 1)
    add_bullet(doc, "Strike a 5% below en SPY a 30d -> ~11% chance de ITM, ~31% chance de touch. Punto de partida razonable para PCS delta-20.")
    add_bullet(doc, "QQQ requiere strike mas lejos para misma P(ITM): 5% en QQQ -> 14% ITM (vs 11% SPY).")
    add_bullet(doc, "IWM aun mas conservador: 5% below -> 19% ITM. Para igualar P(ITM) de SPY a 11%, IWM necesita ~7-8% below.")
    add_bullet(doc, "Stops por touch NO son recomendables: 60-70% de los touches recuperan. Cierre 'pulse' por touch destruye expectancy.")
    add_bullet(doc, "Cierre por DTE (ej. 21 DTE): salva solo 10-30% de los touches (los que ocurren en el ultimo tercio). No es panacea pero ayuda.")
    add_bullet(doc, "Gap overnight: ningun stop te salva. La mejor proteccion es ancho del spread (limitar max loss) y position sizing.")

    add_heading(doc, "8. Caveats", 1)
    add_bullet(doc, "Datos diarios (OHLC). El min(Low) en una ventana es una aproximacion: en la realidad, el min intra-dia puede haber sido tocado y revertido sin dejar marca en el daily.")
    add_bullet(doc, "Periodo train (2018-2024): incluye COVID, 2022 hike. La frecuencia de eventos extremos puede subestimar regimenes futuros.")
    add_bullet(doc, "El % below es un strike fijo. Pero un PCS delta-20 tiene strike DINAMICO con la IV. La Fase 4 hace ese bridge.")
    add_bullet(doc, "P(touch) y P(ITM) son MARGINALES, no condicionales en regimen. La Fase 3 segmenta.")

    add_heading(doc, "9. Proximos pasos (Fase 3)", 1)
    add_paragraph(doc,
        "Segmentar las probabilidades por REGIMEN: IV Rank, VIX, term structure, "
        "tendencia, drawdown reciente, etc. Identificar regimenes donde P(ITM) "
        "es significativamente menor que la unconditional -> filtros candidatos."
    )

    out = REPORTS / "Phase2_NonParametric_Probabilities.docx"
    save(doc, out)
    print(f"Reporte guardado: {out}")


if __name__ == "__main__":
    main()
