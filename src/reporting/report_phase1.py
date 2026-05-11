"""
Reporte Word de Fase 1: distribucion empirica de retornos T-dias.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import REPORTS
from src.reporting.word_builder import (
    add_bullet, add_callout, add_code_block, add_heading,
    add_image, add_paragraph, add_table_from_df, new_document, save,
)


RESULTS = REPORTS / "_phase1_results"
CHARTS = REPORTS / "_phase1_charts"


def fmt(df, cols=None, ndecimals=4):
    if cols is not None:
        df = df[cols]
    fmt_df = df.copy()
    for c in fmt_df.select_dtypes("float").columns:
        fmt_df[c] = fmt_df[c].round(ndecimals)
    return fmt_df


def main():
    doc = new_document(
        "Fase 1 — Distribucion empirica de retornos a 30/35/40/45 dias",
        "Vertical Spreads Edge Research — Comportamiento del precio (puro, sin opciones)",
    )

    # ---- Resumen ejecutivo ----
    add_heading(doc, "Resumen ejecutivo", 1)
    add_paragraph(doc,
        "Esta fase caracteriza la distribucion empirica de retornos T-dias "
        "(T en {30,35,40,45}) para SPY, QQQ e IWM en el train set "
        "(2018-10-03 a 2024-10-03, 1,510 dias por ticker). El objetivo es "
        "responder: que tan extrema es la cola izquierda? como se compara con "
        "la asuncion lognormal de Black-Scholes? hay reversion o momentum a "
        "estos horizontes?",
    )
    add_paragraph(doc, "Hallazgos principales:", bold=True)
    add_bullet(doc, "Normalidad RECHAZADA en los 3 tickers con extremismo: Jarque-Bera p<1e-10 en todos los T.")
    add_bullet(doc, "Skewness negativa (-0.6 a -1.7): cola izquierda significativamente mas pesada.")
    add_bullet(doc, "Excess kurtosis positiva: hasta 7 en SPY a 30d (colas extremadamente gordas).")
    add_bullet(doc, "Reversion a 30d confirmada estadisticamente: returns pasados de 60d se revierten en los proximos 30d (Spearman rho = -0.19 SPY, p<1e-12).")
    add_bullet(doc, "Volatility clustering muy fuerte (ACF de squared returns lag 1 = 0.46 SPY, 0.41 QQQ).")
    add_bullet(doc, "Correlacion cross-ETF SUBE en stress: SPY-IWM pasa de 0.87 a 0.91. Diversificar no protege.")
    add_bullet(doc, "Drawdown intra-ventana brutal: el peor 1% de ventanas SPY 30d tiene -28% de DD interno (aunque pueda terminar OK).")

    # ---- 1. Distribucion ----
    add_heading(doc, "1. Distribucion descriptiva de retornos T-dias", 1)
    add_paragraph(doc,
        "Calculamos el log-return a T dias hacia adelante: "
        "ret_fwd(t,T) = ln(P(t+T) / P(t)). Esta es una serie OVERLAPPING "
        "(retornos consecutivos comparten dias), por lo que las observaciones "
        "no son iid; la 'n efectiva' es menor que la 'n bruta'. Esto se "
        "tendra en cuenta en Fase 7 con block bootstrap."
    )

    add_heading(doc, "1.1 Stats descriptivos", 2)
    dist = pd.read_parquet(RESULTS / "distribution_stats.parquet")
    cols = ["ticker", "T", "n", "mean_pct", "median_pct", "std_pct",
            "skew", "excess_kurt", "min_pct", "max_pct", "jb_pvalue"]
    add_table_from_df(doc, fmt(dist[cols]), float_format="{:.3f}")
    add_paragraph(doc,
        "Lectura: la columna 'jb_pvalue' es del test Jarque-Bera (H0: "
        "normalidad). p<<0.05 -> rechazo. En todos los casos rechazamos "
        "normalidad con p efectivamente igual a cero.",
        italic=True,
    )

    add_paragraph(doc, "Implicaciones:", bold=True)
    add_bullet(doc, "Skew negativa: Black-Scholes con vol constante UNDERSTATES la P(perdida grande).")
    add_bullet(doc, "Kurtosis alta: VaR teorico bajo normalidad SUBESTIMA el riesgo de cola.")
    add_bullet(doc, "QQQ tiene la mayor desviacion (8.7% a 45d) pero menor skew y kurtosis -> distribucion mas simetrica.")
    add_bullet(doc, "IWM tiene la mayor desviacion absoluta y skew -1.4 a 30d.")

    add_heading(doc, "1.2 Percentiles, VaR y Expected Shortfall", 2)
    cols2 = ["ticker", "T", "p1_pct", "p5_pct", "p10_pct", "p50_pct", "p90_pct", "p95_pct", "p99_pct",
             "VaR_5pct", "ES_5pct", "VaR_1pct", "ES_1pct"]
    add_table_from_df(doc, fmt(dist[cols2]), float_format="{:.2f}")
    add_paragraph(doc,
        "VaR_X% = -percentil X% (perdida positiva al X% nivel). "
        "ES_X% = perdida promedio condicional a estar en el X% peor. "
        "Ej: VaR_5% de SPY a 30d = 9.1% significa que en el 5% de las "
        "ventanas, la perdida fue >= 9.1%. ES_5% = 13.6% (promedio en ese 5%).",
        italic=True,
    )

    add_heading(doc, "1.3 Comparacion empirico vs lognormal (BSM)", 2)
    cols3 = ["ticker", "T", "VaR_5_emp_pct", "VaR_5_theo_pct", "VaR_5_excess_pct",
             "VaR_1_emp_pct", "VaR_1_theo_pct", "VaR_1_excess_pct", "tail_ratio_p1_p99"]
    add_table_from_df(doc, fmt(dist[cols3]), float_format="{:.2f}")
    add_paragraph(doc,
        "VaR_excess = VaR empirico - VaR teorico (asumiendo lognormal con "
        "misma media y desvio). >0 -> empirico es MAS pesado que el modelo "
        "(BSM subestima el riesgo). tail_ratio_p1_p99 mide la asimetria de "
        "colas: >1 indica cola izquierda mas extrema.",
        italic=True,
    )

    add_paragraph(doc, "Hallazgo notable:", bold=True)
    add_paragraph(doc,
        "SPY y QQQ muestran el patron esperado: VaR empirico > VaR teorico "
        "(BSM subestima). Pero IWM en T=30 tiene VaR_5_excess negativo "
        "(empirico 11.8% vs teorico 14.0%): la sigma estimada incluye eventos "
        "extremos (COVID, 2022) que inflan la varianza, pero el percentil 5 "
        "efectivo no es tan extremo. Esto significa que el modelo lognormal "
        "no es uniformemente sesgado: depende del activo y horizonte. "
        "tail_ratio: SPY e IWM tienen colas izquierdas casi 2x mas extremas "
        "que las derechas."
    )

    add_heading(doc, "1.4 Histogramas y QQ-plots", 2)
    add_paragraph(doc, "SPY 30d (representativo):")
    add_image(doc, CHARTS / "hist_SPY_T30.png", width_inches=6.5)
    add_image(doc, CHARTS / "qq_SPY_T30.png", width_inches=4.5)
    add_paragraph(doc, "QQQ 30d:")
    add_image(doc, CHARTS / "hist_QQQ_T30.png", width_inches=6.5)
    add_image(doc, CHARTS / "qq_QQQ_T30.png", width_inches=4.5)
    add_paragraph(doc, "IWM 30d:")
    add_image(doc, CHARTS / "hist_IWM_T30.png", width_inches=6.5)
    add_image(doc, CHARTS / "qq_IWM_T30.png", width_inches=4.5)
    add_paragraph(doc,
        "Los QQ-plots muestran la firma clasica de retornos financieros: "
        "linealidad razonable en el centro, desviacion fuerte en las colas "
        "(especialmente la izquierda).",
        italic=True,
    )

    # ---- 2. Drawdown intra-ventana ----
    add_heading(doc, "2. Drawdown intra-ventana (path-dependent)", 1)
    add_paragraph(doc,
        "Para cada ventana T-dias, calculamos el max drawdown intra-ventana: "
        "1 - min(Close[t..t+T]) / Close[t]. Esto importa porque, aunque la "
        "ventana TERMINE OK, el spread puede haber sufrido mark-to-market "
        "muy negativo durante el periodo, gatillando stops o presion psicologica."
    )
    dd = pd.read_parquet(RESULTS / "intra_drawdown.parquet")
    add_table_from_df(doc, fmt(dd), float_format="{:.2f}")
    add_paragraph(doc, "Lectura crucial:", bold=True)
    add_bullet(doc, "Mediana de DD a 30d: SPY 2.3%, QQQ 2.8%, IWM 3.8% -> en condiciones tipicas, el DD intra-ventana es chico.")
    add_bullet(doc, "P95 de DD: SPY 12.4%, QQQ 14.6%, IWM 14.7% -> en el 5% de las ventanas, el DD pasa de 12-15%.")
    add_bullet(doc, "P99 de DD: SPY 28.0%, IWM 36.1% -> el peor 1% es brutal, aun si la ventana termina OK.")
    add_bullet(doc, "Implicacion para PCS con stop a 2x credito: los stops se gatillaran frecuentemente en estas ventanas, convirtiendo perdidas que hubieran sido temporales en realizadas.")

    # ---- 3. Peores ventanas historicas ----
    add_heading(doc, "3. Peores ventanas historicas (T=30)", 1)
    add_paragraph(doc,
        "Top 10 peores ret_fwd a 30 dias por ticker. Identifica los eventos "
        "que dominan la cola izquierda."
    )
    ww = pd.read_parquet(RESULTS / "worst_windows.parquet")
    for t in ["SPY", "QQQ", "IWM"]:
        add_paragraph(doc, f"{t}:", bold=True)
        sub = ww[(ww["ticker"] == t) & (ww["T"] == 30)].copy()
        sub["Open Date"] = pd.to_datetime(sub["Open Date"]).dt.date
        sub["Close Date"] = pd.to_datetime(sub["Close Date"]).dt.date
        cols_w = ["Open Date", "Close Date", "Return % (log)"]
        add_table_from_df(doc, fmt(sub[cols_w]), float_format="{:.2f}")
    add_paragraph(doc,
        "Casi todos los peores casos provienen de febrero-marzo 2020 (COVID), "
        "septiembre 2022 (Fed hiking) y abril 2022. Concentracion temporal "
        "alta: la cola izquierda no es uniforme, viene en clusters.",
        italic=True,
    )

    # ---- 4. Yearly extremes ----
    add_heading(doc, "4. Mejor / peor por anio", 1)
    for t in ["SPY", "QQQ", "IWM"]:
        add_paragraph(doc, f"{t} - retornos extremos a 30 dias por anio:", bold=True)
        ye = pd.read_parquet(RESULTS / f"yearly_extremes_{t}.parquet")
        add_table_from_df(doc, fmt(ye), float_format="{:.2f}")
        add_image(doc, CHARTS / f"yearly_{t}_T30.png", width_inches=6.5)

    add_paragraph(doc, "Lectura:", bold=True)
    add_bullet(doc, "2020 (COVID) y 2022 (Fed hike) son los peores aniversarios. 2019 y 2021 son los mejores.")
    add_bullet(doc, "frac_neg_5pct_T30: % de dias del anio donde ret_fwd 30d fue < -5%. En 2020 fue ~30% en SPY, 2022 ~10%. Identifica anios donde abrir PCS era peligroso.")

    # ---- 5. Dependencia temporal ----
    add_heading(doc, "5. Dependencia temporal: autocorrelacion y vol clustering", 1)
    add_paragraph(doc,
        "Bajo random walk puro (asuncion BSM), los returns deberian ser "
        "iid -> autocorrelacion ~ 0 en todos los lags. Si vemos autocorrelacion "
        "significativa, hay informacion explotable o dependencia que el modelo "
        "ignora."
    )

    add_heading(doc, "5.1 Autocorrelacion (returns y squared returns)", 2)
    ac = pd.read_parquet(RESULTS / "autocorrelation.parquet")
    add_table_from_df(doc, fmt(ac[["ticker", "lag", "acf_returns", "acf_squared_returns"]]),
                      float_format="{:.4f}")
    add_paragraph(doc, "Hallazgos:", bold=True)
    add_bullet(doc, "Lag 1 returns: -0.14 SPY, -0.13 QQQ, -0.07 IWM -> reversion a 1 dia (mean reverting daily).")
    add_bullet(doc, "Lag 2 returns: positivo (~0.05-0.10) -> patron complejo de corto plazo.")
    add_bullet(doc, "Squared returns lag 1: 0.46 SPY, 0.41 QQQ, 0.26 IWM -> volatility clustering MUY fuerte.")
    add_bullet(doc, "Squared returns persiste: aun a lag 20 sigue siendo 0.10. Vol es predecible (en cierta medida).")

    add_heading(doc, "5.2 Test Ljung-Box", 2)
    lb = pd.read_parquet(RESULTS / "ljung_box.parquet")
    add_table_from_df(doc, fmt(lb), float_format="{:.4f}")
    add_paragraph(doc,
        "H0: no autocorrelacion hasta el lag k. p<0.05 -> rechazo. En "
        "TODOS los casos rechazamos H0 con p<<0.001 -> autocorrelacion "
        "significativa tanto en returns como en squared returns.",
        italic=True,
    )

    add_paragraph(doc, "Charts de ACF:", bold=True)
    for t in ["SPY", "QQQ", "IWM"]:
        add_paragraph(doc, f"{t}:")
        add_image(doc, CHARTS / f"acf_{t}.png", width_inches=6.5)

    add_heading(doc, "5.3 Test de momentum vs reversion", 2)
    mr = pd.read_parquet(RESULTS / "momentum_reversion.parquet")
    add_table_from_df(doc, fmt(mr), float_format="{:.4f}")
    add_callout(doc,
        "TODAS las combinaciones (lookback, fwd) en los 3 tickers muestran "
        "REVERSION (rho < 0). El efecto mas fuerte: lookback=60d, fwd=30d "
        "-> SPY rho=-0.19 (p<1e-12), QQQ rho=-0.17, IWM rho=-0.13. "
        "Implicacion: returns de los ultimos 60d se asocian negativamente "
        "con los proximos 30d. Abrir PCS despues de drawdowns recientes "
        "puede tener edge.",
        color="success",
    )

    # ---- 6. Cross-ETF correlation ----
    add_heading(doc, "6. Correlacion cross-ETF y comportamiento en stress", 1)

    add_heading(doc, "6.1 Correlaciones unconditional", 2)
    cd = pd.read_parquet(RESULTS / "corr_daily.parquet")
    c30 = pd.read_parquet(RESULTS / "corr_30d.parquet")
    add_paragraph(doc, "Returns diarios:", bold=True)
    add_table_from_df(doc, cd.reset_index(names="Ticker"), float_format="{:.4f}")
    add_paragraph(doc, "Returns 30 dias hacia adelante:", bold=True)
    add_table_from_df(doc, c30.reset_index(names="Ticker"), float_format="{:.4f}")

    add_heading(doc, "6.2 Correlacion en stress (SPY <= -2% diario)", 2)
    cs = pd.read_parquet(RESULTS / "corr_stress.parquet")
    sm = pd.read_parquet(RESULTS / "stress_meta.parquet")
    add_paragraph(doc, f"Trigger: {sm['trigger'].iloc[0]}. n_stress = {int(sm['n_stress'].iloc[0])} de {int(sm['n_total'].iloc[0])} dias.")
    add_table_from_df(doc, cs.reset_index(names="Ticker"), float_format="{:.4f}")

    add_callout(doc,
        "En stress: SPY-QQQ baja levemente (0.93 -> 0.88), pero SPY-IWM SUBE "
        "(0.87 -> 0.91). Las small caps colapsan junto con las large caps en "
        "los peores dias. Diversificar PCS entre los 3 ETFs NO reduce el "
        "riesgo de cola. Capacidad total debe asumir correlation = 1 en stress.",
        color="warning",
    )

    add_paragraph(doc, "Charts de correlacion:", bold=True)
    add_image(doc, CHARTS / "corr_daily.png", width_inches=4.5)
    add_image(doc, CHARTS / "corr_30d.png", width_inches=4.5)
    add_image(doc, CHARTS / "corr_stress.png", width_inches=4.5)

    # ---- Caveats ----
    add_heading(doc, "7. Caveats de Fase 1", 1)
    add_bullet(doc, "Returns OVERLAPPING: las ventanas T-dias consecutivas comparten datos. La n bruta sobreestima la informacion independiente. CIs honestos requieren block bootstrap (Fase 7).")
    add_bullet(doc, "Periodo train (2018-2026) incluye COVID y 2022 hike, pero NO la GFC 2008. Si el regimen futuro se parece a 2008, los stats de cola estan subestimados.")
    add_bullet(doc, "Las stats no son condicionales: agregan TODOS los dias. La Fase 3 segmenta por regimen para encontrar donde la cola es realmente peligrosa.")
    add_bullet(doc, "Reversion estadisticamente significativa NO implica explotabilidad: el rho es bajo (~0.15) y p<<0.05 puede provenir de muestras grandes con efecto chico. Validar en regimenes (Fase 3) y P&L (Fase 6).")

    # ---- Implicaciones para la estrategia ----
    add_heading(doc, "8. Implicaciones para la estrategia PCS", 1)
    add_bullet(doc, "BSM con vol constante subestima la cola izquierda en SPY/QQQ. Esperar mas perdidas que las que da el modelo.")
    add_bullet(doc, "Vol clustering: despues de un dia de vol alta, los siguientes tendran vol alta. Esperar drawdowns intra-trade despues de spikes.")
    add_bullet(doc, "Reversion a 30d: filtrar entradas con drawdown de 60d previo puede mejorar el edge (validamos en Fase 3).")
    add_bullet(doc, "Diversificar entre los 3 no reduce riesgo de cola: dimensionar como si fuera un solo trade en stress.")
    add_bullet(doc, "Stops a 2x credito gatillaran en el ~5% de ventanas (DD intra > 12% en SPY). Considerar stops mas amplios o cierre por delta/touch.")

    # ---- Proximos pasos ----
    add_heading(doc, "9. Proximos pasos (Fase 2)", 1)
    add_paragraph(doc,
        "Construir grilla no-parametrica de P(close ITM) y P(touch) para "
        "(ticker, T, x% below). Esto responde DIRECTAMENTE: 'si pongo el "
        "strike X% debajo del spot, cuantas veces termina ITM?'. Sin asumir "
        "ningun modelo de opciones."
    )

    out = REPORTS / "Phase1_Price_Distribution.docx"
    save(doc, out)
    print(f"Reporte guardado: {out}")


if __name__ == "__main__":
    main()
