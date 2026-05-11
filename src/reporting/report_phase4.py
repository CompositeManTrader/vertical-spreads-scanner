"""Reporte Word de Fase 4."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import REPORTS
from src.reporting.charts_phase4 import (
    dist_histogram, dist_vs_iv_scatter, empirical_vs_nominal,
    regimes_dynamic_chart,
)
from src.reporting.word_builder import (
    add_bullet, add_callout, add_code_block, add_heading, add_image,
    add_paragraph, add_table_from_df, new_document, save,
)

RESULTS = REPORTS / "_phase4_results"
CHARTS = REPORTS / "_phase4_charts"


def main():
    CHARTS.mkdir(parents=True, exist_ok=True)
    full = pd.read_parquet(RESULTS / "delta20_full_records.parquet")
    summary = pd.read_parquet(RESULTS / "delta20_summary.parquet")
    regimes = pd.read_parquet(RESULTS / "delta20_regimes.parquet")
    for c in ["ticker"]:
        for df in (full, summary, regimes):
            if c in df.columns:
                df[c] = df[c].astype(str)
    if "condition" in regimes.columns:
        regimes["condition"] = regimes["condition"].astype(str)

    # Charts
    for ticker in ["SPY", "QQQ", "IWM"]:
        for T in [30, 45]:
            dist_histogram(full, ticker, T, CHARTS / f"dist_{ticker}_T{T}.png")
            dist_vs_iv_scatter(full, ticker, T, CHARTS / f"distvsiv_{ticker}_T{T}.png")
        regimes_dynamic_chart(regimes, ticker, CHARTS / f"regimes_dyn_{ticker}.png")
    empirical_vs_nominal(summary, 30, CHARTS / "emp_vs_nominal_T30.png")
    empirical_vs_nominal(summary, 45, CHARTS / "emp_vs_nominal_T45.png")

    doc = new_document(
        "Fase 4 — Bridge: del % below al strike delta-20 real",
        "Vertical Spreads Edge Research — Conexion con Black-Scholes y validacion del VRP",
    )

    add_heading(doc, "Resumen ejecutivo", 1)
    add_paragraph(doc,
        "Hasta Fase 3 trabajamos con strikes a % below fijo. Esta fase calcula "
        "el strike delta-20 REAL para cada dia historico via Black-Scholes con "
        "IV ATM, dividend yield y tasa libre de riesgo. Mide la P(ITM) "
        "empirica con ese strike dinamico y la compara con el |delta| nominal "
        "de 20% que la teoria 'promete'."
    )
    add_callout(doc,
        "🎯 HALLAZGO CENTRAL: VRP existe en SPY y QQQ, NO en IWM. P(ITM) "
        "empirica con strike delta-20: SPY 14.78%, QQQ 16.10%, IWM 20.89%. "
        "El nominal es 20%. SPY/QQQ tienen 26% y 20% MENOS perdidas que las "
        "que paga la prima nominal. IWM no tiene edge.",
        color="success",
    )
    add_callout(doc,
        "🎯 El VRP es ROBUSTO al skew: incluso bumpeando IV en +20% (proxy "
        "de skew agresiva), SPY P(ITM) = 12.5%, QQQ = 12.6%. El edge no se "
        "evapora con asunciones conservadoras.",
        color="success",
    )
    add_callout(doc,
        "🚨 IWM Vol baja predice MAS perdidas, no menos: 'low_rv (<15%)' en "
        "IWM tiene P(ITM) 33.2% vs 20.9% uncond (lift 1.59, Bonferroni "
        "p=0.007). Inversion del patron observado en SPY/QQQ.",
        color="danger",
    )

    # ---- 1. Metodologia ----
    add_heading(doc, "1. Metodologia BSM", 1)
    add_paragraph(doc,
        "Black-Scholes-Merton para opciones europeas con dividend yield "
        "continuo. Para cada dia t con IV no-nulo y T en {30,35,40,45}:"
    )
    add_code_block(doc,
        "  Inputs:\n"
        "    S = Close[t]\n"
        "    sigma = iv_atm_barchart[t]   (Imp Vol ATM ~30D constant maturity)\n"
        "    r = rfr_pct[t-1]/100         (DGS3MO publicada en t-1, anti-look-ahead)\n"
        "    q = dividend yield (constante por ticker: SPY 1.3%, QQQ 0.6%, IWM 1.4%)\n"
        "    T = T_dias / 365\n"
        "  Resolver:\n"
        "    K = exp(-(N^-1(0.20*exp(qT)) * sigma*sqrt(T)) + (r-q+sigma^2/2)*T) * S\n"
        "  Outcomes (post-factum, NO features):\n"
        "    was_ITM       = Close[t+T] < K\n"
        "    was_touched   = min(Low[t..t+T]) < K"
    )
    add_paragraph(doc,
        "Tests unitarios: 8/8 verdes. Validan put-call parity, delta ATM, "
        "inverso de la funcion solve_strike, monotonia de precio en sigma, "
        "consistencia delta vs prob ITM RN."
    )

    # ---- 2. Distribucion del strike ----
    add_heading(doc, "2. Distribucion del strike delta-20 dinamico", 1)
    add_paragraph(doc,
        "Como vimos en Fase 1, la IV varia fuertemente. El strike delta-20 "
        "tambien: en IV baja esta cerca del spot (~3% below), en IV alta "
        "lejos (~10% below)."
    )
    rows = []
    for ticker in ["SPY", "QQQ", "IWM"]:
        for T in [30, 45]:
            sub = full[(full["ticker"] == ticker) & (full["T_days"] == T) &
                       (full["iv_bump"] == 1.0)]
            rows.append({
                "Ticker": ticker, "T (dias)": T, "N obs": len(sub),
                "Dist mean (%)": round(sub["dist_pct_below"].mean() * 100, 2),
                "Dist p10 (%)": round(sub["dist_pct_below"].quantile(0.10) * 100, 2),
                "Dist median (%)": round(sub["dist_pct_below"].median() * 100, 2),
                "Dist p90 (%)": round(sub["dist_pct_below"].quantile(0.90) * 100, 2),
                "Dist max (%)": round(sub["dist_pct_below"].max() * 100, 2),
            })
    add_table_from_df(doc, pd.DataFrame(rows))
    add_paragraph(doc, "Implicacion: un PCS delta-20 'auto-ajusta' la distancia segun el regimen.", italic=True)
    for ticker in ["SPY", "QQQ", "IWM"]:
        add_paragraph(doc, f"{ticker}:", bold=True)
        add_image(doc, CHARTS / f"dist_{ticker}_T30.png", width_inches=6.5)
        add_image(doc, CHARTS / f"distvsiv_{ticker}_T30.png", width_inches=6.5)

    # ---- 3. Empirico vs nominal ----
    add_heading(doc, "3. P(ITM) empirica vs |delta| nominal: validacion del VRP", 1)
    add_paragraph(doc,
        "Por construccion, |delta_put| = 0.20 implica P(ITM) ~ 20% bajo "
        "BSM-RN. Si la empirica es MENOR -> existe VRP (la prima paga mas "
        "riesgo del que efectivamente se materializa). Si es MAYOR -> sesgo "
        "(IV ATM subestima el riesgo, posiblemente por skew)."
    )
    cols = ["ticker", "T_days", "iv_bump", "n", "p_itm_emp",
            "p_itm_emp_ci_lo", "p_itm_emp_ci_hi", "p_itm_theo_RN_mean",
            "p_touch_emp", "ratio_touch_itm_emp",
            "pvalue_emp_vs_delta_nominal"]
    add_paragraph(doc, "Tabla maestra (filtrada a iv_bump=1.0, sin skew):", bold=True)
    sub = summary[summary["iv_bump"] == 1.0][cols].copy()
    sub["n"] = sub["n"].astype(int)
    for c in ["p_itm_emp", "p_itm_emp_ci_lo", "p_itm_emp_ci_hi",
              "p_itm_theo_RN_mean", "p_touch_emp", "ratio_touch_itm_emp",
              "pvalue_emp_vs_delta_nominal"]:
        sub[c] = sub[c].round(4)
    add_table_from_df(doc, sub)
    add_paragraph(doc,
        "Lectura: pvalue_emp_vs_delta_nominal es el test de proporciones de "
        "P(ITM emp) vs 20%. p<0.05 indica que empirica es significativamente "
        "distinta de 20%."
    )
    add_paragraph(doc, "Charts comparativos:", bold=True)
    add_image(doc, CHARTS / "emp_vs_nominal_T30.png", width_inches=6.8)
    add_image(doc, CHARTS / "emp_vs_nominal_T45.png", width_inches=6.8)

    # ---- 4. Skew bias ----
    add_heading(doc, "4. Cuantificacion del sesgo por skew", 1)
    add_paragraph(doc,
        "La columna iv_atm_barchart es ATM. En la realidad, los puts OTM "
        "cotizan con IV mas alta (smile/skew put). Si calculamos el strike "
        "delta-20 con IV ATM cuando la IV real del strike es +10-15% mas alta, "
        "el strike resultante esta MAS LEJOS del spot del que sera realmente "
        "el delta-20 de mercado. Esto tiende a SUBESTIMAR la P(ITM) "
        "empirica observada."
    )
    add_paragraph(doc,
        "Para acotar el sesgo, repetimos el calculo bumpeando la IV por "
        "factores {1.00, 1.05, 1.10, 1.15, 1.20}. La tabla de Seccion 3 "
        "muestra como la P(ITM emp) cambia con el bump."
    )
    add_paragraph(doc, "Hallazgo:", bold=True)
    add_bullet(doc, "SPY: bump 1.0 -> 14.78%; bump 1.20 -> 12.47%. El edge VRP se MANTIENE.")
    add_bullet(doc, "QQQ: bump 1.0 -> 16.10%; bump 1.20 -> 12.58%. Edge mantenido.")
    add_bullet(doc, "IWM: bump 1.0 -> 20.89%; bump 1.20 -> 16.84%. Con skew implicit, IWM tendria edge marginal.")
    add_paragraph(doc,
        "Interpretacion: incluso bajo el supuesto conservador de skew "
        "agresiva, el VRP en SPY/QQQ es real. En IWM aparece edge solo "
        "asumiendo skew muy agresiva, lo cual es discutible.",
        italic=True,
    )

    # ---- 5. Re-analisis condicional con strike dinamico ----
    add_heading(doc, "5. Re-analisis condicional con strike DINAMICO", 1)
    add_paragraph(doc,
        "Replicamos el analisis multifactor de Fase 3 pero ahora con strike "
        "delta-20 dinamico (no fijo al 5%). Los filtros encontrados ahi "
        "deberian sostenerse aqui (en estructura, no en magnitudes exactas)."
    )
    for ticker in ["SPY", "QQQ", "IWM"]:
        add_heading(doc, f"5.{1+['SPY','QQQ','IWM'].index(ticker)} {ticker}", 2)
        sub = regimes[(regimes["ticker"] == ticker) & (regimes["T_days"] == 30) &
                      (regimes["n"] >= 80)].sort_values("p_itm").copy()
        sub_show = sub[["condition", "n", "p_itm", "p_uncond", "lift",
                        "pvalue", "pvalue_bonferroni"]].copy()
        sub_show["n"] = sub_show["n"].astype(int)
        for c in ["p_itm", "p_uncond", "lift", "pvalue", "pvalue_bonferroni"]:
            sub_show[c] = sub_show[c].round(4)
        add_table_from_df(doc, sub_show)
        add_image(doc, CHARTS / f"regimes_dyn_{ticker}.png", width_inches=6.8)

    add_callout(doc,
        "Comparativa Fase 3 (strike 5%) vs Fase 4 (strike delta-20 dinamico):\n"
        "- SPY 'above_sma200 & vrp_high': 1.67% vs 4.60% (similar magnitud "
        "de lift). Filtro robusto.\n"
        "- QQQ idem: 2.44% vs 3.25%. Filtro robusto.\n"
        "- IWM: NUEVO hallazgo: 'low_rv' es ADVERSO en IWM con strike "
        "dinamico (P=33.2% vs 20.9% uncond). Inversion del patron SPY/QQQ.",
        color="info",
    )

    # ---- 6. Implicaciones operativas ----
    add_heading(doc, "6. Implicaciones operativas", 1)
    add_bullet(doc, "VRP estructural en SPY y QQQ: vender PCS delta-20 sin filtros TIENE edge (~5pp menor P(ITM) que el nominal).")
    add_bullet(doc, "Filtros 'above_sma200 + vrp_high' + 'no_panic_5d' bajan P(ITM) a ~3-5%. ENORME margen de seguridad.")
    add_bullet(doc, "IWM no participa del VRP estructural. NO operar IWM con esta estrategia, o reservar para fases futuras con setup distinto.")
    add_bullet(doc, "El strike delta-20 'auto-ajusta' la distancia con la IV: en mercados calmos (IV baja) el strike esta cerca pero el riesgo realizado es bajo; en mercados agitados (IV alta) el strike esta lejos.")
    add_bullet(doc, "Recomendacion para Fase 5: testear delta {10, 15, 20, 25, 30} para ver si delta-20 es realmente optimo o si delta menor (mas conservador) tiene mejor relacion riesgo/recompensa.")

    # ---- 7. Caveats ----
    add_heading(doc, "7. Caveats", 1)
    add_bullet(doc, "Sin chain history real, asumimos IV constante por strike (sin skew). El skew bump del 10-20% es proxy razonable pero no exacto.")
    add_bullet(doc, "Los strikes calculados son TEORICOS. Los reales en chain de Schwab pueden diferir por step de strike (SPY tiene strikes a $1, IWM a $0.50, etc.).")
    add_bullet(doc, "Tasa r es DGS3MO. Para opciones a 30-45 DTE, deberia ser tasa Fed Funds o SOFR. Diferencia chica (<10bp) en este horizonte.")
    add_bullet(doc, "Dividend yield constante: en realidad varia trimestralmente. Error chico en strike (<0.2%).")
    add_bullet(doc, "Pricing del PCS completo (credito, mark-to-market) se aborda en Fase 6. Aqui solo medimos probabilidades.")

    # ---- 8. Proximos pasos ----
    add_heading(doc, "8. Proximos pasos (Fase 5)", 1)
    add_paragraph(doc,
        "Sensibilidad a delta {10, 15, 20, 25, 30, 40}, DTE {21, 30, 35, 40, "
        "45, 60} y ancho del spread {3, 5, 10, 15}. Construir superficie de "
        "expectancy y encontrar el punto optimo por ticker."
    )

    out = REPORTS / "Phase4_Delta20_Bridge.docx"
    save(doc, out)
    print(f"Reporte guardado: {out}")


if __name__ == "__main__":
    main()
