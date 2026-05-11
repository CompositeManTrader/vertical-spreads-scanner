"""Reporte Word de Fase 3."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import REPORTS
from src.reporting.charts_phase3 import multifactor_chart, regime_bar
from src.reporting.word_builder import (
    add_bullet, add_callout, add_heading, add_image, add_paragraph,
    add_table_from_df, new_document, save,
)

RESULTS = REPORTS / "_phase3_results"
CHARTS = REPORTS / "_phase3_charts"


def main():
    CHARTS.mkdir(parents=True, exist_ok=True)
    uv = pd.read_parquet(RESULTS / "regime_univariate.parquet")
    mf = pd.read_parquet(RESULTS / "regime_multifactor.parquet")
    for c in ["ticker", "regime", "bucket", "outcome", "condition"]:
        if c in uv.columns:
            uv[c] = uv[c].astype(str)
        if c in mf.columns:
            mf[c] = mf[c].astype(str)

    # Bonferroni en univariado tambien
    uv["pvalue_bonferroni"] = (uv["pvalue"] * len(uv)).clip(upper=1.0)

    # ----------- Generar charts -----------
    KEY_REGIMES = [
        "iv_rank_252", "VIX_Close", "term_structure_ratio", "vrp",
        "price_to_sma200", "dd_60d", "days_since_ath", "rv_20d",
    ]
    for ticker in ["SPY", "QQQ", "IWM"]:
        for regime in KEY_REGIMES:
            regime_bar(uv, regime, ticker, 30, 0.05,
                       CHARTS / f"regime_{ticker}_{regime}_T30_x5.png")
        multifactor_chart(mf, ticker, 30, 0.05,
                          CHARTS / f"multifactor_{ticker}_T30_x5.png")

    # ----------- Documento -----------
    doc = new_document(
        "Fase 3 — Analisis condicional por regimen",
        "Vertical Spreads Edge Research — Donde existe el edge estadistico",
    )

    add_heading(doc, "Resumen ejecutivo", 1)
    add_paragraph(doc,
        "Esta es la fase mas importante del research. Segmentamos el train "
        "set (2018-10-03 a 2024-10-03, 1,510 dias) por 30 regimenes distintos "
        "y medimos como cambia P(ITM) condicionalmente. Combinaciones "
        "multifactor predefinidas EX-ANTE para evitar p-hacking."
    )
    add_paragraph(doc, "Hallazgos principales:", bold=True)
    add_callout(doc,
        "🎯 Filtro mas robusto (SPY/QQQ): 'above SMA200 + VRP en quintil top'. "
        "Reduce P(ITM 5% T=30) de 11-14% a 1.7-2.4% (lift 0.15-0.17, "
        "Bonferroni-significativo p<0.01). Edge real validado.",
        color="success",
    )
    add_callout(doc,
        "🚨 IWM: NINGUNA combinacion supera Bonferroni. El edge no es robusto "
        "para small caps en los filtros probados. Implicacion: la estrategia "
        "PCS sobre IWM requiere setup distinto, o no operar.",
        color="danger",
    )
    add_bullet(doc, "Filtros UNIVARIADOS validados: iv_rank_252 < 20%, low_rv, vrp_high, above_sma200, days_since_ath > 180.")
    add_bullet(doc, "Filtros ADVERSOS validados: below_sma200, dd_60d > 5%, vrp_low, drawdown alto, rv_20d 15-20%.")
    add_bullet(doc, "Hallazgo CONTRAINTUITIVO: 'IV Rank low' (mercado calmo) tiene MENOR P(ITM), no mayor. La sabiduria 'vender en IV alto' es matizada: lo que importa es VRP (IV vs RV), no IV absoluto.")

    # ---- 1. Setup metodologico ----
    add_heading(doc, "1. Metodologia", 1)
    add_paragraph(doc,
        "Pasos: (1) construimos features de regimen point-in-time (todos sin "
        "look-ahead). (2) bucketizamos por feature (quintiles globales para "
        "exploracion + thresholds absolutos para reglas operativas). (3) "
        "para cada bucket, calculamos P(ITM) con strike fijo a 5% y 7% "
        "below, en T=30 y T=45 dias. (4) Comparamos contra unconditional "
        "via z-test de proporciones. (5) Aplicamos Bonferroni para "
        "multiplicidad de tests."
    )
    add_bullet(doc, "Anti-look-ahead: features calculadas con rolling cerrado en t.")
    add_bullet(doc, "Anti-overfitting: combinaciones definidas EX-ANTE, no buscadas a posteriori.")
    add_bullet(doc, "Bonferroni: con N tests, threshold de significancia = 0.05 / N.")
    add_bullet(doc, "Min n por bucket: 50-100 para reportarse como confiable.")

    # ---- 2. Univariado ----
    add_heading(doc, "2. Analisis univariado por regimen", 1)
    add_paragraph(doc,
        "Para cada regimen, mostramos los buckets ordenados por menor P(ITM) "
        "(mejor) y mayor (peor). Significancia Bonferroni con escala n_tests "
        "= 2,536 (univariado completo)."
    )

    for ticker in ["SPY", "QQQ", "IWM"]:
        add_heading(doc, f"2.{1+['SPY','QQQ','IWM'].index(ticker)} {ticker}", 2)

        sub_itm = uv[(uv["ticker"] == ticker) & (uv["T"] == 30) &
                     (uv["x_pct_below"] == 0.05) & (uv["outcome"] == "itm") &
                     (uv["n"] >= 100)].copy()
        if sub_itm.empty:
            continue

        # Top buckets MEJORES (menor P)
        add_paragraph(doc, "Top 10 buckets con MENOR P(ITM) (n>=100, T=30, x=5%):", bold=True)
        best = sub_itm.nsmallest(10, "p_outcome")[
            ["regime", "bucket", "n", "p_outcome", "p_uncond", "lift", "pvalue", "pvalue_bonferroni"]
        ].copy()
        best["n"] = best["n"].astype(int)
        for c in ["p_outcome", "p_uncond", "lift", "pvalue", "pvalue_bonferroni"]:
            best[c] = best[c].round(4)
        add_table_from_df(doc, best)

        add_paragraph(doc, "Top 10 buckets con MAYOR P(ITM) (a EVITAR):", bold=True)
        worst = sub_itm.nlargest(10, "p_outcome")[
            ["regime", "bucket", "n", "p_outcome", "p_uncond", "lift", "pvalue", "pvalue_bonferroni"]
        ].copy()
        worst["n"] = worst["n"].astype(int)
        for c in ["p_outcome", "p_uncond", "lift", "pvalue", "pvalue_bonferroni"]:
            worst[c] = worst[c].round(4)
        add_table_from_df(doc, worst)

        # Charts de regimes clave
        add_paragraph(doc, "Charts por regimen clave:", bold=True)
        for reg in ["iv_rank_252", "vrp", "price_to_sma200", "dd_60d"]:
            chart = CHARTS / f"regime_{ticker}_{reg}_T30_x5.png"
            if chart.exists():
                add_image(doc, chart, width_inches=6.5)

    # ---- 3. Multifactor ----
    add_heading(doc, "3. Combinaciones multifactor", 1)
    add_paragraph(doc,
        "Combinaciones AND de 2-3 features definidas EX-ANTE. La idea: si dos "
        "features son ambos buenos predictores de P(ITM) baja, su interseccion "
        "deberia ser AUN mejor."
    )
    add_paragraph(doc,
        "Las condiciones probadas son fijas (no busqueda a posteriori): "
        "above_sma200, iv_rank_low, vrp_high, low_rv, no_panic_5d, "
        "strong_uptrend, contango, low_vix, far_from_ath, y combinaciones "
        "AND de hasta 3 de ellas. Tambien condiciones ADVERSAS para "
        "validacion: below_sma200, in_panic_5d, vrp_low.",
        italic=True,
    )

    for ticker in ["SPY", "QQQ", "IWM"]:
        add_heading(doc, f"3.{1+['SPY','QQQ','IWM'].index(ticker)} {ticker}", 2)
        sub_mf = mf[(mf["ticker"] == ticker) & (mf["T"] == 30) &
                    (mf["x_pct_below"] == 0.05) & (mf["n"] >= 80)].copy()
        sub_mf = sub_mf.sort_values("p_outcome")
        sub_show = sub_mf[["condition", "n", "p_outcome", "p_uncond", "lift",
                            "pvalue", "pvalue_bonferroni"]].copy()
        sub_show["n"] = sub_show["n"].astype(int)
        for c in ["p_outcome", "p_uncond", "lift"]:
            sub_show[c] = sub_show[c].round(4)
        for c in ["pvalue", "pvalue_bonferroni"]:
            sub_show[c] = sub_show[c].round(4)
        add_table_from_df(doc, sub_show)
        chart = CHARTS / f"multifactor_{ticker}_T30_x5.png"
        if chart.exists():
            add_image(doc, chart, width_inches=6.8)

    # ---- 4. Veredicto por ticker ----
    add_heading(doc, "4. Veredicto por ticker", 1)

    add_heading(doc, "4.1 SPY", 2)
    add_paragraph(doc,
        "Filtros operativos sugeridos para SPY PCS 30-DTE 5% below:"
    )
    add_bullet(doc, "Filtro PRIMARIO recomendado: 'above SMA200 AND VRP in top quintile'. P(ITM) = 1.67% (vs 10.95% uncond). n=239 dias. Bonferroni-significativo. Lift 0.15.")
    add_bullet(doc, "Filtro alternativo simple (mas dias operables): 'IV Rank < 20%'. P(ITM) = 2.93%, n=580.")
    add_bullet(doc, "Filtro robusto (compromiso n vs edge): 'low_rv (RV20d<15%)'. P(ITM) = 5.08%, n=788.")
    add_bullet(doc, "EVITAR: below_sma200 (P(ITM) = 17.4%, lift 1.6).")

    add_heading(doc, "4.2 QQQ", 2)
    add_paragraph(doc,
        "Edge presente y robusto:"
    )
    add_bullet(doc, "Filtro PRIMARIO: 'above SMA200 AND VRP in top quintile'. P(ITM) = 2.44% (vs 14.39%). Bonferroni p<0.0001.")
    add_bullet(doc, "Filtro alternativo: 'iv_rank_low (<20)'. P(ITM) = 4.55%, n=440.")
    add_bullet(doc, "EVITAR: dd_high & below_sma200 (P(ITM) = 28.6%, lift 2.0).")

    add_heading(doc, "4.3 IWM", 2)
    add_callout(doc,
        "IWM no muestra ningun filtro robusto al test Bonferroni. Las "
        "combinaciones que reducen P(ITM) en SPY/QQQ no replican en small "
        "caps. Posibles explicaciones: (a) base mas volatil borra el edge; "
        "(b) factores que mueven IWM son distintos (sensibilidad a tasas, "
        "ciclos sectoriales); (c) sample size insuficiente.",
        color="warning",
    )
    add_bullet(doc, "Recomendacion: NO operar PCS sobre IWM con la estrategia disenada para SPY/QQQ.")
    add_bullet(doc, "Alternativa: investigar Phase 4+ con strikes mas alejados (delta-15 o delta-10) y/o filtros especificos a small caps.")

    # ---- 5. Hallazgos contraintuitivos ----
    add_heading(doc, "5. Hallazgos contraintuitivos", 1)
    add_callout(doc,
        "MITO: 'Vender prima cuando IV es alto'. REALIDAD: IV Rank ALTO (>60) "
        "no muestra ventaja consistente. IV Rank BAJO (<20) tiene MENOR P(ITM). "
        "Lo que importa NO es IV absoluto, sino el VRP (IV - RV): cuando "
        "IV ATM > RV realizada, vender prima paga. Cuando IV es alta porque "
        "RV es alta, no hay edge.",
        color="info",
    )
    add_callout(doc,
        "MITO: 'En panic_5d_window el premium esta inflado, vender'. "
        "REALIDAD: in_panic_5d en SPY no muestra ventaja (P(ITM)=10.9%, lift=1.0). "
        "En QQQ es ADVERSO (P(ITM)=17.7%). Vender despues de un panic day "
        "es atrapar el cuchillo.",
        color="info",
    )
    add_callout(doc,
        "VALIDACION: 'price > SMA200' como filtro alcista funciona en SPY/QQQ "
        "(reduce P(ITM) ~30%). Es el filtro mas simple y replicable.",
        color="success",
    )

    # ---- 6. Caveats ----
    add_heading(doc, "6. Caveats y limitaciones", 1)
    add_bullet(doc, "Sample = 1,510 dias de train. Filtros con n<100 son ruido.")
    add_bullet(doc, "Quintiles globales tienen leak (usan stats del train completo). Para reglas operativas usamos buckets ABSOLUTOS (thresholds), no quintiles.")
    add_bullet(doc, "Bonferroni es conservador (controla family-wise error). Filtros con pvalue~0.01 pero pvalue_bonferroni>0.05 NO se aceptan.")
    add_bullet(doc, "Combinaciones multifactor reducen n drasticamente: above_sma200 & vrp_high & no_panic_5d tiene n=210 (de 1,510). Suficiente para CIs razonables, justo para decisiones de capital.")
    add_bullet(doc, "Validacion final del filtro en HOLDOUT (Fase 7). Si se cae alli, no es robusto.")
    add_bullet(doc, "Filtros encontrados son CORRELACIONALES, no causales. Pueden romperse si cambia el regimen.")

    # ---- 7. Proximos pasos ----
    add_heading(doc, "7. Proximos pasos (Fase 4)", 1)
    add_paragraph(doc,
        "Conectar el analisis con el strike delta-20 real. Hasta ahora usamos "
        "strikes a 5% y 7% below (fijos). Pero un PCS delta-20 tiene strike "
        "DINAMICO con la IV (en IV baja el delta-20 esta cerca, en IV alta "
        "lejos). Phase 4 hace ese bridge."
    )

    out = REPORTS / "Phase3_Regime_Analysis.docx"
    save(doc, out)
    print(f"Reporte guardado: {out}")


if __name__ == "__main__":
    main()
