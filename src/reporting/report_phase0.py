"""
Reporte Word de Fase 0: Setup, ETL y validaciones anti-look-ahead.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import (
    DATA_CLEAN, DATA_EXTERNAL, DATA_END, IV_AVAILABLE_FROM,
    REPORTS, TEST_START, TRAIN_END,
)
from src.etl.build_panel import diagnose_iv_rank_lookahead
from src.reporting.word_builder import (
    add_bullet, add_callout, add_code_block, add_heading, add_numbered,
    add_paragraph, add_table_from_df, new_document, save,
)


def main():
    doc = new_document(
        "Fase 0 — Setup, ETL y Validaciones Anti-Look-Ahead",
        "Vertical Spreads Edge Research — Reporte de Infraestructura",
    )

    # ---- Resumen ejecutivo ----
    add_heading(doc, "Resumen ejecutivo", 1)
    add_paragraph(doc,
        "Esta fase establece la infraestructura de datos y el codigo de honor "
        "anti-look-ahead que se respeta en todas las fases siguientes. Se cargan "
        "los datasets historicos de SPY/QQQ/IWM (2000-2026), VIX/VIX3M y master "
        "de volatilidad, se baja la tasa libre de riesgo DGS3MO de FRED, se "
        "construyen calendarios macro (FOMC, CPI, NFP, earnings season) y se "
        "valida que ningun indicador derivado use informacion futura."
    )
    add_paragraph(doc,
        "HALLAZGO CRITICO: el IV Rank reportado por Barchart NO coincide con "
        "el calculo canonico (252 dias point-in-time). La correlacion es "
        "0.83-0.88 (no ~1) y el max diff llega a 0.86 en escala [0,1]. Por eso "
        "usamos NUESTRO recalculo en todo el research.",
        bold=True,
    )
    add_callout(doc,
        "Decision: el IV Rank de Barchart se DESCARTA. Se usa iv_rank_252 "
        "calculado por nosotros con ventana cerrada en t.",
        color="danger",
    )

    # ---- Stack tecnologico ----
    add_heading(doc, "1. Stack tecnologico", 1)
    pkgs_df = pd.DataFrame([
        {"Paquete": "Python",                     "Version": "3.13.12"},
        {"Paquete": "pandas",                     "Version": "3.0.2"},
        {"Paquete": "numpy",                      "Version": "2.4.4"},
        {"Paquete": "scipy",                      "Version": "1.17.1"},
        {"Paquete": "statsmodels",                "Version": "0.14.6"},
        {"Paquete": "pandas-market-calendars",    "Version": "5.3.2"},
        {"Paquete": "matplotlib / seaborn",       "Version": "3.10.9 / 0.13.2"},
        {"Paquete": "python-docx",                "Version": "1.2.0"},
        {"Paquete": "pyarrow (parquet)",          "Version": "24.0.0"},
        {"Paquete": "openpyxl (xlsx)",            "Version": "3.1.5"},
    ])
    add_table_from_df(doc, pkgs_df)

    add_paragraph(doc,
        "Entorno: miniconda Python 3.13 en C:\\Users\\Windows\\miniconda3. "
        "Tests con pytest 9.0.3."
    )

    # ---- Datasets ----
    add_heading(doc, "2. Datasets cargados", 1)

    add_heading(doc, "2.1 ETFs (Barchart .xlsx)", 2)
    rows = []
    for t in ["SPY", "QQQ", "IWM"]:
        df = pd.read_parquet(DATA_CLEAN / f"{t}.parquet")
        rows.append({
            "Ticker": t,
            "Filas": len(df),
            "Desde": df["Date"].min().date(),
            "Hasta": df["Date"].max().date(),
            "IV no-nulos": df["iv_atm_barchart"].notna().sum(),
            "IV invalidos limpiados": (df["Date"].notna()).sum() - df["iv_atm_barchart"].notna().sum(),
        })
    add_table_from_df(doc, pd.DataFrame(rows), int_format="{:,}")

    add_paragraph(doc,
        "11 filas con IV exactamente igual a 0 (datos rotos de Barchart en "
        "fechas puntuales: 2021-09-29, 2021-10-04, 2022-02-17, 2024-02-02, "
        "2024-02-05, 2024-02-09, 2025-12-01) fueron seteadas a NaN. La fila "
        "se mantiene; solo iv_atm_barchart se invalida."
    )

    add_heading(doc, "2.2 Master de volatilidad", 2)
    add_bullet(doc, "VIX OHLC y close: 2000-01 -> 2026-03 (cobertura 100%)")
    add_bullet(doc, "VIX3M close: desde 2009-01 (term structure real)")
    add_bullet(doc, "VVIX close: desde 2012-03 (vol of vol)")
    add_bullet(doc, "Contango_pct, VIX/VIX3M ratio: pre-calculados en master, copiados al panel")
    add_bullet(doc, "M1/M2 futuros VIX: precio y DTE")
    add_bullet(doc, "VXX OHLC: desde 2018-01 (cubre nuestra ventana IV-based)")

    add_heading(doc, "2.3 Tasa libre de riesgo (DGS3MO de FRED)", 2)
    rates = pd.read_parquet(DATA_EXTERNAL / "rates_dgs3mo.parquet")
    add_paragraph(doc,
        f"Bajado desde el endpoint publico de FRED. {len(rates):,} filas, "
        f"desde {rates['Date'].min().date()} hasta {rates['Date'].max().date()}. "
        f"Rango: min={rates['DGS3MO_pct'].min():.2f}%, max={rates['DGS3MO_pct'].max():.2f}%, "
        f"ultimo={rates['DGS3MO_pct'].iloc[-1]:.2f}%."
    )
    add_callout(doc,
        "Anti-look-ahead: la tasa para decision en t es DGS3MO publicada en t-1 "
        "(FRED publica con un dia de delay). Implementado mediante shift de "
        "+1 dia antes del merge.",
        color="info",
    )

    add_heading(doc, "2.4 Calendarios macro", 2)
    cals = []
    for f, label in [("calendar_fomc.parquet", "FOMC meetings"),
                     ("calendar_cpi.parquet", "CPI releases"),
                     ("calendar_nfp.parquet", "NFP releases"),
                     ("calendar_earnings_season.parquet", "Earnings season days")]:
        df = pd.read_parquet(DATA_EXTERNAL / f)
        cals.append({"Calendario": label, "Filas": len(df),
                     "Desde": df["Date"].min().date(), "Hasta": df["Date"].max().date()})
    add_table_from_df(doc, pd.DataFrame(cals), int_format="{:,}")
    add_paragraph(doc,
        "Todos los calendarios son CONOCIDOS EX-ANTE: las fechas de FOMC se "
        "anuncian con 12 meses de anticipacion; CPI y NFP siguen calendarios "
        "publicados por BLS al inicio del ano. Earnings season clusters son "
        "ventanas predecibles (~enero, abril, julio, octubre). No introducen "
        "look-ahead bias."
    )

    # ---- IV Rank validation ----
    add_heading(doc, "3. Validacion del IV Rank de Barchart", 1)
    add_paragraph(doc,
        "El IV Rank canonico se define como:"
    )
    add_code_block(doc,
        "IV_Rank(t) = (IV(t) - min(IV[t-251:t+1])) /\n"
        "             (max(IV[t-251:t+1]) - min(IV[t-251:t+1]))"
    )
    add_paragraph(doc,
        "Es decir, posicion de IV(t) entre el min y max de los ultimos 252 dias "
        "(inclusivo en t, sin tocar t+1). Si Barchart hubiera recalculado "
        "retroactivamente con la historia completa al momento del export "
        "(2026-03), el IV Rank de fechas pasadas estaria contaminado con "
        "informacion futura.",
    )

    add_paragraph(doc,
        "Comparamos su IV Rank vs nuestro calculo canonico:"
    )

    diag = []
    for t in ["SPY", "QQQ", "IWM"]:
        df = pd.read_parquet(DATA_CLEAN / f"{t}.parquet")
        d = diagnose_iv_rank_lookahead(df, t)
        diag.append(d)
    diag_df = pd.DataFrame(diag)
    diag_df.columns = ["Ticker", "N obs", "Correlacion", "MAE", "Max |diff|", "Bias medio"]
    add_table_from_df(doc, diag_df, float_format="{:.4f}", int_format="{:,}")

    add_paragraph(doc, "Interpretacion:", bold=True)
    add_bullet(doc, "Correlacion 0.83-0.88: muy lejos de 1.0. Barchart NO replica el canonico.")
    add_bullet(doc, "MAE 0.05-0.07 sobre escala [0,1]: divergencias de ~5-7 puntos en promedio.")
    add_bullet(doc, "Max |diff| ~0.86: en algun punto difieren casi completamente.")
    add_bullet(doc, "Bias positivo (~0.04): Barchart tiende a reportar IV Rank ligeramente mas alto.")

    add_callout(doc,
        "Conclusion: el IV Rank de Barchart usa una metodologia distinta "
        "(posiblemente ventana 1 anio calendario, normalizacion proporcional, "
        "o calibracion propia). Para garantizar zero look-ahead, en TODAS las "
        "fases siguientes se usa NUESTRO iv_rank_252 (calculado con rolling "
        "cerrado en t).",
        color="danger",
    )

    # ---- Train/test split ----
    add_heading(doc, "4. Train / Test split sellado", 1)
    add_paragraph(doc,
        f"Train: {IV_AVAILABLE_FROM} a {TRAIN_END} (~6 anios). Test (holdout): "
        f"{TEST_START} a {DATA_END} (~17 meses)."
    )
    add_callout(doc,
        "El test queda SELLADO: ningun analisis de exploracion, fitting, o "
        "tuneo de parametros lo toca. Una sola pasada al final (Fase 7).",
        color="warning",
    )

    splits = []
    for t in ["SPY", "QQQ", "IWM"]:
        train = pd.read_parquet(DATA_CLEAN / "train" / f"{t}.parquet")
        test = pd.read_parquet(DATA_CLEAN / "test" / f"{t}.parquet")
        splits.append({
            "Ticker": t,
            "Train rows": len(train),
            "Train first": train["Date"].min().date(),
            "Train last": train["Date"].max().date(),
            "Test rows": len(test),
            "Test first": test["Date"].min().date(),
            "Test last": test["Date"].max().date(),
        })
    add_table_from_df(doc, pd.DataFrame(splits), int_format="{:,}")

    # ---- Tests anti-look-ahead ----
    add_heading(doc, "5. Tests unitarios anti-look-ahead", 1)
    add_paragraph(doc,
        "Filosofia: para todo indicador f(serie), se exige que "
        "f(serie[:t+1])[t] == f(serie)[t] (point-in-time consistency). Si f "
        "usa info futura, los valores divergiran al truncar la serie."
    )
    add_paragraph(doc, "20 tests, 20 verdes. Cobertura:")
    add_bullet(doc, "iv_rank_252, iv_percentile_252: ventana cerrada en t.")
    add_bullet(doc, "realized_vol: rolling cerrado.")
    add_bullet(doc, "drawdown_from_high: expanding y rolling.")
    add_bullet(doc, "days_since_ath: usa expanding max (no max global).")
    add_bullet(doc, "sma, sma_slope: rolling cerrado.")
    add_bullet(doc, "log_return_back vs log_return_forward: separacion explicita features vs labels.")
    add_bullet(doc, "Panel integrity: fechas ascendentes, sin duplicados, IV en rango valido, rfr presente.")
    add_bullet(doc, "Train/test no-overlap y sentinel del holdout existente.")

    # ---- Caveats ----
    add_heading(doc, "6. Caveats reconocidos", 1)
    add_numbered(doc, "Sin chain history: creditos teoricos, no historicos.")
    add_numbered(doc, "Sin skew historica: la columna Imp Vol es ATM. La skew se aproxima en Fase 4.")
    add_numbered(doc, "Solo 7.5 anos con IV: ~1,510 dias de train (~252 dias/ano de trading).")
    add_numbered(doc, "Regimen 2018-2026: post-QE, COVID, alta inflacion. No incluye GFC ni dot-com.")
    add_numbered(doc, "Survivorship: nulo para SPY/QQQ/IWM (indices liquidos).")
    add_numbered(doc, "Datos diarios: no podemos modelar stops intra-dia exactos.")

    # ---- Proximos pasos ----
    add_heading(doc, "7. Proximos pasos (Fase 1)", 1)
    add_paragraph(doc,
        "Caracterizar la distribucion empirica de retornos a 30/35/40/45 dias "
        "para los 3 ETFs (puro precio, sin opciones todavia)."
    )
    add_bullet(doc, "Stats descriptivos: media, mediana, desvio, skewness, kurtosis.")
    add_bullet(doc, "Test Jarque-Bera de normalidad.")
    add_bullet(doc, "Percentiles, VaR empirico, ES.")
    add_bullet(doc, "Peores 10 ventanas historicas y analisis de drawdown intra-trade.")
    add_bullet(doc, "Autocorrelacion de retornos y volatility clustering (Ljung-Box).")
    add_bullet(doc, "Correlacion cross-ETF y comparacion con lognormal (BSM assumption).")

    out = REPORTS / "Phase0_Setup_and_Validation.docx"
    save(doc, out)
    print(f"Reporte guardado: {out}")


if __name__ == "__main__":
    main()
