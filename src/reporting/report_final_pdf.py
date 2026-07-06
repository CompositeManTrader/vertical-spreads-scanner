# -*- coding: utf-8 -*-
"""
Genera el reporte PDF final completo del research de vertical spreads.
Incluye: hallazgos de todas las fases, backtest hold-to-expiry, reglas
de entrada/salida, tests de estres, caveats y guia del notebook Colab.
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

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

RESULTS = ROOT / "reports" / "_backtest_holdexpiry"
CHARTS = ROOT / "reports" / "_final_pdf_charts"
OUT_PDF = ROOT / "reports" / "REPORTE_FINAL_Vertical_Spreads.pdf"

GREEN = colors.HexColor("#1f6f3d")
DARKBLUE = colors.HexColor("#1a3a5c")
LIGHTGREY = colors.HexColor("#f0f2f5")
RED = colors.HexColor("#a33")


# ============================================================================
# CHARTS
# ============================================================================

def make_charts():
    CHARTS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3,
    })

    tr = pd.read_parquet(RESULTS / "trades_haircut087.parquet")
    tr["strategy"] = tr["strategy"].astype(str)
    tr["expiry_date"] = pd.to_datetime(tr["expiry_date"])
    tr["open_date"] = pd.to_datetime(tr["open_date"])

    # --- Equity curves (PnL realizado en expiry) ---
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for strat, color, label in [("E1_BASELINE", "#888888", "E1 Baseline (sin filtros)"),
                                 ("E2_VRP_HARVEST", "#1f6f3d", "E2 VRP Harvest (filtrada)")]:
        sub = tr[tr["strategy"] == strat].sort_values("expiry_date")
        ax.plot(sub["expiry_date"], sub["pnl"].cumsum(), lw=1.6, color=color, label=label)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("PnL acumulado realizado (USD, 1 contrato)")
    ax.set_title("Curvas de equity — PnL realizado al vencimiento (haircut 0.87)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(CHARTS / "equity.png"); plt.close(fig)

    # --- PnL por anio ---
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    tr["year"] = tr["open_date"].dt.year
    piv = tr.groupby(["year", "strategy"])["pnl"].sum().unstack()
    piv = piv.rename(columns={"E1_BASELINE": "E1 Baseline",
                               "E2_VRP_HARVEST": "E2 VRP Harvest",
                               "E3_PULLBACK": "E3 Pullback"})
    piv.plot(kind="bar", ax=ax, color=["#888888", "#1f6f3d", "#c9a227"], width=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("PnL (USD)")
    ax.set_xlabel("")
    ax.set_title("PnL por año de apertura")
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(CHARTS / "yearly.png"); plt.close(fig)

    # --- Estres ---
    s = pd.read_parquet(RESULTS / "summary.parquet")
    s["strategy"] = s["strategy"].astype(str)
    s["segment"] = s["segment"].astype(str)
    full = s[s["segment"] == "FULL"]
    scenarios = [
        ("Base\n(0.87)", 0.87, 0.0), ("Haircut\n0.80", 0.80, 0.0),
        ("0.87 +\nskew 0.5pt", 0.87, 0.005), ("0.87 +\nskew 1pt", 0.87, 0.01),
        ("PEOR CASO\n0.80 + skew 1pt", 0.80, 0.01),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    x = np.arange(len(scenarios))
    w = 0.38
    for i, (strat, color, label) in enumerate([
            ("E1_BASELINE", "#888888", "E1 Baseline"),
            ("E2_VRP_HARVEST", "#1f6f3d", "E2 VRP Harvest")]):
        vals = []
        for _, h, sk in scenarios:
            row = full[(full["strategy"] == strat) & (full["haircut"] == h) &
                       (full["skew_bump"] == sk)]
            vals.append(row["expectancy"].iloc[0] if len(row) else np.nan)
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=color, label=label, alpha=0.9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (2 if v >= 0 else -6),
                    f"${v:.0f}", ha="center", fontsize=8,
                    fontweight="bold" if v > 0 else "normal")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _, _ in scenarios], fontsize=8)
    ax.set_ylabel("Expectancy USD / trade")
    ax.set_title("Test de estrés del pricing: expectancy por escenario (mismos trades)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "stress.png"); plt.close(fig)


# ============================================================================
# PDF
# ============================================================================

def build_pdf():
    styles = getSampleStyleSheet()
    st_title = ParagraphStyle("T", parent=styles["Title"], fontSize=22, textColor=DARKBLUE, spaceAfter=6)
    st_sub = ParagraphStyle("S", parent=styles["Normal"], fontSize=12, textColor=colors.grey, alignment=TA_CENTER)
    st_h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, textColor=DARKBLUE, spaceBefore=16, spaceAfter=8)
    st_h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12.5, textColor=GREEN, spaceBefore=12, spaceAfter=6)
    st_body = ParagraphStyle("B", parent=styles["Normal"], fontSize=9.8, leading=13.5, alignment=TA_JUSTIFY, spaceAfter=6)
    st_bullet = ParagraphStyle("BU", parent=st_body, leftIndent=16, bulletIndent=6, spaceAfter=3)
    st_code = ParagraphStyle("C", parent=styles["Code"], fontSize=8, leading=10.5,
                              backColor=LIGHTGREY, borderPadding=6, spaceAfter=8, spaceBefore=4)
    st_note = ParagraphStyle("N", parent=st_body, fontSize=8.8, textColor=colors.grey)
    st_alert = ParagraphStyle("A", parent=st_body, textColor=RED, fontName="Helvetica-Bold")
    st_ok = ParagraphStyle("OK", parent=st_body, textColor=GREEN, fontName="Helvetica-Bold")

    def table(data, colw=None, fs=8.2, header_bg=DARKBLUE):
        t = Table(data, colWidths=colw, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), header_bg),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), fs),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHTGREY]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ]))
        return t

    def B(text):
        return Paragraph(f"• {text}", st_bullet)

    story = []
    W = 6.9 * inch

    # ================= PORTADA =================
    story.append(Spacer(1, 1.1 * inch))
    story.append(Paragraph("Vertical Spreads con Edge Estadístico", st_title))
    story.append(Paragraph("Research completo, backtest confiable y reglas operativas", st_sub))
    story.append(Spacer(1, 0.35 * inch))
    story.append(Paragraph("Put Credit Spreads sobre SPY y QQQ — Estrategia E2 «VRP Harvest»",
                            ParagraphStyle("pc", parent=st_sub, fontSize=13, textColor=GREEN)))
    story.append(Spacer(1, 0.5 * inch))
    cover = [
        ["Período analizado", "oct-2018 a mar-2026 (7.4 años, 1,863 días por ticker)"],
        ["Trades backtesteados", "837 (hold-to-expiry, fidelidad exacta al terminal)"],
        ["Estrategia recomendada", "E2 VRP Harvest: 159 trades, 88.1% win, $66/trade, PF 2.7"],
        ["Validación", "Holdout purgado + estrés de pricing + review adversarial (16 agentes)"],
        ["Veredicto", "GO con E2. E1 baseline NO sobrevive el peor caso de pricing. E3 descartada."],
    ]
    t = table([[Paragraph(f"<b>{a}</b>", st_body), Paragraph(b, st_body)] for a, b in cover],
              colw=[1.9 * inch, 5.0 * inch], fs=9.5, header_bg=colors.white)
    t.setStyle(TableStyle([("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.white)]))
    story.append(t)
    story.append(Spacer(1, 0.9 * inch))
    story.append(Paragraph("Este documento NO constituye asesoramiento financiero. Uso personal.", st_note))
    story.append(PageBreak())

    # ================= 1. RESUMEN EJECUTIVO =================
    story.append(Paragraph("1. Resumen ejecutivo", st_h1))
    story.append(Paragraph(
        "Este research partió de una pregunta simple: <b>¿vender put credit spreads (PCS) sobre índices "
        "tiene edge estadístico real, y bajo qué condiciones?</b> Tras 8 fases de análisis sobre 25 años de "
        "precios y 7.4 años de volatilidad implícita, más un backtest final auditado adversarialmente, la "
        "respuesta es: <b>sí, pero solo en SPY y QQQ, y solo cuando la prima está objetivamente cara "
        "respecto del riesgo que después se materializa</b> (Variance Risk Premium alto).", st_body))
    story.append(Paragraph(
        "La estrategia ganadora, <b>E2 «VRP Harvest»</b>, entra una vez por semana como máximo, solo cuando "
        "(a) la volatilidad implícita supera a la realizada por un margen amplio y (b) la estructura temporal "
        "del VIX no está invertida. Vende un PCS delta-30 a 45 días con ancho $5 y lo mantiene hasta el "
        "vencimiento, sin stop loss. En el backtest: <b>88.1% de aciertos, $66 de ganancia esperada por "
        "trade neta de costos, profit factor 2.7, y max drawdown realizado de $2,056</b> — casi 6 veces "
        "menor que operar sin filtros.", st_body))
    story.append(Paragraph(
        "Igual de importante: el backtest fue diseñado para que puedas confiar en él. El payoff de cada trade "
        "se calcula de forma EXACTA con el precio de cierre del día de vencimiento (sin modelos intermedios), "
        "el único componente modelado (el crédito de entrada) fue estresado hasta su peor caso, y el código "
        "fue auditado por un panel adversarial de 16 agentes que encontró y corrigió 8 defectos antes de "
        "producir los números finales.", st_body))

    story.append(Paragraph("1.1 La tabla que resume todo", st_h2))
    data = [["Estrategia", "N", "Win %", "$/trade", "PnL total", "PF", "Max DD", "¿Sobrevive peor caso?"],
            ["E1 Baseline (sin filtros)", "651", "81.9%", "$39", "$25,493", "1.6", "$11,750", "NO (-$7/trade)"],
            ["E2 VRP Harvest", "159", "88.1%", "$66", "$10,538", "2.7", "$2,056", "SÍ (+$22/trade)"],
            ["E3 Pullback", "27", "77.8%", "$17", "$445", "1.2", "$755", "NO (n insuficiente)"]]
    story.append(table(data, colw=[1.55*inch, 0.45*inch, 0.55*inch, 0.6*inch, 0.75*inch, 0.4*inch, 0.7*inch, 1.55*inch]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "«Peor caso» = crédito real 20% menor que el modelo BSM + 1 punto de vol de skew entre las patas "
        "del spread, aplicado a los MISMOS trades. Es la prueba ácida: si la estrategia solo gana cuando el "
        "modelo de precios es optimista, no tiene edge real. E2 la pasa; E1 no.", st_note))
    story.append(PageBreak())

    # ================= 2. METODOLOGIA DE CONFIABILIDAD =================
    story.append(Paragraph("2. Por qué puedes confiar en este backtest", st_h1))
    story.append(Paragraph("2.1 El principio de fidelidad exacta al terminal", st_h2))
    story.append(Paragraph(
        "El problema clásico de backtestear opciones sin datos históricos de la cadena es que hay que "
        "modelar los precios de las opciones, y el modelo puede mentir. Este backtest lo resuelve con una "
        "decisión de diseño: <b>todas las estrategias mantienen el spread hasta el vencimiento</b>. Al "
        "vencimiento, el valor del spread NO depende de ningún modelo: es aritmética pura.", st_body))
    story.append(Paragraph(
        "PnL = crédito_neto − 100 × max(0, min(K_short − S_vencimiento, ancho))", st_code))
    story.append(Paragraph(
        "Donde S_vencimiento es el precio de cierre real del ETF ese día (dato observado, no modelado). "
        "Lo ÚNICO modelado es el crédito cobrado al entrar: se calcula con Black-Scholes y se multiplica "
        "por un «haircut» de 0.87 calibrado contra una orden real simulada en Schwab (el modelo daba ~$118 "
        "para un spread que el mercado pagaba $103). Además se descuentan $2.80 de comisiones por trade.", st_body))

    story.append(Paragraph("2.2 Los cinco blindajes anti-autoengaño", st_h2))
    for txt in [
        "<b>Parámetros pre-registrados:</b> delta -0.30, DTE 45, ancho $5, y las reglas de los filtros se "
        "fijaron ANTES de correr el backtest, basados en el research previo. Cero re-ajuste después de ver resultados.",
        "<b>Split train/holdout purgado:</b> los datos se dividieron en entrenamiento (hasta oct-2024) y "
        "validación (oct-2024 a mar-2026). Los trades que cruzan la frontera se excluyen de ambos segmentos "
        "para que no haya contaminación de información.",
        "<b>Sensibilidad des-contaminada:</b> la decisión de entrar usa siempre el haircut base; los escenarios "
        "de estrés solo revalúan el crédito de los MISMOS trades. Así el escenario 0.80 es una cota inferior "
        "genuina (un error clásico es que cada escenario opere trades distintos, lo que invalida la comparación).",
        "<b>Estrés de skew:</b> el modelo BSM usa una sola IV para ambas patas, pero en la realidad la pata "
        "larga (más OTM) cotiza con IV más alta, reduciendo el crédito. Se estresó con +0.5 y +1 punto de vol.",
        "<b>Auditoría adversarial:</b> 3 revisores independientes (look-ahead, corrección de payoff, honestidad "
        "estadística) más un verificador escéptico por hallazgo. 10 hallazgos, 8 confirmados, todos corregidos. "
        "Los 2 refutados se descartaron con evidencia.",
    ]:
        story.append(B(txt))

    story.append(Paragraph("2.3 Anti-look-ahead", st_h2))
    story.append(Paragraph(
        "Toda señal evaluada el día t usa exclusivamente información disponible al cierre de t: medias "
        "móviles y percentiles con ventanas cerradas en t, tasa libre de riesgo publicada el día anterior, "
        "dividend yield vigente en el año del trade (no el de fin de muestra). El proyecto incluye 20 tests "
        "unitarios automáticos que verifican la propiedad point-in-time de cada indicador: para todo t, "
        "f(serie[:t+1])[t] == f(serie)[t]. Si un indicador usara el futuro, el test falla.", st_body))
    story.append(PageBreak())

    # ================= 3. HALLAZGOS DEL RESEARCH =================
    story.append(Paragraph("3. Hallazgos del research (lo que aprendimos antes del backtest)", st_h1))
    story.append(Paragraph(
        "El backtest final no salió de la nada: destila 8 fases de análisis previo. Estos son los hallazgos "
        "que definieron las reglas, cada uno con su evidencia.", st_body))

    story.append(Paragraph("3.1 El VRP existe en SPY y QQQ — no en IWM", st_h2))
    story.append(Paragraph(
        "El hallazgo central. Un put delta-20 «promete» por construcción ~20% de probabilidad de terminar "
        "ITM. Medimos la frecuencia real en 6 años de datos: <b>SPY 14.8%, QQQ 16.1%, IWM 20.9%</b>. En SPY "
        "y QQQ el mercado paga sistemáticamente más prima de la que el riesgo justifica (los compradores de "
        "protección — institucionales, productos estructurados — son insensibles al precio). Ese exceso es "
        "el Variance Risk Premium y es la fuente estructural del edge. IWM no lo tiene: vender prima ahí es "
        "apostar sin ventaja. <b>Regla derivada: operar solo SPY y QQQ.</b>", st_body))

    story.append(Paragraph("3.2 El stop loss destruye el edge (validado con 90 configuraciones)", st_h2))
    story.append(Paragraph(
        "El hallazgo más contraintuitivo. Simulamos stop loss a 2x el crédito: el win rate cae de 86% a 71% "
        "y la expectancy se vuelve NEGATIVA. La razón está en el comportamiento del precio: el 60-70% de las "
        "veces que el subyacente toca el strike corto, recupera antes del vencimiento. El stop convierte "
        "sustos temporales en pérdidas realizadas, y sistemáticamente corta trades que iban a ganar. "
        "<b>Regla derivada: sin stop loss; el riesgo se controla con el ancho del spread (pérdida máxima "
        "definida) y el position sizing.</b>", st_body))
    data = [["Regla de gestión", "Win rate", "Expectancy", "Veredicto"],
            ["Hold to expiry", "85.8%", "+$20", "Aceptable"],
            ["Take profit 50% (sin SL)", "94.3%", "+$27", "El mejor con gestión activa"],
            ["Stop loss 2x (sin TP)", "70.8%", "-$3", "DESTRUYE el edge"],
            ["TP 50% + SL 2x", "83.8%", "+$10", "El SL arruina al TP"]]
    story.append(table(data, colw=[2.2*inch, 1.1*inch, 1.2*inch, 2.4*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Nota: el TP 50% mejora los resultados pero requiere modelar los precios intra-trade (proxy BSM), "
        "por lo que el backtest final usa hold-to-expiry (verificable sin modelo). El TP 50% queda como "
        "mejora opcional para el trading en vivo, donde los precios son reales.", st_note))

    story.append(Paragraph("3.3 Delta-20 no es el óptimo: los costos ahogan el crédito chico", st_h2))
    story.append(Paragraph(
        "La sabiduría retail dice «vendé delta-20». Con costos reales de Schwab ($2.80 por trade), el "
        "crédito de un delta-20 (~$85/contrato) pierde 3.3% en fricción, y su relación crédito/pérdida "
        "máxima (~17%) es pobre. El delta-30 cobra ~$120-150 con mejor relación (~30-42%) y resultó "
        "consistentemente superior en Sharpe y expectancy en la grilla de 432 combinaciones que simulamos "
        "(deltas 10-40, DTE 21-60, anchos $3-15). <b>Regla derivada: delta-30, no delta-20.</b> Tu propia "
        "orden simulada en Schwab (SPY 721/716, crédito $103) era de hecho un delta-30.", st_body))

    story.append(Paragraph("3.4 DTE 45-60 supera a 21-30", st_h2))
    story.append(Paragraph(
        "En la misma grilla, el patrón fue monotónico: a mayor DTE, mayor win rate y mayor expectancy por "
        "trade. DTE 45 es el compromiso práctico entre captura de theta y rotación de capital. "
        "<b>Regla derivada: DTE 45.</b>", st_body))

    story.append(Paragraph("3.5 El precio revierte a 30-60 días y las correlaciones suben en pánico", st_h2))
    story.append(Paragraph(
        "Dos hallazgos del análisis de precio puro (25 años): (1) los retornos de 60 días anticorrelacionan "
        "con los de los 30 siguientes (Spearman -0.13 a -0.19, p&lt;1e-5) — el mercado sobre-reacciona y "
        "revierte; (2) la correlación SPY-IWM sube de 0.87 a 0.91 en días de pánico — diversificar entre "
        "índices NO protege la cola. <b>Reglas derivadas: el sizing debe asumir que todas las posiciones "
        "pierden juntas en un crash; máximo 2-3 spreads concurrentes.</b>", st_body))

    story.append(Paragraph("3.6 Colas gordas: Black-Scholes subestima el riesgo", st_h2))
    story.append(Paragraph(
        "Los retornos de 30-45 días rechazan la normalidad con contundencia (Jarque-Bera p≈0, skew -0.6 a "
        "-1.7, curtosis hasta 7). El peor 1% de ventanas de 30 días tuvo caídas internas de hasta -28% en "
        "SPY. Esto refuerza dos decisiones: pérdida máxima SIEMPRE definida (nunca puts desnudos) y "
        "distancia del strike que se auto-ajusta con la IV (el delta-30 se aleja solo cuando sube la vol).", st_body))
    story.append(PageBreak())

    # ================= 4. LAS TRES ESTRATEGIAS =================
    story.append(Paragraph("4. Las tres estrategias backtesteadas", st_h1))
    story.append(Paragraph(
        "Las tres comparten la misma estructura de trade (PCS delta-30, DTE 45, ancho $5, hold-to-expiry, "
        "crédito mínimo $55 neto) y difieren solo en CUÁNDO entran. Esto aísla el valor de cada filtro: "
        "E1 es el control científico; E2 y E3 deben ganarle para justificar sus parámetros extra.", st_body))

    # --- E1 ---
    story.append(Paragraph("4.1 E1 — Baseline incondicional (el control)", st_h2))
    story.append(Paragraph(
        "<b>Entrada:</b> el primer día hábil de CADA semana, en SPY y en QQQ, sin mirar ninguna condición "
        "de mercado. Es la apuesta pura al VRP incondicional: «vender prima siempre paga».", st_body))
    story.append(Paragraph(
        "<b>Resultado:</b> rentable en el caso base ($39/trade, PF 1.6) pero con drawdown alto ($11,750, "
        "concentrado en 2022: -$7,251 ese año) y — lo decisivo — <b>su expectancy cae a $0 con skew de 1 "
        "punto y a -$7 en el peor caso de pricing</b>. Su edge vive dentro de la banda de error del modelo "
        "de crédito.", st_body))
    story.append(Paragraph(
        "Veredicto: NO OPERABLE tal cual. Su función fue de control — y la cumplió: demostró que el filtro "
        "de E2 agrega valor real, no ruido.", st_alert))

    # --- E2 ---
    story.append(Paragraph("4.2 E2 — VRP Harvest (LA RECOMENDADA)", st_h2))
    story.append(Paragraph(
        "<b>Tesis:</b> vender prima solo cuando está objetivamente cara respecto del riesgo realizado "
        "reciente, y nunca cuando el mercado de volatilidad está en modo pánico.", st_body))
    story.append(Paragraph("<b>Reglas de entrada</b> (evaluadas al cierre del primer día hábil de cada semana, por ticker):", st_body))
    for txt in [
        "<b>Condición 1 — Prima cara:</b> VRP = IV_ATM_30d − RV_20d ≥ percentil 60 de su propio rolling de "
        "252 días. IV_ATM es la volatilidad implícita at-the-money a ~30 días; RV_20d es la volatilidad "
        "realizada anualizada de los últimos 20 días de retornos. Cuando la implícita excede a la realizada "
        "por un margen que históricamente estuvo en el 40% superior, la prima está «gorda».",
        "<b>Condición 2 — Término no invertido:</b> VIX3M ≥ VIX. Cuando el VIX spot supera al de 3 meses "
        "(backwardation), el mercado está en pánico agudo y el riesgo de gap domina. Este filtro apagó la "
        "estrategia casi por completo en 2022 (solo 6 trades, todos ganadores) mientras el baseline perdía $7,251.",
        "<b>Condición 3 — Crédito suficiente:</b> el crédito neto estimado debe superar $55 por contrato "
        "(11% del ancho). Si la prima no paga, el trade no se toma.",
    ]:
        story.append(B(txt))
    story.append(Paragraph("<b>El trade</b> (idéntico en las tres estrategias):", st_body))
    for txt in [
        "Vender 1 put con strike en la grilla de $1 más cercano al delta -0.30 (resuelto con Black-Scholes "
        "usando la IV ATM del día, la tasa T-bill 3M y el dividend yield del año).",
        "Comprar 1 put $5 más abajo (define la pérdida máxima: $500 − crédito).",
        "Vencimiento: el primer día de negociación ≥ 45 días calendario desde la entrada.",
    ]:
        story.append(B(txt))
    story.append(Paragraph("<b>Reglas de salida:</b>", st_body))
    for txt in [
        "Mantener hasta el vencimiento. Sin stop loss (destruye el edge — sección 3.2). Sin touch stop.",
        "Opcional para vivo (no incluido en el backtest): orden GTC de take profit al 50% del crédito. El "
        "research previo indica que mejora el win rate a ~94% y reduce el drawdown ~65%.",
        "Sizing: pérdida máxima del spread (~$400) ≤ 2% del capital. Máximo 2-3 spreads abiertos en total "
        "(SPY y QQQ colapsan juntos en crashes).",
    ]:
        story.append(B(txt))

    data = [["Métrica E2", "FULL (7.4 años)", "TRAIN (purgado)", "HOLDOUT (nunca visto)"],
            ["Trades", "159", "126", "33"],
            ["Win rate", "88.1%", "88.9%", "84.8%"],
            ["Expectancy/trade", "$66.3", "$69.2", "$55.0"],
            ["Profit factor", "2.7", "2.9", "2.1"],
            ["Max DD realizado", "$2,056", "$1,970", "$923"],
            ["Peor trade", "-$402", "-$402", "-$378"],
            ["Rachas de pérdidas máx.", "5", "5", "3"]]
    story.append(table(data, colw=[1.7*inch, 1.7*inch, 1.7*inch, 1.8*inch]))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "El holdout (oct-2024 → mar-2026) confirma la generalización: $55/trade con PF 2.1, sin re-ajuste "
        "alguno de parámetros. Frecuencia esperada: ~25 trades/año → ~$1,600/año por contrato en el caso base.", st_ok))

    # --- E3 ---
    story.append(Paragraph("4.3 E3 — Pullback Premium Harvest (descartada por ahora)", st_h2))
    story.append(Paragraph(
        "<b>Tesis:</b> entrar tras un susto (drawdown 5-10% desde el máximo de 60 días) dentro de un uptrend "
        "intacto (precio &gt; SMA200) con la IV inflada (≥1.1x su media de 60 días), explotando la reversión "
        "documentada y el VRP post-pánico. <b>Resultado:</b> solo 27 señales en 7.4 años, PF 1.2, holdout "
        "negativo (n=5), y expectancy negativa bajo estrés de skew. La idea económica sigue siendo plausible, "
        "pero no hay evidencia suficiente para operarla. Queda archivada hasta tener más datos o una "
        "definición de señal menos restrictiva.", st_body))
    story.append(PageBreak())

    # ================= 5. RESULTADOS COMPLETOS =================
    story.append(Paragraph("5. Resultados completos del backtest", st_h1))

    story.append(Paragraph("5.1 Curvas de equity", st_h2))
    story.append(Image(str(CHARTS / "equity.png"), width=W, height=W * 4.2 / 8.5))
    story.append(Paragraph(
        "E1 (gris) acumula más PnL total por operar 4x más seguido, pero con el drawdown de 2022 a cuestas. "
        "E2 (verde) tiene una curva mucho más limpia: menos trades, mejor calidad. Nota: es PnL REALIZADO "
        "al vencimiento; el drawdown mark-to-market intra-trade que sentirás en vivo es mayor.", st_note))

    story.append(Paragraph("5.2 PnL por año", st_h2))
    story.append(Image(str(CHARTS / "yearly.png"), width=W, height=W * 3.8 / 8.5))
    story.append(Paragraph(
        "El año que separa a las estrategias es 2022: E1 perdió $7,251 vendiendo prima contra un bear "
        "market; E2 casi no operó (el filtro VIX3M≥VIX la mantuvo afuera) y cerró positiva. En 2024 E2 tuvo "
        "su único año negativo (-$814, 12 trades) — honestidad del dato: ningún filtro es perfecto.", st_note))

    story.append(Paragraph("5.3 El test de estrés del pricing (la prueba ácida)", st_h2))
    story.append(Image(str(CHARTS / "stress.png"), width=W, height=W * 4.0 / 8.5))
    story.append(Paragraph(
        "Cada barra revalúa el crédito de los MISMOS trades bajo un supuesto de pricing más adverso. E1 "
        "colapsa a $0 con skew de 1 punto y es negativa en el peor caso. E2 conserva +$22/trade (PF 1.5) "
        "incluso en el escenario más castigado — y ese escenario es doblemente conservador, porque el "
        "haircut 0.87 ya incorpora el skew de la orden real contra la que se calibró.", st_note))

    story.append(Paragraph("5.4 Peores trades de E2 (validación de que son eventos reales)", st_h2))
    data = [["Ticker", "Apertura", "Strike corto", "S al vencimiento", "PnL"],
            ["QQQ", "2020-01-13", "215", "203.90 (crash COVID)", "-$402"],
            ["SPY", "2020-01-13", "322", "297.51 (crash COVID)", "-$397"],
            ["SPY", "2020-01-21", "325", "297.46 (crash COVID)", "-$397"],
            ["SPY", "2023-08-14", "440", "428.52 (corrección ago-23)", "-$384"],
            ["QQQ", "2024-06-17", "475", "459.66 (rotación jun-24)", "-$382"]]
    story.append(table(data, colw=[0.7*inch, 1.1*inch, 1.2*inch, 2.5*inch, 0.9*inch]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Todos los peores trades corresponden a eventos de mercado conocidos, no a artefactos del modelo. "
        "Las entradas de enero 2020 son legítimas: ningún filtro point-in-time podía ver el COVID venir.", st_note))
    story.append(PageBreak())

    # ================= 6. CAVEATS =================
    story.append(Paragraph("6. Limitaciones que debes tener presentes", st_h1))
    for txt in [
        "<b>El crédito sigue siendo una estimación.</b> El haircut 0.87 se calibró con UNA orden real. La "
        "acción de mayor valor pendiente: recolectar 5-10 cotizaciones simuladas en Schwab (especialmente en "
        "días de VIX alto) para recalibrar. El estrés a 0.80 acota el riesgo, pero más puntos de calibración = más confianza.",
        "<b>Los trades se solapan.</b> Con cadencia semanal y DTE 45 hay hasta ~14 posiciones concurrentes "
        "en rachas. Los 159 trades de E2 no son 159 observaciones independientes; la significancia "
        "estadística efectiva es menor que la nominal.",
        "<b>El drawdown reportado es sobre PnL realizado.</b> El mark-to-market intra-trade será peor en "
        "vivo. Prepárate psicológicamente para ver spreads en rojo que terminan verdes (el 60-70% de los "
        "touches recuperan).",
        "<b>SPY y QQQ se eligieron con información de la muestra</b> (el research mostró VRP ahí). El "
        "holdout mitiga este sesgo de selección pero no lo elimina del todo.",
        "<b>La historia de IV cubre solo 2018-2026.</b> Los únicos stress tests reales dentro de muestra son "
        "COVID y 2022. Un régimen tipo 2008 o una década lateral no están representados.",
        "<b>El vencimiento del backtest es aproximado</b> (primer día hábil ≥ 45 días; no necesariamente un "
        "vencimiento listado). Con los vencimientos diarios/semanales actuales de SPY/QQQ el error es de ±1-2 días.",
        "<b>Riesgo de asignación temprana no modelado:</b> puts cortos muy ITM cerca de ex-dividendo pueden "
        "asignarse antes. Con spread de riesgo definido el impacto es acotado, pero existe.",
    ]:
        story.append(B(txt))

    # ================= 7. CHECKLIST OPERATIVO =================
    story.append(Paragraph("7. Checklist operativo de E2 (para imprimir)", st_h1))
    data = [["Paso", "Acción"],
            ["1. Cadencia", "Cada primer día hábil de la semana, después del cierre, evaluar SPY y QQQ."],
            ["2. Filtro VRP", "Calcular VRP = IV_ATM_30d − RV_20d. ¿Está en el 40% superior de su último año? Si no, NO operar."],
            ["3. Filtro término", "¿VIX3M ≥ VIX? Si el VIX spot está por encima (backwardation), NO operar."],
            ["4. Armar el spread", "Short put ≈ delta -0.30, long put $5 abajo, vencimiento ~45 días."],
            ["5. Validar crédito", "¿Crédito neto ≥ $55 por contrato? Si no, pasar."],
            ["6. Sizing", "Pérdida máx. del spread ≤ 2% del capital. Máx. 2-3 spreads abiertos en total."],
            ["7. Gestión", "Mantener a vencimiento. SIN stop loss. Opcional: TP GTC al 50% del crédito."],
            ["8. Registro", "Anotar crédito real vs. modelado (recalibra el haircut con el tiempo)."]]
    story.append(table(data, colw=[1.3*inch, 5.6*inch], fs=9))

    # ================= 8. COLAB =================
    story.append(Paragraph("8. Cómo correr el código en Google Colab", st_h1))
    story.append(Paragraph(
        "Todo el backtest está reproducido en un notebook auto-contenido que corre en la nube, sin instalar "
        "nada en tu máquina. El notebook descarga los precios de Yahoo Finance y usa VIX/VXN como proxy de "
        "la IV (calibrado con la relación medida en el research: IV_ATM ≈ 0.85 × VIX). Incluye una celda "
        "opcional para subir tus archivos de Barchart y reproducir el backtest exacto.", st_body))
    story.append(Paragraph("Pasos:", st_body))
    for i, txt in enumerate([
        "Abrir el link directo: <b>colab.research.google.com/github/CompositeManTrader/"
        "vertical-spreads-scanner/blob/main/notebooks/vrp_harvest_backtest.ipynb</b> (requiere estar "
        "logueado en una cuenta Google).",
        "En el menú: <b>Entorno de ejecución → Ejecutar todas</b> (Runtime → Run all). La primera celda "
        "instala yfinance (~30 segundos).",
        "Esperar ~2-3 minutos. El notebook descarga datos, corre las 3 estrategias con los 8 escenarios de "
        "estrés y muestra las tablas y gráficos.",
        "(Opcional) Para reproducir el backtest EXACTO con tu IV de Barchart: ejecutar la última sección y "
        "subir tus archivos SPY DAILY.xlsx y QQQ DAILY.xlsx cuando lo pida.",
        "Para experimentar: cambiar los parámetros de la celda «PARÁMETROS» (delta, DTE, percentil del VRP) "
        "y re-ejecutar desde ahí. ADVERTENCIA: si encuentras parámetros «mejores» probándolos contra toda la "
        "muestra, eso es overfitting — cualquier cambio debería re-validarse contra el holdout.",
    ], start=1):
        story.append(B(f"<b>Paso {i}.</b> {txt}"))

    story.append(Paragraph("8.1 Estructura del notebook", st_h2))
    data = [["Sección", "Contenido"],
            ["1. Setup", "Instalación de yfinance e imports"],
            ["2. Parámetros", "Todos los parámetros pre-registrados, editables en un solo lugar"],
            ["3. Datos", "Descarga SPY, QQQ, ^VIX, ^VXN, ^VIX3M desde Yahoo Finance"],
            ["4. Black-Scholes", "put_price y solve_put_strike_for_delta (el módulo de pricing)"],
            ["5. Features", "RV20, VRP, percentil rolling, SMA200 — todo point-in-time"],
            ["6. Motor", "build_trade + señales E1/E2/E3 + loop del backtest"],
            ["7. Resultados", "Tablas por estrategia/segmento + los 8 escenarios de estrés"],
            ["8. Gráficos", "Equity curves, PnL anual, sensibilidad"],
            ["9. Opcional", "Subir xlsx de Barchart para reproducción exacta"]]
    story.append(table(data, colw=[1.5*inch, 5.4*inch], fs=9))

    # ================= 9. ROADMAP =================
    story.append(Paragraph("9. Hoja de ruta recomendada", st_h1))
    for txt in [
        "<b>Semanas 1-8 — Paper trading:</b> operar E2 en simulado en Schwab. Registrar cada cotización real "
        "vs. el crédito modelado para recalibrar el haircut con datos propios.",
        "<b>Meses 3-4 — Capital mínimo:</b> 1 contrato por señal. Validar fills, slippage y la mecánica del "
        "vencimiento. La pérdida máxima por trade es ~$400: dimensionar la cuenta en consecuencia (≥$20k para el 2%).",
        "<b>Mes 6+ — Escalar:</b> subir tamaño solo con track record propio positivo y manteniendo la regla del 2%.",
        "<b>Anual — Re-validar:</b> re-correr el backtest con datos actualizados. Si el VRP de SPY/QQQ se "
        "comprime estructuralmente (P(ITM) empírica ≈ nominal), el edge se acabó y hay que parar.",
        "<b>Pendientes de investigación:</b> extender el backtest a 2000-2018 con VIX como proxy de IV "
        "(incluiría la GFC), la familia «vol-spike fade» (entrar cuando el término VUELVE a contango tras "
        "un pánico — el punto ciego actual), y re-testear E3 con señales menos restrictivas.",
    ]:
        story.append(B(txt))

    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "Repositorio con todo el código: github.com/CompositeManTrader/vertical-spreads-scanner — "
        "Backtest: src/backtest/holdexpiry_backtest.py — Notebook: notebooks/vrp_harvest_backtest.ipynb", st_note))

    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=letter,
                             topMargin=0.75 * inch, bottomMargin=0.75 * inch,
                             leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                             title="Vertical Spreads con Edge Estadistico — Reporte Final",
                             author="Vertical Spreads Research")
    doc.build(story)
    print(f"PDF generado: {OUT_PDF}")


if __name__ == "__main__":
    make_charts()
    build_pdf()
