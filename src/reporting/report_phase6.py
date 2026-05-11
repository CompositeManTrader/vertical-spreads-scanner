"""Reporte Word de Fase 6."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import REPORTS, TOTAL_COST_PER_TRADE
from src.reporting.charts_phase6 import (
    equity_curves_chart, metrics_bar_compare, pnl_distribution,
)
from src.reporting.word_builder import (
    add_bullet, add_callout, add_code_block, add_heading, add_image,
    add_paragraph, add_table_from_df, new_document, save,
)

RESULTS = REPORTS / "_phase6_results"
CHARTS = REPORTS / "_phase6_charts"


def main():
    CHARTS.mkdir(parents=True, exist_ok=True)
    all_trades = pd.read_parquet(RESULTS / "all_trades.parquet")
    summary = pd.read_parquet(RESULTS / "summary.parquet")
    portfolio_summary = pd.read_parquet(RESULTS / "portfolio_summary.parquet")
    portfolio_trades = pd.read_parquet(RESULTS / "portfolio_trades.parquet")

    for c in ["ticker", "config"]:
        if c in all_trades.columns:
            all_trades[c] = all_trades[c].astype(str)
        if c in summary.columns:
            summary[c] = summary[c].astype(str)
        if c in portfolio_summary.columns:
            portfolio_summary[c] = portfolio_summary[c].astype(str)
        if c in portfolio_trades.columns:
            portfolio_trades[c] = portfolio_trades[c].astype(str)

    # Charts
    for ticker in ["SPY", "QQQ", "IWM"]:
        equity_curves_chart(all_trades, ticker, CHARTS / f"equity_{ticker}.png")
    for ticker in ["SPY", "QQQ", "IWM"]:
        sub = all_trades[(all_trades["ticker"] == ticker) & (all_trades["config"] == "FILTRADA")]
        if not sub.empty:
            pnl_distribution(sub, f"{ticker} FILTRADA", CHARTS / f"pnl_dist_FILTRADA_{ticker}.png")
        sub = all_trades[(all_trades["ticker"] == ticker) & (all_trades["config"] == "VANILLA-baseline")]
        if not sub.empty:
            pnl_distribution(sub, f"{ticker} VANILLA baseline", CHARTS / f"pnl_dist_VANILLA_{ticker}.png")
    pnl_distribution(portfolio_trades, "PORTFOLIO SPY+QQQ FILTRADA",
                     CHARTS / "pnl_dist_PORTFOLIO.png")
    metrics_bar_compare(summary, "sharpe_per_trade",
                         "Sharpe per-trade (mayor = mejor)",
                         CHARTS / "bar_sharpe.png")
    metrics_bar_compare(summary, "win_rate", "Win rate", CHARTS / "bar_winrate.png")
    metrics_bar_compare(summary, "max_drawdown_usd",
                         "Max drawdown USD (menor = mejor)",
                         CHARTS / "bar_maxdd.png")

    # Equity curve portfolio
    from src.reporting.charts_phase6 import setup
    import matplotlib.pyplot as plt
    setup()
    p = portfolio_trades.sort_values("exit_date").copy()
    p["cum"] = p["pnl_net_after_costs"].cumsum()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(p["exit_date"], p["cum"], color="#1f6f3d", lw=1.6)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"PORTFOLIO SPY+QQQ FILTRADA - Equity curve (n={len(p)} trades)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("P&L acumulado USD por contrato")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(CHARTS / "equity_PORTFOLIO.png")
    plt.close(fig)

    # ============== Documento ==============
    doc = new_document(
        "Fase 6 — Simulacion P&L con gestion activa, costos y filtros",
        "Vertical Spreads Edge Research — Validacion final del edge en train",
    )

    add_heading(doc, "Resumen ejecutivo", 1)
    add_paragraph(doc,
        "Esta es la fase culminante del research en train. Simulamos 5 "
        "configuraciones de estrategia + 1 portfolio sobre los 3 ETFs, "
        "incorporando:"
    )
    add_bullet(doc, "Pricing BSM intra-trade con mark-to-market diario.")
    add_bullet(doc, "Reglas de gestion: TP 50% credit, SL 2x credit, time stop a 14 DTE.")
    add_bullet(doc, f"Costos Schwab realistas: ${TOTAL_COST_PER_TRADE:.2f} por trade completo (4 legs + slippage).")
    add_bullet(doc, "Filtro de regimen 'above_sma200 & vrp_high' (top quintile) en config FILTRADA.")
    add_bullet(doc, "Portfolio multi-ticker: SPY + QQQ FILTRADAS combinadas.")

    add_callout(doc,
        "🎯 RESULTADO CENTRAL: la config FILTRADA con delta-30, T=45, w=$5, "
        "TP/SL/time stops alcanza Sharpe 1.89 en SPY (1.95 en QQQ), winRate "
        "97-98%, max drawdown ~$300-434 sobre el train (1 contrato/trade). "
        "El portfolio combinado SPY+QQQ tiene Sharpe 1.92 y max DD $353.",
        color="success",
    )
    add_callout(doc,
        "🚨 IWM sigue siendo el outlier negativo: incluso filtrada, Sharpe "
        "0.51 y max DD $1,522. Confirma Fase 4: NO operar IWM con esta "
        "estrategia.",
        color="warning",
    )
    add_callout(doc,
        "🚨 CAVEAT IMPORTANTE: Sharpe per-trade NO es Sharpe anualizado. "
        "Los trades son OVERLAPPING. Bootstrap en Fase 7 dara CI realistas. "
        "Ademas pricing sin skew sobreestima credito (es decir, P&L real "
        "sera menor en los trades perdedores). Tomar el Sharpe 1.9 como "
        "indicativo, no como anualizado real.",
        color="danger",
    )

    # ---- 1. Setup ----
    add_heading(doc, "1. Setup metodologico", 1)
    add_paragraph(doc,
        "Para cada configuracion (delta_short, T_days, width) y cada dia t "
        "de entrada en el train (con IV disponible y >=T_days al final del "
        "dataset), se simula el trade siguiendo este pseudocodigo:"
    )
    add_code_block(doc,
        "1. En t (apertura):\n"
        "   K_short = solve_strike_for_delta(S(t), T, r(t), q, IV(t), delta_short)\n"
        "   K_long  = K_short - width\n"
        "   credit  = put_price(K_short) - put_price(K_long)\n"
        "\n"
        "2. Por cada dia d=1..T (mark-to-market):\n"
        "   T_remain = (T - d) / 365\n"
        "   Revaluar spread: spread_val = put(K_short) - put(K_long)  (con S(t+d), IV(t+d))\n"
        "   pnl = credit - spread_val\n"
        "   - Si touch_stop habilitado y Low(t+d) <= K_short -> cerrar\n"
        "   - Si pnl >= TP * credit -> cerrar (take profit)\n"
        "   - Si pnl <= -SL * credit -> cerrar (stop loss)\n"
        "   - Si T_remain <= time_stop -> cerrar\n"
        "\n"
        "3. Si llega a expiry: pnl = credit - max(0, K_short - max(K_long, S(T)))\n"
        "\n"
        "4. P&L neto = pnl_per_share * 100 - costo_total\n"
        f"   costo_total = ${TOTAL_COST_PER_TRADE:.2f} (Schwab approx, 4 legs + slippage)"
    )

    # ---- 2. Tabla maestra ----
    add_heading(doc, "2. Tabla maestra de metricas", 1)
    cols_show = ["config", "ticker", "n_trades", "win_rate", "expectancy_per_contract",
                 "sharpe_per_trade", "sortino", "max_drawdown_usd",
                 "max_consecutive_losses", "ES_5pct", "profit_factor",
                 "min_pnl", "max_pnl"]
    summary_show = summary[cols_show].copy()
    for c in cols_show:
        if summary_show[c].dtype == "float64":
            summary_show[c] = summary_show[c].round(3)
    summary_show["n_trades"] = summary_show["n_trades"].astype(int)
    summary_show["max_consecutive_losses"] = summary_show["max_consecutive_losses"].astype(int)
    add_table_from_df(doc, summary_show.sort_values(["config", "ticker"]))

    add_paragraph(doc, "Lectura:", bold=True)
    add_bullet(doc, "VANILLA-baseline: delta-30, T=45, w=$5, gestion full. Sharpe 0.30-0.38 segun ticker.")
    add_bullet(doc, "VANILLA-conservative (delta-20): Sharpe drops to 0.02-0.09. Costos comen el credit chico (delta-20 da credit ~$0.23, costos $2.80 = 12% del credit).")
    add_bullet(doc, "VANILLA-aggressive (delta-40, T=60): mejor Sharpe sin filtro (0.28-0.60).")
    add_bullet(doc, "FILTRADA (above_sma200 & vrp_high): JUMP a Sharpe 1.89-1.95 en SPY/QQQ. n cae a ~250 trades pero risk-adjusted explota.")
    add_bullet(doc, "HOLD-TO-EXPIRY (sin gestion): expectancy MAS ALTA en absolute terms (full credit captured) pero max_drawdown ENORME ($28-37k). Confirma valor del time stop.")

    # ---- 3. Equity curves ----
    add_heading(doc, "3. Equity curves comparativas", 1)
    for ticker in ["SPY", "QQQ", "IWM"]:
        add_paragraph(doc, f"{ticker}:", bold=True)
        add_image(doc, CHARTS / f"equity_{ticker}.png", width_inches=6.8)

    # ---- 4. Distribucion P&L ----
    add_heading(doc, "4. Distribucion de P&L por trade", 1)
    add_paragraph(doc, "FILTRADA vs VANILLA baseline (SPY representativo):")
    add_image(doc, CHARTS / "pnl_dist_FILTRADA_SPY.png", width_inches=6.5)
    add_image(doc, CHARTS / "pnl_dist_VANILLA_SPY.png", width_inches=6.5)
    add_paragraph(doc,
        "FILTRADA tiene cola izquierda mucho mas comprimida: muy pocos "
        "stop_loss gatillan. La distribucion concentra masa en pequenos "
        "winners (TP 50% del credit).",
        italic=True,
    )

    # ---- 5. Comparativas en barras ----
    add_heading(doc, "5. Comparativas visuales", 1)
    add_paragraph(doc, "Sharpe per-trade:", bold=True)
    add_image(doc, CHARTS / "bar_sharpe.png", width_inches=6.8)
    add_paragraph(doc, "Win rate:", bold=True)
    add_image(doc, CHARTS / "bar_winrate.png", width_inches=6.8)
    add_paragraph(doc, "Max drawdown:", bold=True)
    add_image(doc, CHARTS / "bar_maxdd.png", width_inches=6.8)

    # ---- 6. Portfolio ----
    add_heading(doc, "6. Portfolio SPY + QQQ (FILTRADA, equal weight)", 1)
    cols_p = ["n_trades", "win_rate", "expectancy_per_contract", "sharpe_per_trade",
              "sortino", "max_drawdown_usd", "max_consecutive_losses", "ES_5pct"]
    p_show = portfolio_summary[["ticker", "config"] + cols_p].copy()
    for c in cols_p:
        if p_show[c].dtype == "float64":
            p_show[c] = p_show[c].round(3)
    p_show["n_trades"] = p_show["n_trades"].astype(int)
    p_show["max_consecutive_losses"] = p_show["max_consecutive_losses"].astype(int)
    add_table_from_df(doc, p_show)
    add_image(doc, CHARTS / "equity_PORTFOLIO.png", width_inches=6.8)
    add_image(doc, CHARTS / "pnl_dist_PORTFOLIO.png", width_inches=6.5)

    add_callout(doc,
        "Diversificar SPY+QQQ con el filtro mantiene Sharpe alto (1.92) y "
        "duplica la cantidad de trades (n=492). Pero NO reduce el max DD "
        "proporcionalmente porque las correlaciones SPY-QQQ son altas, "
        "especialmente en stress (Fase 1: corr 0.93 unconditional, 0.88 stress). "
        "Diversificar sirve para mas trades, no para reducir el riesgo de cola.",
        color="info",
    )

    # ---- 7. Razones de salida ----
    add_heading(doc, "7. Distribucion de razones de salida", 1)
    add_paragraph(doc, "Cuantos trades cierran por cada razon (% de total):")
    rows = []
    for cfg, g in all_trades.groupby("config"):
        for reason in ["take_profit", "stop_loss", "time_stop", "expiry", "touch_stop"]:
            n = (g["exit_reason"] == reason).sum()
            rows.append({
                "config": cfg, "exit_reason": reason,
                "n": int(n), "pct": round(100 * n / len(g), 1),
            })
    add_table_from_df(doc, pd.DataFrame(rows))

    # ---- 8. Caveats ----
    add_heading(doc, "8. Caveats criticos (para no engañarse)", 1)
    add_callout(doc,
        "1. Sharpe per-trade vs anualizado: si la FILTRADA produce 40 "
        "trades/anio, el Sharpe anualizado nominal seria 1.89*sqrt(40)≈12. "
        "Pero los trades son OVERLAPPING (correlacion entre trades adyacentes "
        "es alta). La n efectiva es menor. Bootstrap en Fase 7 dara CI honestos.",
        color="warning",
    )
    add_bullet(doc, "Sin skew real: usamos IV ATM constante por strike. En la realidad, los puts OTM cotizan a IV mayor -> credit recibido sera menor en mercado real -> P&L menor en trades ganadores; spread valdra mas en trades perdedores -> mark-to-market peor. Estimacion: Sharpe real podria ser 30-40% menor.")
    add_bullet(doc, "Sin slippage activo: asumimos que el cierre se ejecuta al precio teorico EOD. En realidad hay 1-3% de slippage adicional sobre el spread.")
    add_bullet(doc, "Bid-ask: opciones sobre IWM tienen spreads mas anchos. Costo efectivo es mayor.")
    add_bullet(doc, "Touch_stop deshabilitado (ya vimos en Fase 2 que cierra trades que recuperan en 60-70% de los casos).")
    add_bullet(doc, "Costo Schwab: $2.80/trade. Si el trader paga mas (broker caro), el edge se erosiona.")
    add_bullet(doc, "Filtro 'above_sma200 & vrp_high' usa quantile global del train para definir 'vrp_high' (top quintile). Para reglas operativas en vivo, hay que recalcular el threshold con expanding window. La degradacion sera marginal (Fase 7 valida).")
    add_bullet(doc, "Sample 1,510 dias. Periodo 2018-2024. Si el regimen cambia (ej. era de tasas altas estancadas), los filtros pueden no funcionar igual.")
    add_bullet(doc, "El filtro reduce n trades a ~16% del total. Significa muchos meses sin operar (paciencia requerida).")

    # ---- 9. Recomendacion preliminar ----
    add_heading(doc, "9. Recomendacion preliminar (sujeta a validacion en Fase 7)", 1)
    add_paragraph(doc, "Configuracion recomendada para SPY/QQQ:", bold=True)
    add_bullet(doc, "Delta short: -0.30")
    add_bullet(doc, "DTE: 45")
    add_bullet(doc, "Ancho del spread: $5 (compromiso entre Sharpe y eficiencia capital con costos)")
    add_bullet(doc, "Filtro de entrada: precio > SMA200 AND VRP en top quintile (recalculado expanding)")
    add_bullet(doc, "Take profit: 50% del credito recibido")
    add_bullet(doc, "Stop loss: 2x el credito recibido")
    add_bullet(doc, "Time stop: cerrar a 14 DTE remanentes (independiente de PnL)")
    add_bullet(doc, "NO touch stop")
    add_bullet(doc, "Position sizing: max loss <= 2% del capital por trade")
    add_bullet(doc, "Solo SPY y QQQ (NO IWM)")
    add_bullet(doc, "Maximo 2-3 trades concurrentes (evitar concentracion en stress)")

    # ---- 10. Proximos pasos ----
    add_heading(doc, "10. Proximos pasos (Fase 7 - validacion final)", 1)
    add_bullet(doc, "Walk-forward analysis con re-tuneo cada anio.")
    add_bullet(doc, "Block bootstrap (block size = 30 dias) para CI honestos del Sharpe.")
    add_bullet(doc, "Monte Carlo de orden de trades para distribucion de max DD.")
    add_bullet(doc, "Stress test en escenarios COVID, 2022 Q3, hipotetico crash -15%.")
    add_bullet(doc, "Sensibilidad a costos (+50%, +100%) y skew bump (+10%, +20%).")
    add_bullet(doc, "Validacion en HOLDOUT sellado (2024-10-04 a 2026-03-12). Una sola pasada.")
    add_bullet(doc, "Veredicto final go/no-go con criterios pre-definidos.")

    out = REPORTS / "Phase6_Strategy_Simulation.docx"
    save(doc, out)
    print(f"Reporte guardado: {out}")


if __name__ == "__main__":
    main()
