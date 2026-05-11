"""
Streamlit dashboard para el Vertical Spreads Scanner.

Run:
    streamlit run streamlit_app.py

Provee:
- Live scan de SPY y QQQ con filtros del research.
- Trade setup detallado si hay senal.
- Historial de scans (SQLite) con filtros y outcome tracking.
- Stats agregadas del forward-test.
"""

import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.scanner.data_feed import get_feed
from src.scanner.filter_engine import compute_features_for_today, evaluate_filter
from src.scanner.signal_logger import DB_PATH, log_scan
from src.scanner.trade_builder import build_pcs_setup


# ============================================================================
# CONFIG GLOBAL DE LA APP
# ============================================================================

st.set_page_config(
    page_title="Vertical Spreads Scanner",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# Defaults segun research final (MASTER_Final_Report.docx)
DEFAULT_TICKERS = ["SPY", "QQQ"]
DEFAULT_DELTA = -0.30
DEFAULT_DTE = 45
DEFAULT_WIDTH = 5.0
DEFAULT_TP_PCT = 0.50
DEFAULT_SL_MULT = 2.0
DEFAULT_TIME_STOP = 14
DEFAULT_VRP_THR = 0.03
DEFAULT_RFR = 3.7


# ============================================================================
# Sidebar - configuracion
# ============================================================================

st.sidebar.title("Configuracion")
provider = st.sidebar.radio(
    "Data provider",
    options=["yfinance", "schwab"],
    index=0,
    help="yfinance es gratis. Schwab requiere OAuth (.env configurado).",
)

tickers = st.sidebar.multiselect(
    "Tickers a scannear",
    options=["SPY", "QQQ", "IWM"],
    default=DEFAULT_TICKERS,
    help="Research recomienda solo SPY y QQQ. IWM no tiene VRP estructural.",
)

with st.sidebar.expander("Parametros del trade", expanded=False):
    delta_short = st.number_input("Delta short", value=DEFAULT_DELTA, step=0.05,
                                   max_value=-0.05, min_value=-0.50,
                                   help="Negativo. Default -0.30 segun research.")
    dte = st.number_input("DTE", value=DEFAULT_DTE, step=5, min_value=14, max_value=90)
    width = st.number_input("Width ($)", value=DEFAULT_WIDTH, step=1.0, min_value=1.0)
    tp_pct = st.slider("Take profit (% del credito)", 0.10, 0.90, DEFAULT_TP_PCT, 0.05)
    sl_mult = st.slider("Stop loss (x credito)", 1.0, 4.0, DEFAULT_SL_MULT, 0.5)
    time_stop = st.number_input("Time stop (DTE remanentes)", value=DEFAULT_TIME_STOP,
                                 step=1, min_value=0, max_value=30)

with st.sidebar.expander("Filtros", expanded=False):
    vrp_threshold = st.slider("VRP threshold (vol points)",
                               0.00, 0.10, DEFAULT_VRP_THR, 0.005,
                               help="VRP = IV ATM - RV20d. Default 0.03 (~quintile 80 historico).")
    rfr_pct = st.number_input("Risk-free rate (%)", value=DEFAULT_RFR, step=0.10)

st.sidebar.markdown("---")
log_to_db = st.sidebar.checkbox("Loggear scan a SQLite", value=False,
                                 help="Solo activar en runs reales (no exploracion).")


# ============================================================================
# Cache de feed (evita llamadas duplicadas en re-renders)
# ============================================================================

@st.cache_data(ttl=300)  # 5 min
def cached_scan(provider: str, ticker: str) -> dict:
    feed = get_feed(provider)
    history = feed.fetch_history(ticker, years=2)
    iv, iv_meta = feed.fetch_atm_iv_30d(ticker)
    return {
        "history": history,
        "iv": iv,
        "iv_meta": iv_meta,
        "scanned_at": datetime.now().isoformat(timespec="seconds"),
    }


@st.cache_data(ttl=300)
def cached_vix(provider: str):
    feed = get_feed(provider)
    return feed.fetch_vix_close()


# ============================================================================
# HEADER
# ============================================================================

st.title("Vertical Spreads Scanner")
st.caption(f"Provider: **{provider}** | Filtro: above_sma200 AND vrp >= {vrp_threshold*100:.1f}pp | "
            f"Trade: delta {delta_short}, DTE {dte}, ${width} ancho")

vix, vix_meta = cached_vix(provider)
top_cols = st.columns(4)
top_cols[0].metric("VIX", f"{vix:.2f}" if vix else "n/a", help=vix_meta.get("date", ""))
top_cols[1].metric("Tickers", ", ".join(tickers) if tickers else "—")
top_cols[2].metric("Delta short", f"{delta_short}")
top_cols[3].metric("DTE", f"{dte}")


# ============================================================================
# SECCION 1: scan actual por ticker
# ============================================================================

st.markdown("---")
st.header("1. Senales actuales (EOD)")

if not tickers:
    st.warning("Seleccionar al menos 1 ticker en sidebar.")
else:
    for ticker in tickers:
        with st.expander(f"**{ticker}**", expanded=True):
            try:
                with st.spinner(f"Bajando datos de {ticker}..."):
                    cache = cached_scan(provider, ticker)
            except Exception as e:
                st.error(f"Error cargando {ticker}: {e}")
                continue

            history = cache["history"]
            iv = cache["iv"]
            iv_meta = cache["iv_meta"]
            if iv is None:
                st.error(f"No se pudo obtener IV ATM 30D: {iv_meta}")
                continue

            features = compute_features_for_today(history, iv)
            fr = evaluate_filter(features, vrp_threshold=vrp_threshold)

            cols = st.columns(5)
            cols[0].metric("Spot", f"${features['close']:.2f}")
            cols[1].metric("IV ATM 30D", f"{features['iv_atm_today']*100:.2f}%")
            cols[2].metric("RV 20d", f"{features['rv_20d']*100:.2f}%")
            cols[3].metric("VRP", f"{features['vrp']*100:+.2f}pp",
                           delta=f"{(features['vrp']-vrp_threshold)*100:+.2f}pp vs thr")
            cols[4].metric("Spot vs SMA200", f"{features['price_to_sma200_pct']*100:+.2f}%")

            # Filter status
            f_cols = st.columns(2)
            with f_cols[0]:
                if fr["above_sma200_pass"]:
                    st.success(f"above_sma200: PASS")
                else:
                    st.error(f"above_sma200: FAIL")
            with f_cols[1]:
                if fr["vrp_pass"]:
                    st.success(f"vrp >= {vrp_threshold*100:.1f}pp: PASS  ({features['vrp']*100:+.2f}pp)")
                else:
                    st.error(f"vrp >= {vrp_threshold*100:.1f}pp: FAIL  ({features['vrp']*100:+.2f}pp)")

            # Si pasa: trade setup
            setup = None
            if fr["filter_pass"]:
                st.success("### **SENAL: ABRIR TRADE**")
                setup = build_pcs_setup(
                    ticker=ticker, spot=features["close"], iv=features["iv_atm_today"],
                    rfr_pct=rfr_pct, delta_short=delta_short, dte=dte, width=width,
                    tp_pct=tp_pct, sl_mult=sl_mult, time_stop_dte=time_stop,
                )
                cs = st.columns(4)
                cs[0].metric("K_short (sell)", f"${setup['K_short_recommended']:.2f}",
                              delta=f"delta {setup['K_short_delta_actual']:+.3f}")
                cs[1].metric("K_long (buy)", f"${setup['K_long_recommended']:.2f}",
                              delta=f"delta {setup['K_long_delta_actual']:+.3f}")
                cs[2].metric("Credito BSM", f"${setup['credit_per_contract_BSM']:.2f}",
                              delta=f"{setup['credit_to_maxloss_ratio']*100:.0f}% del max loss")
                cs[3].metric("Max loss", f"${setup['max_loss_per_contract']:.2f}")

                cs2 = st.columns(3)
                cs2[0].metric("Take profit", f"${setup['tp_per_contract']:.2f}",
                               help=f"@ {tp_pct*100:.0f}% del credito")
                cs2[1].metric("Stop loss", f"${setup['sl_per_contract']:.2f}",
                               help=f"@ -{sl_mult}x credito")
                cs2[2].metric("Time stop date", setup["time_stop_date"])

                st.info(f"**Expiracion**: {setup['expiry_date']} ({setup['dte_at_open']} DTE)  |  "
                         f"**Costo total estimado**: ${setup['estimated_costs_total']:.2f}")
            else:
                st.warning("### Sin senal: filtro NO pasa")

            # Mini chart con SMA200
            df_chart = history.tail(252).copy()
            df_chart["SMA200"] = df_chart["Close"].rolling(200, min_periods=200).mean()
            df_chart["SMA50"] = df_chart["Close"].rolling(50, min_periods=50).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_chart["Date"], y=df_chart["Close"],
                                       name="Close", line=dict(color="#3470b8", width=1.5)))
            fig.add_trace(go.Scatter(x=df_chart["Date"], y=df_chart["SMA50"],
                                       name="SMA50", line=dict(color="orange", width=1, dash="dot")))
            fig.add_trace(go.Scatter(x=df_chart["Date"], y=df_chart["SMA200"],
                                       name="SMA200", line=dict(color="red", width=1, dash="dash")))
            if setup:
                fig.add_hline(y=setup["K_short_recommended"], line_dash="dash",
                                line_color="green", annotation_text=f"K_short ${setup['K_short_recommended']}")
                fig.add_hline(y=setup["K_long_recommended"], line_dash="dot",
                                line_color="green", annotation_text=f"K_long ${setup['K_long_recommended']}")
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0),
                                showlegend=True, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # Log scan?
            if log_to_db:
                if st.button(f"Loggear scan {ticker}", key=f"log_{ticker}"):
                    log_scan(ticker, features, fr, setup,
                              notes=f"streamlit/{provider}, vix={vix}")
                    st.success("Loggeado a SQLite.")


# ============================================================================
# SECCION 2: historial de scans (SQLite)
# ============================================================================

st.markdown("---")
st.header("2. Historial de scans loggeados")

if DB_PATH.exists():
    with sqlite3.connect(DB_PATH) as conn:
        df_scans = pd.read_sql_query(
            "SELECT * FROM scans ORDER BY id DESC LIMIT 200", conn
        )
    if df_scans.empty:
        st.info("Sin scans loggeados todavia. Activar 'Loggear scan a SQLite' en sidebar.")
    else:
        df_scans["scan_timestamp"] = pd.to_datetime(df_scans["scan_timestamp"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Total scans", len(df_scans))
        c2.metric("Filtro PASS", int(df_scans["filter_pass"].sum()))
        pass_rate = df_scans["filter_pass"].mean() * 100 if len(df_scans) else 0
        c3.metric("Pass rate", f"{pass_rate:.1f}%")

        st.dataframe(
            df_scans[["scan_timestamp", "ticker", "market_date", "spot",
                       "iv_atm", "rv_20d", "vrp", "above_sma200", "filter_pass"]],
            use_container_width=True, hide_index=True,
        )

        # Plot historico VRP por ticker
        st.subheader("VRP historico (loggeado)")
        if len(df_scans) > 1:
            fig2 = px.line(df_scans.sort_values("scan_timestamp"),
                              x="scan_timestamp", y="vrp", color="ticker",
                              markers=True)
            fig2.add_hline(y=vrp_threshold, line_dash="dash", line_color="red",
                              annotation_text=f"Threshold {vrp_threshold*100:.1f}pp")
            fig2.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("DB de scans aun no creada. Loggear primer scan para inicializarla.")


# ============================================================================
# SECCION 3: research summary (referencia)
# ============================================================================

st.markdown("---")
st.header("3. Research reference (resumen MASTER report)")

with st.expander("Hallazgos clave del research"):
    st.markdown("""
- **VRP estructural** confirmado en SPY y QQQ (NO IWM).
- **Filtro `above_sma200 & vrp_high` (top quintile)**: reduce P(ITM) ~85%.
- **Sharpe en train (FILTRADA)**: SPY 1.89, QQQ 1.95. Bootstrap CI [1.4-5.0].
- **Holdout (sellado)**: SPY/QQQ winRate **100%**, n=35-48.
- **Robustez**: pasa costos 3x, mejora con skew, **se apaga en 2022 stress**.
- **Mitos derribados**:
    - Delta-20 NO es optimo (delta-30/40 mejor risk-adjusted con costos).
    - "Vender en IV alto" sin medir VRP es trampa.
    - Diversificar SPY+QQQ NO reduce riesgo (corr 0.93).
- **NO operar IWM** con esta estrategia.
    """)

with st.expander("Reglas operativas"):
    st.markdown(f"""
| Parametro | Valor |
|---|---|
| Subyacentes | SPY, QQQ |
| Estructura | Put Credit Spread (PCS) |
| Delta short | {delta_short} |
| DTE | {dte} |
| Ancho | ${width} |
| Take profit | {tp_pct*100:.0f}% del credito |
| Stop loss | {sl_mult}x credito |
| Time stop | {time_stop} DTE remanentes |
| Filtro | above_sma200 AND vrp >= {vrp_threshold*100:.1f}pp |
| Sizing | Max loss <= 2% capital por trade |
| Concurrentes | Max 2-3 trades |
| Costos asumidos | $2.80/trade (Schwab) |
    """)

st.caption(f"App generada por Vertical Spreads Edge Research. "
            f"Provider activo: {provider}. DB: {DB_PATH}")
