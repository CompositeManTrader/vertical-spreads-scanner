"""
ETL principal: construye el panel limpio por ticker, listo para Fase 1+.

Salida (por ticker, parquet):
- Date (datetime, ascendente)
- OHLCV
- iv_atm_barchart (raw IV from Barchart, fraction)
- iv_rank_barchart (raw, fraction)
- iv_percentile_barchart (raw, fraction)
- iv_rank_252 (RECALCULADO por nosotros, anti-look-ahead)
- iv_percentile_252 (RECALCULADO por nosotros, anti-look-ahead)
- pc_volume_ratio, pc_oi_ratio
- ret_back_1d (log return)
- VIX_Close, VIX3M_Close, VVIX_Close, VXX_Close (del master)
- Contango_pct, VIX_VIX3M_Ratio
- DGS3MO_pct (rezagado un dia para evitar look-ahead)

Tambien:
- Reporta tests de calidad y diagnostico de IV Rank (Barchart vs nuestro).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import (
    DATA_CLEAN, DATA_EXTERNAL, ETF_PATHS, IV_AVAILABLE_FROM,
    MASTER_VOL_PATH, TRAIN_END,
)
from src.analysis.indicators import (
    iv_percentile, iv_rank, log_return_back,
)
from src.etl.loaders import (
    load_calendar, load_etf_daily, load_master_vol, load_rates,
)


MASTER_COLS_KEEP = [
    "Date", "VIX_Close", "VIX3M_Close", "VVIX_Close",
    "Contango_pct", "VIX_VIX3M_Ratio",
    "VXX_Close", "M1_Price", "M2_Price",
]


def build_ticker_panel(ticker: str, etf_path: Path,
                       master: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """Arma el panel diario completo para un ticker."""
    print(f"\n--- {ticker} ---")
    df = load_etf_daily(etf_path)
    print(f"  Filas crudas: {len(df):,}  ({df['Date'].min().date()} -> {df['Date'].max().date()})")

    # Sanity: validar IV en rango razonable. IV == 0 es bug de Barchart (datos
    # rotos en dias puntuales), lo descartamos junto con valores fuera de
    # rango. La fila se mantiene; solo se invalida iv_atm_barchart -> NaN.
    iv = df["iv_atm_barchart"]
    bad_iv_mask = (iv <= 0) | (iv > 5)
    bad_iv_count = bad_iv_mask.sum()
    if bad_iv_count:
        print(f"  WARN: {bad_iv_count} valores de IV invalidos (0 o fuera de rango), se setean a NaN")
        print(f"        Fechas: {df.loc[bad_iv_mask, 'Date'].dt.date.tolist()}")
        df.loc[bad_iv_mask, "iv_atm_barchart"] = np.nan

    # Sanity: detectar splits (cambios > 30% en un dia que no son crisis)
    df["_pct_chg"] = df["Close"].pct_change()
    big_jumps = df[df["_pct_chg"].abs() > 0.30][["Date", "Close", "_pct_chg"]]
    if len(big_jumps):
        print(f"  WARN: {len(big_jumps)} dias con cambio > 30%:")
        print(big_jumps.head(5).to_string(index=False))
    df = df.drop(columns=["_pct_chg"])

    # Returns log diarios (back-looking, feature legitimo)
    df["ret_back_1d"] = log_return_back(df["Close"], periods=1)

    # IV Rank y IV Percentile RECALCULADOS por nosotros (sin look-ahead).
    df["iv_rank_252"] = iv_rank(df["iv_atm_barchart"], lookback=252)
    df["iv_percentile_252"] = iv_percentile(df["iv_atm_barchart"], lookback=252)

    # Merge con master de volatilidad (VIX, term structure, etc.)
    m = master[MASTER_COLS_KEEP].copy()
    df = df.merge(m, on="Date", how="left")

    # Merge con tasa libre de riesgo, rezagada 1 dia (anti-look-ahead).
    r = rates.copy()
    r["Date"] = r["Date"] + pd.Timedelta(days=1)  # mover fecha 1 dia adelante
    r = r.rename(columns={"DGS3MO_pct": "rfr_pct"})
    df = df.merge(r, on="Date", how="left")
    # Forward-fill rate solo para dias sin publicacion (fines de semana etc.)
    df["rfr_pct"] = df["rfr_pct"].ffill()

    # Filtro temporal: dejar solo desde IV_AVAILABLE_FROM.
    df_iv = df[df["Date"] >= pd.Timestamp(IV_AVAILABLE_FROM)].copy().reset_index(drop=True)
    print(f"  Filas con IV disponible (>= {IV_AVAILABLE_FROM}): {len(df_iv):,}")
    print(f"  IV no-nulos en ventana: {df_iv['iv_atm_barchart'].notna().sum():,}")

    return df_iv


def diagnose_iv_rank_lookahead(panel: pd.DataFrame, ticker: str) -> dict:
    """
    Compara IV Rank Barchart vs nuestro recalculo.
    Si Barchart usaron look-ahead (ventana > pasado), las series divergiran.
    Si usaron min/max global, la divergencia sera enorme.
    Si usaron 252 dias hacia atras (correcto), R^2 deberia ser ~1.
    """
    valid = panel.dropna(subset=["iv_rank_barchart", "iv_rank_252"])
    if len(valid) == 0:
        return {"ticker": ticker, "n": 0, "note": "sin overlap"}

    barchart = valid["iv_rank_barchart"].values
    ours = valid["iv_rank_252"].values
    diff = barchart - ours
    corr = float(np.corrcoef(barchart, ours)[0, 1])
    mae = float(np.mean(np.abs(diff)))
    max_abs_diff = float(np.max(np.abs(diff)))
    bias = float(np.mean(diff))

    return {
        "ticker": ticker,
        "n": int(len(valid)),
        "corr": corr,
        "mae": mae,
        "max_abs_diff": max_abs_diff,
        "bias_mean": bias,
    }


def main():
    DATA_CLEAN.mkdir(parents=True, exist_ok=True)

    print("Cargando master de volatilidad...")
    master = load_master_vol(MASTER_VOL_PATH)
    print(f"  Master: {len(master):,} filas, {master['Date'].min().date()} -> {master['Date'].max().date()}")

    print("\nCargando tasa libre de riesgo (DGS3MO)...")
    rates = load_rates(DATA_EXTERNAL / "rates_dgs3mo.parquet")
    print(f"  Rates: {len(rates):,} filas, last={rates['DGS3MO_pct'].iloc[-1]:.2f}%")

    diagnostics = []
    for ticker, path in ETF_PATHS.items():
        panel = build_ticker_panel(ticker, path, master, rates)
        out_path = DATA_CLEAN / f"{ticker}.parquet"
        panel.to_parquet(out_path, index=False)
        print(f"  Guardado: {out_path}  ({panel.shape[0]:,} filas x {panel.shape[1]} cols)")

        diag = diagnose_iv_rank_lookahead(panel, ticker)
        diagnostics.append(diag)

    print("\n" + "=" * 70)
    print("DIAGNOSTICO IV RANK: Barchart vs nuestro recalculo (252d, sin look-ahead)")
    print("=" * 70)
    diag_df = pd.DataFrame(diagnostics)
    print(diag_df.to_string(index=False))

    print("""
Interpretacion:
- corr ~ 1.00 y mae < 0.05 -> Barchart NO tiene look-ahead, sus valores OK.
- corr alto pero mae > 0.10 -> Barchart usa otra ventana o normalizacion.
- corr bajo o bias sistematico -> Barchart pudo usar info futura.
En todos los casos, usaremos NUESTRO iv_rank_252 (point-in-time garantizado).
""")


if __name__ == "__main__":
    main()
