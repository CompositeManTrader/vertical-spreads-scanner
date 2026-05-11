"""
Loaders para datasets crudos. Funciones puras: input path, output DataFrame
ya normalizado (Date como datetime ascendente, columnas tipadas).

ANTI-LOOK-AHEAD: aqui solo se cargan datos. No se calculan features con
ventanas. Esa logica vive en build_panel.py.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# ETF xlsx (Barchart format: header en fila 1, datos desde fila 2)
# ---------------------------------------------------------------------------

ETF_RAW_COLUMNS = [
    "Date", "Symbol", "Name", "Close", "Open", "High", "Low",
    "Volume", "OpenInterest", "SaleCondition",
    "Imp Vol", "IV Chg", "IV Rank", "IV Percentile",
    "Put/Call Volume Ratio", "Call Volume", "Put Volume", "Options Vol",
    "Put/Call Open Interest Ratio", "Call OI", "Put OI", "Total OI",
]


def load_etf_daily(path: Path | str) -> pd.DataFrame:
    """
    Carga un xlsx Barchart de ETF diario.
    Convierte la fecha de serial Excel a datetime y ordena ascendente.
    """
    df = pd.read_excel(path, header=1, engine="openpyxl")

    # Validar columnas esperadas
    missing = set(ETF_RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en {path}: {missing}")

    # Convertir fecha (puede venir como serial Excel int o como datetime).
    if pd.api.types.is_numeric_dtype(df["Date"]):
        # Excel serial date: dia 1 = 1900-01-01 (con bug del leap year 1900).
        # pd.to_datetime con origin="1899-12-30" maneja esto correctamente.
        df["Date"] = pd.to_datetime(df["Date"], origin="1899-12-30", unit="D")
    else:
        df["Date"] = pd.to_datetime(df["Date"])

    # Tipos numericos
    numeric_cols = [
        "Close", "Open", "High", "Low", "Volume", "OpenInterest",
        "Imp Vol", "IV Chg", "IV Rank", "IV Percentile",
        "Put/Call Volume Ratio", "Call Volume", "Put Volume", "Options Vol",
        "Put/Call Open Interest Ratio", "Call OI", "Put OI", "Total OI",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)

    # Renombrar Imp Vol y IV Rank/Percentile a snake_case para uso interno;
    # Imp Vol viene como FRACCION (0.17 = 17%). IV Rank y IV Percentile vienen
    # como fraccion (0.74 = 74%).
    df = df.rename(columns={
        "Imp Vol": "iv_atm_barchart",
        "IV Rank": "iv_rank_barchart",
        "IV Percentile": "iv_percentile_barchart",
        "Put/Call Volume Ratio": "pc_volume_ratio",
        "Put/Call Open Interest Ratio": "pc_oi_ratio",
    })

    return df


# ---------------------------------------------------------------------------
# Master volatility CSV
# ---------------------------------------------------------------------------

def load_master_vol(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# FRED rates (parquet local)
# ---------------------------------------------------------------------------

def load_rates(path: Path | str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Calendarios macro (parquet local)
# ---------------------------------------------------------------------------

def load_calendar(path: Path | str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_values("Date").reset_index(drop=True)
    return df
