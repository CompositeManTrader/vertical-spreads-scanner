"""
Descarga datos externos:
- DGS3MO (Treasury 3M yield) desde FRED CSV publico.
- Calendario FOMC (hardcoded desde 2018, fechas oficiales).
- Calendario CPI (hardcoded desde 2018, fechas BLS).
- Calendario NFP (hardcoded primer viernes con excepciones).
- Earnings season clusters (computado).

Todos los calendarios son CONOCIDOS EX-ANTE (publicados con anticipacion),
por lo que no introducen look-ahead bias.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import DATA_EXTERNAL


# ---------------------------------------------------------------------------
# DGS3MO (3-month Treasury yield) desde FRED
# ---------------------------------------------------------------------------

def fetch_dgs3mo() -> pd.DataFrame:
    """
    Descarga DGS3MO daily desde FRED (publico, sin auth).
    Devuelve DataFrame con columnas Date, DGS3MO (yield anual en %).
    Anti-look-ahead: FRED publica con 1 dia de lag. Para usar la tasa en
    decisiones del dia t, usaremos r(t) = DGS3MO(t-1) en el resto del codigo.
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
    df = pd.read_csv(url, parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "Date", "DGS3MO": "DGS3MO_pct"})
    # FRED usa "." para missing
    df["DGS3MO_pct"] = pd.to_numeric(df["DGS3MO_pct"], errors="coerce")
    df = df.dropna(subset=["DGS3MO_pct"]).sort_values("Date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# FOMC meetings (8 por anio aprox; fechas oficiales conocidas con anticipacion)
# Fuente: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# Cada meeting tipicamente dura 2 dias; reportamos el SEGUNDO dia (cuando se
# anuncia la decision a las 14:00 ET).
# ---------------------------------------------------------------------------

FOMC_MEETING_DATES = [
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020 (incluye reuniones de emergencia)
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (anunciado por la Fed)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16",
]


def fomc_calendar() -> pd.DataFrame:
    df = pd.DataFrame({"Date": pd.to_datetime(FOMC_MEETING_DATES)})
    df["is_FOMC"] = True
    return df.sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# CPI release dates
# Fuente: BLS publica calendario anual con 1 anio de anticipacion.
# Aprox: ~10mo dia habil del mes siguiente al mes de referencia.
# Para precision, usamos fechas oficiales BLS publicadas.
# ---------------------------------------------------------------------------

CPI_RELEASE_DATES = [
    # 2018
    "2018-01-12", "2018-02-14", "2018-03-13", "2018-04-11", "2018-05-10",
    "2018-06-12", "2018-07-12", "2018-08-10", "2018-09-13", "2018-10-11",
    "2018-11-14", "2018-12-12",
    # 2019
    "2019-01-11", "2019-02-13", "2019-03-12", "2019-04-10", "2019-05-10",
    "2019-06-12", "2019-07-11", "2019-08-13", "2019-09-12", "2019-10-10",
    "2019-11-13", "2019-12-11",
    # 2020
    "2020-01-14", "2020-02-13", "2020-03-11", "2020-04-10", "2020-05-12",
    "2020-06-10", "2020-07-14", "2020-08-12", "2020-09-11", "2020-10-13",
    "2020-11-12", "2020-12-10",
    # 2021
    "2021-01-13", "2021-02-10", "2021-03-10", "2021-04-13", "2021-05-12",
    "2021-06-10", "2021-07-13", "2021-08-11", "2021-09-14", "2021-10-13",
    "2021-11-10", "2021-12-10",
    # 2022
    "2022-01-12", "2022-02-10", "2022-03-10", "2022-04-12", "2022-05-11",
    "2022-06-10", "2022-07-13", "2022-08-10", "2022-09-13", "2022-10-13",
    "2022-11-10", "2022-12-13",
    # 2023
    "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12", "2023-05-10",
    "2023-06-13", "2023-07-12", "2023-08-10", "2023-09-13", "2023-10-12",
    "2023-11-14", "2023-12-12",
    # 2024
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15",
    "2024-06-12", "2024-07-11", "2024-08-14", "2024-09-11", "2024-10-10",
    "2024-11-13", "2024-12-11",
    # 2025
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13",
    "2025-06-11", "2025-07-15", "2025-08-12", "2025-09-11", "2025-10-15",
    "2025-11-13", "2025-12-10",
    # 2026
    "2026-01-14", "2026-02-11", "2026-03-11",
]


def cpi_calendar() -> pd.DataFrame:
    df = pd.DataFrame({"Date": pd.to_datetime(CPI_RELEASE_DATES)})
    df["is_CPI"] = True
    return df.sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# NFP releases (Non-Farm Payrolls): tipicamente primer viernes del mes,
# con excepciones por feriados.
# ---------------------------------------------------------------------------

def _first_friday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    # Day of week: Monday=0, Friday=4
    offset = (4 - d.dayofweek) % 7
    return d + pd.Timedelta(days=offset)


def nfp_calendar(start_year: int = 2018, end_year: int = 2026) -> pd.DataFrame:
    dates = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            d = _first_friday(y, m)
            if d <= pd.Timestamp(f"{end_year}-12-31"):
                dates.append(d)
    df = pd.DataFrame({"Date": dates})
    df["is_NFP"] = True
    return df.sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Earnings season cluster (SPX): semanas 2-6 despues del cierre de cada Q.
# Trimestres calendarios: Q4 cierra Dec 31, Q1 cierra Mar 31, etc.
# Earnings season pico: ~mid-Jan a mid-Feb (Q4), mid-Apr a mid-May (Q1), etc.
# ---------------------------------------------------------------------------

def earnings_season_calendar(start_year: int = 2018, end_year: int = 2026) -> pd.DataFrame:
    """
    Marca dias dentro de earnings season pico (~3 semanas tras cierre de Q,
    enero/abril/julio/octubre).
    Anti-look-ahead: estos clusters son predecibles (no dependen de resultados).
    """
    rows = []
    for y in range(start_year, end_year + 1):
        # 4 ventanas: enero (Q4), abril (Q1), julio (Q2), octubre (Q3)
        for month_start in [1, 4, 7, 10]:
            start = pd.Timestamp(year=y, month=month_start, day=10)
            end = start + pd.Timedelta(days=35)  # ~5 semanas
            for d in pd.date_range(start, end, freq="D"):
                if d.dayofweek < 5:  # weekdays
                    rows.append({"Date": d, "is_earnings_season": True})
    return pd.DataFrame(rows).drop_duplicates("Date").sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)

    print("Bajando DGS3MO de FRED...")
    rates = fetch_dgs3mo()
    rates.to_parquet(DATA_EXTERNAL / "rates_dgs3mo.parquet")
    print(f"  OK  {len(rates):,} filas, {rates['Date'].min().date()} -> {rates['Date'].max().date()}")
    print(f"  Stats: min={rates['DGS3MO_pct'].min():.2f}%  max={rates['DGS3MO_pct'].max():.2f}%  last={rates['DGS3MO_pct'].iloc[-1]:.2f}%")

    print("\nGenerando calendario FOMC...")
    fomc = fomc_calendar()
    fomc.to_parquet(DATA_EXTERNAL / "calendar_fomc.parquet")
    print(f"  OK  {len(fomc)} reuniones desde {fomc['Date'].min().date()} hasta {fomc['Date'].max().date()}")

    print("\nGenerando calendario CPI...")
    cpi = cpi_calendar()
    cpi.to_parquet(DATA_EXTERNAL / "calendar_cpi.parquet")
    print(f"  OK  {len(cpi)} releases")

    print("\nGenerando calendario NFP...")
    nfp = nfp_calendar()
    nfp.to_parquet(DATA_EXTERNAL / "calendar_nfp.parquet")
    print(f"  OK  {len(nfp)} releases")

    print("\nGenerando calendario earnings season...")
    es = earnings_season_calendar()
    es.to_parquet(DATA_EXTERNAL / "calendar_earnings_season.parquet")
    print(f"  OK  {len(es)} dias en clusters")


if __name__ == "__main__":
    main()
