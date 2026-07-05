"""
Diagnostico: que es la columna 'Imp Vol' de los datasets de ETFs?

Hipotesis:
  H1: ATM IV constant-maturity 30D (lo ideal). Deberia coincidir con VIX/100 en SPY.
  H2: IV de la front-month ATM (similar pero con efecto de calendario).
  H3: Otra cosa (IV promedio de toda la cadena, IV de un strike especifico, etc.).
"""

import pandas as pd
import numpy as np

ROOT = r"C:\Users\Windows\Desktop\Nueva carpeta\Bases de datos mercado"
SPY_PATH = rf"{ROOT}\SPY ETF\DAILY\SPY DAILY.xlsx"
QQQ_PATH = rf"{ROOT}\QQQ ETF\DAILY\QQQ DAILY.xlsx"
IWM_PATH = rf"{ROOT}\IWM ETF\DAILY\IWM DAILY.xlsx"
VIX_PATH = rf"{ROOT}\Volatility\VIX\VIX_spot.csv"


def load_etf(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def load_vix(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"])
    df = df.rename(columns={"Time": "Date", "Latest": "VIX"})
    return df[["Date", "VIX"]].sort_values("Date").reset_index(drop=True)


def diagnose(name: str, etf: pd.DataFrame, vix: pd.DataFrame) -> None:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    print(f"Filas totales: {len(etf):,}")
    print(f"Rango fechas: {etf['Date'].min().date()}  ->  {etf['Date'].max().date()}")
    iv = etf["Imp Vol"].dropna()
    print(f"Filas con Imp Vol: {len(iv):,}  ({100*len(iv)/len(etf):.1f}%)")
    if len(iv):
        print(f"Imp Vol disponible desde: {etf.loc[etf['Imp Vol'].notna(), 'Date'].min().date()}")
        print(f"Imp Vol stats: min={iv.min():.4f}  max={iv.max():.4f}  mean={iv.mean():.4f}  median={iv.median():.4f}")

    # Merge con VIX
    m = etf[["Date", "Imp Vol"]].merge(vix, on="Date", how="inner").dropna()
    if not len(m):
        print("Sin overlap con VIX.")
        return
    m["IV_pts"] = m["Imp Vol"] * 100  # asumiendo que Imp Vol esta en fraccion (0.17 = 17%)
    m["diff"] = m["IV_pts"] - m["VIX"]
    m["ratio"] = m["IV_pts"] / m["VIX"]
    print(f"\nOverlap con VIX: {len(m):,} dias")
    print(f"Correlacion IV*100 vs VIX: {m['IV_pts'].corr(m['VIX']):.4f}")
    print(f"Diferencia (IV*100 - VIX) en puntos:  mean={m['diff'].mean():+.3f}  std={m['diff'].std():.3f}  median={m['diff'].median():+.3f}")
    print(f"Ratio IV*100 / VIX:                   mean={m['ratio'].mean():.4f}  std={m['ratio'].std():.4f}  median={m['ratio'].median():.4f}")
    print(f"Percentiles del diff (puntos): p5={m['diff'].quantile(0.05):+.2f}  p50={m['diff'].quantile(0.5):+.2f}  p95={m['diff'].quantile(0.95):+.2f}")

    # Sample reciente
    print("\nUltimas 5 obs (overlap):")
    print(m.tail(5).to_string(index=False))


def main():
    vix = load_vix(VIX_PATH)
    print(f"VIX cargado: {len(vix):,} filas, {vix['Date'].min().date()} -> {vix['Date'].max().date()}")

    for name, path in [("SPY", SPY_PATH), ("QQQ", QQQ_PATH), ("IWM", IWM_PATH)]:
        diagnose(name, load_etf(path), vix)


if __name__ == "__main__":
    main()
