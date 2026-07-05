"""Inspecciona el master file de volatilidad: que columnas hay, desde cuando, gaps."""

import pandas as pd

PATH = r"C:\Users\Windows\Desktop\Master_Historico_Completo_2000_2026.csv"

df = pd.read_csv(PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print(f"Total filas: {len(df):,}")
print(f"Rango fechas: {df['Date'].min().date()}  ->  {df['Date'].max().date()}")
print(f"Total columnas: {len(df.columns)}")
print()
print(f"{'Columna':<25} {'Non-null':>10} {'%':>7}  {'Primera fecha con dato':>25}")
print("-" * 75)
for c in df.columns:
    if c == "Date":
        continue
    n = df[c].notna().sum()
    pct = 100 * n / len(df)
    if n:
        first = df.loc[df[c].notna(), "Date"].min().date()
    else:
        first = "N/A"
    print(f"{c:<25} {n:>10,} {pct:>6.1f}%  {str(first):>25}")
