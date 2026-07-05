"""
Constantes globales del proyecto.

CONVENCION ANTI-LOOK-AHEAD:
- Toda decision en t usa info <= t.
- Returns hacia adelante (labels) se nombran ret_fwd_*.
- Returns hacia atras (features) se nombran ret_back_*.
- Stats normalizadas usan expanding/rolling cerradas en t.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Users\Windows\Desktop\Codigos\Vertical Spreads")

DATA_RAW_BASE = Path(r"C:\Users\Windows\Desktop\Nueva carpeta\Bases de datos mercado")
MASTER_VOL_PATH = Path(r"C:\Users\Windows\Desktop\Master_Historico_Completo_2000_2026.csv")

DATA_CLEAN = PROJECT_ROOT / "data" / "clean"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
REPORTS = PROJECT_ROOT / "reports"

ETF_PATHS = {
    "SPY": DATA_RAW_BASE / "SPY ETF" / "DAILY" / "SPY DAILY.xlsx",
    "QQQ": DATA_RAW_BASE / "QQQ ETF" / "DAILY" / "QQQ DAILY.xlsx",
    "IWM": DATA_RAW_BASE / "IWM ETF" / "DAILY" / "IWM DAILY.xlsx",
}

VIX_SPOT_PATH = DATA_RAW_BASE / "Volatility" / "VIX" / "VIX_spot.csv"
VXX_PATH = DATA_RAW_BASE / "Volatility" / "VXX" / "VXX_daily.csv"

# ---------------------------------------------------------------------------
# Constantes financieras
# ---------------------------------------------------------------------------
# Dividend yields constantes por ticker (anualizado).
# Source: trailing 12m yield aprox a marzo 2026.
DIVIDEND_YIELDS = {
    "SPY": 0.013,
    "QQQ": 0.006,
    "IWM": 0.014,
}

# Dividend yields POINT-IN-TIME por anio (trailing 12m vigente al inicio de
# cada anio, valores publicos aproximados). Evita usar el yield de fin de
# muestra retroactivamente (hallazgo del code review adversarial).
DIVIDEND_YIELDS_BY_YEAR = {
    "SPY": {2018: 0.018, 2019: 0.019, 2020: 0.018, 2021: 0.013, 2022: 0.014,
            2023: 0.015, 2024: 0.013, 2025: 0.012, 2026: 0.013},
    "QQQ": {2018: 0.008, 2019: 0.008, 2020: 0.007, 2021: 0.005, 2022: 0.006,
            2023: 0.006, 2024: 0.006, 2025: 0.006, 2026: 0.006},
    "IWM": {2018: 0.012, 2019: 0.013, 2020: 0.013, 2021: 0.010, 2022: 0.012,
            2023: 0.014, 2024: 0.013, 2025: 0.013, 2026: 0.014},
}

# Costos de trading (Schwab approx, por trade completo: 2 legs apertura + 2 legs cierre).
COMMISSION_PER_CONTRACT = 0.65  # USD por contrato por leg
COMMISSION_PER_TRADE = COMMISSION_PER_CONTRACT * 4  # = 2.60
SLIPPAGE_PER_TRADE = 0.20  # USD aproximado por trade completo
TOTAL_COST_PER_TRADE = COMMISSION_PER_TRADE + SLIPPAGE_PER_TRADE  # = 2.80

# ---------------------------------------------------------------------------
# Rangos temporales
# ---------------------------------------------------------------------------
# IV disponible en datasets desde:
IV_AVAILABLE_FROM = "2018-10-03"

# Train/test split SELLADO. NO modificar tras explorar resultados.
TRAIN_END = "2024-10-03"     # inclusive, ultimo dia de train
TEST_START = "2024-10-04"    # inclusive, primer dia de test
DATA_END = "2026-03-12"      # ultimo dia con dato del master

# ---------------------------------------------------------------------------
# Parametros de la estrategia base
# ---------------------------------------------------------------------------
DTE_GRID = [30, 35, 40, 45]
DELTA_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
WIDTH_GRID = [3, 5, 10, 15]
PCT_BELOW_GRID = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]

# ---------------------------------------------------------------------------
# Reglas estadisticas
# ---------------------------------------------------------------------------
MIN_OBS_PER_REGIME = 100  # buckets con menos no se reportan
BONFERRONI_FAMILY_SIZE = 17  # ~17 regimenes testeados; ajustar
SIGNIFICANCE_LEVEL = 0.05
