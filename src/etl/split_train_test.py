"""
Split train/test SELLADO. Una vez ejecutado, no se modifica.

Train: IV_AVAILABLE_FROM (2018-10-03) -> TRAIN_END (2024-10-03), inclusive.
Test:  TEST_START (2024-10-04) -> DATA_END (2026-03-12), inclusive.

ANTI-LOOK-AHEAD:
- Toda exploracion, fitting, seleccion de filtros y parametros se hace
  EXCLUSIVAMENTE sobre train.
- Test se evalua UNA SOLA VEZ al final del proyecto, sin re-tuneo.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import (
    DATA_CLEAN, IV_AVAILABLE_FROM, TEST_START, TRAIN_END, DATA_END,
)


def split_panel(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = DATA_CLEAN / f"{ticker}.parquet"
    df = pd.read_parquet(src)
    df["Date"] = pd.to_datetime(df["Date"])

    train = df[(df["Date"] >= pd.Timestamp(IV_AVAILABLE_FROM)) &
               (df["Date"] <= pd.Timestamp(TRAIN_END))].copy().reset_index(drop=True)
    test = df[(df["Date"] >= pd.Timestamp(TEST_START)) &
              (df["Date"] <= pd.Timestamp(DATA_END))].copy().reset_index(drop=True)

    return train, test


def main():
    (DATA_CLEAN / "train").mkdir(parents=True, exist_ok=True)
    (DATA_CLEAN / "test").mkdir(parents=True, exist_ok=True)

    print(f"Train window: {IV_AVAILABLE_FROM} -> {TRAIN_END}")
    print(f"Test  window: {TEST_START} -> {DATA_END}")
    print()

    summary = []
    for ticker in ["SPY", "QQQ", "IWM"]:
        train, test = split_panel(ticker)

        # Validar que no hay overlap
        if len(train) and len(test):
            assert train["Date"].max() < test["Date"].min(), \
                f"OVERLAP en {ticker}!"

        train_path = DATA_CLEAN / "train" / f"{ticker}.parquet"
        test_path = DATA_CLEAN / "test" / f"{ticker}.parquet"
        train.to_parquet(train_path, index=False)
        test.to_parquet(test_path, index=False)

        summary.append({
            "ticker": ticker,
            "train_rows": len(train),
            "train_first": train["Date"].min().date() if len(train) else None,
            "train_last": train["Date"].max().date() if len(train) else None,
            "test_rows": len(test),
            "test_first": test["Date"].min().date() if len(test) else None,
            "test_last": test["Date"].max().date() if len(test) else None,
        })

    print(pd.DataFrame(summary).to_string(index=False))

    # SELLAR el holdout: marcar el archivo como read-only en disco mediante
    # un sentinel file que diga "no tocar".
    sentinel = DATA_CLEAN / "test" / "_HOLDOUT_SEALED_DO_NOT_USE_DURING_RESEARCH.txt"
    sentinel.write_text(
        "Este directorio contiene el HOLDOUT TEST.\n"
        "NO debe ser leido durante exploracion ni tuneo de parametros.\n"
        "Solo se evalua una vez al final del proyecto (Fase 7.2).\n"
        f"Sellado: train hasta {TRAIN_END}, test desde {TEST_START}.\n"
    )
    print(f"\nHoldout sellado con sentinel: {sentinel.name}")


if __name__ == "__main__":
    main()
