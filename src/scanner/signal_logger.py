"""
Logger de señales en SQLite para forward-test.

Cada vez que el scanner corre, guarda:
- Snapshot de features (date, spot, IV, RV, VRP, SMA, etc.)
- Si el filtro paso o no
- Si paso, los detalles del trade sugerido
- Outcome (a llenar manualmente o por proceso aparte)

Construye el dataset para validar el research en vivo.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "scanner_signals" / "signals.sqlite"


SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_timestamp TEXT NOT NULL,
    ticker TEXT NOT NULL,
    market_date TEXT NOT NULL,
    spot REAL,
    iv_atm REAL,
    rv_20d REAL,
    vrp REAL,
    sma_50 REAL,
    sma_200 REAL,
    above_sma200 INTEGER,
    vrp_threshold REAL,
    filter_pass INTEGER,
    trade_setup_json TEXT,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_scan_ts ON scans(scan_timestamp);
CREATE INDEX IF NOT EXISTS idx_ticker_date ON scans(ticker, market_date);
"""


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def log_scan(ticker: str, features: dict, filter_result: dict,
             trade_setup: dict | None, notes: str = ""):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO scans
               (scan_timestamp, ticker, market_date, spot, iv_atm, rv_20d, vrp,
                sma_50, sma_200, above_sma200, vrp_threshold, filter_pass,
                trade_setup_json, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(timespec="seconds"),
                ticker,
                str(features["date"].date() if hasattr(features["date"], "date")
                    else features["date"]),
                features["close"],
                features["iv_atm_today"],
                features["rv_20d"],
                features["vrp"],
                features["sma_50"],
                features["sma_200"],
                features["above_sma200"],
                filter_result.get("vrp_threshold_used"),
                int(filter_result["filter_pass"]),
                json.dumps(trade_setup) if trade_setup else None,
                notes,
            ),
        )
        conn.commit()
