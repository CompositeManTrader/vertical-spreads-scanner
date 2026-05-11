"""
Scanner principal: corre EOD para cada ticker, evalua el filtro y notifica.

Uso:
    python -m src.scanner.main

Output:
- Print en consola con detalles del estado actual.
- Si el filtro pasa: detalles del trade sugerido + alert.
- Log a SQLite (data/scanner_signals/signals.sqlite) para forward-test.

Tickers: SPY y QQQ (NO IWM segun research).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.scanner.data_feed import (
    fetch_atm_iv_30d, fetch_history, fetch_vix_close,
)
from src.scanner.filter_engine import compute_features_for_today, evaluate_filter
from src.scanner.signal_logger import log_scan
from src.scanner.trade_builder import build_pcs_setup

TICKERS = ["SPY", "QQQ"]

# Config segun research final (ver MASTER_Final_Report.docx)
DELTA_SHORT = -0.30
DTE = 45
WIDTH = 5.0
TP_PCT = 0.50
SL_MULT = 2.0
TIME_STOP_DTE = 14
VRP_THRESHOLD = 0.03  # 3 vol points

# Risk-free rate fallback (DGS3MO actual; idealmente bajar de FRED en vivo)
RFR_PCT_FALLBACK = 3.7


SEP = "=" * 78


def colorize(s: str, color: str = "default") -> str:
    """ANSI colors si terminal soporta."""
    codes = {
        "green": "\033[92m", "yellow": "\033[93m", "red": "\033[91m",
        "blue": "\033[94m", "bold": "\033[1m", "end": "\033[0m",
    }
    if color in codes:
        return f"{codes[color]}{s}{codes['end']}"
    return s


def scan_ticker(ticker: str, vrp_threshold: float = VRP_THRESHOLD) -> dict:
    print(f"\n{SEP}\n{colorize(ticker, 'bold')}\n{SEP}")

    # 1. Historia + IV
    hist = fetch_history(ticker, years=2)
    iv, iv_meta = fetch_atm_iv_30d(ticker)
    if iv is None:
        print(f"  ERROR obteniendo IV: {iv_meta}")
        return {"ticker": ticker, "error": iv_meta}

    # 2. Features
    feats = compute_features_for_today(hist, iv)

    # 3. Filtro
    fr = evaluate_filter(feats, vrp_threshold=vrp_threshold)

    # 4. Print estado
    print(f"  Fecha:                {feats['date'].date()}")
    print(f"  Spot:                 ${feats['close']:.2f}")
    print(f"  IV ATM 30D:           {feats['iv_atm_today']*100:.2f}%   (yfinance, {iv_meta.get('method','?')})")
    print(f"  RV 20d:               {feats['rv_20d']*100:.2f}%")
    print(f"  VRP (IV - RV):        {feats['vrp']*100:+.2f}pp   (threshold: {vrp_threshold*100:.1f}pp)")
    print(f"  SMA50:                ${feats['sma_50']:.2f}")
    print(f"  SMA200:               ${feats['sma_200']:.2f}")
    print(f"  Spot vs SMA200:       {feats['price_to_sma200_pct']*100:+.2f}%")

    print(f"\n  Filtros:")
    color1 = "green" if fr["above_sma200_pass"] else "red"
    color2 = "green" if fr["vrp_pass"] else "red"
    print(f"    above_sma200:   {colorize('PASS' if fr['above_sma200_pass'] else 'FAIL', color1)}")
    print(f"    vrp >= {vrp_threshold*100:.1f}pp: {colorize('PASS' if fr['vrp_pass'] else 'FAIL', color2)}  (vrp={feats['vrp']*100:+.2f}pp)")

    # 5. Si pasa: trade setup
    setup = None
    if fr["filter_pass"]:
        print(f"\n  {colorize('>>> SENAL: FILTRO PASA. ABRIR TRADE. <<<', 'green')}")
        setup = build_pcs_setup(
            ticker=ticker, spot=feats["close"], iv=feats["iv_atm_today"],
            rfr_pct=RFR_PCT_FALLBACK, delta_short=DELTA_SHORT,
            dte=DTE, width=WIDTH, tp_pct=TP_PCT, sl_mult=SL_MULT,
            time_stop_dte=TIME_STOP_DTE,
        )
        print(f"\n  Trade sugerido:")
        print(f"    Estructura:        Put Credit Spread (vender K_short, comprar K_long)")
        print(f"    K_short (sell):    ${setup['K_short_recommended']:>7.2f}  (delta {setup['K_short_delta_actual']:+.3f})")
        print(f"    K_long  (buy):     ${setup['K_long_recommended']:>7.2f}  (delta {setup['K_long_delta_actual']:+.3f})")
        print(f"    Width:             ${setup['width_recommended']}")
        print(f"    Expiracion:        {setup['expiry_date']}  (DTE: {setup['dte_at_open']})")
        print(f"    Credito BSM teor:  ${setup['credit_per_share_BSM']:.3f}/share = ${setup['credit_per_contract_BSM']:.2f}/contract")
        print(f"    Max loss:          ${setup['max_loss_per_share']:.3f}/share = ${setup['max_loss_per_contract']:.2f}/contract")
        print(f"    Credit/MaxLoss:    {setup['credit_to_maxloss_ratio']*100:.1f}%")
        print(f"\n  Reglas de gestion:")
        print(f"    Take profit:       cerrar @ ${setup['tp_per_contract']:.2f}/contract (50% del credito)")
        print(f"    Stop loss:         cerrar @ ${setup['sl_per_contract']:.2f}/contract (-2x credito)")
        print(f"    Time stop:         cerrar el {setup['time_stop_date']} ({TIME_STOP_DTE} DTE remanentes)")
        print(f"    Costos estimados:  ${setup['estimated_costs_total']:.2f} por trade completo")
    else:
        print(f"\n  {colorize('--- SIN SENAL: filtro NO pasa. No abrir trade. ---', 'yellow')}")
        if not fr["above_sma200_pass"]:
            print(f"    Razon: precio debajo de SMA200")
        if not fr["vrp_pass"]:
            print(f"    Razon: VRP {feats['vrp']*100:+.2f}pp < threshold {vrp_threshold*100:.1f}pp")

    # 6. Log a SQLite
    log_scan(ticker, feats, fr, setup, notes=f"yfinance prototype, RFR fallback {RFR_PCT_FALLBACK}%")

    return {"ticker": ticker, "features": feats, "filter": fr, "setup": setup}


def main():
    print(f"\n{SEP}")
    print(f"  VERTICAL SPREADS SCANNER  --  {Path(__file__).parent.name}")
    print(f"  Config: delta_short={DELTA_SHORT}, DTE={DTE}, width=${WIDTH}, vrp_thr={VRP_THRESHOLD*100:.1f}pp")
    print(f"{SEP}")

    # VIX (informativo)
    vix, vix_meta = fetch_vix_close()
    if vix:
        print(f"\n  VIX:  {vix:.2f}  ({vix_meta.get('date')})")

    results = []
    for t in TICKERS:
        try:
            r = scan_ticker(t)
            results.append(r)
        except Exception as e:
            print(f"  ERROR scanning {t}: {e}")

    # Resumen final
    print(f"\n{SEP}\n  RESUMEN\n{SEP}")
    for r in results:
        if "error" in r:
            print(f"  {r['ticker']}: ERROR")
            continue
        passed = r["filter"]["filter_pass"]
        sym = "[OK]" if passed else "[--]"
        print(f"  {sym}  {r['ticker']:5s} - filtro {'PASS' if passed else 'no pasa'}")
    print(f"\n  Logs: data/scanner_signals/signals.sqlite")
    print()


if __name__ == "__main__":
    main()
