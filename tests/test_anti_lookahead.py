"""
Tests anti-look-ahead.

Filosofia: para cualquier indicador f(serie), si f es point-in-time entonces
            f(serie[:t+1])[t] == f(serie)[t]
es decir: el valor de f en t no cambia si trunco la serie a partir de t+1.
Si cambia -> f esta usando informacion futura.

Ejecutar con: pytest tests/test_anti_lookahead.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.indicators import (
    days_since_ath, drawdown_from_high, iv_percentile, iv_rank,
    log_return_back, log_return_forward, realized_vol, sma, sma_slope,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_series():
    """Serie sintetica de 500 puntos con un rango razonable."""
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.012, 500)
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices)


@pytest.fixture
def random_iv():
    """IV sintetica entre 0.10 y 0.50."""
    np.random.seed(7)
    return pd.Series(np.random.uniform(0.10, 0.50, 500))


# ---------------------------------------------------------------------------
# Helper: assert point-in-time consistency
# ---------------------------------------------------------------------------

def assert_pit_consistent(func, series, t_test_indices, **kwargs):
    """
    Verifica que func(series[:t+1])[t] == func(series)[t] para varios t.
    Si difiere -> func usa info futura.
    """
    full = func(series, **kwargs)
    for t in t_test_indices:
        truncated = func(series.iloc[:t + 1], **kwargs)
        full_val = full.iloc[t]
        trunc_val = truncated.iloc[t]
        if pd.isna(full_val) and pd.isna(trunc_val):
            continue
        if pd.isna(full_val) or pd.isna(trunc_val):
            raise AssertionError(
                f"NaN mismatch en t={t}: full={full_val}  trunc={trunc_val}"
            )
        assert np.isclose(full_val, trunc_val, atol=1e-12), \
            f"LOOK-AHEAD detectado en t={t}: full={full_val}  trunc={trunc_val}"


# ---------------------------------------------------------------------------
# Tests anti-look-ahead por funcion
# ---------------------------------------------------------------------------

class TestPointInTime:

    def test_iv_rank_no_lookahead(self, random_iv):
        assert_pit_consistent(iv_rank, random_iv, [251, 300, 400, 499], lookback=252)

    def test_iv_percentile_no_lookahead(self, random_iv):
        assert_pit_consistent(iv_percentile, random_iv, [251, 300, 400, 499], lookback=252)

    def test_realized_vol_no_lookahead(self, random_series):
        rets = log_return_back(random_series)
        assert_pit_consistent(realized_vol, rets, [50, 100, 200, 499], window=20)

    def test_drawdown_expanding_no_lookahead(self, random_series):
        assert_pit_consistent(drawdown_from_high, random_series, [50, 100, 200, 499])

    def test_drawdown_rolling_no_lookahead(self, random_series):
        assert_pit_consistent(drawdown_from_high, random_series, [100, 200, 499], lookback=60)

    def test_days_since_ath_no_lookahead(self, random_series):
        assert_pit_consistent(days_since_ath, random_series, [50, 100, 200, 499])

    def test_sma_no_lookahead(self, random_series):
        assert_pit_consistent(sma, random_series, [50, 100, 200, 499], window=50)

    def test_sma_slope_no_lookahead(self, random_series):
        assert_pit_consistent(sma_slope, random_series, [100, 200, 499], window=50, slope_lookback=20)

    def test_log_return_back_no_lookahead(self, random_series):
        assert_pit_consistent(log_return_back, random_series, [10, 100, 499], periods=5)


# ---------------------------------------------------------------------------
# Tests semanticos: ret_fwd debe ser distinto de ret_back
# ---------------------------------------------------------------------------

class TestForwardVsBackward:

    def test_forward_label_is_future(self, random_series):
        """ret_fwd(t, k) debe ser ln(P(t+k)/P(t))."""
        k = 30
        fwd = log_return_forward(random_series, periods=k)
        for t in [50, 100, 400]:
            expected = np.log(random_series.iloc[t + k] / random_series.iloc[t])
            assert np.isclose(fwd.iloc[t], expected), \
                f"fwd ret en t={t} mal calculado"

    def test_backward_feature_is_past(self, random_series):
        """ret_back(t, k) debe ser ln(P(t)/P(t-k))."""
        k = 5
        back = log_return_back(random_series, periods=k)
        for t in [10, 100, 400]:
            expected = np.log(random_series.iloc[t] / random_series.iloc[t - k])
            assert np.isclose(back.iloc[t], expected), \
                f"back ret en t={t} mal calculado"

    def test_forward_truncation_produces_nan(self, random_series):
        """Las ultimas k posiciones de ret_fwd deben ser NaN (no hay futuro)."""
        k = 30
        fwd = log_return_forward(random_series, periods=k)
        assert fwd.iloc[-k:].isna().all(), "ret_fwd debe tener NaN al final"
        # Y los anteriores no deben ser NaN (excepto los iniciales si los hay)
        assert not fwd.iloc[100:-k].isna().any(), "ret_fwd no debe tener NaN en el medio"


# ---------------------------------------------------------------------------
# Tests sobre el panel real
# ---------------------------------------------------------------------------

class TestPanelIntegrity:

    @pytest.fixture
    def spy_panel(self):
        from config.settings import DATA_CLEAN
        return pd.read_parquet(DATA_CLEAN / "SPY.parquet")

    def test_panel_dates_ascending(self, spy_panel):
        assert spy_panel["Date"].is_monotonic_increasing

    def test_panel_no_duplicate_dates(self, spy_panel):
        assert not spy_panel["Date"].duplicated().any()

    def test_iv_in_valid_range(self, spy_panel):
        iv = spy_panel["iv_atm_barchart"].dropna()
        assert (iv > 0).all() and (iv < 5).all(), "IV fuera de rango razonable"

    def test_iv_rank_recalculated_in_valid_range(self, spy_panel):
        rk = spy_panel["iv_rank_252"].dropna()
        assert (rk >= 0).all() and (rk <= 1).all(), "iv_rank_252 fuera de [0,1]"

    def test_iv_percentile_recalculated_in_valid_range(self, spy_panel):
        pc = spy_panel["iv_percentile_252"].dropna()
        assert (pc >= 0).all() and (pc <= 1).all(), "iv_percentile_252 fuera de [0,1]"

    def test_rates_lagged_one_day(self, spy_panel):
        """
        rfr_pct(t) debe usar DGS3MO publicado <= t-1.
        Test indirecto: la ultima fecha del panel debe tener rfr_pct asignado
        (despues del lag y forward fill).
        """
        last_row = spy_panel.iloc[-1]
        assert pd.notna(last_row["rfr_pct"]), "rfr_pct debe estar disponible al final"


# ---------------------------------------------------------------------------
# Tests de split train/test
# ---------------------------------------------------------------------------

class TestTrainTestSplit:

    def test_no_overlap_dates(self):
        from config.settings import DATA_CLEAN
        for ticker in ["SPY", "QQQ", "IWM"]:
            train = pd.read_parquet(DATA_CLEAN / "train" / f"{ticker}.parquet")
            test = pd.read_parquet(DATA_CLEAN / "test" / f"{ticker}.parquet")
            assert train["Date"].max() < test["Date"].min(), \
                f"{ticker}: overlap entre train y test"

    def test_holdout_sentinel_exists(self):
        from config.settings import DATA_CLEAN
        sentinel = DATA_CLEAN / "test" / "_HOLDOUT_SEALED_DO_NOT_USE_DURING_RESEARCH.txt"
        assert sentinel.exists(), "Sentinel del holdout debe existir"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
