"""Charts especificos de Fase 2."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 130,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
    })


def heatmap_prob(grid: pd.DataFrame, ticker: str, prob_col: str, title: str,
                 out_path: Path) -> Path:
    """Heatmap (T, x%) de una probabilidad."""
    setup()
    sub = grid[grid["ticker"] == ticker].copy()
    pivot = sub.pivot(index="T", columns="x_pct_below", values=prob_col)
    pivot = pivot.sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="RdYlGn_r",
                cbar_kws={"label": "%"}, ax=ax,
                xticklabels=[f"{c:.0%}" for c in pivot.columns])
    ax.set_xlabel("Strike % debajo del spot")
    ax.set_ylabel("DTE")
    ax.set_title(f"{title} - {ticker}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def prob_curve(grid: pd.DataFrame, ticker: str, T: int, out_path: Path) -> Path:
    """Curva P(ITM) y P(touch) vs x% con bandas Wilson."""
    setup()
    sub = grid[(grid["ticker"] == ticker) & (grid["T"] == T)].sort_values("x_pct_below")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = sub["x_pct_below"] * 100
    ax.plot(x, sub["p_itm"] * 100, "-o", color="#cc4444", label="P(ITM at expiry)")
    ax.fill_between(x, sub["p_itm_ci_lo"] * 100, sub["p_itm_ci_hi"] * 100,
                    color="#cc4444", alpha=0.15)
    ax.plot(x, sub["p_touch"] * 100, "-s", color="#3470b8", label="P(touch durante la vida)")
    ax.fill_between(x, sub["p_touch_ci_lo"] * 100, sub["p_touch_ci_hi"] * 100,
                    color="#3470b8", alpha=0.15)
    ax.plot(x, sub["p_touch_recovered"] * 100, ":^", color="#888888",
            label="P(touch & recovered)")
    ax.set_xlabel("Strike % debajo del spot")
    ax.set_ylabel("Probabilidad (%)")
    ax.set_title(f"{ticker} - T={T} dias - Probabilidades empiricas (CI 95% Wilson)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def ratio_curve(grid: pd.DataFrame, ticker: str, out_path: Path) -> Path:
    """Ratio P(touch)/P(ITM) vs x% para todos los DTE."""
    setup()
    sub = grid[grid["ticker"] == ticker].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    for T in sorted(sub["T"].unique()):
        s = sub[sub["T"] == T].sort_values("x_pct_below")
        ax.plot(s["x_pct_below"] * 100, s["ratio_touch_itm"], "-o", label=f"T={T}d")
    ax.axhline(2.0, color="black", ls="--", lw=1, alpha=0.6,
               label="Random walk (theory: 2x)")
    ax.set_xlabel("Strike % debajo del spot")
    ax.set_ylabel("Ratio P(touch) / P(ITM)")
    ax.set_title(f"{ticker} - Ratio empirico touch / ITM")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def gap_risk_chart(gaps: pd.DataFrame, out_path: Path) -> Path:
    setup()
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.18
    xs = np.array([0.01, 0.02, 0.03, 0.05, 0.07])
    for i, t in enumerate(["SPY", "QQQ", "IWM"]):
        sub = gaps[gaps["ticker"] == t].sort_values("x_pct")
        ax.bar(np.arange(len(xs)) + (i - 1) * width, sub["p_overnight_gap_down"] * 100,
               width=width, label=f"{t} overnight gap-down", alpha=0.85)
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels([f">{x:.0%}" for x in xs])
    ax.set_ylabel("% de dias")
    ax.set_xlabel("Magnitud del gap-down (apertura vs cierre previo)")
    ax.set_title("Frecuencia de gap-downs overnight")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
