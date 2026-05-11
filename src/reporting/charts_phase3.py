"""Charts especificos de Fase 3."""

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
        "axes.grid": True, "grid.alpha": 0.3, "font.size": 9,
    })


def regime_bar(df: pd.DataFrame, regime: str, ticker: str, T: int, x: float,
               out_path: Path) -> Path:
    """Barras P(ITM) por bucket de un regime, con CI."""
    setup()
    sub = df[(df["ticker"] == ticker) & (df["regime"] == regime) &
             (df["T"] == T) & (df["x_pct_below"] == x) & (df["outcome"] == "itm")].copy()
    sub = sub[sub["n"] >= 50].copy()
    if sub.empty:
        return None
    p_uncond = sub["p_uncond"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x_pos = np.arange(len(sub))
    ax.bar(x_pos, sub["p_outcome"] * 100, color="#3470b8", alpha=0.85)
    err_lo = (sub["p_outcome"] - sub["ci_lo"]) * 100
    err_hi = (sub["ci_hi"] - sub["p_outcome"]) * 100
    ax.errorbar(x_pos, sub["p_outcome"] * 100,
                yerr=[err_lo, err_hi], fmt="none", color="black", capsize=3)
    ax.axhline(p_uncond * 100, color="red", ls="--", lw=1.2,
               label=f"Unconditional = {p_uncond*100:.2f}%")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sub["bucket"], rotation=15)
    for i, (n, p) in enumerate(zip(sub["n"], sub["p_outcome"])):
        ax.text(i, p * 100 + 0.5, f"n={int(n)}", ha="center", fontsize=7)
    ax.set_ylabel("P(ITM) %")
    ax.set_title(f"{ticker} - {regime} - P(ITM) at T={T}d, strike {x:.0%} below")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def multifactor_chart(mf: pd.DataFrame, ticker: str, T: int, x: float,
                      out_path: Path) -> Path:
    setup()
    sub = mf[(mf["ticker"] == ticker) & (mf["T"] == T) &
             (mf["x_pct_below"] == x) & (mf["n"] >= 80)].copy()
    sub = sub.sort_values("p_outcome")
    fig, ax = plt.subplots(figsize=(9, 5))
    p_uncond = sub["p_uncond"].iloc[0]
    colors = ["#44aa44" if p < p_uncond else "#cc4444" for p in sub["p_outcome"]]
    y_pos = np.arange(len(sub))
    bars = ax.barh(y_pos, sub["p_outcome"] * 100, color=colors, alpha=0.85)
    err_lo = (sub["p_outcome"] - sub["ci_lo"]) * 100
    err_hi = (sub["ci_hi"] - sub["p_outcome"]) * 100
    ax.errorbar(sub["p_outcome"] * 100, y_pos,
                xerr=[err_lo, err_hi], fmt="none", color="black", capsize=3, lw=0.8)
    ax.axvline(p_uncond * 100, color="black", ls="--", lw=1.2,
               label=f"Unconditional = {p_uncond*100:.2f}%")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["condition"], fontsize=8)
    for i, (n, p, pb) in enumerate(zip(sub["n"], sub["p_outcome"], sub["pvalue_bonferroni"])):
        marker = " ***" if pb < 0.05 else ""
        ax.text(p * 100 + 0.3, i, f" n={int(n)}{marker}", va="center", fontsize=7)
    ax.set_xlabel("P(ITM) %")
    ax.set_title(f"{ticker} - Multifactor combinations - T={T}d, strike {x:.0%} below\n*** = significativo Bonferroni p<0.05")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
