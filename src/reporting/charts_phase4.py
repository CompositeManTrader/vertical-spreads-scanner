"""Charts especificos de Fase 4."""

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


def dist_histogram(records: pd.DataFrame, ticker: str, T_days: int,
                   out_path: Path) -> Path:
    setup()
    sub = records[(records["ticker"] == ticker) & (records["T_days"] == T_days) &
                  (records["iv_bump"] == 1.0)]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.hist(sub["dist_pct_below"] * 100, bins=40, color="#3470b8", alpha=0.85,
            edgecolor="white")
    ax.axvline(sub["dist_pct_below"].mean() * 100, color="red", ls="--",
               label=f"Mean = {sub['dist_pct_below'].mean()*100:.2f}%")
    ax.set_xlabel("Strike % debajo del spot (delta-20 dinamico)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"{ticker} - Distancia delta-20 strike to spot, T={T_days}d")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def dist_vs_iv_scatter(records: pd.DataFrame, ticker: str, T_days: int,
                       out_path: Path) -> Path:
    setup()
    sub = records[(records["ticker"] == ticker) & (records["T_days"] == T_days) &
                  (records["iv_bump"] == 1.0)]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.scatter(sub["iv_atm"] * 100, sub["dist_pct_below"] * 100,
               s=4, alpha=0.4, color="#3470b8")
    ax.set_xlabel("IV ATM Barchart (%)")
    ax.set_ylabel("Strike delta-20 % below spot")
    ax.set_title(f"{ticker} - Distancia delta-20 vs IV (T={T_days}d)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def empirical_vs_nominal(summary: pd.DataFrame, T_days: int,
                         out_path: Path) -> Path:
    setup()
    sub = summary[summary["T_days"] == T_days].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    bumps = sorted(sub["iv_bump"].unique())
    x = np.arange(len(bumps))
    colors = {"SPY": "#1f77b4", "QQQ": "#2ca02c", "IWM": "#d62728"}
    for i, ticker in enumerate(["SPY", "QQQ", "IWM"]):
        s = sub[sub["ticker"] == ticker].sort_values("iv_bump")
        offset = (i - 1) * width
        bars = ax.bar(x + offset, s["p_itm_emp"] * 100, width=width,
                      label=ticker, color=colors[ticker], alpha=0.85)
        for j, (b, p) in enumerate(zip(bumps, s["p_itm_emp"])):
            ax.text(j + offset, p * 100 + 0.3, f"{p*100:.1f}", ha="center", fontsize=7)
    ax.axhline(20, color="black", ls="--", lw=1.2,
               label="|Δ|=0.20 nominal (BSM expectation)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:.2f}" for b in bumps])
    ax.set_xlabel("IV bump (multiplicador del IV ATM, simula skew)")
    ax.set_ylabel("P(ITM) empirica %")
    ax.set_title(f"P(ITM) empirica con strike delta-20 vs |delta| nominal, T={T_days}d")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def regimes_dynamic_chart(d: pd.DataFrame, ticker: str, out_path: Path) -> Path:
    setup()
    sub = d[(d["ticker"] == ticker) & (d["T_days"] == 30) & (d["n"] >= 80)].copy()
    sub = sub.sort_values("p_itm")
    fig, ax = plt.subplots(figsize=(9, 5))
    p_uncond = sub["p_uncond"].iloc[0]
    colors = ["#44aa44" if p < p_uncond else "#cc4444" for p in sub["p_itm"]]
    y_pos = np.arange(len(sub))
    bars = ax.barh(y_pos, sub["p_itm"] * 100, color=colors, alpha=0.85)
    err_lo = (sub["p_itm"] - sub["ci_lo"]) * 100
    err_hi = (sub["ci_hi"] - sub["p_itm"]) * 100
    ax.errorbar(sub["p_itm"] * 100, y_pos,
                xerr=[err_lo, err_hi], fmt="none", color="black", capsize=3, lw=0.8)
    ax.axvline(p_uncond * 100, color="black", ls="--", lw=1.2,
               label=f"Unconditional = {p_uncond*100:.2f}%")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["condition"], fontsize=8)
    for i, (n, p, pb) in enumerate(zip(sub["n"], sub["p_itm"], sub["pvalue_bonferroni"])):
        marker = " ***" if pb < 0.05 else ""
        ax.text(p * 100 + 0.3, i, f" n={int(n)}{marker}", va="center", fontsize=7)
    ax.set_xlabel("P(ITM) %")
    ax.set_title(f"{ticker} - Multifactor con strike DINAMICO delta-20, T=30d")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
