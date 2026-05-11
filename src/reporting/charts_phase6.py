"""Charts especificos de Fase 6."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 130,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "font.size": 9,
    })


def equity_curves_chart(all_trades: pd.DataFrame, ticker: str,
                        out_path: Path) -> Path:
    setup()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sub = all_trades[all_trades["ticker"] == ticker].copy()
    for cfg, g in sub.groupby("config"):
        g = g.sort_values("exit_date").copy()
        g["cum"] = g["pnl_net_after_costs"].cumsum()
        ax.plot(g["exit_date"], g["cum"], label=f"{cfg} (n={len(g)})", lw=1.4)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("P&L acumulado USD por contrato")
    ax.set_title(f"{ticker} - Equity curves comparativas (1 contrato/trade)")
    ax.legend(fontsize=8, loc="upper left")
    ax.axhline(0, color="black", lw=0.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def pnl_distribution(trades: pd.DataFrame, label: str, out_path: Path) -> Path:
    setup()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    p = trades["pnl_net_after_costs"]
    ax.hist(p, bins=40, color="#3470b8", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.axvline(p.mean(), color="red", ls="--", lw=1.3, label=f"Mean = ${p.mean():.1f}")
    ax.set_xlabel("P&L por contrato (USD, neto de costos)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"{label} - Distribucion de P&L por trade")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def metrics_bar_compare(summary: pd.DataFrame, metric: str, title: str,
                        out_path: Path) -> Path:
    setup()
    pivot = summary.pivot(index="config", columns="ticker", values=metric)
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", ax=ax, alpha=0.85)
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(title="Ticker")
    ax.axhline(0, color="black", lw=0.5)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
