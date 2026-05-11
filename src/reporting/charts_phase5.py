"""Charts especificos de Fase 5."""

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


def heatmap_metric(s: pd.DataFrame, ticker: str, width: float, metric: str,
                    title: str, out_path: Path) -> Path:
    setup()
    sub = s[(s["ticker"] == ticker) & (s["width"] == width)].copy()
    sub["delta_short"] = sub["delta_short"].abs()
    pivot = sub.pivot(index="T_days", columns="delta_short", values=metric)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                center=0 if metric == "sharpe" else None,
                ax=ax, cbar_kws={"label": metric})
    ax.set_xlabel("Delta short")
    ax.set_ylabel("DTE")
    ax.set_title(f"{ticker} - {title} - width=${width}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def curve_metric_by_delta(s: pd.DataFrame, ticker: str, T_days: int,
                          metric: str, out_path: Path) -> Path:
    setup()
    sub = s[(s["ticker"] == ticker) & (s["T_days"] == T_days)].copy()
    sub["delta_short"] = sub["delta_short"].abs()
    fig, ax = plt.subplots(figsize=(7, 4))
    for w in sorted(sub["width"].unique()):
        ss = sub[sub["width"] == w].sort_values("delta_short")
        ax.plot(ss["delta_short"], ss[metric], "-o", label=f"width=${w}")
    ax.set_xlabel("Delta short |Δ|")
    ax.set_ylabel(metric)
    ax.set_title(f"{ticker} - {metric} vs delta - T={T_days}d")
    ax.legend()
    if metric == "sharpe":
        ax.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def credit_loss_scatter(s: pd.DataFrame, ticker: str, out_path: Path) -> Path:
    setup()
    sub = s[s["ticker"] == ticker].copy()
    sub["delta_short"] = sub["delta_short"].abs()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sc = ax.scatter(sub["full_loss_rate"] * 100, sub["expectancy_pct_of_maxloss"] * 100,
                    c=sub["delta_short"], s=80, cmap="viridis", alpha=0.85, edgecolors="black", lw=0.4)
    plt.colorbar(sc, ax=ax, label="Delta short")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Full loss rate (%)")
    ax.set_ylabel("Expectancy / max loss (%)")
    ax.set_title(f"{ticker} - Frontera riesgo (full_loss_rate) vs retorno (expectancy/maxloss)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
