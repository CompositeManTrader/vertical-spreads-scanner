"""Plot helpers que guardan PNGs listos para insertar en Word."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    })


def histogram_with_normal(returns: pd.Series, title: str, out_path: Path,
                          bins: int = 40) -> Path:
    setup_style()
    r = returns.dropna() * 100  # a porcentaje
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.hist(r, bins=bins, density=True, alpha=0.65, color="#3470b8",
            edgecolor="white", label="Empirico")

    mu, sigma = r.mean(), r.std(ddof=1)
    x = np.linspace(r.min(), r.max(), 300)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=1.7,
            label=f"Normal(μ={mu:.2f}%, σ={sigma:.2f}%)")

    ax.axvline(np.percentile(r, 5), ls="--", color="black", lw=1, alpha=0.7,
               label=f"P5 emp = {np.percentile(r, 5):.2f}%")
    ax.set_xlabel("Log return (%)")
    ax.set_ylabel("Densidad")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def qq_plot(returns: pd.Series, title: str, out_path: Path) -> Path:
    setup_style()
    r = returns.dropna() * 100
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    stats.probplot(r, dist="norm", plot=ax)
    ax.set_title(title)
    ax.get_lines()[0].set_markerfacecolor("#3470b8")
    ax.get_lines()[0].set_markersize(3)
    ax.get_lines()[1].set_color("red")
    ax.get_lines()[1].set_linewidth(1.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def autocorrelation_plot(returns: pd.Series, title: str, out_path: Path,
                         max_lag: int = 60) -> Path:
    from statsmodels.tsa.stattools import acf
    setup_style()
    r = returns.dropna()
    r2 = (r ** 2).dropna()
    acf_r = acf(r, nlags=max_lag, fft=False)
    acf_r2 = acf(r2, nlags=max_lag, fft=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    n = len(r)
    ci = 1.96 / np.sqrt(n)

    axes[0].bar(range(max_lag + 1), acf_r, width=0.8, color="#3470b8", alpha=0.8)
    axes[0].axhline(ci, color="red", ls="--", lw=0.9)
    axes[0].axhline(-ci, color="red", ls="--", lw=0.9)
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set_title(f"ACF returns - {title}")
    axes[0].set_xlabel("Lag (dias)")
    axes[0].set_ylabel("ACF")

    axes[1].bar(range(max_lag + 1), acf_r2, width=0.8, color="#cc4444", alpha=0.8)
    axes[1].axhline(ci, color="red", ls="--", lw=0.9)
    axes[1].axhline(-ci, color="red", ls="--", lw=0.9)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_title(f"ACF squared returns (vol clustering) - {title}")
    axes[1].set_xlabel("Lag (dias)")
    axes[1].set_ylabel("ACF")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def correlation_heatmap(corr: pd.DataFrame, title: str, out_path: Path) -> Path:
    import seaborn as sns
    setup_style()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlBu_r",
                center=0, vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def yearly_extremes_bar(yearly: pd.DataFrame, ticker: str, T: int,
                        out_path: Path) -> Path:
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    years = yearly["year"].values
    worst = yearly[f"worst_ret_pct_T{T}"].values
    best = yearly[f"best_ret_pct_T{T}"].values
    x = np.arange(len(years))
    w = 0.4
    ax.bar(x - w / 2, worst, w, color="#cc4444", label="Peor ret T-dias")
    ax.bar(x + w / 2, best, w, color="#44aa44", label="Mejor ret T-dias")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Log return (%)")
    ax.set_title(f"{ticker} - Mejor y peor retorno {T}d por año")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
