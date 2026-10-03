"""
research/quant_style.py
========================
Shared matplotlib style for all QuantAgent research plots.

Style: "quant-arb" — minimal, publication-ready, robust statistical arbitrage.

Usage (one line at the top of any plot script):
    from research.quant_style import apply, C, annotate_band, hline, vline

Palette C (indexed):
    C.BLUE   C.RED   C.GREEN   C.AMBER   C.PURPLE   C.TEAL
    C.GRAY   C.DARK

apply() must be called before any plt.figure() call.
"""
from __future__ import annotations
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ── Palette ──────────────────────────────────────────────────────────────────

class C:
    BLUE   = "#1A6EBD"
    RED    = "#C0392B"
    GREEN  = "#229954"
    AMBER  = "#D68910"
    PURPLE = "#7D3C98"
    TEAL   = "#117A65"
    GRAY   = "#888888"
    DARK   = "#1A1A1A"
    LGRAY  = "#CCCCCC"
    BG     = "#FAFAFA"

    # Fills (same hue, 20% alpha equivalent hex)
    BLUE_F   = "#D6E8F7"
    RED_F    = "#FADBD8"
    GREEN_F  = "#D5F5E3"
    AMBER_F  = "#FDEBD0"

# ── Style application ─────────────────────────────────────────────────────────

def apply(backend: str = "Agg") -> None:
    """
    Apply the quant-arb style globally.
    Call once at the top of every plot script, before any figure is created.
    """
    matplotlib.use(backend)
    try:
        plt.style.use("quant-arb")
    except OSError:
        # Fallback if .mplstyle not found — apply inline
        plt.rcParams.update({
            "figure.facecolor":    "white",
            "axes.facecolor":      "#FAFAFA",
            "axes.edgecolor":      "#CCCCCC",
            "axes.linewidth":      0.8,
            "axes.spines.top":     False,
            "axes.spines.right":   False,
            "axes.grid":           True,
            "axes.axisbelow":      True,
            "grid.color":          "#E5E5E5",
            "grid.linewidth":      0.6,
            "grid.linestyle":      "-",
            "lines.linewidth":     1.4,
            "xtick.direction":     "out",
            "ytick.direction":     "out",
            "xtick.color":         "#555555",
            "ytick.color":         "#555555",
            "xtick.labelsize":     8,
            "ytick.labelsize":     8,
            "axes.labelsize":      9,
            "axes.titlesize":      10,
            "axes.titleweight":    "bold",
            "legend.fontsize":     8,
            "legend.facecolor":    "white",
            "legend.edgecolor":    "#DDDDDD",
            "font.family":         "sans-serif",
            "text.color":          "#1A1A1A",
            "figure.dpi":          150,
            "savefig.dpi":         200,
            "savefig.bbox":        "tight",
            "savefig.facecolor":   "white",
            "patch.edgecolor":     "none",
        })

# ── Convenience helpers ───────────────────────────────────────────────────────

def hline(ax, y: float, label: str | None = None,
          color: str = C.GRAY, lw: float = 0.9, ls: str = "--") -> None:
    """Horizontal reference line."""
    ax.axhline(y, color=color, lw=lw, ls=ls, zorder=1)
    if label:
        ax.text(ax.get_xlim()[1], y, f"  {label}", va="center",
                fontsize=7, color=color)


def vline(ax, x, label: str | None = None,
          color: str = C.AMBER, lw: float = 1.0, ls: str = ":") -> None:
    """Vertical event marker."""
    ax.axvline(x, color=color, lw=lw, ls=ls, zorder=2)
    if label:
        ax.text(x, ax.get_ylim()[1], f" {label}", va="top",
                fontsize=7, color=color, rotation=90)


def band(ax, x, y_lo, y_hi,
         color: str = C.BLUE, alpha: float = 0.15, label: str | None = None) -> None:
    """Shaded confidence / range band."""
    ax.fill_between(x, y_lo, y_hi, color=color, alpha=alpha,
                    label=label, linewidth=0)


def annotate_stat(ax, x, y, text: str, color: str = C.DARK) -> None:
    """Small stat annotation box."""
    ax.annotate(text, xy=(x, y),
                fontsize=7, color=color,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=C.LGRAY, lw=0.6, alpha=0.9))


def label_last(ax, series_x, series_y, fmt: str = "{:.4f}",
               color: str = C.BLUE, offset: tuple = (4, 0)) -> None:
    """Label the last point of a series."""
    ax.annotate(fmt.format(series_y.iloc[-1]),
                xy=(series_x.iloc[-1], series_y.iloc[-1]),
                xytext=offset, textcoords="offset points",
                fontsize=7.5, color=color, va="center")


def pct_formatter(ax_attr: str = "y") -> None:
    """Apply % formatter to an axis inside the caller's ax — call as:
       pct_formatter() after setting ax.
    Use: ax.yaxis.set_major_formatter(pct_fmt())
    """
    return ticker.FuncFormatter(lambda v, _: f"{v:+.3f}%")


def pct_fmt() -> ticker.FuncFormatter:
    return ticker.FuncFormatter(lambda v, _: f"{v:+.3f}%")


def price_fmt() -> ticker.FuncFormatter:
    return ticker.FuncFormatter(lambda v, _: f"${v:,.0f}")


def watermark(fig, text: str = "QuantAgent Research | Not Financial Advice") -> None:
    fig.text(0.99, 0.01, text, ha="right", va="bottom",
             fontsize=6.5, color=C.LGRAY, style="italic")
