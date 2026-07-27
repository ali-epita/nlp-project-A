"""Shared figure styling for the analysis scripts.

The categorical palette is an Okabe-Ito subset in a fixed order, validated for
colour-vision-deficiency separation and lightness on a white surface. Colours
follow entities (a strategy, a model) consistently across figures; magnitude
grids use a single-hue sequential ramp. Bars carry direct value labels, which
also serves as the required relief for the two lower-contrast hues.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed-order categorical palette (Okabe-Ito subset, CVD-validated).
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
SEQUENTIAL_CMAP = "Blues"
INK = "#1a1a1a"
MUTED = "#666666"
GRID = "#d9d9d9"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )


def bar_labels(ax, bars, fmt: str = "{:.3f}", fontsize: int = 8) -> None:
    """Direct value labels above each bar, in ink rather than the series colour."""
    for b in bars:
        ax.annotate(
            fmt.format(b.get_height()),
            (b.get_x() + b.get_width() / 2, b.get_height()),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            fontsize=fontsize,
            color=INK,
        )


def save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path.name}")
