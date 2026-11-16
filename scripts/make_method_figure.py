#!/usr/bin/env python3
"""Draw the audited shared caching abstraction without loading model code.

The PDF contains vector shapes and text. Tensor boundaries remain specific to
each backbone; the policy does not necessarily observe the current latent or
block input. Run from any directory with Python and matplotlib installed.

    python scripts/make_method_figure.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 6.875, 2.65
INK = "#283438"
BLUE = "#466F87"
CYAN = "#78B6C8"
YELLOW = "#F2D98E"
GRAY = "#666E72"
STYLE = {
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 9.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "savefig.facecolor": "white",
}


def draw_framework():
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, WIDTH), ylim=(.45, 3.10))
    ax.set_aspect("equal")
    ax.set_axis_off()

    def text(x, y, value, **kwargs):
        color = kwargs.pop("color", INK)
        return ax.text(x, y, value, ha="center", va="center", color=color,
                       fontsize=9.5, **kwargs)

    def box(x, y, width, height, label, *, fill="white", accent=False):
        ax.add_patch(FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.035",
            linewidth=.8, edgecolor=BLUE if accent else GRAY,
            facecolor=fill, zorder=3))
        text(x + width / 2, y + height / 2, label, linespacing=1.25, zorder=4)

    def arrow(start, end, *, dashed=False, color=INK, head=True):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>" if head else "-",
            mutation_scale=8, linewidth=.85, color=color,
            linestyle=(0, (3, 2)) if dashed else "-",
            shrinkA=0, shrinkB=0, zorder=2))

    def route(points, *, color=INK, head=True):
        """Draw an orthogonal path with a single arrowhead at its destination."""
        for start, end in zip(points[:-2], points[1:-1]):
            arrow(start, end, color=color, head=False)
        arrow(points[-2], points[-1], color=color, head=head)

    # A policy control path is separate from the current-feature data path.
    text(1.82, 2.96, "Policy signals / history")
    box(1.07, 2.36, 1.50, .43, "WHEN: reuse policy", fill=CYAN, accent=True)
    arrow((1.82, 2.87), (1.82, 2.81), color=BLUE)
    arrow((1.82, 2.34), (1.82, 1.54), dashed=True, color=BLUE)

    text(4.77, 2.96, "Fixed within each backbone", fontweight="bold")
    text(4.77, 2.75, "WHICH: full block stack")
    text(4.77, 2.55, "HOW: residual reuse")

    # Current features are prepared at every stage, including reuse stages.
    text(.68, 2.02, r"Stage $k$: $(z_k,t_k,c)$")
    arrow((.68, 1.90), (.68, 1.79))
    box(.10, 1.24, 1.16, .52, "Current feature\npreparation")
    arrow((1.28, 1.50), (1.78, 1.50))
    text(1.52, 1.67, r"$a_k$")
    ax.add_patch(Circle((1.82, 1.50), .032, facecolor=BLUE, edgecolor="none", zorder=5))

    # Both alternatives yield the current block-stack output b_k. Refresh
    # replaces the cache; reuse adds the held residual to the CURRENT input.
    box(2.73, 1.73, 1.74, .65,
        "Compute full block stack\n" + r"$b_k=F_k(a_k)$" + "\n" +
        r"Refresh: $r_k\leftarrow b_k-a_k$", fill=CYAN, accent=True)
    box(2.73, .65, 1.74, .55,
        "Reuse last residual\n" + r"$b_k=a_k+r_\tau$", fill=YELLOW, accent=True)
    route([(1.85, 1.50), (2.10, 1.50), (2.10, 2.055), (2.71, 2.055)], color=BLUE)
    route([(1.85, 1.50), (2.10, 1.50), (2.10, .925), (2.71, .925)])
    text(2.39, 2.21, "refresh", color=BLUE)
    text(2.39, .77, "reuse")

    route([(4.49, 2.055), (4.68, 2.055), (4.68, 1.50)], head=False)
    route([(4.49, .925), (4.68, .925), (4.68, 1.50)], head=False)
    ax.add_patch(Circle((4.68, 1.50), .024, facecolor=INK, edgecolor="none", zorder=4))
    arrow((4.70, 1.50), (4.85, 1.50))
    box(4.87, 1.24, .75, .52, "Output\nhead")
    arrow((5.64, 1.50), (6.00, 1.50))
    text(5.82, 1.69, r"$\widehat v_k$")
    box(6.02, 1.24, .75, .52, "Sampler")
    arrow((6.395, 1.22), (6.395, .98))
    text(6.395, .84, r"$z_{k+1}$")

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "report" / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig = draw_framework()
        for suffix in ("pdf", "png"):
            destination = args.output_dir / f"caching_framework.{suffix}"
            fig.savefig(destination, dpi=300)
            print(destination)
        plt.close(fig)


if __name__ == "__main__":
    main()
