#!/usr/bin/env python3
"""Redraw the four single-column report figures from audited summaries, offline.

No model, GPU, evaluator or upstream repository is loaded. Confidence intervals
are read from the tables, not re-estimated. Defaults target a 6.5-inch paper column.

    python scripts/make_report_figures.py
    python scripts/make_report_figures.py --output-dir /tmp/report-figures --formats pdf png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("TeaCache", "MagCache", "SeaCache", "SenCache")
# Shared report palette supplied by the group. Uncached remains a neutral reference.
COLORS = {"TeaCache": "#466F87", "MagCache": "#78B6C8", "SeaCache": "#F2D98E",
          "SenCache": "#E48578", "Fixed": "#98AF1E"}
MARKER_EDGE = "#666666"
MARKERS = dict(zip((*METHODS, "Fixed"), ("o", "s", "^", "D", "x")))
METHOD_LABELS = {"teacache": "TeaCache", "magcache": "MagCache", "seacache": "SeaCache",
                 "sencache": "SenCache", "fixed": "Fixed"}
STYLE = {"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9.5,
         "axes.labelsize": 10, "axes.titlesize": 10.5, "xtick.labelsize": 9,
         "ytick.labelsize": 9, "legend.fontsize": 9, "axes.spines.top": False,
         "axes.spines.right": False, "axes.edgecolor": "#777777", "axes.linewidth": .65,
         "xtick.color": "#333333", "ytick.color": "#333333", "xtick.major.size": 3,
         "ytick.major.size": 3, "xtick.major.width": .6, "ytick.major.width": .6,
         "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"}


def read_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        raise ValueError(f"Expected a nonempty list of rows: {path}")
    return rows


def single_count(rows: list[dict], key: str) -> int:
    values = {r[key] for r in rows}
    if len(values) != 1:
        raise ValueError(f"Inconsistent {key}: {values}")
    return int(values.pop())


def lpips_points(rows: list[dict]):
    x = np.array([r["e2e_speedup_ratio_of_totals"] for r in rows], dtype=float)
    y = np.array([r["lpips"] for r in rows], dtype=float)
    low = np.array([r["lpips_ci95_low"] for r in rows], dtype=float)
    high = np.array([r["lpips_ci95_high"] for r in rows], dtype=float)
    if not np.isfinite([x, y, low, high]).all() or np.any(low > y) or np.any(y > high):
        raise ValueError("Invalid LPIPS values or confidence intervals")
    return x, y, np.vstack((y - low, high - y))


def clean_axis(ax, xlabel=r"Generation speedup ($\times$)", ylabel=r"LPIPS to Uncached $\downarrow$"):
    ax.set_xlabel(xlabel, labelpad=3)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=4)
    ax.grid(axis="y", color="#E6E6E6", linewidth=.55)
    ax.tick_params(pad=2)


def method_handles(methods=METHODS, port=False):
    return [Line2D([], [], color=COLORS[m], marker=MARKERS[m], linewidth=1.35,
                   markersize=4.5, markeredgecolor=MARKER_EDGE, markeredgewidth=.55,
                   label=("SenCache (port)" if m == "SenCache" and port else
                          "Fixed control" if m == "Fixed" else m)) for m in methods]


def save_figure(fig, directory: Path, name: str, formats: list[str]) -> None:
    for fmt in formats:
        path = directory / f"{name}.{fmt}"
        fig.savefig(path, dpi=300)
        print(path)
    plt.close(fig)


def wan_speed_fidelity(primary, grid):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.65), sharey=True)
    cached = [r for r in primary if r["method"] in METHODS]
    for method in METHODS:
        points = sorted((r for r in grid if r["method"] == method), key=lambda r: r["threshold"])
        x, y, error = lpips_points(points)
        axes[0].errorbar(x, y, yerr=error, color=COLORS[method], marker=MARKERS[method],
                         linewidth=1.35, markersize=4.2, capsize=2, elinewidth=.65,
                         markeredgecolor=MARKER_EDGE, markeredgewidth=.55)
        row = next(r for r in cached if r["method"] == method)
        x, y, error = lpips_points([row])
        axes[1].errorbar(x, y, yerr=error, color=COLORS[method], marker=MARKERS[method],
                         linestyle="none", markersize=5, capsize=2.5, elinewidth=.8,
                         markeredgecolor=MARKER_EDGE, markeredgewidth=.55)
    axes[0].set_title(f"(a) Development · {single_count(grid, 'n_pairs')} pairs", loc="left", pad=5)
    axes[1].set_title(f"(b) Test · {single_count(cached, 'n_pairs')} pairs", loc="left", pad=5)
    axes[0].set_xlim(1.20, 1.76)
    axes[0].set_xticks([1.2, 1.4, 1.6, 1.7])
    axes[1].set_xlim(1.44, 1.69)
    axes[1].set_xticks([1.45, 1.55, 1.65])
    axes[0].set_ylim(-.005, .205)
    axes[0].set_yticks([0, .05, .10, .15, .20])
    clean_axis(axes[0])
    clean_axis(axes[1], ylabel=None)
    fig.legend(handles=method_handles(), loc="upper center", bbox_to_anchor=(.54, 1.01),
               ncol=4, frameon=False, handlelength=1.4, handletextpad=.45, columnspacing=1.3)
    fig.subplots_adjust(left=.10, right=.985, bottom=.20, top=.77, wspace=.22)
    return fig


def wan_component_tradeoffs(rows):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.70))
    variant_labels = {"official": "Full method", "raw_proxy": "Raw proxy",
                      "no_gain_normalization": "No gain norm.",
                      "latent_only": "Latent only", "time_only": "Time only"}
    for ax, method in zip(axes, ("SeaCache", "SenCache")):
        subset = [r for r in rows if r["method"] == method]
        variants = ("official", "raw_proxy", "no_gain_normalization") if method == "SeaCache" \
            else ("official", "latent_only", "time_only")
        for i, variant in enumerate(variants):
            points = sorted((r for r in subset if r["variant"] == variant), key=lambda r: r["threshold"])
            x, y, error = lpips_points(points)
            marker = (MARKERS[method], "o", "x")[i]
            ax.errorbar(x, y, yerr=error, marker=marker, linestyle=("-", "--", ":")[i],
                        color=COLORS[method], label=variant_labels[variant], capsize=2,
                        linewidth=1.3, markersize=4, elinewidth=.65,
                        markerfacecolor=COLORS[method] if i == 0 else "white",
                        markeredgecolor=MARKER_EDGE, markeredgewidth=.65)
        panel = "a" if method == "SeaCache" else "b"
        ax.set_title(f"({panel}) {method} · {single_count(subset, 'n_pairs')} pairs", loc="left", pad=6)
        clean_axis(ax)
        ax.legend(frameon=False, loc="upper left", handlelength=1.6,
                  handletextpad=.5, borderaxespad=.2, labelspacing=.25)
    # Linear axes retain the zero-deviation / all-refresh point and the full raw-proxy range.
    axes[0].set_xlim(.95, 2.47)
    axes[0].set_ylim(-.02, .79)
    axes[0].set_xticks([1.0, 1.5, 2.0, 2.4])
    axes[0].set_yticks([0, .2, .4, .6])
    axes[1].set_xlim(1.61, 1.87)
    axes[1].set_ylim(0, .49)
    axes[1].set_xticks([1.65, 1.75, 1.85])
    axes[1].set_yticks([0, .15, .30, .45])
    fig.subplots_adjust(left=.10, right=.985, bottom=.20, top=.86, wspace=.34)
    return fig


def flux_speed_fidelity(rows):
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    names = (*METHODS, "Fixed")
    for method in names:
        points = sorted((r for r in rows if METHOD_LABELS.get(r["method"]) == method),
                        key=lambda r: r["median_speedup"])
        ax.plot([r["median_speedup"] for r in points], [r["lpips"] for r in points],
                color=COLORS[method], marker=MARKERS[method], markersize=4.5,
                linestyle="--" if method == "Fixed" else "-", linewidth=1.4,
                markeredgecolor=COLORS[method] if method == "Fixed" else MARKER_EDGE,
                markeredgewidth=.65, zorder=3)
    # Reference budgets aid comparison; data points retain their actual measured speedups.
    for budget, label, alignment in [(1.72, "≈1.7×", "center"), (2.27, "≈2.27×", "right"),
                                      (2.45, "≈2.45×", "left"), (2.98, "≈3×", "center")]:
        ax.vlines(budget, 0, .432, color="#BBBBBB", linewidth=.55, linestyle=(0, (2, 3)), zorder=1)
        ax.text(budget, .442, label, ha=alignment, va="bottom", fontsize=9, color="#555555")
    ax.set_xlim(1.15, 3.43)
    ax.set_ylim(0, .475)
    ax.set_xticks([1.25, 1.75, 2.25, 2.75, 3.25])
    ax.set_yticks([0, .1, .2, .3, .4])
    clean_axis(ax, xlabel=r"Median paired generation speedup ($\times$)")
    fig.legend(handles=method_handles(names, port=True), loc="upper center",
               bbox_to_anchor=(.54, 1.005), ncol=5, frameon=False, handlelength=1.3,
               handletextpad=.4, columnspacing=.9)
    fig.subplots_adjust(left=.10, right=.985, bottom=.18, top=.82)
    return fig


def flux_modal_schedules(rows):
    data = {r["config_id"]: r for r in rows}
    groups = [("(a) Near 2.45×", ("teacache_0.5", "magcache_0.12", "seacache_0.6", "sencache_2")),
              ("(b) Near 2.27×", ("sencache_1.0", "fixed_12"))]
    labels = {"teacache_0.5": "TeaCache 0.5", "magcache_0.12": "MagCache 0.12",
              "seacache_0.6": "SeaCache 0.6", "sencache_2": "SenCache 2 (port)",
              "sencache_1.0": "SenCache 1 (port)", "fixed_12": "Fixed 12"}
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 2.75),
                             gridspec_kw={"height_ratios": [4, 2], "hspace": .72})
    for ax, (title, configs) in zip(axes, groups):
        patterns = [data[c]["modal_schedule"] for c in configs]
        if any(len(s) != 28 or set(s) - {"0", "1"} for s in patterns):
            raise ValueError("Expected binary modal schedules with 28 steps")
        color_order = (*METHODS, "Fixed")
        cells = np.zeros((len(configs), 28), dtype=int)
        for i, (config, pattern) in enumerate(zip(configs, patterns)):
            code = color_order.index(METHOD_LABELS[data[config]["method"]]) + 1
            cells[i] = [code if c == "1" else 0 for c in pattern]
        # Draw each cell as a vector polygon in the PDF, rather than a raster image.
        ax.pcolormesh(np.arange(29) - .5, np.arange(len(configs) + 1) - .5, cells,
                      cmap=ListedColormap(["#F2F2F2", *(COLORS[m] for m in color_order)]),
                      vmin=0, vmax=5, shading="flat", edgecolors="white", linewidth=.4,
                      rasterized=False)
        ax.set_ylim(len(configs) - .5, -.5)
        ax.set_yticks(range(len(configs)),
                      [f"{labels[c]}  ·  {data[c]['median_speedup']:.2f}×  ·  "
                       f"{100 * data[c]['modal_share']:.0f}%" if data[c]['modal_share'] == 1 else
                       f"{labels[c]}  ·  {data[c]['median_speedup']:.2f}×  ·  "
                       f"{100 * data[c]['modal_share']:.1f}%" for c in configs], fontsize=9)
        ticks = [0, 4, 9, 14, 19, 24, 27]
        ax.set_xticks(ticks, [str(t + 1) for t in ticks])
        ax.set_xlim(-.5, 27.5)
        ax.set_xticks(np.arange(-.5, 28, 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(configs), 1), minor=True)
        ax.grid(False)
        ax.grid(which="minor", color="white", linewidth=.7)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.tick_params(axis="x", length=2, pad=2)
        ax.set_title(title, loc="left", fontsize=10, pad=4)
        ax.axvline(5.5, color="#555555", linestyle=(0, (1, 2)), linewidth=.65)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].set_xlabel("Denoising step", labelpad=2)
    fig.text(.025, .957, "Method · speedup · modal share", fontsize=9, va="top")
    fig.legend(handles=[Patch(facecolor="#777777", label="Refresh (method color)"),
                        Patch(facecolor="#F2F2F2", edgecolor="#BBBBBB", label="Reuse")],
               loc="upper center", bbox_to_anchor=(.69, 1.0), ncol=2, frameon=False,
               fontsize=9, handlelength=1.2, handletextpad=.45, columnspacing=1)
    fig.subplots_adjust(left=.375, right=.985, top=.81, bottom=.17)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "analysis/report_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "report/figures")
    parser.add_argument("--formats", nargs="+", choices=("pdf", "png"), default=["pdf"])
    args = parser.parse_args()
    data = {name: read_rows(args.data_dir / f"{name}.json") for name in
            ("wan_primary", "wan_development_grid", "wan_component_ablations", "flux_all_configs")}
    plt.rcParams.update(STYLE)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = {"wan_speed_fidelity": wan_speed_fidelity(data["wan_primary"], data["wan_development_grid"]),
               "wan_component_tradeoffs": wan_component_tradeoffs(data["wan_component_ablations"]),
               "flux_speed_fidelity": flux_speed_fidelity(data["flux_all_configs"]),
               "flux_modal_schedules": flux_modal_schedules(data["flux_all_configs"])}
    for name, fig in figures.items():
        save_figure(fig, args.output_dir, name, args.formats)


if __name__ == "__main__":
    main()
