"""Rebuild the report's qualitative panels from verified, portable image inputs.

No generation or image enhancement is performed. Input provenance and exact Wan
frame-cell crop boxes are recorded in report/figures/qualitative_inputs/manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


WIDTH = 6.5
DPI = 300
GAP = 4 / DPI  # Four pixels at the preview/export resolution.
METHODS = ["baseline", "teacache", "magcache", "seacache", "sencache"]
NAMES = ["Uncached", "TeaCache", "MagCache", "SeaCache", "SenCache"]
FLUX_CONFIGS = ["uncached", "teacache_0.5", "magcache_0.12", "seacache_0.6", "sencache_2"]


def load_inputs(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["flux"] + manifest["wan"]:
        path = directory / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Input checksum mismatch: {path}")
        with Image.open(path) as image:
            if list(image.size) != record["dimensions"]:
                raise ValueError(f"Input dimensions mismatch: {path}")
    return manifest


def image_cell(figure, path, x, y, width, height, figure_height):
    axes = figure.add_axes([x / WIDTH, y / figure_height, width / WIDTH, height / figure_height])
    with Image.open(path) as image:
        # PDF embeds the existing image samples without a smoothing filter.
        axes.imshow(image.copy(), interpolation="none", aspect="equal")
    axes.set_axis_off()


def label(figure, text, x, y, figure_height, *, ha="center", va="center"):
    figure.text(x / WIDTH, y / figure_height, text, ha=ha, va=va, fontsize=9, color="black")


def flux_panel(inputs, records, output):
    by_key = {(record["prompt_id"], record["config_id"]): record for record in records}
    left, right, top, bottom = 0.74, 0.02, 0.29, 0.025
    size = (WIDTH - left - right - 4 * GAP) / 5
    height = bottom + 2 * size + GAP + top
    figure = plt.figure(figsize=(WIDTH, height), facecolor="white")
    headers = NAMES[:-1] + ["SenCache (port)"]
    for column, name in enumerate(headers):
        label(figure, name, left + column * (size + GAP) + size / 2,
              height - 0.12, height)
    for row, (prompt_id, title) in enumerate((("db_053", "(a) Counting"), ("db_101", "(b) Appearance"))):
        y = height - top - (row + 1) * size - row * GAP
        marker, text = title.split(" ", 1)
        label(figure, f"{marker}\n{text}", left - 0.07, y + size / 2, height, ha="right")
        for column, config in enumerate(FLUX_CONFIGS):
            record = by_key[(prompt_id, config)]
            image_cell(figure, inputs / record["file"], left + column * (size + GAP), y,
                       size, size, height)
    figure.savefig(output / "flux_cases_restyled.pdf", dpi=DPI, facecolor="white")
    plt.close(figure)


def wan_panel(inputs, records, output):
    by_key = {(record["case"], record["method"], record["frame"]): record for record in records}
    left, right, top, bottom, group_gap = 0.74, 0.02, 0.43, 0.025, 0.11
    width = (WIDTH - left - right - group_gap - 4 * GAP) / 6
    # The first rasterized row is one pixel taller. Keep each input's aspect ratio;
    # both source panels have identical row dimensions and align exactly.
    heights = []
    for method in METHODS:
        record = by_key[("success", method, 1)]
        heights.append(width * record["dimensions"][1] / record["dimensions"][0])
    height = bottom + sum(heights) + 4 * GAP + top
    figure = plt.figure(figsize=(WIDTH, height), facecolor="white")
    group_width = 3 * width + 2 * GAP
    for group, (case, title) in enumerate((("success", "(a) Low deviation"),
                                          ("failure", "(b) High deviation"))):
        start = left + group * (group_width + group_gap)
        label(figure, title, start + group_width / 2, height - 0.105, height)
        for column, frame in enumerate((1, 41, 81)):
            label(figure, f"Frame {frame}", start + column * (width + GAP) + width / 2,
                  height - 0.295, height)
        y = height - top
        for row, (method, name, cell_height) in enumerate(zip(METHODS, NAMES, heights)):
            y -= cell_height
            if group == 0:
                label(figure, name, left - 0.07, y + cell_height / 2, height, ha="right")
            for column, frame in enumerate((1, 41, 81)):
                record = by_key[(case, method, frame)]
                if record["dimensions"] != by_key[("success", method, frame)]["dimensions"]:
                    raise ValueError("Wan frame cells must have matching dimensions across cases")
                image_cell(figure, inputs / record["file"], start + column * (width + GAP), y,
                           width, cell_height, height)
            y -= GAP
    figure.savefig(output / "wan_cases_restyled.pdf", dpi=DPI, facecolor="white")
    plt.close(figure)


def main():
    default_output = Path(__file__).resolve().parents[1] / "report" / "figures"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=default_output / "qualitative_inputs")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9, "text.color": "black",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "pdf.compression": 9})
    manifest = load_inputs(args.inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    flux_panel(args.inputs, manifest["flux"], args.output_dir)
    wan_panel(args.inputs, manifest["wan"], args.output_dir)
    print(args.output_dir / "flux_cases_restyled.pdf")
    print(args.output_dir / "wan_cases_restyled.pdf")


if __name__ == "__main__":
    main()
