"""Qualitative grid for the DrawBench x FLUX runs: randomly drawn prompts (fixed RNG seed, not hand-picked),
one row per prompt, one column per configuration.

    python scripts/qualitative_drawbench.py --runs RUN1 RUN2 --out results/drawbench_flux/examples_2.45x.jpg
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# About 2.45x for every method (summary_all.csv), plus the reference.
DEFAULT = ["uncached", "teacache_0.5", "magcache_0.12", "seacache_0.6", "sencache_2"]
LABEL = {"uncached": "Uncached (1.00x)", "teacache_0.5": "TeaCache 0.5", "magcache_0.12": "MagCache 0.12",
         "seacache_0.6": "SeaCache 0.6", "sencache_2": "SenCache 2 (our port)"}


def font(size, bold=False):
    import matplotlib
    d = Path(matplotlib.get_data_path()) / "fonts/ttf"
    return ImageFont.truetype(str(d / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def find(runs, cfg, pid):
    for r in runs:
        p = Path(r) / cfg / f"{pid}.png"
        if p.is_file():
            return p
    raise FileNotFoundError(f"{cfg}/{pid}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", nargs="+", default=DEFAULT)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile", type=int, default=300)
    a = ap.parse_args()
    pids = sorted(p.stem for p in (Path(a.runs[0]) / "uncached").glob("db_*.png"))
    pick = sorted(np.random.default_rng(a.seed).choice(pids, a.n, replace=False))
    metrics = {}
    for r in a.runs:
        f = Path(r) / "per_image_metrics.csv"
        if f.is_file():
            import csv
            for row in csv.DictReader(open(f)):
                metrics[(row["config_id"], row["prompt_id"])] = row
    T, pad, head, cap = a.tile, 8, 46, 70
    W = 230 + len(a.configs) * (T + pad)
    H = head + a.n * (T + cap + pad)
    g = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(g)
    for j, c in enumerate(a.configs):
        d.text((230 + j * (T + pad) + T // 2, head // 2), LABEL.get(c, c), font=font(18, True), fill="black",
               anchor="mm")
    for i, pid in enumerate(pick):
        y = head + i * (T + cap + pad)
        prompt = json.loads((Path(a.runs[0]) / "uncached" / f"{pid}.json").read_text())["prompt"]
        d.multiline_text((8, y + 4), "\n".join(textwrap.wrap(f"{pid}: {prompt}", 24)[:9]), font=font(14),
                         fill="black", spacing=3)
        for j, c in enumerate(a.configs):
            x = 230 + j * (T + pad)
            g.paste(Image.open(find(a.runs, c, pid)).convert("RGB").resize((T, T), Image.LANCZOS), (x, y))
            m = metrics.get((c, pid))
            if m and m.get("lpips"):
                d.text((x + T // 2, y + T + 18), f"LPIPS {float(m['lpips']):.3f}  PSNR {float(m['psnr']):.1f}",
                       font=font(14), fill="black", anchor="mm")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    g.save(a.out, quality=90)
    print(a.out, list(pick))


if __name__ == "__main__":
    main()
