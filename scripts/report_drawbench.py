"""Merge DrawBench x FLUX runs into report tables and the speed-fidelity figure.

Inputs: per_image_metrics.csv + per-job json of one or more runs (first run's
uncached is the reference; later runs must reproduce it -- checked by caller).
Outputs in --out: summary_all.csv, per_image_metrics_all.csv, pairwise_lpips.csv, schedules.csv,
speed_vs_lpips.png.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHOD = {"teacache": "TeaCache", "magcache": "MagCache", "seacache": "SeaCache", "sencache": "SenCache (our port)",
          "fixed": "Static schedule (control)", "uncached": "Uncached"}
# Validated categorical slots 1-4 (dataviz reference palette, light); control in neutral ink.
COLOR = {"teacache": "#2a78d6", "magcache": "#eb6834", "seacache": "#1baf7a", "sencache": "#eda100",
         "fixed": "#6b6a64"}
MARKER = {"teacache": "o", "magcache": "s", "seacache": "^", "sencache": "D", "fixed": "x"}
PAIRS = [  # (a, b): matched speed unless noted
    ("seacache_0.3", "sencache_0.5"), ("seacache_0.3", "fixed_16"), ("sencache_0.5", "fixed_16"),
    ("seacache_0.3", "teacache_0.25"), ("sencache_0.6", "magcache_0.06"), ("sencache_0.8", "fixed_14"),
    ("sencache_1.0", "seacache_0.45"), ("sencache_1.0", "magcache_0.09"), ("fixed_12", "seacache_0.45"),
    ("fixed_12", "magcache_0.09"), ("fixed_12", "sencache_1.0"),
    ("sencache_2", "seacache_0.6"), ("sencache_2", "magcache_0.12"), ("sencache_2", "teacache_0.5"),
    ("seacache_0.6", "magcache_0.12"), ("fixed_10", "magcache_0.24"), ("fixed_9", "magcache_0.24"),
    ("fixed_10", "teacache_0.6"), ("magcache_0.24", "teacache_0.6")]


def load(runs):
    frames, scheds = [], []
    for k, run in enumerate(runs):
        df = pd.read_csv(Path(run) / "per_image_metrics.csv")
        if k:
            df = df[df.config_id != "uncached"]
        frames.append(df)
        for d in sorted(Path(run).iterdir()):
            if not d.is_dir() or d.name == "logs" or (k and d.name == "uncached"):
                continue
            recs = [json.loads(open(f).read()) for f in glob.glob(str(d / "db_*.json"))]
            if recs:
                pats = collections.Counter("".join(map(str, r["decisions"])) for r in recs)
                top, n = pats.most_common(1)[0]
                scheds.append({"config_id": d.name, "distinct_schedules": len(pats), "modal_schedule": top,
                               "modal_share": n / len(recs),
                               "compute_steps_min": min(r["compute_steps"] for r in recs),
                               "compute_steps_max": max(r["compute_steps"] for r in recs),
                               "computed_in_first_6": float(np.mean([sum(r["decisions"][:6]) for r in recs]))})
    d = pd.concat(frames)
    if d.duplicated(["config_id", "prompt_id"]).any():
        raise SystemExit("duplicate config/prompt across runs")
    return d, pd.DataFrame(scheds)


def bootstrap(diff, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(diff, (n, len(diff))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d, sched = load(a.runs)
    s = (d.groupby(["config_id", "method", "threshold"], dropna=False)
         .agg(n=("prompt_id", "count"), median_speedup=("speedup", "median"),
              median_generation_s=("generation_s", "median"), mean_refresh_ratio=("refresh_ratio", "mean"),
              psnr=("psnr", "mean"), ssim=("ssim", "mean"), lpips=("lpips", "mean"),
              clip_score=("clip_score", "mean"), image_reward=("image_reward", "mean"))
         .reset_index().merge(sched, on="config_id", how="left").sort_values("median_speedup"))
    s.to_csv(out / "summary_all.csv", index=False)
    d.sort_values(["config_id", "prompt_id"]).to_csv(out / "per_image_metrics_all.csv", index=False)
    sched.to_csv(out / "schedules.csv", index=False)

    L = d.pivot_table(index="prompt_id", columns="config_id", values="lpips")
    sp = s.set_index("config_id").median_speedup
    rows = []
    for x, y in PAIRS:
        if x not in L or y not in L:
            continue
        diff = (L[x] - L[y]).dropna().values
        lo, hi = bootstrap(diff)
        rows.append({"a": x, "b": y, "speedup_a": sp[x], "speedup_b": sp[y], "mean_lpips_a_minus_b": diff.mean(),
                     "ci95_low": lo, "ci95_high": hi, "a_wins_pct": 100 * (diff < 0).mean(), "n": len(diff)})
    pd.DataFrame(rows).to_csv(out / "pairwise_lpips.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b9b8b0", "axes.labelcolor": "#2b2b29",
                         "xtick.color": "#5d5c57", "ytick.color": "#5d5c57"})
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    ax.grid(True, color="#e6e5df", linewidth=0.6)
    ax.set_axisbelow(True)
    for m in ["teacache", "magcache", "seacache", "sencache", "fixed"]:
        g = s[s.method == m].sort_values("median_speedup")
        if g.empty:
            continue
        ax.plot(g.median_speedup, g.lpips, color=COLOR[m], linewidth=2 if m != "fixed" else 1.4,
                linestyle="--" if m == "fixed" else "-", marker=MARKER[m], markersize=6,
                markeredgecolor="#fcfcfb" if m != "fixed" else COLOR[m], markeredgewidth=1.2 if m != "fixed" else 1.6,
                label=METHOD[m], zorder=3 if m != "fixed" else 2)
    ax.set_xlabel("Median paired speedup vs. uncached (same GPU)")
    ax.set_ylabel("Mean LPIPS vs. uncached (lower = closer)")
    ax.set_xlim(1.15, 3.45)
    ax.set_ylim(0, None)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("FLUX.1-dev, DrawBench (200 prompts, seed 0, 28 steps, 1024$^2$)", fontsize=9, color="#2b2b29",
                 loc="left")
    fig.tight_layout()
    fig.savefig(out / "speed_vs_lpips.png")
    print(s[["config_id", "median_speedup", "mean_refresh_ratio", "psnr", "ssim", "lpips", "clip_score",
             "image_reward", "distinct_schedules", "computed_in_first_6"]].round(3).to_string(index=False))
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
