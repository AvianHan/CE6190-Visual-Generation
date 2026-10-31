#!/usr/bin/env python3
"""Plot measured aggregate curves and real refresh schedules. Never simulates results."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--trace", action="append", default=[], help="Real trace.json paths, one or more")
    a = p.parse_args(); dst = Path(a.output_dir); dst.mkdir(parents=True, exist_ok=True)
    summaries = json.loads(Path(a.summary).read_text()); groups = defaultdict(list)
    for s in summaries:
        if s["metrics"]["lpips"]["mean"] is not None:
            groups[s["policy"]["method"]].append(s)
    if not groups:
        raise ValueError("Measured LPIPS summaries required")
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for method, values in sorted(groups.items()):
        values.sort(key=lambda s:s["ratio_of_total_times"])
        x=[s["ratio_of_total_times"] for s in values]; y=[s["metrics"]["lpips"]["mean"] for s in values]
        ax.plot(x,y,"o-",label=method)
    ax.set(xlabel="Measured end-to-end speedup (higher is faster)", ylabel="LPIPS to uncached reference (lower is closer)", title="Measured speed-fidelity trade-off")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(dst/"speed_lpips.png",dpi=180); plt.close(fig)
    if a.trace:
        masks=[];labels=[]
        for file in a.trace:
            data=json.loads(Path(file).read_text()); events=data["repeats"][-1]
            for branch in ("conditional","unconditional"):
                chosen=[e for e in events if e["branch"]==branch]
                masks.append([int(e["refresh"]) for e in chosen]); labels.append(Path(file).parent.name[:9]+" "+branch)
        if len({len(x) for x in masks})!=1:
            raise ValueError("Cannot compare schedules with different sampling-step counts")
        fig,ax=plt.subplots(figsize=(10,max(2,len(masks)*.25)))
        ax.imshow(np.asarray(masks),aspect="auto",interpolation="nearest",cmap="Blues",vmin=0,vmax=1)
        ax.set(xlabel="Sampling iteration",title="Actual refresh schedules (blue = full block computation)")
        ax.set_yticks(range(len(labels)),labels,fontsize=7); fig.tight_layout(); fig.savefig(dst/"refresh_schedules.png",dpi=180);plt.close(fig)
        m=np.asarray(masks,dtype=np.float64); diversity=float(np.mean([np.mean(m[i]!=m[j]) for i in range(len(m)) for j in range(i+1,len(m))])) if len(m)>1 else 0.
        (dst/"schedule_summary.json").write_text(json.dumps({"mean_refresh_ratio":float(m.mean()),"mean_pairwise_hamming_fraction":diversity,
           "warning":"Compare same method/config/branch across prompts to interpret content-adaptive schedule diversity; mixed traces are only descriptive."},indent=2))


if __name__=="__main__":main()
