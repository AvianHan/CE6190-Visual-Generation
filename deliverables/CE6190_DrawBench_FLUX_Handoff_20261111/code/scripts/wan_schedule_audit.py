"""Audit the compute/reuse schedules of finished Wan2.1 runs (no GPU, no new experiments).

Reads trace.json + metrics.json of completed runs, either from a run snapshot ZIP (e.g. the
CE6190_Local_Run_Snapshot_*.zip in deliverables/CE6190_Friend_Handoff_20261106/backups/) or from result
directories (<root>/<run_id>/{trace,metrics}.json), and writes, per method x threshold: number of runs and
prompt-seed pairs, mean refresh ratio, number of distinct schedules (conditional branch, first timing repeat),
the most common schedule and its share, and how many of the first 20% of steps are computed.

    python scripts/wan_schedule_audit.py --source deliverables/.../backups/CE6190_Local_Run_Snapshot_...zip \
        --out results/wan_schedules/schedules_wan.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import zipfile
from pathlib import Path


def iter_runs(source):
    src = Path(source)
    if src.suffix == ".zip":
        z = zipfile.ZipFile(src)
        names = set(z.namelist())
        for n in sorted(names):
            if n.endswith("/trace.json") and n.replace("trace.json", "metrics.json") in names:
                yield (json.loads(z.read(n.replace("trace.json", "metrics.json"))), json.loads(z.read(n)))
    else:
        for t in sorted(src.glob("*/trace.json")):
            m = t.with_name("metrics.json")
            if m.is_file():
                yield json.loads(m.read_text()), json.loads(t.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    groups = collections.defaultdict(list)
    for s in a.source:
        for m, t in iter_runs(s):
            if m.get("status") != "complete":
                continue
            c = m["config"]
            rep = t["repeats"][0]
            sched = "".join("1" if e["refresh"] else "0" for e in rep if e["branch"] == "conditional")
            groups[(c["method"], c.get("threshold"))].append((c["pair_id"], sched, m["refresh_ratio"], m["gpu_name"]))
    rows = []
    for (meth, th), v in sorted(groups.items(), key=lambda kv: (kv[0][0], float(kv[0][1] or 0))):
        pats = collections.Counter(s for _, s, _, _ in v)
        top, n = pats.most_common(1)[0]
        steps = len(top)
        early = int(round(0.2 * steps))
        rows.append({"method": meth, "threshold": th, "runs": len(v), "pairs": len({p for p, *_ in v}),
                     "gpus": ";".join(sorted({g for *_, g in v})),
                     "mean_refresh_ratio": round(sum(r for _, _, r, _ in v) / len(v), 4),
                     "distinct_schedules": len(pats), "modal_share": round(n / len(v), 3),
                     "steps": steps, f"computed_in_first_{early}": top[:early].count("1"),
                     "modal_schedule": top})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print({k: r[k] for k in ("method", "threshold", "runs", "mean_refresh_ratio", "distinct_schedules",
                                 "modal_share")}, r["modal_schedule"])


if __name__ == "__main__":
    main()
