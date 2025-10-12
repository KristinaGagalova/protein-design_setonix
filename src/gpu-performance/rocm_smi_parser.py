#!/usr/bin/env python3
from pathlib import Path
import re
import json
import argparse
from datetime import datetime

TSTAMP_RE = re.compile(r"^===== (?P<ts>[\d\-T:+]+) =====$")
GPU_USE_RE = re.compile(r"^GPU\[(?P<gpu>\d+)\]\s*:\s*GPU use \(%\):\s*(?P<pct>\d+)", re.IGNORECASE)
VRAM_TOTAL_RE = re.compile(r"^GPU\[(?P<gpu>\d+)\]\s*:\s*VRAM Total Memory \(B\):\s*(?P<bytes>\d+)", re.IGNORECASE)
VRAM_USED_RE  = re.compile(r"^GPU\[(?P<gpu>\d+)\]\s*:\s*VRAM Total Used Memory \(B\):\s*(?P<bytes>\d+)", re.IGNORECASE)
TEMP_RE = re.compile(r"^GPU\[(?P<gpu>\d+)\]\s*:\s*Temperature \(Sensor (?P<sensor>edge|junction|memory|HBM \d)\) \(C\):\s*(?P<temp>[\d.]+)", re.IGNORECASE)

def parse_log(path: Path):
    rows = []
    if not path.exists():
        raise FileNotFoundError(f"Log not found: {path}")
    cur_ts = None
    per_ts = {}

    def flush_ts():
        nonlocal per_ts, cur_ts, rows
        if cur_ts is None or not per_ts:
            return
        for gpu, m in per_ts.items():
            rows.append({
                "timestamp": cur_ts.isoformat(),
                "gpu": int(gpu),
                "gpu_use_pct": m.get("gpu_use_pct"),
                "vram_total_bytes": m.get("vram_total_bytes"),
                "vram_used_bytes": m.get("vram_used_bytes"),
                "temp_edge_c": m.get("temp_edge_c"),
                "temp_junction_c": m.get("temp_junction_c"),
                "temp_memory_c": m.get("temp_memory_c"),
            })
        per_ts = {}

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            m = TSTAMP_RE.match(line.strip())
            if m:
                flush_ts()
                ts_str = m.group("ts")
                try:
                    cur_ts = datetime.fromisoformat(ts_str)
                except Exception:
                    cur_ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
                continue
            m = GPU_USE_RE.match(line)
            if m:
                g = m.group("gpu"); pct = int(m.group("pct"))
                per_ts.setdefault(g, {})
                per_ts[g]["gpu_use_pct"] = pct
                continue
            m = VRAM_TOTAL_RE.match(line)
            if m:
                g = m.group("gpu"); b = int(m.group("bytes"))
                per_ts.setdefault(g, {})
                per_ts[g]["vram_total_bytes"] = b
                continue
            m = VRAM_USED_RE.match(line)
            if m:
                g = m.group("gpu"); b = int(m.group("bytes"))
                per_ts.setdefault(g, {})
                per_ts[g]["vram_used_bytes"] = b
                continue
            m = TEMP_RE.match(line)
            if m:
                g = m.group("gpu")
                per_ts.setdefault(g, {})
                sensor = m.group("sensor").lower().replace(" ", "_")
                per_ts[g][f"temp_{sensor}_c"] = float(m.group("temp"))
                continue
    flush_ts()
    return rows

def summarize(rows):
    from datetime import datetime
    def avg(x): return sum(x)/len(x) if x else None
    def to_gib(b): return b/(1024**3) if b is not None else None

    if not rows:
        return {"samples": 0}

    by_gpu = {}
    for r in rows:
        by_gpu.setdefault(r["gpu"], []).append(r)

    out = {"samples": len(rows), "gpus": {}}
    for gpu, lst in by_gpu.items():
        ts = [datetime.fromisoformat(r["timestamp"]) for r in lst if r.get("timestamp")]
        ts.sort()
        duration_s = (ts[-1]-ts[0]).total_seconds() if len(ts)>1 else 0.0

        gpu_use = [r["gpu_use_pct"] for r in lst if r.get("gpu_use_pct") is not None]
        vram_used = [r["vram_used_bytes"] for r in lst if r.get("vram_used_bytes") is not None]
        vram_total = [r["vram_total_bytes"] for r in lst if r.get("vram_total_bytes") is not None]

        t_edge = [r["temp_edge_c"] for r in lst if r.get("temp_edge_c") is not None]
        t_junc = [r["temp_junction_c"] for r in lst if r.get("temp_junction_c") is not None]
        t_mem  = [r["temp_memory_c"] for r in lst if r.get("temp_memory_c") is not None]

        out["gpus"][gpu] = {
            "samples": len(lst),
            "duration_seconds": duration_s,
            "gpu_use_pct": {
                "avg": avg(gpu_use),
                "peak": max(gpu_use) if gpu_use else None,
                "pct_of_samples_at_100": (sum(1 for v in gpu_use if v==100)/len(gpu_use)*100.0) if gpu_use else None,
            },
            "vram_used_bytes": {
                "avg": avg(vram_used),
                "peak": max(vram_used) if vram_used else None,
                "avg_gib": to_gib(avg(vram_used)),
                "peak_gib": to_gib(max(vram_used) if vram_used else None),
            },
            "vram_total_bytes": max(vram_total) if vram_total else None,
            "vram_total_gib": to_gib(max(vram_total) if vram_total else None),
            "temps_c": {
                "edge": {"avg": avg(t_edge), "peak": max(t_edge) if t_edge else None},
                "junction": {"avg": avg(t_junc), "peak": max(t_junc) if t_junc else None},
                "memory": {"avg": avg(t_mem), "peak": max(t_mem) if t_mem else None},
            },
            "time_range": {
                "start": ts[0].isoformat() if ts else None,
                "end": ts[-1].isoformat() if ts else None,
            }
        }
    return out

def main():
    ap = argparse.ArgumentParser(description="Parse ROCm SMI sampling log and summarize metrics.")
    ap.add_argument("--log", required=True, help="Path to rocm_smi_<JOBID>.log")
    ap.add_argument("--csv", help="Output CSV path (optional)")
    ap.add_argument("--summary", help="Output JSON summary path (optional)")
    ap.add_argument("--plot", help="Output PNG plot path (optional)")
    args = ap.parse_args()

    path = Path(args.log)
    rows = parse_log(path)

    # Write CSV
    import csv
    csv_path = Path(args.csv) if args.csv else path.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp","gpu","gpu_use_pct","vram_total_bytes","vram_used_bytes","temp_edge_c","temp_junction_c","temp_memory_c"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Summary JSON
    summary = summarize(rows)
    json_path = Path(args.summary) if args.summary else path.with_suffix(".summary.json")
    json_path.write_text(json.dumps(summary, indent=2))

    # Optional plot
    if args.plot:
        import matplotlib.pyplot as plt
        from datetime import datetime
        by_gpu = {}
        for r in rows:
            by_gpu.setdefault(r["gpu"], []).append(r)
        for gpu, lst in by_gpu.items():
            lst = sorted(lst, key=lambda r: r["timestamp"])
            ts = [datetime.fromisoformat(r["timestamp"]) for r in lst]
            util = [r["gpu_use_pct"] if r["gpu_use_pct"] is not None else float('nan') for r in lst]
            vram = [r["vram_used_bytes"]/(1024**3) if r["vram_used_bytes"] is not None else float('nan') for r in lst]

            plt.figure()
            plt.plot(ts, util)
            plt.xlabel("Time")
            plt.ylabel("GPU use (%)")
            plt.title(f"GPU {gpu} utilization")
            plt.tight_layout()
            util_png = Path(args.plot).with_name(Path(args.plot).stem + f"_gpu{gpu}_util.png")
            plt.savefig(util_png)
            plt.close()

            plt.figure()
            plt.plot(ts, vram)
            plt.xlabel("Time")
            plt.ylabel("VRAM used (GiB)")
            plt.title(f"GPU {gpu} VRAM usage")
            plt.tight_layout()
            vram_png = Path(args.plot).with_name(Path(args.plot).stem + f"_gpu{gpu}_vram.png")
            plt.savefig(vram_png)
            plt.close()

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote summary JSON: {json_path}")
    if args.plot:
        print(f"Wrote plots near: {args.plot} (per-GPU files)")

if __name__ == "__main__":
    main()
