#!/usr/bin/env python3
"""
rocm_smi_parser.py
Parse ROCm-SMI sampling logs, write CSV + JSON summary, optionally plot.

Features
- Python 3.6–compatible ISO8601 parsing (handles "YYYY-mm-ddTHH:MM:SS+08:00")
- Parses GPU util, VRAM total/used, temps (edge/junction/memory)
- Summarises per-GPU stats to JSON
- Optional SU calculation via `sacct`:
    SU = (elapsed_sec/3600) * nodes * rate
    default rate: 512 for 'gpu' partitions else 128 (override with --rate)
- Combined plot for ANY number of GPUs:
    One figure with two subplots:
      top   = GPU utilization (%) for all GPUs
      bottom= VRAM used (GiB) for all GPUs

Usage
  python3 rocm_smi_parser.py --log rocm_smi_<JOBID>.log --jobid <JOBID> --plot /path/out.png
  python3 rocm_smi_parser.py --log rocm_smi_<JOBID>.log --elapsed-sec 135 --nodes 1 --partition gpu
"""

from pathlib import Path
import re
import json
import argparse
import subprocess
from datetime import datetime

# ---------- Regex ----------
TSTAMP_RE = re.compile(r"^===== (?P<ts>[\d\-T:+]+) =====$")
GPU_USE_RE = re.compile(r"^GPU\[(?P<gpu>\d+)\]\s*:\s*GPU use \(%\):\s*(?P<pct>\d+)", re.IGNORECASE)
VRAM_TOTAL_RE = re.compile(r"^GPU\[(?P<gpu>\d+)\]\s*:\s*VRAM Total Memory \(B\):\s*(?P<bytes>\d+)", re.IGNORECASE)
VRAM_USED_RE  = re.compile(r"^GPU\[(?P<gpu>\d+)\]\s*:\s*VRAM Total Used Memory \(B\):\s*(?P<bytes>\d+)", re.IGNORECASE)
TEMP_RE = re.compile(r"^GPU\[(?P<gpu>\d+)\]\s*:\s*Temperature \(Sensor (?P<sensor>edge|junction|memory|HBM \d)\) \(C\):\s*(?P<temp>[\d.]+)", re.IGNORECASE)

# ---------- Python 3.6-friendly ISO time parsing ----------
ISO_TZ_RE = re.compile(r"^(?P<base>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?P<tz>Z|[+-]\d{2}:\d{2})$")

def parse_iso_ts(ts_str: str) -> datetime:
    """
    Parse 'YYYY-MM-DDTHH:MM:SS+08:00' or '...Z' or naive '...'.
    Compatible with Python 3.6 (no datetime.fromisoformat).
    """
    m = ISO_TZ_RE.match(ts_str)
    if m:
        base = m.group("base")
        tz = m.group("tz")
        if tz == "Z":
            return datetime.strptime(base + "+0000", "%Y-%m-%dT%H:%M:%S%z")
        return datetime.strptime(base + tz.replace(":", ""), "%Y-%m-%dT%H:%M:%S%z")
    # try naive
    try:
        return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        if "." in ts_str:
            base = ts_str.split(".", 1)[0]
            return datetime.strptime(base, "%Y-%m-%dT%H:%M:%S")
        raise

# ---------- Parser ----------
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
                cur_ts = parse_iso_ts(ts_str)
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
    def avg(x): return sum(x)/len(x) if x else None
    def to_gib(b): return b/(1024**3) if b is not None else None

    if not rows:
        return {"samples": 0}

    by_gpu = {}
    for r in rows:
        by_gpu.setdefault(r["gpu"], []).append(r)

    out = {"samples": len(rows), "gpus": {}}
    for gpu, lst in by_gpu.items():
        ts = [parse_iso_ts(r["timestamp"]) for r in lst if r.get("timestamp")]
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

# ---------- Slurm helpers for SU ----------
def fetch_slurm(jobid: str):
    """
    Returns dict with elapsed_sec, nodes, partition for the jobid using sacct.
    """
    fmt = "ElapsedRaw,AllocNodes,Partition"
    cmd = ["sacct", "-j", jobid, "--noheader", f"--format={fmt}"]
    try:
        out = subprocess.check_output(cmd, universal_newlines=True)
    except Exception as e:
        return {"elapsed_sec": None, "nodes": None, "partition": None, "error": str(e)}
    # More robust: -P (pipe-delimited)
    cmd_p = ["sacct", "-P", "-j", jobid, "--noheader", f"--format={fmt}"]
    try:
        out_p = subprocess.check_output(cmd_p, universal_newlines=True).strip()
        rec = out_p.splitlines()[0].split("|")
        elapsed = int(rec[0]) if rec[0].isdigit() else None
        nodes = int(rec[1]) if rec[1].isdigit() else None
        part = rec[2] if rec[2] else None
        return {"elapsed_sec": elapsed, "nodes": nodes, "partition": part, "raw": out_p}
    except Exception as e:
        # Fallback best-effort parse
        line = next((l for l in out.splitlines() if l.strip()), "")
        parts = [p.strip() for p in line.split()]
        elapsed = None
        nodes = None
        part = None
        try:
            if parts and parts[0].isdigit():
                elapsed = int(parts[0])
            if len(parts) > 1 and parts[1].isdigit():
                nodes = int(parts[1])
            if len(parts) > 2:
                part = parts[2]
        except Exception:
            pass
        return {"elapsed_sec": elapsed, "nodes": nodes, "partition": part, "raw": out, "error": str(e)}

def compute_su(elapsed_sec: int, nodes: int, partition: str, rate_override: float = None):
    """
    SU = (elapsed_sec/3600) * nodes * rate
    Default rate: 512 for 'gpu' partitions, 128 otherwise (cpu).
    """
    if elapsed_sec is None or nodes is None:
        return None
    part = (partition or "").lower()
    default_rate = 512.0 if "gpu" in part else 128.0
    rate = float(rate_override) if rate_override is not None else default_rate
    sus = (elapsed_sec / 3600.0) * nodes * rate
    return {"elapsed_sec": elapsed_sec, "nodes": nodes, "partition": partition, "rate_su_per_node_hour": rate, "sus": sus}

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Parse ROCm SMI sampling log and summarize metrics, with optional SU calc.")
    ap.add_argument("--log", required=True, help="Path to rocm_smi_<JOBID>.log")
    ap.add_argument("--csv", help="Output CSV path (optional)")
    ap.add_argument("--summary", help="Output JSON summary path (optional)")
    ap.add_argument("--plot", help="Output PNG path (we will write *_combined_all.png next to it)")

    # SU options
    ap.add_argument("--jobid", help="Slurm JobID to pull elapsed/nodes/partition via sacct")
    ap.add_argument("--elapsed-sec", type=int, help="Elapsed seconds (if not using --jobid)")
    ap.add_argument("--nodes", type=int, help="Allocated nodes (if not using --jobid)")
    ap.add_argument("--partition", help="Partition name (e.g., gpu or cpu) (if not using --jobid)")
    ap.add_argument("--rate", type=float, help="Override SU rate (SU per node-hour). Default: 512 for gpu, 128 otherwise.")
    args = ap.parse_args()

    path = Path(args.log)
    rows = parse_log(path)

    # Write CSV
    import csv
    csv_path = Path(args.csv) if args.csv else path.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "timestamp","gpu","gpu_use_pct","vram_total_bytes","vram_used_bytes",
            "temp_edge_c","temp_junction_c","temp_memory_c"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Summary from log
    summary = summarize(rows)

    # Slurm / SU section
    slurm_info = {}
    su_info = None
    if args.jobid:
        slurm_info = fetch_slurm(args.jobid)
        su_info = compute_su(slurm_info.get("elapsed_sec"), slurm_info.get("nodes"),
                             slurm_info.get("partition"), args.rate)
    else:
        if args.elapsed_sec is not None and args.nodes is not None:
            su_info = compute_su(args.elapsed_sec, args.nodes, args.partition or "", args.rate)
            slurm_info = {"elapsed_sec": args.elapsed_sec, "nodes": args.nodes, "partition": args.partition}

    if slurm_info:
        summary["slurm"] = slurm_info
    if su_info:
        summary["su"] = su_info

    # Summary JSON
    json_path = Path(args.summary) if args.summary else path.with_suffix(".summary.json")
    json_path.write_text(json.dumps(summary, indent=2))

    # Optional combined plot (any number of GPUs)
    if args.plot:
        import matplotlib.pyplot as plt

        # Group by GPU id
        by_gpu = {}
        for r in rows:
            by_gpu.setdefault(r["gpu"], []).append(r)

        if not by_gpu:
            print("No GPU samples found; skipping plots.")
        else:
            # Prepare series per GPU
            gpu_series = {}
            for gpu, lst in by_gpu.items():
                lst = sorted(lst, key=lambda r: r["timestamp"])
                ts = [parse_iso_ts(r["timestamp"]) for r in lst]
                util = [r["gpu_use_pct"] if r.get("gpu_use_pct") is not None else float('nan') for r in lst]
                vram = [(r["vram_used_bytes"]/(1024.0**3)) if r.get("vram_used_bytes") is not None else float('nan') for r in lst]
                gpu_series[gpu] = {"ts": ts, "util": util, "vram_gib": vram}

            # One combined figure: 2 subplots (Util %, VRAM GiB), all GPUs
            fig, (ax_u, ax_v) = plt.subplots(nrows=2, ncols=1, figsize=(12, 7), dpi=150, sharex=True)

            # Util subplot
            for gpu in sorted(gpu_series.keys()):
                s = gpu_series[gpu]
                ax_u.plot(s["ts"], s["util"], label=f"GPU {gpu}")
            ax_u.set_ylabel("GPU use (%)")
            ax_u.set_title("GPU Utilization (all GPUs)")
            ax_u.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            ax_u.legend(loc="upper left", ncol=2)

            # VRAM subplot
            for gpu in sorted(gpu_series.keys()):
                s = gpu_series[gpu]
                ax_v.plot(s["ts"], s["vram_gib"], label=f"GPU {gpu}")
            ax_v.set_ylabel("VRAM used (GiB)")
            ax_v.set_xlabel("Time")
            ax_v.set_title("VRAM Usage (all GPUs)")
            ax_v.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

            plt.tight_layout()
            combined_png = Path(args.plot).with_name(Path(args.plot).stem + "_combined_all.png")
            plt.savefig(combined_png)
            plt.close()
            print(f"Wrote combined plot: {combined_png}")

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote summary JSON: {json_path}")

if __name__ == "__main__":
    main()
