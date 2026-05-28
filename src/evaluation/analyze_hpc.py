"""
Post-hoc analysis of the HPC benchmark JSONs.

Inputs:
    Comma-separated JSON files produced by `benchmark_hpc.py`. Any
    combination of Experiment 1 (task parallelism) and Experiment 2
    (data parallelism, strong or weak scaling) results is accepted; the
    analyser dispatches per-experiment automatically.

Outputs:
    - PNG plots in `--plot-dir`:
        * exp1_task_parallelism.png  (mean batch wall vs max_workers)
        * exp2_strong_scaling.png    (wall, speedup, efficiency vs N_workers)
        * exp2_weak_scaling.png      (wall vs N_workers; expected flat)
    - Markdown summary in `--report`:
        Tables of speedup, parallel efficiency, and per-agent latencies,
        ready for inclusion in the Evaluation chapter, *System
        Performance* section. The markdown is intentionally
        report-grade: short, with the key numbers in tables, no prose.

The script avoids hardcoding any C-index/AI-side metric — it is a
pure performance-analysis tool.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _is_exp1(blob: dict) -> bool:
    return blob.get("experiment") == 1


def _is_exp2(blob: dict) -> bool:
    return blob.get("experiment") == 2


def _safe_import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: WPS433

        return plt
    except ImportError:
        print(
            "[analyze] matplotlib not available; skipping plots, only writing report."
        )
        return None


# ---------------------------------------------------------------------------
# Experiment 1 — task parallelism analysis
# ---------------------------------------------------------------------------


def analyze_exp1(blob: dict, plot_dir: Path, plt) -> dict:
    """Compute speedup over the max_workers=1 baseline using medians."""
    runs = blob.get("runs", [])
    workers_list = sorted({r["max_workers"] for r in runs})
    if not workers_list:
        return {"error": "no runs in exp1 blob"}

    walls_by_w: dict[int, list[float]] = {w: [] for w in workers_list}
    for r in runs:
        walls_by_w[r["max_workers"]].append(r["batch_wall_seconds"])

    median_walls = {w: statistics.median(v) for w, v in walls_by_w.items() if v}
    baseline = median_walls.get(min(workers_list), float("inf"))
    speedup = {
        w: (baseline / median_walls[w]) if median_walls[w] > 0 else 0.0
        for w in median_walls
    }
    efficiency = {w: speedup[w] / w for w in speedup}

    if plt is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig, ax1 = plt.subplots(figsize=(6, 4))
        xs = sorted(median_walls.keys())
        ax1.plot(xs, [median_walls[x] for x in xs], "o-", label="Median wall (s)")
        ax1.set_xlabel("max_workers (concurrent agent dispatch)")
        ax1.set_ylabel("Median batch wall time (s)")
        ax1.set_xticks(xs)
        ax2 = ax1.twinx()
        ax2.plot(xs, [speedup[x] for x in xs], "s--", color="tab:red", label="Speedup")
        ax2.plot(
            xs, [efficiency[x] for x in xs], "^:", color="tab:green", label="Efficiency"
        )
        ax2.set_ylabel("Speedup / Efficiency")
        ax2.axhline(1.0, color="grey", linewidth=0.5)
        ax1.set_title("Experiment 1 — Task parallelism (single vLLM)")
        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(plot_dir / "exp1_task_parallelism.png", dpi=150)
        plt.close(fig)

    return {
        "experiment": 1,
        "median_walls": median_walls,
        "speedup_over_workers1": speedup,
        "parallel_efficiency": efficiency,
        "n_runs_per_workers": {w: len(walls_by_w[w]) for w in workers_list},
    }


# ---------------------------------------------------------------------------
# Experiment 2 — data parallelism analysis
# ---------------------------------------------------------------------------


def analyze_exp2(blob: dict, plot_dir: Path, plt) -> dict:
    """Strong/weak scaling tables and plots."""
    mode = blob.get("config", {}).get("scaling_mode", "strong")
    runs = blob.get("runs", [])
    walls = {r["n_workers"]: r["wall_seconds"] for r in runs}
    if not walls:
        return {"error": "no runs in exp2 blob"}

    workers_list = sorted(walls.keys())
    baseline = walls[min(workers_list)]
    speedup = {w: baseline / walls[w] if walls[w] > 0 else 0.0 for w in workers_list}
    efficiency = {w: speedup[w] / w for w in workers_list}

    # Throughput: patients per second
    throughput = {}
    for r in runs:
        n_pat = r.get("total_patients_processed", 0)
        wall = r.get("wall_seconds", 0)
        throughput[r["n_workers"]] = n_pat / wall if wall > 0 else 0.0

    if plt is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].plot(workers_list, [walls[w] for w in workers_list], "o-")
        axes[0].set_xlabel("N workers (= N GPUs)")
        axes[0].set_ylabel("Total wall time (s)")
        axes[0].set_xticks(workers_list)
        axes[0].set_title(f"{mode.capitalize()} scaling — wall time")

        axes[1].plot(
            workers_list,
            [speedup[w] for w in workers_list],
            "s-",
            label="Measured speedup",
        )
        axes[1].plot(
            workers_list, workers_list, "--", color="grey", label="Ideal (linear)"
        )
        axes[1].set_xlabel("N workers")
        axes[1].set_ylabel("Speedup")
        axes[1].set_xticks(workers_list)
        axes[1].set_title("Speedup vs ideal")
        axes[1].legend()

        axes[2].plot(workers_list, [efficiency[w] for w in workers_list], "^-")
        axes[2].axhline(1.0, color="grey", linewidth=0.5)
        axes[2].set_xlabel("N workers")
        axes[2].set_ylabel("Parallel efficiency")
        axes[2].set_xticks(workers_list)
        axes[2].set_ylim(0, 1.1)
        axes[2].set_title("Parallel efficiency = speedup / N")

        fig.suptitle(f"Experiment 2 — Data parallelism ({mode} scaling)", fontsize=12)
        fig.tight_layout()
        fig.savefig(plot_dir / f"exp2_{mode}_scaling.png", dpi=150)
        plt.close(fig)

    return {
        "experiment": 2,
        "scaling_mode": mode,
        "wall_seconds": walls,
        "speedup_over_workers1": speedup,
        "parallel_efficiency": efficiency,
        "throughput_patients_per_second": throughput,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def build_report(results: list[dict]) -> str:
    lines = ["# HPC benchmark summary", ""]
    lines.append(
        "Auto-generated by `analyze_hpc.py`. The numbers below are the ones "
        "that go into the Evaluation chapter, *System Performance* section. "
        "Speedup is computed against the smallest-workers configuration of "
        "each experiment; parallel efficiency is `speedup / N_workers`."
    )
    lines.append("")

    for res in results:
        if res.get("experiment") == 1:
            lines.append("## Experiment 1 — Task parallelism (intra-patient)")
            lines.append("")
            lines.append(
                "| max_workers | median batch wall (s) | speedup | efficiency |"
            )
            lines.append("|:-:|:-:|:-:|:-:|")
            for w in sorted(res["median_walls"].keys()):
                lines.append(
                    f"| {w} | {res['median_walls'][w]:.2f} "
                    f"| {res['speedup_over_workers1'][w]:.2f} "
                    f"| {res['parallel_efficiency'][w]:.2f} |"
                )
            lines.append("")
        elif res.get("experiment") == 2:
            mode = res.get("scaling_mode", "strong")
            lines.append(
                f"## Experiment 2 — Data parallelism ({mode} scaling, inter-patient)"
            )
            lines.append("")
            header = (
                "| N_workers | wall (s) | speedup | efficiency | throughput (pat/s) |"
            )
            sep = "|:-:|:-:|:-:|:-:|:-:|"
            lines.extend([header, sep])
            for w in sorted(res["wall_seconds"].keys()):
                lines.append(
                    f"| {w} | {res['wall_seconds'][w]:.1f} "
                    f"| {res['speedup_over_workers1'][w]:.2f} "
                    f"| {res['parallel_efficiency'][w]:.2f} "
                    f"| {res['throughput_patients_per_second'][w]:.3f} |"
                )
            lines.append("")
    lines.append("---")
    lines.append(
        "Plots: `plots/exp1_task_parallelism.png`, `plots/exp2_strong_scaling.png`, "
        "`plots/exp2_weak_scaling.png` (when available)."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        required=True,
        help="Comma-separated paths to benchmark_hpc.py JSON outputs.",
    )
    ap.add_argument("--plot-dir", default="results/hpc/plots")
    ap.add_argument("--report", default="results/hpc/hpc_summary.md")
    args = ap.parse_args()

    plt = _safe_import_pyplot()
    plot_dir = Path(args.plot_dir)

    results = []
    for raw in args.inputs.split(","):
        p = Path(raw.strip())
        if not p.exists():
            print(f"[analyze] missing input: {p}")
            continue
        blob = _load_json(p)
        if _is_exp1(blob):
            results.append(analyze_exp1(blob, plot_dir, plt))
        elif _is_exp2(blob):
            results.append(analyze_exp2(blob, plot_dir, plt))
        else:
            print(f"[analyze] unknown experiment id in {p}; skipping")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(results))
    print(f"[analyze] wrote report: {report_path}")
    if plt is not None:
        print(f"[analyze] wrote plots:  {plot_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
