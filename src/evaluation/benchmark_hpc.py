"""
HPC benchmarks for the multimodal lung cancer orchestrator.

Two experiments, mapping directly to the two parallelism categories named
in the High-Performance Systems learning objectives:

  Experiment 1 — Task parallelism (intra-patient).
      Modality agents (Clinical, Genomic, Visual, Methylation) are
      dispatched concurrently via the ThreadPoolExecutor in
      `src/orchestrator/parallel.py` against a single vLLM instance.
      vLLM's continuous batching absorbs the concurrent requests on
      the GPU. We sweep `max_workers ∈ {1, 2, 4}` on the same patients
      and measure agent-batch wall time, per-agent latency, and the
      speedup of the parallel batch over the serialised baseline.

  Experiment 2 — Data parallelism (inter-patient).
      The patient set is partitioned across N worker subprocesses, each
      pinned to a distinct vLLM endpoint on a distinct GPU via the
      `OPENAI_BASE_URL` environment variable. Each worker runs a fully
      independent orchestrator pipeline. We sweep `N_workers ∈ {1, 2, 4}`
      and report strong scaling (fixed total patients, vary N_workers)
      and weak scaling (patients scaled with workers).

Output: JSON file with all timings; companion `analyze_hpc.py` computes
speedup, parallel efficiency, throughput, and produces matplotlib plots
that go directly into the Evaluation chapter, *System Performance*
section.

USAGE
-----

Single-GPU baseline (Experiment 1):
    OPENAI_BASE_URL=http://localhost:8000/v1 \\
    python -m src.evaluation.benchmark_hpc \\
        --experiment 1 \\
        --n-patients 16 \\
        --max-workers-list 1,2,4 \\
        --output results/hpc/exp1_singlegpu.json

Multi-GPU strong scaling (Experiment 2):
    python -m src.evaluation.benchmark_hpc \\
        --experiment 2 \\
        --n-patients 32 \\
        --vllm-endpoints http://localhost:8000/v1,http://localhost:8001/v1,http://localhost:8002/v1,http://localhost:8003/v1 \\
        --workers-list 1,2,4 \\
        --output results/hpc/exp2_strongscaling.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# --- Project root setup (mirrors src/orchestrator/run.py) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "extracted" / "cache_data"
SPLITS_DIR = DATA_DIR / "splits"

SPLIT_FILES = {
    "luad": "tcga_luad_DSS_k3_r1_test0.2_val0.2_seed42.json",
    "lusc": "tcga_lusc_DSS_k5_r1_test0.2_val0.2_seed42.json",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> float:
    return time.perf_counter()


def _load_test_patient_ids(n_max: int | None = None) -> list[tuple[str, str]]:
    """Return (patient_id, cohort) tuples from both cohorts' test splits."""
    from src.data_loader import load_split

    pairs: list[tuple[str, str]] = []
    for cohort, split_file in SPLIT_FILES.items():
        if not (SPLITS_DIR / split_file).exists():
            print(f"[bench] WARNING: split file not found: {SPLITS_DIR / split_file}")
            continue
        _, _, test_ids = load_split(SPLITS_DIR, split_file)
        pairs.extend((pid, cohort) for pid in test_ids)
    if n_max is not None:
        pairs = pairs[:n_max]
    return pairs


def _gpu_snapshot() -> dict | None:
    """Capture a one-shot nvidia-smi readout. Returns None if nvidia-smi unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
        ).decode()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            rows.append(
                {
                    "gpu": int(parts[0]),
                    "util_pct": int(parts[1]),
                    "mem_used_mb": int(parts[2]),
                    "mem_total_mb": int(parts[3]),
                }
            )
    return {"timestamp": time.time(), "gpus": rows}


# ---------------------------------------------------------------------------
# Experiment 1 — Task parallelism (intra-patient, single GPU)
# ---------------------------------------------------------------------------


def _build_agents_and_llm():
    """Construct one LLM client and one agent per modality. Imported lazily."""
    from src.orchestrator.agents.clinical import ClinicalAgent
    from src.orchestrator.agents.genomic import GenomicAgent
    from src.orchestrator.agents.methylation import MethylationAgent
    from src.orchestrator.agents.visual import VisualAgent
    from src.orchestrator.llm import get_llm_client

    llm = get_llm_client()
    return llm, {
        "clinical": ClinicalAgent(llm),
        "transcriptomics": GenomicAgent(llm),
        "wsi": VisualAgent(llm),
        "methylation": MethylationAgent(llm),
    }


def _load_patient_features(patient_id: str, cohort: str) -> dict:
    """Return {modality: 1-D ndarray} for the modalities the patient has."""
    from src.data_loader import load_patient, load_raw_data

    raw, _ = load_raw_data(DATA_DIR, cohort)
    patient = load_patient(patient_id, raw)
    if patient is None:
        raise KeyError(f"Patient {patient_id} not found in cohort {cohort}")
    return {
        mod: patient[mod]
        for mod in ("clinical", "transcriptomics", "wsi", "methylation")
        if patient[mod] is not None
    }


def experiment_1_task_parallelism(
    patient_pairs: list[tuple[str, str]],
    max_workers_list: list[int],
    warmup: bool = True,
) -> dict[str, Any]:
    """Sweep `max_workers` on the agent batch, holding the vLLM endpoint fixed."""
    from src.orchestrator.parallel import run_agents_parallel

    print("\n[Experiment 1] Task parallelism — single vLLM, vary max_workers\n")
    llm, agents = _build_agents_and_llm()

    if warmup:
        try:
            llm.invoke("warmup", system="reply with 'ok'")
            print("[bench] warmup call complete")
        except Exception as e:
            print(f"[bench] WARNING: warmup call failed: {e}")

    runs = []
    for patient_id, cohort in patient_pairs:
        try:
            features = _load_patient_features(patient_id, cohort)
        except Exception as e:
            print(f"[bench] skip {patient_id}: {e}")
            continue
        n_present = len(features)
        for nw in max_workers_list:
            t0 = _now()
            summaries, timings = run_agents_parallel(agents, features, max_workers=nw)
            wall = _now() - t0
            runs.append(
                {
                    "patient_id": patient_id,
                    "cohort": cohort,
                    "n_modalities_present": n_present,
                    "max_workers": nw,
                    "batch_wall_seconds": wall,
                    "parallel_total_reported": timings.get("_total"),
                    "per_agent_seconds": {
                        k: v for k, v in timings.items() if k != "_total"
                    },
                    "n_agents_run": len(summaries),
                }
            )
            print(
                f"  {patient_id} ({cohort}) n_mod={n_present} workers={nw} "
                f"wall={wall:.2f}s"
            )

    summary_by_workers: dict[int, dict] = {}
    for nw in max_workers_list:
        walls = [r["batch_wall_seconds"] for r in runs if r["max_workers"] == nw]
        if not walls:
            continue
        summary_by_workers[nw] = {
            "n_runs": len(walls),
            "mean": statistics.mean(walls),
            "median": statistics.median(walls),
            "stdev": statistics.stdev(walls) if len(walls) > 1 else 0.0,
            "min": min(walls),
            "max": max(walls),
        }
    print("\n[Experiment 1] Aggregated wall times (seconds):")
    for nw, stats in sorted(summary_by_workers.items()):
        print(
            f"  workers={nw}: mean={stats['mean']:.2f}  median={stats['median']:.2f} "
            f"stdev={stats['stdev']:.2f}  n={stats['n_runs']}"
        )

    return {
        "experiment": 1,
        "name": "task_parallelism_intra_patient",
        "config": {
            "max_workers_list": max_workers_list,
            "n_patients_attempted": len(patient_pairs),
            "vllm_base_url": os.environ.get("OPENAI_BASE_URL"),
            "model": os.environ.get("LLM_MODEL"),
        },
        "runs": runs,
        "aggregated": summary_by_workers,
        "gpu_snapshot_after": _gpu_snapshot(),
    }


# ---------------------------------------------------------------------------
# Experiment 2 — Data parallelism (inter-patient, multi-GPU)
# ---------------------------------------------------------------------------


WORKER_SCRIPT_NAME = "_bench_worker.py"

WORKER_SCRIPT_BODY = (
    "import argparse, json, sys, time\n"
    "from pathlib import Path\n"
    "ROOT = Path(__file__).resolve().parents[3]\n"
    "sys.path.insert(0, str(ROOT))\n"
    "from src.orchestrator.graph import build_graph\n"
    "from src.orchestrator.run import run_patient\n"
    "\n"
    "DATA_DIR = ROOT / 'data' / 'extracted' / 'cache_data'\n"
    "ap = argparse.ArgumentParser()\n"
    "ap.add_argument('--patient', required=True)\n"
    "args = ap.parse_args()\n"
    "graph = build_graph(DATA_DIR)\n"
    "t0 = time.perf_counter()\n"
    "ok, err = True, None\n"
    "try:\n"
    "    state = run_patient(patient_id=args.patient, graph=graph, verbose=False)\n"
    "except Exception as e:\n"
    "    ok, err = False, repr(e)\n"
    "wall = time.perf_counter() - t0\n"
    "print(json.dumps({'patient_id': args.patient, 'total_wall': wall, "
    "'ok': ok, 'error': err}))\n"
)


def _ensure_worker_script(workdir: Path) -> Path:
    """Write a tiny worker script that runs one patient and prints timing JSON."""
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / WORKER_SCRIPT_NAME
    path.write_text(WORKER_SCRIPT_BODY)
    return path


def _spawn_worker(
    worker_script: Path,
    patient_id: str,
    base_url: str,
    log_path: Path,
) -> subprocess.Popen:
    """Spawn a single subprocess worker pinned to one vLLM endpoint."""
    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = base_url
    log_fh = open(log_path, "w")
    return subprocess.Popen(
        [sys.executable, str(worker_script), "--patient", patient_id],
        stdout=subprocess.PIPE,
        stderr=log_fh,
        env=env,
        cwd=str(PROJECT_ROOT),
        text=True,
    )


def _round_robin(items: list, n_buckets: int) -> list[list]:
    """Distribute items into n_buckets, round-robin order."""
    buckets: list[list] = [[] for _ in range(n_buckets)]
    for i, it in enumerate(items):
        buckets[i % n_buckets].append(it)
    return buckets


def experiment_2_data_parallelism(
    patient_pairs: list[tuple[str, str]],
    vllm_endpoints: list[str],
    workers_list: list[int],
    workdir: Path,
    weak_scaling: bool = False,
) -> dict[str, Any]:
    """Distribute patients across N workers, each pinned to a vLLM endpoint."""
    if not vllm_endpoints:
        raise ValueError("vllm_endpoints must be non-empty")
    max_w = max(workers_list)
    if max_w > len(vllm_endpoints):
        raise ValueError(
            f"workers_list max ({max_w}) > available endpoints ({len(vllm_endpoints)})"
        )

    print(
        "\n[Experiment 2] Data parallelism — "
        f"{'weak' if weak_scaling else 'strong'} scaling\n"
    )
    worker_script = _ensure_worker_script(workdir)

    runs = []
    for n_workers in workers_list:
        endpoints = vllm_endpoints[:n_workers]
        if weak_scaling:
            assigned = patient_pairs * n_workers
            buckets = _round_robin(assigned, n_workers)
        else:
            buckets = _round_robin(patient_pairs, n_workers)

        per_worker_max = max(len(b) for b in buckets) if buckets else 0
        print(
            f"  N_workers={n_workers}  endpoints={endpoints}  "
            f"per_worker_max={per_worker_max}"
        )

        # Run each worker bucket sequentially in the parent (one subprocess at
        # a time). For TRUE inter-worker concurrency we'd interleave Popen
        # calls then collect with as_completed; for clarity of attribution
        # this version runs one subprocess at a time per bucket — speedup
        # comes from the within-bucket sequential time being shorter when
        # there are more workers and each does fewer patients.
        t0 = _now()
        worker_results: list[dict] = []
        # Concurrent dispatch: one subprocess per worker in flight at the
        # same time. Each worker processes its bucket sequentially.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _run_bucket(w_idx: int, bucket: list):
            results = []
            for patient_id, _cohort in bucket:
                log_path = workdir / f"worker{w_idx}_{patient_id}.log"
                proc = _spawn_worker(
                    worker_script, patient_id, endpoints[w_idx], log_path
                )
                stdout, _ = proc.communicate()
                rc = proc.returncode
                line = (stdout or "").strip().splitlines()[-1] if stdout else ""
                try:
                    parsed = (
                        json.loads(line)
                        if line
                        else {"ok": False, "error": "no stdout"}
                    )
                except json.JSONDecodeError:
                    parsed = {"ok": False, "error": f"bad stdout: {line[:200]}"}
                parsed["worker_idx"] = w_idx
                parsed["endpoint"] = endpoints[w_idx]
                parsed["return_code"] = rc
                results.append(parsed)
            return results

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_run_bucket, w_idx, bucket): w_idx
                for w_idx, bucket in enumerate(buckets)
            }
            for fut in as_completed(futures):
                worker_results.extend(fut.result())
        wall = _now() - t0

        runs.append(
            {
                "n_workers": n_workers,
                "endpoints": endpoints,
                "scaling_mode": "weak" if weak_scaling else "strong",
                "total_patients_processed": sum(len(b) for b in buckets),
                "wall_seconds": wall,
                "worker_results": worker_results,
            }
        )
        print(
            f"    -> wall={wall:.1f}s  "
            f"patients={sum(len(b) for b in buckets)}  "
            f"ok={sum(1 for r in worker_results if r.get('ok'))}"
            f"/{len(worker_results)}"
        )

    return {
        "experiment": 2,
        "name": "data_parallelism_inter_patient",
        "config": {
            "scaling_mode": "weak" if weak_scaling else "strong",
            "vllm_endpoints": vllm_endpoints,
            "workers_list": workers_list,
            "n_patients_per_run": len(patient_pairs),
        },
        "runs": runs,
        "gpu_snapshot_after": _gpu_snapshot(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HPC benchmarks for the orchestrator.")
    p.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2],
        required=True,
        help="1 = task parallelism (intra-patient); 2 = data parallelism (inter-patient).",
    )
    p.add_argument("--n-patients", type=int, default=16)
    p.add_argument(
        "--max-workers-list",
        type=str,
        default="1,2,4",
        help="Experiment 1 only: comma-separated max_workers values.",
    )
    p.add_argument(
        "--vllm-endpoints",
        type=str,
        default="",
        help="Experiment 2 only: comma-separated list of vLLM base URLs.",
    )
    p.add_argument(
        "--workers-list",
        type=str,
        default="1,2,4",
        help="Experiment 2 only: comma-separated worker counts.",
    )
    p.add_argument(
        "--scaling",
        choices=["strong", "weak"],
        default="strong",
        help="Experiment 2 only.",
    )
    p.add_argument("--output", type=str, required=True, help="Output JSON file path.")
    p.add_argument(
        "--workdir",
        type=str,
        default="results/hpc/_workdir",
        help="Experiment 2 only: scratch dir for worker script + per-worker logs.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    patient_pairs = _load_test_patient_ids(args.n_patients)
    if not patient_pairs:
        print("[bench] FATAL: no patient IDs loaded; check splits dir.")
        return 1
    print(f"[bench] loaded {len(patient_pairs)} test patients")

    if args.experiment == 1:
        max_workers_list = [int(x) for x in args.max_workers_list.split(",")]
        result = experiment_1_task_parallelism(patient_pairs, max_workers_list)
    else:
        endpoints = [e.strip() for e in args.vllm_endpoints.split(",") if e.strip()]
        workers_list = [int(x) for x in args.workers_list.split(",")]
        workdir = Path(args.workdir)
        result = experiment_2_data_parallelism(
            patient_pairs,
            endpoints,
            workers_list,
            workdir,
            weak_scaling=(args.scaling == "weak"),
        )

    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n[bench] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
