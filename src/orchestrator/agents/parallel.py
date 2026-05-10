"""
Parallel execution of modality agents.

Each agent issues an HTTP call to the vLLM server, which is I/O-bound,
so a ThreadPoolExecutor gives true concurrency. With a single vLLM
instance, continuous batching on the GPU absorbs concurrent requests;
with multiple vLLM instances on different GPUs, this becomes genuine
task-parallelism across devices.

This is the parallelism point measured in the HPC benchmark chapter.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from src.orchestrator.agents.base import AgentSummary, ModalityAgent

logger = logging.getLogger(__name__)


def run_agents_parallel(
    agents: dict[str, ModalityAgent],
    modality_features: dict[str, np.ndarray],
    max_workers: int | None = None,
) -> tuple[dict[str, AgentSummary], dict[str, float]]:
    """
    Dispatch one agent per available modality, concurrently.

    Args:
        agents:             {modality_key: ModalityAgent instance}.
        modality_features:  {modality_key: 1-D feature vector}. Only
                            modalities present here are dispatched.
        max_workers:        Pool size. Defaults to len(modality_features).

    Returns:
        summaries: {modality_key: AgentSummary}
        timings:   {modality_key: wall-clock seconds, "_total": batch wall-clock}
    """
    targets = {
        mod: feats
        for mod, feats in modality_features.items()
        if mod in agents and feats is not None
    }
    if not targets:
        return {}, {"_total": 0.0}

    workers = max_workers or len(targets)
    summaries: dict[str, AgentSummary] = {}
    timings: dict[str, float] = {}

    def _run(mod: str, feats: np.ndarray) -> tuple[str, AgentSummary, float]:
        t0 = time.perf_counter()
        summary = agents[mod].analyze(feats)
        return mod, summary, time.perf_counter() - t0

    t_batch = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run, mod, feats): mod for mod, feats in targets.items()}
        for fut in as_completed(futures):
            mod = futures[fut]
            try:
                mod_, summary, dt = fut.result()
                summaries[mod_] = summary
                timings[mod_] = dt
            except Exception as e:
                logger.error(
                    f"Agent for '{mod}' raised in parallel pool: {e}. Using stub."
                )
                summaries[mod] = agents[mod]._stub_summary(
                    np.asarray(targets[mod]).flatten(),
                    reason=f"pool error: {e}",
                )
                timings[mod] = -1.0

    timings["_total"] = time.perf_counter() - t_batch
    return summaries, timings
