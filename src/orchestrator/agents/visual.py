"""
Visual (WSI) understanding agent — analogical retrieval mode.

Without tile-level features the slide-level embedding is opaque, so the
agent characterises this WSI BY ANALOGY: it retrieves the top-N most
similar WSI embeddings from a reference pool and describes the patient
through the clinical profiles of those neighbours. This avoids
hallucinating direct morphological observations the embedding cannot
support.

The pool argument follows the same shape as build_pool_index() in
generator.py — a list of dicts with at least 'patient_id', 'wsi', and
'clinical' keys.
"""

from __future__ import annotations

import numpy as np

from src.orchestrator.agents.base import ModalityAgent
from src.orchestrator.llm import BaseLLMClient


class VisualAgent(ModalityAgent):
    modality = "wsi"

    SYSTEM_PROMPT = (
        "You are an expert in computational histopathology. You analyse "
        "slide-level WSI embeddings (1024-dim foundation-model features). "
        "You DO NOT have direct visual access to tiles or regions of "
        "interest, so your analysis MUST rely on (a) global embedding "
        "statistics and (b) similarity to reference patients with known "
        "clinical profiles.\n\n"
        "Be explicit that conclusions are inferential, not based on "
        "direct image inspection. Avoid asserting specific morphological "
        "features (e.g. 'high tumour cellularity', 'necrotic regions') "
        "unless the analogy with neighbours strongly supports it. When "
        "the analogy is weak, say so and lower confidence.\n\n"
        "Respond ONLY in JSON with this schema:\n"
        '{"summary": "<3-5 line analogical narrative>", '
        '"key_features": ["<clinical field name from neighbours>", ...], '
        '"confidence": "high"|"medium"|"low", '
        '"concerns": ["<limitation>", ...]}'
    )

    def __init__(
        self,
        llm: BaseLLMClient,
        metadata: dict | None = None,
        pool: list[dict] | None = None,
        n_neighbors: int = 3,
    ):
        super().__init__(llm, metadata)
        self.pool = pool or []
        self.n_neighbors = n_neighbors
        self.clinical_columns: list[str] = (metadata or {}).get("clinical_columns", [])

    def _build_prompt(self, features: np.ndarray) -> str:
        if features.size == 0 or not np.any(features):
            return (
                "WSI embedding is empty or all zero. Set confidence='low' "
                "and explain under concerns."
            )

        norm = float(np.linalg.norm(features))
        sparsity = float(1.0 - np.count_nonzero(features) / features.size)
        mean = float(features.mean())
        std = float(features.std())

        neighbour_block = self._neighbour_analogy_block(features)

        return (
            f"WSI embedding statistics: dim={features.size}, "
            f"L2 norm={norm:.3f}, sparsity={sparsity:.1%}, "
            f"mean={mean:.4f}, std={std:.4f}.\n\n"
            f"{neighbour_block}\n\n"
            f"Without tile-level features you cannot describe specific "
            f"morphological patterns. Instead, describe this WSI "
            f"embedding BY ANALOGY: what kind of patient does it most "
            f"resemble in the reference cohort, based on the clinical "
            f"profiles of the nearest neighbours? Be explicit that this "
            f"is similarity-based inference, not direct image analysis. "
            f"List the clinical feature names you used as `key_features` "
            f"and any caveats under `concerns`."
        )

    def _neighbour_analogy_block(self, query: np.ndarray) -> str:
        if not self.pool:
            return "Reference cohort unavailable; analogical analysis disabled."

        qnorm = float(np.linalg.norm(query)) + 1e-9
        sims: list[tuple[float, dict]] = []
        for entry in self.pool:
            other = entry.get("wsi")
            if other is None:
                continue
            o = np.asarray(other, dtype=np.float32).flatten()
            if o.size != query.size:
                continue
            cos = float(np.dot(query, o) / (qnorm * (np.linalg.norm(o) + 1e-9)))
            sims.append((cos, entry))

        if not sims:
            return "No reference patients with WSI data; analogical analysis skipped."

        sims.sort(key=lambda x: -x[0])
        top = sims[: self.n_neighbors]

        lines = ["Top similar patients in the reference cohort (cosine sim):"]
        for sim, entry in top:
            pid = entry.get("patient_id", "<unknown>")
            cohort = entry.get("cohort", "")
            active = self._active_clinical(entry.get("clinical"))
            active_str = (
                ", ".join(active[:8]) if active else "(no active clinical fields)"
            )
            lines.append(f"  - {pid} (cohort={cohort}, sim={sim:.3f}): {active_str}")
        return "\n".join(lines)

    def _active_clinical(self, clinical) -> list[str]:
        if clinical is None or not self.clinical_columns:
            return []
        arr = np.asarray(clinical, dtype=np.float32).flatten()
        n = min(len(self.clinical_columns), arr.size)
        return [self.clinical_columns[i] for i in range(n) if arr[i] != 0]
