"""
Methylation understanding agent.

Reads the 16166-dim CpG probe / SNP vector. Without external probe→gene
annotation we cannot meaningfully name biological loci from probe IDs
alone, so the agent reports top probes by |value|, the
hyper/hypo-methylation balance, and stays honest about interpretation
limits.
"""

from __future__ import annotations

import numpy as np

from src.orchestrator.agents.base import ModalityAgent


class MethylationAgent(ModalityAgent):
    modality = "methylation"

    SYSTEM_PROMPT = (
        "You are an expert in cancer epigenetics, specialising in DNA "
        "methylation patterns in lung adenocarcinoma and squamous cell "
        "carcinoma. You analyse a high-dimensional CpG probe vector "
        "(mostly Illumina 450K / EPIC probe IDs).\n\n"
        "Treat positive values as hyper-methylation and negative values "
        "as hypo-methylation.\n\n"
        "You receive only the most prominent probes by absolute value. "
        "Reason about general patterns: hyper- vs hypo-methylation "
        "balance, presence of well-known smoking-related or age-related "
        "signatures if probe identifiers suggest them, and overall "
        "sparsity. Without external annotation you CANNOT reliably name "
        "specific genes from probe IDs alone — be honest about this and "
        "set confidence accordingly.\n\n"
        "Respond ONLY in JSON with this schema:\n"
        '{"summary": "<3-5 line narrative>", '
        '"key_features": ["<probe id from the input>", ...], '
        '"confidence": "high"|"medium"|"low", '
        '"concerns": ["<limitation>", ...]}'
    )

    def _build_prompt(self, features: np.ndarray) -> str:
        if features.size == 0 or not np.any(features):
            return (
                "Methylation vector is empty or all zero. Set "
                "confidence='low' and explain under concerns."
            )

        topk = self._topk_named(features, k=15)
        if topk:
            topk_str = ", ".join(f"{n}={v:.3g}" for n, v in topk)
            metadata_note = ""
            values = np.array([v for _, v in topk], dtype=np.float32)
        else:
            top_idx = np.argsort(np.abs(features))[::-1][:15]
            topk_str = ", ".join(
                f"idx{int(i)}={float(features[i]):.3g}" for i in top_idx
            )
            metadata_note = (
                "Probe-name metadata unavailable — only positional indices "
                "shown. Note this under concerns."
            )
            values = features[top_idx]

        n_pos = int(np.sum(values > 0))
        n_neg = int(np.sum(values < 0))
        sparsity = float(1.0 - np.count_nonzero(features) / features.size)

        return (
            f"Methylation vector: {features.size} CpG probes / SNPs, "
            f"sparsity={sparsity:.1%}.\n\n"
            f"Top-15 probes by |value|: {topk_str}\n\n"
            f"Among those: {n_pos} positive, {n_neg} negative values.\n"
            f"{metadata_note}\n\n"
            f"Summarise the methylation profile, comment on the "
            f"hyper/hypo balance, and list 4-8 of the probe identifiers "
            f"above under `key_features` (copy them exactly). Note any "
            f"interpretation limitations under `concerns`."
        )
