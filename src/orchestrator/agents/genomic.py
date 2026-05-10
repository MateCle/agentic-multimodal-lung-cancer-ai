"""
Genomic understanding agent.

Reads the 1824-dim REACTOME pathway-activity vector and produces a
hallmark-grouped summary that distinguishes up- and down-regulated
pathways. Uses pathway names from metadata['transcriptomics_columns'].
"""

from __future__ import annotations

import numpy as np

from src.orchestrator.agents.base import ModalityAgent


class GenomicAgent(ModalityAgent):
    modality = "transcriptomics"

    SYSTEM_PROMPT = (
        "You are a molecular oncologist with deep expertise in cancer "
        "transcriptomics and pathway biology, specialising in lung "
        "adenocarcinoma (LUAD) and squamous cell carcinoma (LUSC). You "
        "analyse REACTOME pathway activity scores where each entry "
        "represents enrichment / activity of a specific biological "
        "pathway in a single patient.\n\n"
        "Group the most active pathways by cancer-hallmark family — e.g. "
        "EMT and migration, immune signalling, DNA damage repair, "
        "metabolism, proliferation, apoptosis — and identify the "
        "dominant biological theme. Distinguish strongly upregulated "
        "(positive scores) from downregulated (negative scores) "
        "pathways.\n\n"
        "Respond ONLY in JSON with this schema:\n"
        '{"summary": "<3-5 line narrative grouping pathways by hallmark>", '
        '"key_features": ["<pathway name>", ...], '
        '"confidence": "high"|"medium"|"low", '
        '"concerns": ["<data-quality issue>", ...]}'
    )

    def _build_prompt(self, features: np.ndarray) -> str:
        if features.size == 0 or not np.any(features):
            return (
                "Transcriptomic vector is empty or all zero. Set "
                "confidence='low' and explain under concerns. "
                "Leave key_features empty."
            )

        topk = self._topk_named(features, k=20)
        if not topk:
            return (
                "Transcriptomic vector available but no pathway-name "
                "metadata. Set confidence='low' and note this under "
                "concerns. Leave key_features empty."
            )

        up = [(n, v) for n, v in topk if v > 0]
        down = [(n, v) for n, v in topk if v < 0]

        up_str = (
            "; ".join(f"{n} ({v:+.2f})" for n, v in up[:12])
            if up
            else "(none in top-20)"
        )
        down_str = (
            "; ".join(f"{n} ({v:+.2f})" for n, v in down[:12])
            if down
            else "(none in top-20)"
        )

        return (
            f"Top REACTOME pathways for this patient (1824 total).\n\n"
            f"UPREGULATED (positive scores):\n  {up_str}\n\n"
            f"DOWNREGULATED (negative scores):\n  {down_str}\n\n"
            f"Identify the dominant cancer-hallmark families, comment "
            f"on the up/down balance, and list 4-8 of the pathway names "
            f"above under `key_features` (copy the names exactly)."
        )
