"""
Clinical understanding agent.

Reads the 63-dim clinical feature vector (demographics, TNM staging,
histological subtype, smoking history, treatment) and produces a
physician-readable narrative for the Miner. Clinical data is heavily
one-hot encoded, so we surface ALL non-zero fields by name rather than
top-k by magnitude.
"""

from __future__ import annotations

import numpy as np

from src.orchestrator.agents.base import ModalityAgent


class ClinicalAgent(ModalityAgent):
    modality = "clinical"

    SYSTEM_PROMPT = (
        "You are a clinical oncology expert specialised in lung cancer. "
        "Given a patient's structured clinical record (demographics, TNM "
        "staging, histological subtype, smoking exposure, and prior "
        "treatment), produce a concise, physician-readable summary that "
        "captures prognostic signals relevant to disease-specific survival.\n\n"
        "Focus on: histological subtype (LUAD vs LUSC), AJCC stage, T/N/M "
        "components, age at diagnosis, smoking history, recorded treatment. "
        "Do not invent values that are not present in the input.\n\n"
        "Respond ONLY in JSON with this schema:\n"
        '{"summary": "<3-5 line narrative>", '
        '"key_features": ["<feature name as given>", ...], '
        '"confidence": "high"|"medium"|"low", '
        '"concerns": ["<data-quality issue>", ...]}'
    )

    def _build_prompt(self, features: np.ndarray) -> str:
        active = self._active_named(features, max_n=30)

        if not active:
            return (
                "The patient's clinical record is empty (all 63 features are "
                "zero). Set confidence='low' and describe the absence under "
                "concerns. Leave key_features empty."
            )

        bullet_lines = []
        for name, value in active:
            # One-hot indicator → show name only; continuous (e.g. age) → show value
            if abs(value - 1.0) < 1e-6:
                bullet_lines.append(f"  - {name}")
            else:
                bullet_lines.append(f"  - {name}: {value:.3g}")

        bullet_block = "\n".join(bullet_lines)

        return (
            "Active clinical fields for this patient:\n\n"
            f"{bullet_block}\n\n"
            "Summarise this profile, identify the strongest prognostic signals, "
            "and list 4-8 of the field names above under `key_features` "
            "(copy the names exactly). Flag any clinically expected but "
            "missing fields under `concerns` (e.g., no staging recorded, "
            "no smoking history, no treatment fields)."
        )
