"""
Clinical understanding agent.

Reads the 63-dim clinical feature vector (demographics, TNM staging,
histological subtype, smoking history, treatment) and produces a
physician-readable narrative for the Miner. Clinical data mixes
one-hot indicators with z-scored continuous values, so we surface
active categorical fields and non-zero continuous values rather than
top-k by magnitude.
"""

from __future__ import annotations

import numpy as np

from src.orchestrator.agents.base import ModalityAgent

COLUMN_TYPE_BINARY_01 = "binary_01"
COLUMN_TYPE_BINARY_M11 = "binary_m11"
COLUMN_TYPE_CONTINUOUS = "continuous"


def infer_clinical_column_types(
    pool: list[dict], n_features: int, tol: float = 1e-6
) -> list[str]:
    """Infer clinical column types: binary 0/1, binary -1/1, or continuous."""
    if not pool or n_features <= 0:
        return []

    all_01 = np.ones(n_features, dtype=bool)
    all_m11 = np.ones(n_features, dtype=bool)
    seen = np.zeros(n_features, dtype=bool)
    for entry in pool:
        features = entry.get("features", {})
        clinical = features.get("clinical")
        if clinical is None:
            continue
        arr = np.asarray(clinical, dtype=np.float32).flatten()
        if arr.size != n_features:
            continue
        valid = np.isfinite(arr)
        if not np.any(valid):
            continue
        seen |= valid

        near0 = np.isclose(arr, 0.0, atol=tol)
        near1 = np.isclose(arr, 1.0, atol=tol)
        near_m1 = np.isclose(arr, -1.0, atol=tol)

        is_01 = near0 | near1
        is_m11 = is_01 | near_m1

        all_01 &= (~valid) | is_01
        all_m11 &= (~valid) | is_m11

    types: list[str] = []
    for i in range(n_features):
        if seen[i] and all_01[i]:
            types.append(COLUMN_TYPE_BINARY_01)
        elif seen[i] and all_m11[i]:
            types.append(COLUMN_TYPE_BINARY_M11)
        else:
            types.append(COLUMN_TYPE_CONTINUOUS)

    return types


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
        "Categorical fields are one-hot encoded (0/1 or -1/1; -1 means "
        "not present). Continuous numeric fields are z-scores: negative "
        "means below the cohort mean, positive above, and zero is the "
        "cohort mean. Do not treat negative z-scores as absence."
        "For continuous features (z-scores), describe them strictly as one of: "
        "'slightly above average' (z between 0.1 and 0.5), 'above average' (z "
        "between 0.5 and 1.5), 'well above average' (z > 1.5), and the "
        "corresponding 'below average' phrasings for negative z-scores. Use "
        "'near average' for |z| < 0.1. Do NOT use absolute clinical terms like "
        "'heavy', 'light', 'minimal', 'severe', or convert z-scores into "
        "absolute units (years, pack-years).\n\n"
        "Respond ONLY in JSON with this schema:\n"
        '{"summary": "<3-5 line narrative>", '
        '"key_features": ["<feature name as given>", ...], '
        '"confidence": "high"|"medium"|"low", '
        '"concerns": ["<data-quality issue>", ...]}'
    )

    def __init__(
        self,
        llm,
        metadata: dict | None = None,
        clinical_column_types: list[str] | None = None,
    ):
        super().__init__(llm, metadata)
        self.clinical_column_types = clinical_column_types or []

    def _build_prompt(self, features: np.ndarray) -> str:
        arr = np.asarray(features, dtype=np.float32).flatten()
        n = min(len(self.columns), arr.size)
        tol = 1e-6

        if self.columns and self.clinical_column_types:
            cont_lines = []
            bin_lines = []
            for i in range(n):
                name = self.columns[i]
                val = float(arr[i])
                if not np.isfinite(val):
                    continue
                col_type = (
                    self.clinical_column_types[i]
                    if i < len(self.clinical_column_types)
                    else COLUMN_TYPE_CONTINUOUS
                )
                if col_type in (COLUMN_TYPE_BINARY_01, COLUMN_TYPE_BINARY_M11):
                    if abs(val - 1.0) < tol:
                        bin_lines.append(f"  - {name}")
                else:
                    if abs(val) < tol:
                        continue
                    cont_lines.append(f"  - {name}: {val:.3g} (z-score)")

            bullet_lines = (cont_lines + bin_lines)[:30]
        else:
            active = self._active_named(features, max_n=30)
            bullet_lines = []
            for name, value in active:
                # One-hot indicator → show name only; continuous (e.g. age) → show value
                if abs(value - 1.0) < tol:
                    bullet_lines.append(f"  - {name}")
                elif abs(value + 1.0) < tol:
                    continue
                else:
                    bullet_lines.append(f"  - {name}: {value:.3g} (z-score)")

        if not bullet_lines:
            return (
                "The patient's clinical record is empty (all 63 features are "
                "zero). Set confidence='low' and describe the absence under "
                "concerns. Leave key_features empty."
            )

        bullet_block = "\n".join(bullet_lines)

        return (
            "Active clinical fields for this patient:\n\n"
            f"{bullet_block}\n\n"
            "Interpretation guidance: continuous values are z-scores; "
            "describe them as above/below cohort average and avoid "
            "absolute units (years, cigarettes/day).\n\n"
            "Summarise this profile, identify the strongest prognostic signals, "
            "and list 4-8 of the field names above under `key_features` "
            "(copy the names exactly). Flag any clinically expected but "
            "missing fields under `concerns` (e.g., no staging recorded, "
            "no smoking history, no treatment fields)."
        )
