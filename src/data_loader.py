"""
TCGA Multimodal Data Loader.
Handles loading of patient data, metadata, and train/test splits.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

MODALITY_KEYS = ["clinical", "transcriptomics", "wsi", "methylation"]
MODALITY_DIMS = {
    "clinical": 63,
    "transcriptomics": 1824,
    "wsi": 1024,
    "methylation": 16166,
}


def load_raw_data(data_dir: Path, cohort: str) -> tuple[dict, dict]:
    """Load the prepared data and metadata pickle files for a given cohort."""
    data_path = data_dir / f"tcga_{cohort}_prepared_data.pkl"
    meta_path = data_dir / f"tcga_{cohort}_metadata.pkl"

    with open(data_path, "rb") as f:
        data = pickle.load(f)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    return data, metadata


def load_split(
    splits_dir: Path,
    split_file: str,
    repeat: int = 0,
    fold: int = 0,
) -> tuple[list, list, list]:
    """
    Load patient IDs for train, validation, and test sets
    from a given split JSON file.
    """
    with open(splits_dir / split_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    fold_data = list(splits[f"repeat_{repeat}"][fold].values())
    return fold_data[0], fold_data[1], fold_data[2]


def load_patient(patient_id: str, raw_data: dict) -> Optional[dict]:
    """
    Load and structure data for a single patient.
    Returns None if the patient ID is not found.
    The 'avail' array encodes modality availability as:
    [clinical, transcriptomics, wsi, methylation]
    """
    if patient_id not in raw_data:
        return None

    record = raw_data[patient_id]
    avail = record["avail"]

    # Convertiamo il float in intero per evitare bug di approssimazione
    available = [key for i, key in enumerate(MODALITY_KEYS) if int(avail[i]) == 1]
    missing = [key for i, key in enumerate(MODALITY_KEYS) if int(avail[i]) == 0]

    return {
        "patient_id": patient_id,
        "clinical": record["clinical"] if int(avail[0]) == 1 else None,
        "transcriptomics": record["transcriptomics"] if int(avail[1]) == 1 else None,
        "wsi": record["wsi"] if int(avail[2]) == 1 else None,
        "methylation": record["methylation"] if int(avail[3]) == 1 else None,
        "available_modalities": available,
        "missing_modalities": missing,
        "label": record["label"],
    }


def load_split_patients(patient_ids: list, raw_data: dict) -> list[dict]:
    """Load all patients for a given list of patient IDs."""
    patients = []
    for pid in patient_ids:
        patient = load_patient(pid, raw_data)
        if patient is not None:
            patients.append(patient)
    return patients


def get_modality_stats(patients: list[dict]) -> dict:
    """
    Compute missing modality statistics across a set of patients.
    Returns per-modality counts and a breakdown of modality availability patterns.
    """
    total = len(patients)
    stats = {}

    for modality in MODALITY_KEYS:
        present = sum(1 for p in patients if modality in p["available_modalities"])
        stats[modality] = {
            "present": present,
            "missing": total - present,
            "pct_present": round(present / total * 100, 1),
        }

    pattern_counts = {}
    for p in patients:
        pattern = tuple(p["available_modalities"])
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    stats["modality_patterns"] = pattern_counts
    return stats
