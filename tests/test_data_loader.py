import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import load_patient, load_raw_data, load_split, load_split_patients

DATA_DIR = Path("data/extracted/cache_data")
SPLITS_DIR = DATA_DIR / "splits"


@pytest.fixture(scope="module")
def luad_data():
    data, _ = load_raw_data(DATA_DIR, "luad")
    return data


@pytest.fixture(scope="module")
def train_patients(luad_data):
    train_ids, _, _ = load_split(
        SPLITS_DIR, "tcga_luad_DSS_k3_r1_test0.2_val0.2_seed42.json"
    )
    return load_split_patients(train_ids, luad_data)


def test_load_patient_returns_correct_keys(luad_data):
    patient_id = list(luad_data.keys())[0]
    patient = load_patient(patient_id, luad_data)
    expected_keys = {
        "patient_id",
        "clinical",
        "transcriptomics",
        "wsi",
        "methylation",
        "available_modalities",
        "missing_modalities",
        "label",
    }
    assert set(patient.keys()) == expected_keys


def test_missing_modality_is_none(luad_data):
    # TCGA-05-4244 has avail=[1,1,1,0] so methylation is missing
    patient = load_patient("TCGA-05-4244", luad_data)
    assert patient["methylation"] is None
    assert "methylation" in patient["missing_modalities"]
    assert "methylation" not in patient["available_modalities"]


def test_present_modality_is_array(luad_data):
    patient = load_patient("TCGA-05-4244", luad_data)
    assert isinstance(patient["clinical"], np.ndarray)
    assert isinstance(patient["transcriptomics"], np.ndarray)
    assert isinstance(patient["wsi"], np.ndarray)


def test_modality_shapes(luad_data):
    patient = load_patient("TCGA-05-4244", luad_data)
    assert patient["clinical"].shape == (63,)
    assert patient["transcriptomics"].shape == (1824,)
    assert patient["wsi"].shape == (1024,)


def test_available_modalities_consistent_with_avail(luad_data):
    for patient_id, record in list(luad_data.items())[:20]:
        patient = load_patient(patient_id, luad_data)
        avail = record["avail"]
        modality_keys = ["clinical", "transcriptomics", "wsi", "methylation"]
        for i, key in enumerate(modality_keys):
            if np.isclose(avail[i], 1.0, rtol=1e-09, atol=1e-09):
                assert key in patient["available_modalities"]
                assert patient[key] is not None
            else:
                assert key in patient["missing_modalities"]
                assert patient[key] is None


def test_unknown_patient_returns_none(luad_data):
    assert load_patient("TCGA-FAKE-0000", luad_data) is None


def test_split_sizes(luad_data):
    train_ids, val_ids, test_ids = load_split(
        SPLITS_DIR, "tcga_luad_DSS_k3_r1_test0.2_val0.2_seed42.json"
    )
    assert len(train_ids) == 253
    assert len(val_ids) == 127
    assert len(test_ids) == 95


def test_clinical_always_present(train_patients):
    for patient in train_patients:
        assert "clinical" in patient["available_modalities"]
        assert patient["clinical"] is not None
