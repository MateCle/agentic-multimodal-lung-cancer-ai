import numpy as np

from src.data_loader import MODALITY_DIMS, MODALITY_KEYS

SURVIVAL_EVENT = "DSS"
SURVIVAL_TIME = "DSS.time"


def apply_zero_imputation(patient: dict) -> np.ndarray:
    """
    Concatenate all modality features using zero-imputation for missing ones.
    Strictly enforces dimension checks to prevent inhomogeneous arrays.
    Returns a single flat feature vector.
    """
    vectors = []
    for modality in MODALITY_KEYS:
        expected_dim = MODALITY_DIMS[modality]
        val = patient.get(modality)

        if val is not None:
            val_arr = np.array(val).flatten()
            if val_arr.size == expected_dim:
                vectors.append(val_arr)
            else:
                vectors.append(np.zeros(expected_dim, dtype=np.float32))
        else:
            vectors.append(np.zeros(expected_dim, dtype=np.float32))

    return np.concatenate(vectors)


def build_structured_dataset(
    patients: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build feature matrix X, structured survival labels y, and completeness mask.
    Filters out patients with missing or invalid survival labels.
    Returns y as a Numpy Structured Array required by scikit-survival.
    """
    X, is_complete = [], []
    y_list = []

    for p in patients:
        label = p["label"]
        event = label.get(SURVIVAL_EVENT)
        time = label.get(SURVIVAL_TIME)

        if event is None or time is None:
            continue
        if np.isnan(event) or np.isnan(time):
            continue
        if time <= 0:
            continue

        X.append(apply_zero_imputation(p))

        # scikit-survival format: (Status as boolean, Time as float)
        y_list.append((bool(event), float(time)))

        missing_modalities = [m for m in MODALITY_KEYS if p.get(m) is None]
        is_complete.append(len(missing_modalities) == 0)

    y = np.array(y_list, dtype=[("Status", "?"), ("Time", "<f8")])
    return np.array(X), y, np.array(is_complete)
