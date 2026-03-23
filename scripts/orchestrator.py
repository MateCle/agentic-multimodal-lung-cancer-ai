"""
Smoke-test / acceptance-criteria script for the LangGraph orchestrator.

Run from the project root:
    python scripts/orchestrator.py

Acceptance criteria:
  - Ingests data from the Data Loader.
  - Routes a patient with missing RNA through Miner -> Generator -> Verifier.
  - Reaches END without crashing.
  - Terminal prints show the full execution trace.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.orchestrator.graph import build_graph
from src.orchestrator.state import PatientState
from src.data_loader        import load_raw_data, load_split

DATA_DIR   = Path("data/extracted/cache_data")
SPLITS_DIR = DATA_DIR / "splits"


def _empty_state(patient_id: str) -> PatientState:
    return {
        "patient_id":           patient_id,
        "clinical":             None,
        "transcriptomics":      None,
        "wsi":                  None,
        "methylation":          None,
        "available_modalities": [],
        "missing_modalities":   [],
        "mining_rules":         {},
        "generated_modalities": {},
        "verification_scores":  {},
        "verification_passed":  False,
        "survival_prediction":  None,
        "routing_decision":     "",
        "execution_log":        [],
    }


def run_patient(patient_id: str, graph) -> PatientState:
    print(f"\n{'='*60}")
    print(f"  Patient: {patient_id}")
    print(f"{'='*60}")

    result = graph.invoke(_empty_state(patient_id))

    print("\n  Execution trace:")
    for line in result["execution_log"]:
        print(f"    {line}")

    print(f"\n  Routing decision    : {result['routing_decision']}")
    print(f"  Mining rules        : {list(result['mining_rules'].keys())}")
    print(f"  Verification scores : {result['verification_scores']}")
    print(f"  Verification passed : {result['verification_passed']}")
    print(f"  Survival prediction : {result['survival_prediction']}")
    return result


if __name__ == "__main__":
    graph = build_graph(DATA_DIR)

    # Test 1: three patients from the LUAD test split
    print("\n>>> Test 1: LUAD test-split patients")
    _, _, test_ids = load_split(
        SPLITS_DIR,
        "tcga_luad_DSS_k3_r1_test0.2_val0.2_seed42.json",
    )
    for pid in test_ids[:3]:
        run_patient(pid, graph)

    # Test 2: acceptance criterion — patient with missing RNA
    print("\n>>> Test 2: Acceptance criterion — patient with missing RNA")
    luad_data, _ = load_raw_data(DATA_DIR, "luad")
    missing_rna  = [
        pid for pid, rec in luad_data.items()
        if int(rec["avail"][1]) == 0
    ]

    if missing_rna:
        run_patient(missing_rna[0], graph)
    else:
        print("  [INFO] No LUAD patients with missing RNA — using TCGA-05-4244.")
        run_patient("TCGA-05-4244", graph)
