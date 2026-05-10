"""
CLI entry point for the multimodal lung cancer orchestrator.

Usage:
    # Mock mode (no GPU, no LLM):
    python -m src.orchestrator.run --patient TCGA-05-4244 --verbose

    # Real mode (vLLM on AI-LAB, set env vars first):
    export LLM_PROVIDER=openai
    export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
    export OPENAI_API_KEY=not-needed
    export OPENAI_BASE_URL=http://localhost:8000/v1
    python -m src.orchestrator.run --patient TCGA-05-4244 --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_loader import load_split
from src.orchestrator.graph import build_graph

DATA_DIR = Path("data/extracted/cache_data")
SPLITS_DIR = DATA_DIR / "splits"

SPLIT_FILES = {
    "luad": "tcga_luad_DSS_k3_r1_test0.2_val0.2_seed42.json",
    "lusc": "tcga_lusc_DSS_k5_r1_test0.2_val0.2_seed42.json",
}


def _get_train_ids() -> list[str]:
    """Load training patient IDs from both cohorts."""
    train_ids = []
    for cohort, split_file in SPLIT_FILES.items():
        split_path = SPLITS_DIR / split_file
        if split_path.exists():
            ids, _, _ = load_split(SPLITS_DIR, split_file)
            train_ids.extend(ids)
    return train_ids


def run_patient(
    patient_id: str | None,
    graph,
    verbose: bool = False,
    user_query: str = "",
) -> dict:
    """Run the orchestrator on a single patient."""
    display_id = patient_id or "<from query>"
    print(f"\n{'=' * 60}")
    print(f"  Patient: {display_id}")
    if user_query:
        print(f"  Query:   '{user_query[:80]}{'...' if len(user_query) > 80 else ''}'")
    print(f"{'=' * 60}")

    initial_state = {
        "user_query": user_query,
        "parsed_query": {},
        "patient_id": patient_id or "",
        "cohort": "",
        "clinical": None,
        "transcriptomics": None,
        "wsi": None,
        "methylation": None,
        "available_modalities": [],
        "missing_modalities": [],
        "agent_summaries": {},
        "mining_rules": {},
        "generated_modalities": {},
        "verification_scores": {},
        "verification_passed": False,
        "survival_prediction": None,
        "risk_class": "",
        "top_shap_features": [],
        "shap_feature_details": [],
        "source_map": {},
        "clinical_report": "",
        "routing_decision": "",
        "execution_log": [],
        "correction_hints": {},
    }

    result = graph.invoke(initial_state)

    if verbose:
        print("\n  Execution trace:")
        for line in result["execution_log"]:
            print(f"    {line}")

    print(f"\n  Cohort             : {(result.get('cohort') or 'unknown').upper()}")
    print(f"  Available          : {result.get('available_modalities', [])}")
    print(f"  Missing            : {result.get('missing_modalities', [])}")
    print(f"  Routing            : {result.get('routing_decision', '')}")
    print(f"  Mining rules       : {list(result.get('mining_rules', {}).keys())}")
    print(f"  Verification scores: {result.get('verification_scores', {})}")
    print(f"  Verification passed: {result.get('verification_passed', False)}")
    print(f"  Survival prediction: {result.get('survival_prediction')}")
    print(f"  Risk class         : {result.get('risk_class', 'unknown')}")
    print(f"  Top SHAP features  : {result.get('top_shap_features', [])}")
    print(f"  Source map         : {result.get('source_map', {})}")

    # Print the final clinical report
    report = result.get("clinical_report", "")
    if report:
        print(f"\n{'=' * 60}")
        print("  CLINICAL REPORT")
        print(f"{'=' * 60}\n")
        print(report)
        print()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run the multimodal lung cancer orchestrator."
    )
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Single patient ID to process (e.g., TCGA-05-4244).",
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=3,
        help="Number of test patients to process (default: 3).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Natural-language query (e.g., 'analyze patient TCGA-05-4244'). "
        "If both --patient and --query are given, --query takes priority.",
    )
    parser.add_argument(
        "--cohort",
        type=str,
        default="luad",
        choices=["luad", "lusc"],
        help="Cohort to use for test patients (default: luad).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="coxnet",
        help="Baseline model for Predictor (default: coxnet).",
    )
    parser.add_argument(
        "--imputation",
        type=str,
        default="mice",
        help="Imputation strategy the baseline was trained with (default: mice).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print full execution trace.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode (no LLM calls).",
    )
    args = parser.parse_args()

    if not args.patient and not args.query:
        parser.error("Specify either --patient TCGA-XX-YYYY or --query '<text>'.")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # Get training IDs for the retrieval pool
    train_ids = None if args.mock else _get_train_ids()

    if train_ids:
        print(f"[INFO] Real mode: pool={len(train_ids)} training patients.")
    else:
        print("[INFO] Mock mode: no LLM calls, placeholder outputs.")

    # Build graph
    graph = build_graph(
        data_dir=DATA_DIR,
        model_name=args.model,
        imputation=args.imputation,
        train_patient_ids=train_ids,
    )

    # Run
    if args.query:
        # Query mode: patient_id will be parsed by the LanguageAgent
        run_patient(
            patient_id=None,
            graph=graph,
            verbose=args.verbose,
            user_query=args.query,
        )
    elif args.patient:
        run_patient(
            patient_id=args.patient,
            graph=graph,
            verbose=args.verbose,
        )
    else:
        split_file = SPLIT_FILES.get(args.cohort)
        if split_file:
            _, _, test_ids = load_split(SPLITS_DIR, split_file)
            for pid in test_ids[: args.n_patients]:
                run_patient(
                    patient_id=pid,
                    graph=graph,
                    verbose=args.verbose,
                )
        else:
            print(f"[ERROR] No split file for cohort: {args.cohort}")


if __name__ == "__main__":
    main()
