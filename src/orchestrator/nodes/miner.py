"""
Miner node for the LangGraph orchestrator (AFM2-aligned).

Follows AFM2's Paradigm 3: the Miner calls the LLM directly with
the available modality data to generate mining rules for reconstruction.
No modality sub-agents — the LLM receives feature statistics and
metadata names and reasons about cross-modal relationships.

Usage in graph.py:
    from src.orchestrator.nodes.miner import make_miner_node
    miner = make_miner_node(llm, metadata)
    builder.add_node("miner", miner)
"""

import logging

import numpy as np

from src.orchestrator.llm import BaseLLMClient
from src.orchestrator.state import PatientState

logger = logging.getLogger(__name__)

MINER_SYSTEM_PROMPT = (
    "You are the Miner agent in a multimodal lung cancer survival prediction "
    "system, inspired by the AFM2 framework. Your role is to analyze available "
    "patient data and generate specific mining rules for reconstructing missing "
    "modalities.\n\n"
    "The data modalities are:\n"
    "  - clinical: 63 features (demographics, staging, diagnosis, treatment)\n"
    "  - transcriptomics: 1824 REACTOME pathway activity scores\n"
    "  - wsi: 1024-dim slide-level histopathology embedding\n"
    "  - methylation: 16166 CpG probe / SNP values\n\n"
    "Generate rules that specify:\n"
    "  1. Which features from available modalities are most informative\n"
    "  2. What biological relationships connect available to missing data\n"
    "  3. How to prioritize similar patients for k-NN retrieval\n\n"
    "Be specific and grounded in lung cancer biology."
)


# ---------------------------------------------------------------------------
# Feature statistics for prompt construction
# ---------------------------------------------------------------------------


def _compute_modality_stats(
    features: np.ndarray,
    modality: str,
    metadata: dict | None = None,
) -> str:
    """
    Build a human-readable summary of a modality's features
    for inclusion in the LLM prompt.
    """
    arr = np.array(features).flatten()
    stats_lines = [
        f"  Dimensions: {len(arr)}",
        f"  Non-zero: {np.count_nonzero(arr)}/{len(arr)}",
        f"  Range: [{arr.min():.3f}, {arr.max():.3f}]",
        f"  Mean: {arr.mean():.3f}, Std: {arr.std():.3f}",
    ]

    # Add feature names from metadata where available
    if metadata is not None:
        col_key = f"{modality}_columns"
        columns = metadata.get(col_key, [])

        if modality == "clinical" and columns:
            # Show active (non-zero) clinical features by name
            active = [
                columns[i] for i in range(len(arr)) if i < len(columns) and arr[i] != 0
            ]
            if active:
                stats_lines.append(f"  Active features: {', '.join(active[:15])}")
                if len(active) > 15:
                    stats_lines.append(f"    (and {len(active) - 15} more)")

        elif modality == "transcriptomics" and columns:
            # Show top 5 most active pathways by name
            top_idx = np.argsort(np.abs(arr))[-5:][::-1]
            top_named = [
                f"{columns[i]}={arr[i]:.2f}" for i in top_idx if i < len(columns)
            ]
            stats_lines.append(f"  Top pathways: {', '.join(top_named)}")

        elif modality == "wsi":
            stats_lines.append(f"  L2 norm: {np.linalg.norm(arr):.3f}")
            stats_lines.append(
                f"  Sparsity: {1 - np.count_nonzero(arr) / len(arr):.1%}"
            )

    return "\n".join(stats_lines)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_mining_prompt(
    state: PatientState,
    metadata: dict | None,
) -> str:
    """Build the prompt for the Miner LLM call."""
    available = state["available_modalities"]
    missing = state["missing_modalities"]

    # Summarize each available modality
    available_block = []
    for mod in available:
        features = state.get(mod)
        if features is None:
            continue
        summary = _compute_modality_stats(features, mod, metadata)
        available_block.append(f"[{mod.upper()}]\n{summary}")

    available_text = "\n\n".join(available_block) or "(no data available)"
    missing_text = ", ".join(missing)

    prompt = (
        f"Patient {state['patient_id']} (cohort: {state.get('cohort', 'unknown').upper()}).\n\n"
        f"AVAILABLE modalities:\n\n{available_text}\n\n"
        f"MISSING modalities: {missing_text}\n\n"
        f"For each missing modality, generate a mining rule that guides "
        f"k-NN retrieval-based reconstruction. The rule should specify "
        f"which available features to weight higher when computing "
        f"patient similarity, and what biological relationships to exploit.\n\n"
        f'Respond ONLY in JSON: {{"rules": {{"modality_name": "rule text", ...}}}}'
    )

    return prompt


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------


def make_miner_node(
    llm: BaseLLMClient,
    metadata: dict | None = None,
):
    """
    Returns a Miner closure following AFM2's direct LLM approach.

    The Miner:
        1. Computes statistics of each available modality
        2. Includes feature names from metadata in the prompt
        3. Calls the LLM to generate mining rules
        4. Returns one rule per missing modality

    Args:
        llm:      LLM client (Qwen via vLLM, OpenAI, or mock).
        metadata: Dict from metadata .pkl with *_columns keys.
    """

    def miner_node(state: PatientState) -> dict:
        missing = state["missing_modalities"]
        log_lines = []

        if not missing:
            log_lines.append("[Miner] No missing modalities. Skipping.")
            return {"mining_rules": {}, "execution_log": log_lines}

        # Build and send prompt
        prompt = _build_mining_prompt(state, metadata)

        log_lines.append(
            f"[Miner] Calling LLM for mining rules. "
            f"Available: {state['available_modalities']}, "
            f"Missing: {missing}."
        )

        response = llm.invoke_json(prompt, system=MINER_SYSTEM_PROMPT)
        rules = response.get("rules", {})

        # Ensure a rule exists for every missing modality
        for mod in missing:
            if mod not in rules:
                rules[mod] = (
                    f"No specific rule generated for '{mod}'. "
                    f"Use k-NN with uniform weighting as fallback."
                )

        for mod, rule in rules.items():
            log_lines.append(f"[Miner] Rule for '{mod}': {str(rule)[:100]}...")

        return {"mining_rules": rules, "execution_log": log_lines}

    return miner_node


# ---------------------------------------------------------------------------
# Backward-compatible mock
# ---------------------------------------------------------------------------

_MOCK_RULES: dict[str, str] = {
    "clinical->transcriptomics": (
        "Use age, smoking history, and disease subtype (LUAD/LUSC) "
        "as proxies for gene expression pathway activity."
    ),
    "clinical->wsi": (
        "Infer gross tissue morphology features from staging and "
        "histological subtype fields in the clinical record."
    ),
    "clinical->methylation": (
        "Use age-at-diagnosis and smoking pack-years as signals for "
        "epigenetic clock and tobacco-related methylation patterns."
    ),
    "wsi->transcriptomics": (
        "Extract tile-level spatial features and map tumour cellularity "
        "regions to known expression signatures (e.g. EMT, proliferation)."
    ),
    "wsi->methylation": (
        "Use morphological heterogeneity scores from WSI as a proxy for "
        "copy-number-driven methylation changes."
    ),
    "transcriptomics->methylation": (
        "Leverage pathway-level expression scores (REACTOME) to infer "
        "promoter methylation status via known co-regulation patterns."
    ),
}
_FALLBACK_RULE = (
    "No specific rule available for this modality pair. "
    "Use zero-imputation as a conservative baseline."
)


def miner_node(state: PatientState) -> dict:
    """MOCK fallback: deterministic rules without LLM."""
    available = state["available_modalities"]
    missing = state["missing_modalities"]

    rules: dict[str, str] = {}
    for target in missing:
        rule = _FALLBACK_RULE
        for source in available:
            key = f"{source}->{target}"
            if key in _MOCK_RULES:
                rule = _MOCK_RULES[key]
                break
        rules[target] = rule

    log_lines = [f"[Miner] MOCK rules for: {missing}."]
    for modality, rule in rules.items():
        log_lines.append(f"[Miner] Rule for '{modality}': {rule[:80]}...")

    return {"mining_rules": rules, "execution_log": log_lines}
