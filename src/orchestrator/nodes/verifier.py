"""
Verifier node for the LangGraph orchestrator (AFM2-aligned).

Two-step validation following AFM2:
    1. Distributional check — fast, deterministic sanity check
    2. LLM multi-criteria scoring — 6 clinical criteria, each 0-5,
       overall_score as average (matching AFM2's 6-criteria approach)

If overall_score < threshold, produces correction_hints for the Generator.
The self-refinement loop (Generator <-> Verifier) is bounded to max 3 iterations.

Usage in graph.py:
    from src.orchestrator.nodes.verifier import make_verifier_node, build_pool_stats
    stats = build_pool_stats(pool)
    verifier = make_verifier_node(stats, llm)
    builder.add_node("verifier", verifier)
"""

import numpy as np

from src.data_loader import MODALITY_DIMS, MODALITY_KEYS
from src.orchestrator.llm import BaseLLMClient
from src.orchestrator.state import PatientState

# AFM2 uses generation_threshold=4 and max_generation_verification_step=3
VERIFIER_THRESHOLD = 4.0
N_STD = 3.0
MIN_PATIENTS_FOR_STATS = 5

EVALUATION_CRITERIA = [
    "distributional_plausibility",
    "biological_consistency",
    "cross_modal_coherence",
    "clinical_relevance",
    "pathway_consistency",
    "hallucination_risk",
]

VERIFIER_SYSTEM_PROMPT = (
    "You are the Verifier agent in a multimodal lung cancer survival "
    "prediction system, following the AFM2 framework. Your role is to "
    "evaluate the quality of reconstructed modality features using "
    "six clinical evaluation criteria.\n\n"
    "You receive:\n"
    "  - A distributional analysis showing what % of features are in range\n"
    "  - The mining rule that guided the reconstruction\n"
    "  - Statistics about the generated features\n\n"
    "[EVALUATION CRITERIA]\n"
    "Rate each criterion on a scale of 0-5:\n\n"
    "1. distributional_plausibility (0-5)\n"
    "   Are the generated feature values within the expected statistical\n"
    "   range for this cancer type? Score 5 if all values are plausible,\n"
    "   0 if most are out of range.\n\n"
    "2. biological_consistency (0-5)\n"
    "   Do the generated patterns make biological sense for lung cancer?\n"
    "   For transcriptomics: are pathway activation scores consistent\n"
    "   with known LUAD/LUSC biology? For methylation: are CpG patterns\n"
    "   consistent with the cancer subtype?\n\n"
    "3. cross_modal_coherence (0-5)\n"
    "   Are the generated features consistent with the available\n"
    "   modalities? E.g., if clinical data shows late-stage cancer,\n"
    "   do the generated pathway scores reflect high proliferation?\n\n"
    "4. clinical_relevance (0-5)\n"
    "   Are the generated values compatible with the patient's staging,\n"
    "   diagnosis, and demographic information?\n\n"
    "5. pathway_consistency (0-5)\n"
    "   For transcriptomics: do the REACTOME pathway scores form\n"
    "   biologically plausible co-activation patterns? For other\n"
    "   modalities: do feature correlations match expected patterns?\n\n"
    "6. hallucination_risk (0-5)\n"
    "   Are there features with implausible or contradictory values?\n"
    "   Score 5 if no hallucinations detected, 0 if many features\n"
    "   appear fabricated.\n\n"
    "[OVERALL SCORE]\n"
    "Compute overall_score as the average of the six criteria.\n\n"
    "If overall_score < 4, provide specific feedback on what the\n"
    "Generator should change in the next attempt.\n"
)


# ---------------------------------------------------------------------------
# Pool statistics
# ---------------------------------------------------------------------------


def build_pool_stats(pool: list[dict]) -> dict[str, dict]:
    """Compute per-feature mean and std for each modality from the pool."""
    stats = {}
    for mod in MODALITY_KEYS:
        arrays = [e["features"][mod] for e in pool if mod in e["features"]]

        if len(arrays) < MIN_PATIENTS_FOR_STATS:
            stats[mod] = {
                "mean": np.zeros(MODALITY_DIMS[mod]),
                "std": np.ones(MODALITY_DIMS[mod]),
                "n_patients": len(arrays),
                "valid": False,
            }
            continue

        matrix = np.stack(arrays)
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        std = np.where(std < 1e-8, 1e-8, std)

        stats[mod] = {
            "mean": mean,
            "std": std,
            "n_patients": len(arrays),
            "valid": True,
        }

    return stats


# ---------------------------------------------------------------------------
# Distributional check
# ---------------------------------------------------------------------------


def _check_distributional(generated, mod_stats, n_std=N_STD):
    """Return (fraction_in_range, outlier_indices)."""
    if not mod_stats["valid"]:
        return 1.0, []

    lower = mod_stats["mean"] - n_std * mod_stats["std"]
    upper = mod_stats["mean"] + n_std * mod_stats["std"]
    in_range = (generated >= lower) & (generated <= upper)
    return float(np.mean(in_range)), np.nonzero(~in_range)[0].tolist()


def _format_outliers(outlier_indices, generated, mod_stats, max_show=10):
    """Format outlier info for the LLM prompt and correction hints."""
    if not outlier_indices:
        return ""

    mean, std = mod_stats["mean"], mod_stats["std"]
    details = []
    for idx in outlier_indices[:max_show]:
        z = (generated[idx] - mean[idx]) / std[idx] if std[idx] > 1e-8 else 0.0
        details.append(f"feat_{idx}(z={z:+.1f})")

    text = f"{len(outlier_indices)}/{len(generated)} features out of range. "
    text += f"Worst: {', '.join(details)}"
    if len(outlier_indices) > max_show:
        text += f" (+{len(outlier_indices) - max_show} more)"
    return text


# ---------------------------------------------------------------------------
# LLM verification prompt (AFM2 style: 6 criteria, each 0-5)
# ---------------------------------------------------------------------------


def _build_verification_prompt(
    modality: str,
    dist_score: float,
    outlier_text: str,
    mining_rule: str,
    generated_stats: dict,
) -> str:
    """Build prompt for AFM2-style multi-criteria scoring."""
    prompt = (
        f"Evaluate the reconstructed '{modality}' features for a lung cancer patient.\n\n"
        f"DISTRIBUTIONAL CHECK:\n"
        f"  {dist_score:.1%} of features are within the expected range (±3 std).\n"
    )
    if outlier_text:
        prompt += f"  Outliers: {outlier_text}\n"

    prompt += (
        f"\nMINING RULE used for reconstruction:\n"
        f"  {mining_rule}\n\n"
        f"GENERATED FEATURE STATISTICS:\n"
        f"  Mean: {generated_stats['mean']:.4f}\n"
        f"  Std: {generated_stats['std']:.4f}\n"
        f"  Range: [{generated_stats['min']:.4f}, {generated_stats['max']:.4f}]\n"
        f"  Non-zero: {generated_stats['n_nonzero']}/{generated_stats['n_total']}\n\n"
        f"Respond ONLY in JSON:\n"
        f'{{"distributional_plausibility": <0-5>, '
        f'"biological_consistency": <0-5>, '
        f'"cross_modal_coherence": <0-5>, '
        f'"clinical_relevance": <0-5>, '
        f'"pathway_consistency": <0-5>, '
        f'"hallucination_risk": <0-5>, '
        f'"overall_score": <0-5 average of above>, '
        f'"feedback": "specific improvement suggestions if overall_score < 4"}}'
    )
    return prompt


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------


def _parse_criteria_scores(response: dict) -> tuple[float, dict, str]:
    """
    Extract multi-criteria scores from LLM response.
    Returns (overall_score, criteria_dict, feedback).
    """
    criteria = {}
    for c in EVALUATION_CRITERIA:
        val = response.get(c, 3.0)
        try:
            criteria[c] = max(0.0, min(5.0, float(val)))
        except (TypeError, ValueError):
            criteria[c] = 3.0

    # Use LLM's overall if provided, otherwise compute mean
    overall = response.get("overall_score")
    if overall is not None:
        try:
            overall = max(0.0, min(5.0, float(overall)))
        except (TypeError, ValueError):
            overall = sum(criteria.values()) / len(criteria)
    else:
        overall = sum(criteria.values()) / len(criteria)

    feedback = response.get("feedback", "")
    return overall, criteria, feedback


# ---------------------------------------------------------------------------
# Per-modality evaluation helpers
# ---------------------------------------------------------------------------


def _run_distributional_check(arr, mod_stats):
    """Step 1: fast deterministic check."""
    if mod_stats is not None:
        dist_score, outlier_indices = _check_distributional(arr, mod_stats)
    else:
        dist_score, outlier_indices = 1.0, []

    outlier_text = _format_outliers(
        outlier_indices,
        arr,
        mod_stats or {"mean": np.zeros(len(arr)), "std": np.ones(len(arr))},
    )
    return dist_score, outlier_text


def _compute_gen_stats(arr):
    """Compute summary statistics for generated features."""
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n_nonzero": int(np.count_nonzero(arr)),
        "n_total": len(arr),
    }


def _build_correction_hint(outlier_text, criteria, feedback):
    """Build correction hint from failed verification."""
    hint_parts = []
    if outlier_text:
        hint_parts.append(f"Distribution: {outlier_text}")
    worst = sorted(criteria.items(), key=lambda x: x[1])[:2]
    worst_str = ", ".join(f"{c}={v:.1f}" for c, v in worst)
    hint_parts.append(f"Weakest criteria: {worst_str}")
    if feedback:
        hint_parts.append(f"LLM feedback: {feedback}")
    return " | ".join(hint_parts)


def _evaluate_modality(modality, features, pool_stats, rules, llm, threshold):
    """Evaluate a single generated modality. Returns (score, hint, log_lines)."""
    arr = np.array(features).flatten()
    mod_stats = pool_stats.get(modality)
    log_lines = []

    dist_score, outlier_text = _run_distributional_check(arr, mod_stats)
    log_lines.append(
        f"[Verifier] '{modality}' distributional: {dist_score:.1%} in range."
    )

    gen_stats = _compute_gen_stats(arr)
    prompt = _build_verification_prompt(
        modality=modality,
        dist_score=dist_score,
        outlier_text=outlier_text,
        mining_rule=rules.get(modality, "No rule"),
        generated_stats=gen_stats,
    )

    response = llm.invoke_json(prompt, system=VERIFIER_SYSTEM_PROMPT)
    overall, criteria, feedback = _parse_criteria_scores(response)
    passed = overall >= threshold

    criteria_str = ", ".join(f"{c[:12]}={v:.1f}" for c, v in criteria.items())
    log_lines.append(
        f"[Verifier] '{modality}' criteria: [{criteria_str}]. "
        f"Overall: {overall:.1f}/5 (threshold={threshold}). "
        f"{'PASS' if passed else 'FAIL'}."
    )

    hint = ""
    if not passed:
        hint = _build_correction_hint(outlier_text, criteria, feedback)

    return round(overall, 2), hint, log_lines


# ---------------------------------------------------------------------------
# Verifier node factory
# ---------------------------------------------------------------------------


def make_verifier_node(
    pool_stats: dict[str, dict],
    llm: BaseLLMClient,
    threshold: float = VERIFIER_THRESHOLD,
):
    """
    Returns a Verifier closure following AFM2's multi-criteria approach.

    Step 1: Distributional check (fast, deterministic)
    Step 2: LLM scores 6 clinical criteria, each 0-5, overall as average

    Overall pass: overall_score >= threshold.
    On fail: correction_hints contain distributional outliers + LLM feedback.
    """

    def verifier_node(state: PatientState) -> dict:
        generated = state.get("generated_modalities") or {}
        rules = state.get("mining_rules") or {}
        scores = {}
        hints = {}
        log_lines = []

        for modality, features in generated.items():
            score, hint, mod_logs = _evaluate_modality(
                modality, features, pool_stats, rules, llm, threshold
            )
            scores[modality] = score
            log_lines.extend(mod_logs)
            if hint:
                hints[modality] = hint

        overall_passed = all(s >= threshold for s in scores.values())
        log_lines.append(
            f"[Verifier] Overall: {'PASS' if overall_passed else 'FAIL'}. "
            f"Scores: {scores}"
        )

        return {
            "verification_scores": scores,
            "verification_passed": overall_passed,
            "correction_hints": hints,
            "execution_log": log_lines,
        }

    return verifier_node


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------


def verifier_node(state: PatientState) -> dict:
    """MOCK fallback: random scores 1-5."""
    import random

    generated = state.get("generated_modalities") or {}
    scores = {}
    log_lines = []

    for modality in generated:
        score = round(random.uniform(1.0, 5.0), 1)
        scores[modality] = score
        log_lines.append(
            f"[Verifier] MOCK '{modality}' score: {score}/5 "
            f"(threshold={VERIFIER_THRESHOLD})."
        )

    passed = all(s >= VERIFIER_THRESHOLD for s in scores.values())
    log_lines.append(f"[Verifier] Overall: {'PASS' if passed else 'FAIL'}.")

    return {
        "verification_scores": scores,
        "verification_passed": passed,
        "execution_log": log_lines,
    }
