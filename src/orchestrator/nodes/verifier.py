"""
Verifier nodes for the LangGraph orchestrator (AFM2-aligned).

Two distinct nodes following AFM2:

  pre_verifier  — runs ONCE between Miner and Generator.
                  Reviews raw mining rules, produces refined guidance
                  that conditions the Generator's k-NN retrieval. T=0.

  verifier      — runs after Generator in the self-refinement loop.
                  Scores all N candidates per modality (best-of-N),
                  promotes the winner to generated_modalities, and
                  produces correction_hints for the next retry.
                  T=0, max 3 iterations (enforced by router).

Criteria (AFM2-style, 6-criteria MLLM-as-Judge):
  distributional_plausibility, biological_consistency,
  cross_modal_coherence, clinical_relevance,
  pathway_consistency, hallucination_risk
  — each 0-5; overall = average.
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

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

VERIFIER_PRE_SYSTEM_PROMPT = (
    "You are the pre-generation Verifier in a multimodal lung cancer "
    "survival prediction system following the AFM2 framework. "
    "The Miner agent has produced raw cross-modal reasoning rules. "
    "Your role is to review these rules and produce refined, actionable "
    "guidance for the Generator agent that will perform k-NN retrieval.\n\n"
    "Focus on:\n"
    "  - Biological plausibility of the stated relationships\n"
    "  - Actionable modality weighting instructions for retrieval\n"
    "  - Flagging and correcting any unsupported or vague claims\n\n"
    "Be concise and specific. Your output directly conditions the "
    "Generator's retrieval strategy.\n\n"
    "Respond ONLY in JSON:\n"
    '{"guidance": {"<modality>": "<refined_guidance>", ...}}'
)

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
        expected_dim = MODALITY_DIMS[mod]
        arrays = [
            e["features"][mod]
            for e in pool
            if mod in e["features"] and e["features"][mod].shape == (expected_dim,)
        ]

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
# Per-modality evaluation helpers (shared by pre- and post-verifier)
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
# Pre-generation Verifier (Miner → Verifier-pre → Generator)
# ---------------------------------------------------------------------------


def _build_pre_verification_prompt(
    mining_rules: dict[str, str],
    available_modalities: list[str],
    missing_modalities: list[str],
) -> str:
    rules_block = "\n".join(f"  [{mod}]: {rule}" for mod, rule in mining_rules.items())
    return (
        f"Missing modalities: {', '.join(missing_modalities)}\n"
        f"Available modalities for retrieval: {', '.join(available_modalities)}\n\n"
        f"Raw mining rules from the Miner:\n{rules_block}\n\n"
        f"For each missing modality, produce refined guidance for the Generator.\n"
        f"The guidance should specify:\n"
        f"  1. Which available modalities should carry the most weight for k-NN retrieval\n"
        f"  2. Any biological constraints or red flags in the raw rule\n"
        f"  3. The key biological rationale in 1-2 sentences\n\n"
        f"Respond ONLY in JSON:\n"
        f'{{"guidance": {{"<modality>": "<refined_guidance>", ...}}}}'
    )


def _parse_guidance(
    response: dict, missing_modalities: list[str], mining_rules: dict
) -> dict[str, str]:
    """Extract guidance from LLM response; fall back to raw rules on failure."""
    raw = response.get("guidance", {})
    if not isinstance(raw, dict):
        raw = {}

    guidance = {}
    for mod in missing_modalities:
        val = raw.get(mod, "")
        if isinstance(val, str) and val.strip():
            guidance[mod] = val.strip()
        else:
            # Fall back to raw mining rule so the Generator always has something
            guidance[mod] = mining_rules.get(
                mod, f"Use k-NN with uniform weighting as fallback for '{mod}'."
            )
    return guidance


def make_pre_verifier_node(
    llm: BaseLLMClient,
):
    """
    Returns the pre-generation Verifier node closure.

    Reads mining_rules from state, calls the LLM once (T=0) to produce
    refined guidance per missing modality, and writes guidance to state.
    Also persists both mining_rules and guidance in source_map.
    """

    def pre_verifier_node(state: PatientState) -> dict:
        mining_rules = state.get("mining_rules") or {}
        missing = state.get("missing_modalities") or []
        available = state.get("available_modalities") or []
        log_lines = []

        if not missing:
            log_lines.append("[Verifier-pre] No missing modalities. Skipping.")
            return {"guidance": {}, "execution_log": log_lines}

        prompt = _build_pre_verification_prompt(
            mining_rules=mining_rules,
            available_modalities=available,
            missing_modalities=missing,
        )
        log_lines.append(
            f"[Verifier-pre] Reviewing {len(missing)} mining rules. Missing: {missing}."
        )

        try:
            response = llm.invoke_json(prompt, system=VERIFIER_PRE_SYSTEM_PROMPT)
            guidance = _parse_guidance(response, missing, mining_rules)
        except Exception as e:
            log_lines.append(
                f"[Verifier-pre] LLM call failed: {e}. Falling back to raw rules."
            )
            guidance = {
                mod: mining_rules.get(mod, f"Fallback for '{mod}'.") for mod in missing
            }

        for mod, g in guidance.items():
            log_lines.append(f"[Verifier-pre] Guidance for '{mod}': {g[:100]}...")

        # Persist both raw rules and refined guidance in source_map for auditability
        current_map = dict(state.get("source_map") or {})
        current_map["mining_rules"] = mining_rules
        current_map["guidance"] = guidance

        return {
            "guidance": guidance,
            "source_map": current_map,
            "execution_log": log_lines,
        }

    return pre_verifier_node


# ---------------------------------------------------------------------------
# Post-generation Verifier (best-of-N ranker)
# ---------------------------------------------------------------------------


def _score_candidates(
    modality: str,
    candidates: list,
    pool_stats: dict,
    rules: dict,
    llm: BaseLLMClient,
    threshold: float,
    score_all: bool,
) -> tuple[float, object, str, list[str]]:
    """Evaluate N candidates for one modality; return (best_score, best_arr, best_hint, log_lines)."""
    n = len(candidates)
    eval_list = candidates if score_all else candidates[:1]
    best_score = -1.0
    best_arr = candidates[0]
    best_hint = ""
    log_lines: list[str] = []

    for idx, features in enumerate(eval_list):
        score, hint, mod_logs = _evaluate_modality(
            modality, features, pool_stats, rules, llm, threshold
        )
        log_lines.extend(mod_logs)
        if n > 1 and score_all:
            log_lines.append(
                f"[Verifier] '{modality}' candidate {idx + 1}/{n}: score={score:.1f}."
            )
        if score > best_score:
            best_score = score
            best_arr = features
            best_hint = hint

    return best_score, best_arr, best_hint, log_lines


def make_verifier_node(
    pool_stats: dict[str, dict],
    llm: BaseLLMClient,
    threshold: float = VERIFIER_THRESHOLD,
    score_all_candidates: bool = True,
):
    """
    Returns the post-generation Verifier closure (best-of-N ranker).

    Reads generation_candidates (N arrays per modality) from state.
    For each modality:
      - Scores each candidate with the 6-criteria LLM evaluation
      - Selects the highest-scoring candidate
    Promotes the best candidates to generated_modalities.
    Produces correction_hints for the Generator if best score < threshold.

    Falls back to generated_modalities when generation_candidates is absent
    (backward compatibility with N=1 or mock generator).

    Args:
        score_all_candidates: When True (default), all N candidates are scored
            and the best is selected. When False, only candidate[0] is scored
            (fast path for benchmark runs — equivalent to N=1 cost).
    """

    def verifier_node(state: PatientState) -> dict:
        gen_candidates: dict = state.get("generation_candidates") or {}

        # Backward compat: wrap generated_modalities as single-candidate lists
        if not gen_candidates:
            gen_candidates = {
                mod: [arr]
                for mod, arr in (state.get("generated_modalities") or {}).items()
            }

        # Use guidance if available; fall back to raw mining rules
        rules = state.get("guidance") or state.get("mining_rules") or {}

        scores: dict[str, float] = {}
        best_modalities: dict = {}
        hints: dict[str, str] = {}
        log_lines: list[str] = []

        for modality, candidates in gen_candidates.items():
            if not candidates:
                continue
            best_score, best_arr, best_hint, mod_logs = _score_candidates(
                modality,
                candidates,
                pool_stats,
                rules,
                llm,
                threshold,
                score_all_candidates,
            )
            log_lines.extend(mod_logs)
            scores[modality] = round(best_score, 2)
            best_modalities[modality] = best_arr
            if best_hint:
                hints[modality] = best_hint

        overall_passed = all(s >= threshold for s in scores.values())
        log_lines.append(
            f"[Verifier] Overall: {'PASS' if overall_passed else 'FAIL'}. "
            f"Scores: {scores}"
        )

        return {
            "generated_modalities": best_modalities,
            "verification_scores": scores,
            "verification_passed": overall_passed,
            "correction_hints": hints,
            "execution_log": log_lines,
        }

    return verifier_node


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


def pre_verifier_node(state: PatientState) -> dict:
    """MOCK pre-Verifier: echoes mining rules as guidance unchanged."""
    mining_rules = state.get("mining_rules") or {}
    missing = state.get("missing_modalities") or []
    guidance = {
        mod: mining_rules.get(mod, f"Mock guidance for '{mod}'.") for mod in missing
    }

    current_map = dict(state.get("source_map") or {})
    current_map["mining_rules"] = mining_rules
    current_map["guidance"] = guidance

    return {
        "guidance": guidance,
        "source_map": current_map,
        "execution_log": [f"[Verifier-pre] MOCK: guidance echoed for {missing}."],
    }


def verifier_node(state: PatientState) -> dict:
    """MOCK post-Verifier: random scores 1-5 for each candidate set."""
    import random

    gen_candidates: dict = state.get("generation_candidates") or {}
    if not gen_candidates:
        gen_candidates = {
            mod: [arr] for mod, arr in (state.get("generated_modalities") or {}).items()
        }

    scores: dict[str, float] = {}
    best_modalities: dict = {}
    log_lines: list[str] = []

    for modality, candidates in gen_candidates.items():
        if not candidates:
            continue
        candidate_scores = [round(random.uniform(1.0, 5.0), 1) for _ in candidates]
        best_idx = candidate_scores.index(max(candidate_scores))
        scores[modality] = candidate_scores[best_idx]
        best_modalities[modality] = candidates[best_idx]
        log_lines.append(
            f"[Verifier] MOCK '{modality}' best score: {scores[modality]}/5 "
            f"(candidate {best_idx + 1}/{len(candidates)}, "
            f"threshold={VERIFIER_THRESHOLD})."
        )

    passed = all(s >= VERIFIER_THRESHOLD for s in scores.values())
    log_lines.append(f"[Verifier] Overall: {'PASS' if passed else 'FAIL'}.")

    return {
        "generated_modalities": best_modalities,
        "verification_scores": scores,
        "verification_passed": passed,
        "execution_log": log_lines,
    }
