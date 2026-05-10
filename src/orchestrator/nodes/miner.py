"""
Miner node for the LangGraph orchestrator (AFM2-aligned, two-stage).

Stage 1 — per-modality understanding agents run in parallel for every
available modality. Each agent is a Python class (not a graph node)
that calls the LLM with a domain-specific prompt and returns an
AgentSummary.

Stage 2 — the Miner collects the summaries and performs cross-modal
reasoning with a single LLM call to generate one mining rule per
missing modality.

Usage in graph.py:
    from src.orchestrator.agents import (
        ClinicalAgent, GenomicAgent, VisualAgent, MethylationAgent,
    )
    from src.orchestrator.nodes.miner import make_miner_node

    agents = {
        "clinical":       ClinicalAgent(llm, metadata),
        "transcriptomics": GenomicAgent(llm, metadata),
        "wsi":            VisualAgent(llm, metadata, pool=pool),
        "methylation":    MethylationAgent(llm, metadata),
    }
    builder.add_node("miner", make_miner_node(llm, agents))
"""

import logging

from src.orchestrator.agents import ModalityAgent, run_agents_parallel
from src.orchestrator.llm import BaseLLMClient
from src.orchestrator.state import PatientState

logger = logging.getLogger(__name__)


MINER_SYSTEM_PROMPT = (
    "You are the Miner agent in a multimodal lung cancer survival "
    "prediction system following the AFM2 framework. Per-modality "
    "understanding agents have analysed the patient's available data "
    "and provided structured summaries. Your role is cross-modal "
    "reasoning: for each missing modality, generate ONE mining rule "
    "that:\n"
    "  1. Identifies the strongest biological links from the available "
    "modalities to the missing one,\n"
    "  2. Specifies which available features should weight neighbour "
    "selection in the downstream k-NN retrieval,\n"
    "  3. Echoes any agent concerns that should temper the "
    "reconstruction (e.g. low confidence on a source modality).\n\n"
    "Be specific and grounded in lung cancer biology. Each rule should "
    "be 2-4 sentences."
)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_mining_prompt(state: PatientState, agent_summaries: dict) -> str:
    """Build the cross-modal reasoning prompt from agent summaries."""
    pid = state["patient_id"]
    cohort = (state.get("cohort") or "unknown").upper()
    missing = state["missing_modalities"]

    if agent_summaries:
        agent_block = "\n\n".join(s.to_prompt_block() for s in agent_summaries.values())
    else:
        agent_block = "(no agent summaries available)"

    return (
        f"Patient {pid} (cohort: {cohort}).\n\n"
        f"AGENT SUMMARIES (available modalities):\n\n{agent_block}\n\n"
        f"MISSING modalities: {', '.join(missing)}.\n\n"
        f"For each missing modality, generate a mining rule. Respond "
        f"ONLY in JSON: "
        f'{{"rules": {{"<modality>": "<rule>", ...}}}}.'
    )


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------


def make_miner_node(
    llm: BaseLLMClient,
    agents: dict[str, ModalityAgent],
    metadata: dict | None = None,
):
    """
    Build the Miner LangGraph node.

    Args:
        llm:      LLM client used for the cross-modal reasoning step.
        agents:   {modality_key: ModalityAgent}. Modalities not present
                  in this dict are silently skipped at the agent stage,
                  but the cross-modal step still runs for whatever
                  summaries are available.
        metadata: Reserved for future hooks; agents own their metadata.
    """

    def miner_node(state: PatientState) -> dict:
        missing = state["missing_modalities"]
        log_lines: list[str] = []

        if not missing:
            log_lines.append("[Miner] No missing modalities. Skipping.")
            return {
                "mining_rules": {},
                "agent_summaries": {},
                "execution_log": log_lines,
            }

        # ---- Stage 1: per-modality understanding agents (parallel) ------
        modality_features = {
            mod: state.get(mod)
            for mod in state["available_modalities"]
            if state.get(mod) is not None
        }
        summaries, timings = run_agents_parallel(agents, modality_features)

        for mod in summaries:
            log_lines.append(
                f"[Miner] Agent '{mod}' completed in "
                f"{timings.get(mod, -1):.2f}s "
                f"(confidence={summaries[mod].confidence})."
            )
        per_agent_sum = sum(v for k, v in timings.items() if k != "_total" and v > 0)
        speedup = (
            per_agent_sum / timings["_total"] if timings.get("_total", 0) > 0 else 0.0
        )
        log_lines.append(
            f"[Miner] Agents wall-clock {timings.get('_total', 0):.2f}s "
            f"vs sequential {per_agent_sum:.2f}s "
            f"(speedup x{speedup:.2f}, n={len(summaries)})."
        )

        # ---- Stage 2: cross-modal reasoning -----------------------------
        prompt = _build_mining_prompt(state, summaries)
        log_lines.append(
            f"[Miner] Cross-modal LLM call. "
            f"Available: {state['available_modalities']}. "
            f"Missing: {missing}."
        )

        try:
            response = llm.invoke_json(prompt, system=MINER_SYSTEM_PROMPT)
            rules = response.get("rules", {}) if isinstance(response, dict) else {}
        except Exception as e:
            log_lines.append(f"[Miner] Cross-modal LLM call failed: {e}")
            rules = {}

        # Ensure a rule exists for every missing modality
        for mod in missing:
            if mod not in rules or not str(rules.get(mod, "")).strip():
                rules[mod] = (
                    f"No specific rule generated for '{mod}'. "
                    f"Use k-NN with uniform weighting as fallback."
                )

        for mod, rule in rules.items():
            log_lines.append(f"[Miner] Rule for '{mod}': {str(rule)[:100]}...")

        return {
            "mining_rules": rules,
            "agent_summaries": summaries,
            "execution_log": log_lines,
        }

    return miner_node


# ---------------------------------------------------------------------------
# Backward-compatible deterministic mock (no LLM, no agents)
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
    """MOCK fallback: deterministic rules without LLM, no agents called."""
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

    return {
        "mining_rules": rules,
        "agent_summaries": {},
        "execution_log": log_lines,
    }
