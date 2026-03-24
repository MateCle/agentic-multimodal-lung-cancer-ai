"""
Miner node for the LangGraph orchestrator.
Based on AFM2 (Ke et al.): generates mining rules that guide the Generator
on how to reconstruct each missing modality from the available ones.

MOCK constraint: no LLM calls. Returns hardcoded domain rules per modality pair.
In production, replace _mock_rules() with a GPT-4o or DeepSeek R1 call.
"""
from src.orchestrator.state import PatientState

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


def _mock_rules(available: list[str], missing: list[str]) -> dict[str, str]:
    rules: dict[str, str] = {}
    for target in missing:
        rule = _FALLBACK_RULE
        for source in available:
            key = f"{source}->{target}"
            if key in _MOCK_RULES:
                rule = _MOCK_RULES[key]
                break
        rules[target] = rule
    return rules


def miner_node(state: PatientState) -> dict:
    """
    Generates one mining rule per missing modality based on the available ones.

    In AFM2 this node calls a reasoning LLM to infer how signals in the
    available modalities can guide the reconstruction of the missing ones.
    Here we return deterministic mock rules — same interface, no API call.
    """
    available = state["available_modalities"]
    missing   = state["missing_modalities"]
    rules     = _mock_rules(available, missing)

    log_lines = [f"[Miner] Generating rules for: {missing}."]
    for modality, rule in rules.items():
        log_lines.append(f"[Miner] Rule for '{modality}': {rule[:80]}...")

    return {"mining_rules": rules, "execution_log": log_lines}
