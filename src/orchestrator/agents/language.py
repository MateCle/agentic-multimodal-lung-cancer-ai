"""
Language Agent — natural language interface for the orchestrator.

Two responsibilities, used as separate LangGraph nodes:

  1. parse_query (entry node):
     Takes a free-text user query (state.user_query) and extracts a
     structured (patient_id, cohort) tuple. Sets state.parsed_query and
     state.patient_id (so the rest of the graph can proceed normally).
     If parsing fails, sets a short error report and a sentinel that
     the entry router uses to short-circuit straight to END.

  2. generate_report (exit node):
     Synthesises a markdown clinical report from all orchestrator
     outputs (risk score + class, agent summaries, mining rules,
     verification scores, top SHAP features, source map). Sets
     state.clinical_report.

Both methods fall back to deterministic templates when the LLM client
is unavailable, so smoke tests in mock mode still produce a usable
(if blunt) report.

Scope is intentionally minimal: parse_query only extracts patient_id
and cohort; all other content in the query is ignored. Anything beyond
this — multi-patient queries, conditional routing, follow-up
conversations — is out of scope for this iteration and is documented
as such in the Implementation chapter.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.orchestrator.llm import BaseLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------


_PARSE_SYSTEM = (
    "You are the Language Agent of a multimodal lung cancer survival "
    "prediction system. Your role is to extract a TCGA patient "
    "identifier (and optionally a cohort) from a free-text user "
    "query.\n\n"
    "Patient IDs follow the TCGA convention: TCGA-XX-YYYY where XX and "
    "YYYY are alphanumeric (e.g. TCGA-05-4244, TCGA-49-4505). Cohorts "
    "are 'luad' (lung adenocarcinoma) or 'lusc' (lung squamous cell "
    "carcinoma); they may not be specified.\n\n"
    "Ignore any other content in the query — clinical context, "
    "instructions, requests for additional analysis. Extract ONLY the "
    "patient ID and the cohort if present.\n\n"
    "Respond ONLY in JSON:\n"
    '{"patient_id": "<TCGA-XX-YYYY or null>", '
    '"cohort": "<luad|lusc|null>"}'
)


_REPORT_SYSTEM = (
    "You are a clinical oncology assistant generating a structured "
    "patient report for a multimodal lung cancer survival prediction "
    "system. The report must be in markdown with five sections, in "
    "this exact order:\n\n"
    "## Patient Overview\n"
    "## Risk Assessment\n"
    "## Reasoning Chain\n"
    "## Key Prognostic Features\n"
    "## Caveats and Limitations\n\n"
    "Be concise (3-6 lines per section). Use information from the "
    "input strictly — do not invent values. Be honest about "
    "limitations: if a modality was generated rather than measured, "
    "or if the verification did not pass, surface this clearly in "
    "the caveats. The report is meant for a clinician with knowledge "
    "of TCGA terminology and survival modelling. In the Key Prognostic "
    "Features section, list SHAP feature names exactly as provided; do "
    "not infer patient values from them (e.g., do not say 'female' "
    "unless explicitly stated in the inputs). SHAP details contain raw_value, "
    "modality, column_type, and an active flag; only use those fields and do "
    "not invent extra semantics. Do not convert z-scores into absolute units "
    "(years, pack-years); describe only above/below average when needed.\n\n"
    "Output ONLY the markdown report. No preamble, no JSON, no code "
    "fences."
)


_SHAP_MAX_ITEMS = 6
_SHAP_SECTION_HEADER = "## Key Prognostic Features"
_SHAP_NOTE = (
    "SHAP feature names indicate model influence only; they are not patient "
    "attribute values, and inactive indicators can still be influential."
)
_ZSCORE_NOTE = (
    "Clinical continuous values are z-scores (above/below cohort average), "
    "not absolute units."
)
_ACTIVE_MAGNITUDE_THRESHOLD = 0.5
_ACTIVE_NOTE = (
    "Active patient features are defined as binary indicators with value=1 or "
    f"|value| >= {_ACTIVE_MAGNITUDE_THRESHOLD:.1f} for continuous/score features."
)


def _describe_binary_shap_value(value: float) -> str:
    return "present (value=1)" if value > 0.5 else f"not present (value={value:.0f})"


def _describe_continuous_shap_value(value: float) -> str:
    if value > 0:
        direction = "above"
    elif value < 0:
        direction = "below"
    else:
        direction = "at"
    return f"z-score {value:.3g} ({direction} cohort mean)"


def _describe_score_shap_value(value: float) -> str:
    if value > 0:
        direction = "positive"
    elif value < 0:
        direction = "negative"
    else:
        direction = "neutral"
    return f"score {value:.3g} ({direction})"


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ParsedQuery:
    patient_id: str | None
    cohort: str | None
    raw_query: str
    error: str | None = None  # set when parsing fails

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "cohort": self.cohort,
            "raw_query": self.raw_query,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# LanguageAgent class
# ---------------------------------------------------------------------------


class LanguageAgent:
    """
    Entry/exit interface of the orchestrator.

    Stateless across calls — receives the full PatientState in each
    invocation.
    """

    # Regex used as deterministic fallback when LLM is unavailable.
    # Matches TCGA-XX-YYYY where X and Y are alphanumeric.
    _TCGA_PATTERN = re.compile(r"\bTCGA-[A-Z0-9]{2}-[A-Z0-9]{4}\b", re.IGNORECASE)
    _COHORT_PATTERN = re.compile(r"\b(luad|lusc)\b", re.IGNORECASE)

    def __init__(self, llm: BaseLLMClient | None):
        self.llm = llm

    # -- Entry: parse query ---------------------------------------------------

    def parse_query(self, raw_query: str) -> ParsedQuery:
        """
        Extract patient_id and cohort from free-text query.

        First tries the LLM for robust parsing; falls back to regex on
        failure. If neither produces a valid patient_id, returns a
        ParsedQuery with error set.
        """
        if not raw_query or not raw_query.strip():
            return ParsedQuery(
                patient_id=None,
                cohort=None,
                raw_query=raw_query,
                error="Empty query.",
            )

        # Attempt 1: LLM
        if self.llm is not None:
            try:
                response = self.llm.invoke_json(raw_query, system=_PARSE_SYSTEM)
                pid = response.get("patient_id")
                cohort = response.get("cohort")
                if pid and pid.lower() != "null":
                    return ParsedQuery(
                        patient_id=str(pid).upper(),
                        cohort=str(cohort).lower()
                        if cohort and cohort.lower() != "null"
                        else None,
                        raw_query=raw_query,
                    )
            except Exception as e:
                logger.warning(f"[LanguageAgent] LLM parse failed: {e}. Trying regex.")

        # Attempt 2: regex fallback
        pid_match = self._TCGA_PATTERN.search(raw_query)
        cohort_match = self._COHORT_PATTERN.search(raw_query)
        if pid_match:
            return ParsedQuery(
                patient_id=pid_match.group(0).upper(),
                cohort=cohort_match.group(0).lower() if cohort_match else None,
                raw_query=raw_query,
            )

        # Failed
        return ParsedQuery(
            patient_id=None,
            cohort=None,
            raw_query=raw_query,
            error=(
                "Could not extract a TCGA patient identifier from the query. "
                "Please specify a patient ID like 'TCGA-05-4244'."
            ),
        )

    # -- Exit: generate report ------------------------------------------------

    def generate_report(self, state: dict) -> str:
        """
        Synthesise the final clinical report from the full state.

        Returns markdown. Falls back to deterministic template when LLM
        is unavailable or returns invalid output.
        """
        if self.llm is not None:
            try:
                user_prompt = self._build_report_prompt(state)
                response = self.llm.invoke(user_prompt, system=_REPORT_SYSTEM)
                content = response.content if response else ""
                if content and self._looks_like_valid_report(content):
                    return self._postprocess_report(content.strip(), state)
                else:
                    logger.warning(
                        "[LanguageAgent] LLM returned non-report output "
                        "(possibly mock fallback). Using deterministic template."
                    )
            except Exception as e:
                logger.warning(
                    f"[LanguageAgent] LLM report generation failed: {e}. "
                    f"Using template."
                )

        shap_details = state.get("shap_feature_details", []) or []
        return self._postprocess_report(
            self._template_report(state, shap_details), state
        )

    # -- Prompt construction --------------------------------------------------

    def _build_report_prompt(self, state: dict) -> str:
        """Pack the full state into a single, structured prompt for Qwen."""
        pid = state.get("patient_id", "<unknown>")
        cohort = (state.get("cohort") or "unknown").upper()
        available = state.get("available_modalities", [])
        missing = state.get("missing_modalities", [])
        risk_score = state.get("survival_prediction")
        risk_class = state.get("risk_class", "unknown")
        verification_passed = state.get("verification_passed", False)
        verification_scores = state.get("verification_scores", {}) or {}
        agent_summaries = state.get("agent_summaries", {}) or {}
        mining_rules = state.get("mining_rules", {}) or {}
        top_shap = state.get("top_shap_features", []) or []
        shap_details = state.get("shap_feature_details", []) or []
        source_map = state.get("source_map", {}) or {}

        agent_block = self._format_agent_block(agent_summaries)
        mining_block = self._format_mining_block(mining_rules)
        shap_block = self._format_shap_block(top_shap, shap_details)
        shap_active_block = self._format_active_shap_block(shap_details)
        shap_details_block = self._format_shap_details_block(shap_details)
        source_block = self._format_source_block(source_map)

        risk_score_str = (
            f"{risk_score:.4f}" if risk_score is not None else "unavailable"
        )

        return (
            f"Patient ID: {pid}\n"
            f"Cohort: {cohort}\n"
            f"Available modalities: {available}\n"
            f"Missing modalities: {missing}\n\n"
            f"--- PREDICTION ---\n"
            f"DSS risk score: {risk_score_str}\n"
            f"Risk class (training-cohort tertile): {risk_class}\n\n"
            f"--- DATA PROVENANCE ---\n"
            f"{source_block}\n"
            f"Verification passed: {verification_passed}\n"
            f"Verification scores: {verification_scores}\n\n"
            f"--- AGENT SUMMARIES ---\n"
            f"{agent_block}\n\n"
            f"--- MINING RULES (for missing modalities) ---\n"
            f"{mining_block}\n\n"
            f"--- TOP PROGNOSTIC FEATURES (per-patient SHAP) ---\n"
            f"{shap_block}\n\n"
            "NOTE: SHAP entries are feature names only; do not infer patient "
            "attribute values from them. Use only provided value descriptions.\n\n"
            f"--- ACTIVE PATIENT FEATURES (subset) ---\n"
            f"{shap_active_block}\n\n"
            f"--- SHAP FEATURE DETAILS ---\n"
            f"{shap_details_block}\n\n"
            f"Generate the markdown clinical report following the five-section "
            f"structure specified in the system prompt."
        )

    # -- Deterministic template fallback --------------------------------------

    def _template_report(self, state: dict, shap_details: list) -> str:
        """Build a minimal markdown report without LLM."""
        overview = self._build_overview_section(state)
        risk = self._build_risk_section(state)
        reasoning = self._build_reasoning_section(state)
        features = self._format_shap_section(
            state.get("top_shap_features", []) or [], shap_details
        )
        caveats = self._build_caveats_section(state)
        return "\n\n".join([overview, risk, reasoning, features, caveats])

    def _format_shap_section(self, top_shap: list, shap_details: list) -> str:
        if not shap_details and not top_shap:
            return f"{_SHAP_SECTION_HEADER}\nSHAP analysis unavailable for this prediction."

        lines = [_SHAP_SECTION_HEADER, "Top influential features (SHAP):"]
        if shap_details:
            lines.extend(self._format_shap_feature_lines(shap_details))
        else:
            lines.extend(
                [
                    f"{i + 1}. {name} (|importance|={imp:.4f})"
                    for i, (name, imp) in enumerate(top_shap[:_SHAP_MAX_ITEMS])
                ]
            )
        lines.append("Active patient features (subset):")
        lines.extend(self._format_active_feature_lines(shap_details))
        summary = self._format_shap_summary(shap_details)
        if summary:
            lines.append(summary)
        return "\n".join(lines)

    def _postprocess_report(self, report: str, state: dict) -> str:
        """Ensure the SHAP section is deterministic and not value-inferential."""
        shap_section = self._format_shap_section(
            state.get("top_shap_features", []),
            state.get("shap_feature_details", []),
        )
        report = self._inject_section(report, _SHAP_SECTION_HEADER, shap_section)
        caveat_notes = self._collect_caveat_notes(state)
        return self._append_notes_to_section(
            report, "## Caveats and Limitations", caveat_notes
        )

    def _format_shap_details_block(self, shap_details: list) -> str:
        if not shap_details:
            return "(none)"
        lines = []
        for d in shap_details[:_SHAP_MAX_ITEMS]:
            label = self._format_feature_label(d)
            lines.append(
                f"- {label}: {self._describe_shap_detail(d)} | modality={d['modality']} | "
                f"|importance|={d['importance']:.4f}"
            )
        return "\n".join(lines)

    def _collect_caveat_notes(self, state: dict) -> list[str]:
        notes = [_SHAP_NOTE, _ACTIVE_NOTE]
        if state.get("clinical") is not None:
            notes.append(_ZSCORE_NOTE)

        source_map = state.get("source_map", {}) or {}
        generated = []
        for mod, info in source_map.items():
            if info.get("source") == "generated" and not info.get("verified", False):
                score = info.get("verification_score", 0.0)
                generated.append(f"{mod} (score {score:.2f})")
        if generated:
            notes.append(f"Generated modalities not verified: {', '.join(generated)}.")

        zero_filled = [
            mod for mod, info in source_map.items() if info.get("source") == "zero"
        ]
        if zero_filled:
            notes.append(f"Zero-filled modalities: {', '.join(zero_filled)}.")

        return notes

    def _build_overview_section(self, state: dict) -> str:
        pid = state.get("patient_id", "<unknown>")
        cohort = (state.get("cohort") or "unknown").upper()
        available = state.get("available_modalities", [])
        missing = state.get("missing_modalities", [])
        return (
            f"## Patient Overview\n"
            f"- **Patient ID**: {pid}\n"
            f"- **Cohort**: {cohort}\n"
            f"- **Modalities available**: {', '.join(available) if available else 'none'}\n"
            f"- **Modalities missing**: {', '.join(missing) if missing else 'none'}"
        )

    def _build_risk_section(self, state: dict) -> str:
        risk_score = state.get("survival_prediction")
        risk_class = state.get("risk_class", "unknown")
        risk_score_str = (
            f"{risk_score:.4f}" if risk_score is not None else "unavailable"
        )
        return (
            f"## Risk Assessment\n"
            f"- **DSS risk score**: {risk_score_str}\n"
            f"- **Risk class** (relative to training cohort): **{risk_class}**"
        )

    def _build_reasoning_section(self, state: dict) -> str:
        missing = state.get("missing_modalities", [])
        verification_passed = state.get("verification_passed", False)
        if missing:
            return (
                f"## Reasoning Chain\n"
                f"The orchestrator reconstructed {len(missing)} missing "
                f"modalit{'y' if len(missing) == 1 else 'ies'} "
                f"({', '.join(missing)}) using LLM-guided k-NN retrieval over "
                f"a training cohort of available patients. Verification "
                f"{'passed' if verification_passed else 'did not pass'} the 4.0/5 "
                f"clinical-coherence threshold."
            )
        return (
            "## Reasoning Chain\n"
            "All four modalities were available — no reconstruction was "
            "needed. Inference proceeded directly through the baseline "
            "pipeline."
        )

    def _build_caveats_section(self, state: dict) -> str:
        caveat_lines: list[str] = []
        for mod, info in (state.get("source_map", {}) or {}).items():
            if info.get("source") == "generated" and not info.get("verified", False):
                caveat_lines.append(
                    f"- The **{mod}** modality was reconstructed by the "
                    f"system and did not pass clinical verification "
                    f"(score {info.get('verification_score', 0.0):.2f}/5)."
                )
            elif info.get("source") == "zero":
                caveat_lines.append(
                    f"- The **{mod}** modality was zero-filled "
                    f"(reason: {info.get('reason', 'unknown')})."
                )
        if not caveat_lines:
            caveat_lines.append("- No major data-quality concerns flagged.")
        return "## Caveats and Limitations\n" + "\n".join(caveat_lines)

    @staticmethod
    def _humanize_feature_name(name: str) -> str:
        cleaned = name.replace(".", " ").replace("_", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or name

    def _format_feature_label(self, detail: dict) -> str:
        raw = str(detail.get("name", ""))
        # display = self._humanize_feature_name(raw)
        # return f"{display} ({raw})" if display != raw else raw
        return raw

    def _format_shap_feature_lines(self, shap_details: list) -> list[str]:
        lines = []
        for i, detail in enumerate(shap_details[:_SHAP_MAX_ITEMS]):
            label = self._format_feature_label(detail)
            lines.append(
                f"{i + 1}. {label} (|importance|={detail['importance']:.4f}) - "
                f"{self._describe_shap_detail(detail)}"
            )
        return lines

    def _format_active_feature_lines(self, shap_details: list) -> list[str]:
        if not shap_details:
            return ["- (none)"]
        active, _inactive = self._split_shap_details(shap_details)
        if not active:
            return ["- none above the activity threshold"]
        lines = []
        for detail in active[:_SHAP_MAX_ITEMS]:
            label = self._format_feature_label(detail)
            lines.append(f"- {label}: {self._describe_shap_detail(detail)}")
        return lines

    def _format_shap_summary(self, shap_details: list) -> str:
        if not shap_details:
            return ""
        active, inactive = self._split_shap_details(shap_details)
        if not active:
            return "Summary: No SHAP-listed features are active above threshold."
        counts: dict[str, int] = {}
        for detail in active:
            mod = detail.get("modality", "unknown")
            counts[mod] = counts.get(mod, 0) + 1
        parts = [f"{mod}={count}" for mod, count in sorted(counts.items())]
        inactive_note = "" if not inactive else " Inactive indicators may still matter."
        return f"Summary: Active signals in {', '.join(parts)}.{inactive_note}"

    def _split_shap_details(self, shap_details: list) -> tuple[list[dict], list[dict]]:
        active = []
        inactive = []
        for detail in shap_details:
            if self._is_feature_active(detail):
                active.append(detail)
            else:
                inactive.append(detail)
        return active, inactive

    @staticmethod
    def _is_feature_active(detail: dict) -> bool:
        if "active" in detail:
            return bool(detail.get("active"))
        value = detail.get("raw_value")
        if value is None:
            return False
        column_type = detail.get("column_type")
        if column_type in ("binary_01", "binary_m11"):
            return float(value) > 0.5
        try:
            return abs(float(value)) >= _ACTIVE_MAGNITUDE_THRESHOLD
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _describe_shap_detail(detail: dict) -> str:
        value = detail.get("raw_value")
        column_type = detail.get("column_type", "unknown")
        if value is None:
            return "value unavailable"
        if column_type in ("binary_01", "binary_m11"):
            return _describe_binary_shap_value(float(value))
        if column_type == "continuous":
            return _describe_continuous_shap_value(float(value))
        if column_type == "score":
            return _describe_score_shap_value(float(value))
        if column_type == "embedding":
            return f"latent embedding component {float(value):.3g} (no direct clinical interpretation)"
        return f"value {float(value):.3g}"

    @staticmethod
    def _format_agent_block(agent_summaries: dict) -> str:
        if not agent_summaries:
            return "(no per-modality agent summaries - all modalities present)"
        return "\n\n".join(
            f"[{summary.modality.upper()} - confidence: {summary.confidence}] "
            f"{summary.summary}"
            for summary in agent_summaries.values()
        )

    @staticmethod
    def _format_mining_block(mining_rules: dict) -> str:
        if not mining_rules:
            return "(no missing modalities - no rules generated)"
        return "\n".join(f"- {mod}: {rule}" for mod, rule in mining_rules.items())

    def _format_shap_block(self, top_shap: list, shap_details: list) -> str:
        if shap_details:
            return "\n".join(
                f"  {i + 1}. {self._format_feature_label(detail)} "
                f"(|importance|={detail['importance']:.4f}) - {self._describe_shap_detail(detail)}"
                for i, detail in enumerate(shap_details[:_SHAP_MAX_ITEMS])
            )
        if top_shap:
            return "\n".join(
                f"  {i + 1}. {name} (|importance|={imp:.4f})"
                for i, (name, imp) in enumerate(top_shap[:_SHAP_MAX_ITEMS])
            )
        return "(SHAP unavailable)"

    def _format_active_shap_block(self, shap_details: list) -> str:
        if not shap_details:
            return "(none)"
        active, _inactive = self._split_shap_details(shap_details)
        if not active:
            return "(none above threshold)"
        return "\n".join(
            f"- {self._format_feature_label(detail)}: {self._describe_shap_detail(detail)}"
            for detail in active[:_SHAP_MAX_ITEMS]
        )

    @staticmethod
    def _format_source_block(source_map: dict) -> str:
        if not source_map:
            return "(no source info)"
        lines = []
        for mod, info in source_map.items():
            line = f"- {mod}: {info.get('source', 'unknown')}"
            if info.get("source") == "generated":
                line += (
                    f" (verified={info.get('verified', False)}, "
                    f"score={info.get('verification_score', 0.0):.2f})"
                )
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _append_notes_to_section(report: str, header: str, notes: list[str]) -> str:
        if not notes:
            return report

        pattern = rf"{re.escape(header)}\n.*?(?=\n## |\Z)"
        match = re.search(pattern, report, flags=re.S)
        if match:
            existing = match.group(0).strip()
            body = existing[len(header) :].strip()
            lines = [line.strip() for line in body.splitlines() if line.strip()]
            for note in notes:
                if note not in existing:
                    lines.append(f"- {note}")
            replacement = header + "\n" + "\n".join(lines)
            return re.sub(pattern, replacement, report, flags=re.S).strip()

        appended = header + "\n" + "\n".join(f"- {n}" for n in notes)
        return (report.rstrip() + "\n\n" + appended).strip()

    @staticmethod
    def _inject_section(report: str, header: str, replacement: str) -> str:
        pattern = rf"{re.escape(header)}\n.*?(?=\n## |\Z)"
        if re.search(pattern, report, flags=re.S):
            return re.sub(pattern, replacement, report, flags=re.S).strip()
        return (report.rstrip() + "\n\n" + replacement).strip()

    # -- Heuristic checks ------------------------------------------------------
    @staticmethod
    def _looks_like_valid_report(content: str) -> bool:
        """
        Heuristic check: does the LLM output look like a markdown report
        with the five expected sections, rather than a JSON or other
        unrelated payload from a mock or misrouted prompt?
        """
        s = content.strip()
        if not s:
            return False
        # Reject obvious JSON
        if s.startswith("{") or s.startswith("["):
            return False
        # Require at least one of the five expected section headers
        expected_headers = (
            "## Patient Overview",
            "## Risk Assessment",
            "## Reasoning Chain",
            "## Key Prognostic Features",
            "## Caveats",
        )
        return any(h in s for h in expected_headers)
