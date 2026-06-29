"""Curated source cards for public BashGym data and evaluation resources.

The catalog is deliberately metadata-first. Source cards describe what a source
is safe for before any adapter downloads, converts, or trains on it.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SourceArtifactType(str, Enum):
    SFT_EXAMPLES = "sft_examples"
    DPO_PAIRS = "dpo_pairs"
    REWARD_EXAMPLES = "reward_examples"
    PROCESS_REWARD_EXAMPLES = "process_reward_examples"
    ENVIRONMENT_SPECS = "environment_specs"
    EVAL_MANIFEST = "eval_manifest"
    RAW_CORPUS = "raw_corpus"


class SourceUse(str, Enum):
    SFT = "sft"
    DPO = "dpo"
    REWARD_MODEL = "reward_model"
    PROCESS_REWARD = "process_reward"
    TERMINAL_RL = "terminal_rl"
    EVALUATION = "evaluation"
    RAW_REFERENCE = "raw_reference"


class SourceRisk(str, Enum):
    LICENSE = "license_review"
    EVAL_LEAKAGE = "eval_leakage"
    CONTAMINATION = "contamination"
    PII = "pii_or_sensitive_data"
    GENERATED_LABELS = "generated_or_judged_labels"
    RAW_CORPUS = "large_raw_corpus"
    EXECUTION = "executes_untrusted_code"


TRAINING_USES = {
    SourceUse.SFT,
    SourceUse.DPO,
    SourceUse.REWARD_MODEL,
    SourceUse.PROCESS_REWARD,
    SourceUse.TERMINAL_RL,
}

USE_ARTIFACTS: dict[SourceUse, set[SourceArtifactType]] = {
    SourceUse.SFT: {SourceArtifactType.SFT_EXAMPLES},
    SourceUse.DPO: {SourceArtifactType.DPO_PAIRS},
    SourceUse.REWARD_MODEL: {SourceArtifactType.REWARD_EXAMPLES},
    SourceUse.PROCESS_REWARD: {SourceArtifactType.PROCESS_REWARD_EXAMPLES},
    SourceUse.TERMINAL_RL: {SourceArtifactType.ENVIRONMENT_SPECS},
    SourceUse.EVALUATION: {SourceArtifactType.EVAL_MANIFEST, SourceArtifactType.ENVIRONMENT_SPECS},
    SourceUse.RAW_REFERENCE: {SourceArtifactType.RAW_CORPUS},
}

SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class SourceCard:
    """Machine-readable card for a public training or evaluation source."""

    id: str
    name: str
    homepage: str
    domain: str
    task_family: str
    artifact_types: tuple[SourceArtifactType, ...]
    training_eligible: bool
    eval_only: bool
    license: str
    input_format: str
    adapter: str
    recommended_uses: tuple[SourceUse, ...]
    not_recommended_for: tuple[SourceUse, ...] = ()
    known_risks: tuple[SourceRisk, ...] = ()
    decontam_notes: str = ""
    split_policy: str = ""
    source_quality_notes: str = ""
    repo: str | None = None
    huggingface_id: str | None = None
    data_size: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifact_types"] = [item.value for item in self.artifact_types]
        payload["recommended_uses"] = [item.value for item in self.recommended_uses]
        payload["not_recommended_for"] = [item.value for item in self.not_recommended_for]
        payload["known_risks"] = [item.value for item in self.known_risks]
        return payload

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if not SOURCE_ID_RE.match(self.id):
            errors.append("id must be lowercase kebab/snake style")
        if not self.name.strip():
            errors.append("name is required")
        if not self.homepage.startswith(("https://", "http://")):
            errors.append("homepage must be an http(s) URL")
        if self.repo and not self.repo.startswith(("https://", "http://")):
            errors.append("repo must be an http(s) URL when set")
        if self.eval_only and self.training_eligible:
            errors.append("eval_only sources cannot also be training_eligible")
        if not self.artifact_types:
            errors.append("at least one artifact type is required")
        if not self.recommended_uses:
            errors.append("at least one recommended use is required")
        if self.eval_only and any(use in TRAINING_USES for use in self.recommended_uses):
            errors.append("eval_only sources cannot recommend training uses")
        return errors


CATALOG: tuple[SourceCard, ...] = (
    SourceCard(
        id="harbor_terminal_bench",
        name="Harbor / Terminal-Bench",
        homepage="https://www.harborframework.com/docs/tutorials/running-terminal-bench",
        repo="https://github.com/harbor-framework/harbor",
        domain="terminal_agents",
        task_family="terminal_benchmark",
        artifact_types=(SourceArtifactType.EVAL_MANIFEST, SourceArtifactType.ENVIRONMENT_SPECS),
        training_eligible=False,
        eval_only=True,
        license="See upstream repository",
        data_size="Terminal-Bench datasets exposed through Harbor",
        input_format="Harbor benchmark dataset and run manifest",
        adapter="harbor_terminal_bench",
        recommended_uses=(SourceUse.EVALUATION,),
        not_recommended_for=(SourceUse.SFT, SourceUse.DPO, SourceUse.TERMINAL_RL),
        known_risks=(SourceRisk.EVAL_LEAKAGE, SourceRisk.EXECUTION),
        decontam_notes="Treat benchmark tasks as eval-only unless a separate train split is explicitly documented.",
        split_policy="Group by benchmark task id, domain, and source release.",
        source_quality_notes="Best used as release evidence for terminal-agent behavior.",
    ),
    SourceCard(
        id="swe_bench",
        name="SWE-bench",
        homepage="https://www.swebench.com/",
        repo="https://github.com/swe-bench/SWE-bench",
        huggingface_id="princeton-nlp/SWE-bench",
        domain="software_engineering",
        task_family="repo_issue_resolution",
        artifact_types=(SourceArtifactType.EVAL_MANIFEST, SourceArtifactType.ENVIRONMENT_SPECS),
        training_eligible=False,
        eval_only=True,
        license="MIT",
        data_size="Issue-resolution benchmark splits",
        input_format="GitHub issue/task records and harness manifests",
        adapter="swe_bench",
        recommended_uses=(SourceUse.EVALUATION,),
        not_recommended_for=(SourceUse.SFT, SourceUse.DPO),
        known_risks=(SourceRisk.EVAL_LEAKAGE, SourceRisk.CONTAMINATION, SourceRisk.EXECUTION),
        decontam_notes="Do not train on benchmark test instances; preserve repository and issue ids for contamination checks.",
        split_policy="Group by repository, issue id, and benchmark split.",
        source_quality_notes="Use for external software-engineering claims, not casual data expansion.",
    ),
    SourceCard(
        id="bfcl",
        name="Berkeley Function Calling Leaderboard",
        homepage="https://gorilla.cs.berkeley.edu/leaderboard.html",
        repo="https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard",
        domain="tool_use",
        task_family="function_calling",
        artifact_types=(SourceArtifactType.EVAL_MANIFEST,),
        training_eligible=False,
        eval_only=True,
        license="See upstream repository",
        data_size="Function-calling benchmark cases",
        input_format="Function schemas, prompts, and expected call outputs",
        adapter="bfcl",
        recommended_uses=(SourceUse.EVALUATION,),
        not_recommended_for=(SourceUse.SFT, SourceUse.DPO),
        known_risks=(SourceRisk.EVAL_LEAKAGE,),
        decontam_notes="Keep as eval-only by default; training on benchmark cases weakens tool-use claims.",
        split_policy="Group by function family, category, and benchmark release.",
        source_quality_notes="Useful for argument-level tool-call reliability.",
    ),
    SourceCard(
        id="tau_bench",
        name="tau-bench",
        homepage="https://github.com/sierra-research/tau-bench",
        repo="https://github.com/sierra-research/tau-bench",
        domain="tool_use",
        task_family="business_workflow_agents",
        artifact_types=(SourceArtifactType.EVAL_MANIFEST,),
        training_eligible=False,
        eval_only=True,
        license="See upstream repository",
        data_size="Tool-agent benchmark tasks",
        input_format="Domain workflow tasks, tools, and evaluation state",
        adapter="tau_bench",
        recommended_uses=(SourceUse.EVALUATION,),
        not_recommended_for=(SourceUse.SFT, SourceUse.DPO),
        known_risks=(SourceRisk.EVAL_LEAKAGE,),
        decontam_notes="Use as a heldout business-workflow benchmark unless a train split is separately curated.",
        split_policy="Group by domain, scenario, and task id.",
        source_quality_notes="Good proxy for multi-turn tool-use reliability.",
    ),
    SourceCard(
        id="rewardbench",
        name="RewardBench",
        homepage="https://github.com/allenai/reward-bench",
        repo="https://github.com/allenai/reward-bench",
        domain="reward_modeling",
        task_family="preference_model_eval",
        artifact_types=(SourceArtifactType.EVAL_MANIFEST, SourceArtifactType.REWARD_EXAMPLES),
        training_eligible=False,
        eval_only=True,
        license="See upstream repository",
        data_size="Reward-model benchmark preference pairs",
        input_format="Prompt, chosen response, rejected response, and subset metadata",
        adapter="rewardbench",
        recommended_uses=(SourceUse.EVALUATION,),
        not_recommended_for=(SourceUse.REWARD_MODEL, SourceUse.DPO),
        known_risks=(SourceRisk.EVAL_LEAKAGE, SourceRisk.CONTAMINATION),
        decontam_notes="Keep as reward-model eval evidence; use separate preference datasets for training.",
        split_policy="Group by benchmark subset and prompt id.",
        source_quality_notes="Useful for reward-model sanity checks before using learned rewards.",
    ),
    SourceCard(
        id="cua_rewardbench",
        name="CUARewardBench",
        homepage="https://github.com/Tencent/CUARewardBench",
        repo="https://github.com/Tencent/CUARewardBench",
        domain="computer_use_agents",
        task_family="trajectory_reward_model_eval",
        artifact_types=(SourceArtifactType.EVAL_MANIFEST, SourceArtifactType.REWARD_EXAMPLES),
        training_eligible=False,
        eval_only=True,
        license="See upstream repository",
        data_size="Computer-use reward-model benchmark data",
        input_format="Trajectory or step-level agent preference examples",
        adapter="cua_rewardbench",
        recommended_uses=(SourceUse.EVALUATION,),
        not_recommended_for=(SourceUse.REWARD_MODEL, SourceUse.PROCESS_REWARD),
        known_risks=(SourceRisk.EVAL_LEAKAGE, SourceRisk.CONTAMINATION),
        decontam_notes="Use for CUA reward-model evaluation; do not mix into training splits by default.",
        split_policy="Group by task, trajectory id, and benchmark release.",
        source_quality_notes="Closest public fit for CUA-style ORM/PRM evaluation.",
    ),
    SourceCard(
        id="ultrafeedback_binarized",
        name="UltraFeedback Binarized",
        homepage="https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized",
        huggingface_id="HuggingFaceH4/ultrafeedback_binarized",
        domain="general_alignment",
        task_family="preference_training",
        artifact_types=(SourceArtifactType.DPO_PAIRS, SourceArtifactType.REWARD_EXAMPLES),
        training_eligible=True,
        eval_only=False,
        license="See Hugging Face dataset card",
        data_size="Public binarized preference dataset",
        input_format="Prompt, chosen response, rejected response",
        adapter="hf_preference_pairs",
        recommended_uses=(SourceUse.DPO, SourceUse.REWARD_MODEL),
        not_recommended_for=(SourceUse.TERMINAL_RL,),
        known_risks=(SourceRisk.GENERATED_LABELS, SourceRisk.CONTAMINATION),
        decontam_notes="Keep separate from terminal-agent evals; preserve prompt hashes for pair audits.",
        split_policy="Group by prompt hash and source subset.",
        source_quality_notes="Good seed for generic preference tooling, not enough for terminal-agent skill by itself.",
    ),
    SourceCard(
        id="helpsteer2",
        name="NVIDIA HelpSteer2",
        homepage="https://huggingface.co/datasets/nvidia/HelpSteer2",
        huggingface_id="nvidia/HelpSteer2",
        domain="general_alignment",
        task_family="preference_and_reward_modeling",
        artifact_types=(
            SourceArtifactType.DPO_PAIRS,
            SourceArtifactType.REWARD_EXAMPLES,
            SourceArtifactType.PROCESS_REWARD_EXAMPLES,
        ),
        training_eligible=True,
        eval_only=False,
        license="See Hugging Face dataset card",
        data_size="Public preference/reward dataset",
        input_format="Assistant responses with helpfulness-style annotations",
        adapter="hf_reward_preferences",
        recommended_uses=(SourceUse.REWARD_MODEL, SourceUse.DPO, SourceUse.PROCESS_REWARD),
        not_recommended_for=(SourceUse.TERMINAL_RL,),
        known_risks=(SourceRisk.GENERATED_LABELS, SourceRisk.CONTAMINATION),
        decontam_notes="Use prompt hashes and source ids before mixing with internal preference data.",
        split_policy="Group by prompt id and annotation source.",
        source_quality_notes="Useful for reward-model lane smoke tests before terminal-specific RM data exists.",
    ),
)


def list_sources() -> list[SourceCard]:
    """Return the curated source catalog in display order."""

    return list(CATALOG)


def get_source(source_id: str) -> SourceCard:
    """Return one source card by id."""

    for card in CATALOG:
        if card.id == source_id:
            return card
    raise KeyError(source_id)


def validate_catalog(cards: list[SourceCard] | None = None) -> dict[str, list[str]]:
    """Validate all source cards and return errors keyed by source id."""

    errors: dict[str, list[str]] = {}
    seen: set[str] = set()
    for card in cards or list_sources():
        card_errors = card.validation_errors()
        if card.id in seen:
            card_errors.append("duplicate source id")
        seen.add(card.id)
        if card_errors:
            errors[card.id] = card_errors
    return errors


def _coerce_use(goal: SourceUse | str) -> SourceUse:
    if isinstance(goal, SourceUse):
        return goal
    try:
        return SourceUse(goal)
    except ValueError as exc:
        valid = ", ".join(use.value for use in SourceUse)
        raise ValueError(f"unknown source goal {goal!r}; choose one of {valid}") from exc


def supports_goal(card: SourceCard, goal: SourceUse | str) -> bool:
    """Return whether a source can produce the artifact family for a goal."""

    source_use = _coerce_use(goal)
    return bool(set(card.artifact_types) & USE_ARTIFACTS[source_use])


def validate_source_use(
    card: SourceCard,
    goal: SourceUse | str,
    *,
    allow_eval_only: bool = False,
    override_reason: str | None = None,
) -> dict[str, Any]:
    """Validate a requested source use and return a machine-readable verdict."""

    source_use = _coerce_use(goal)
    blocking: list[str] = []
    warnings: list[str] = []
    if source_use in TRAINING_USES and card.eval_only and not allow_eval_only:
        blocking.append("eval_only_source_for_training")
    if source_use in TRAINING_USES and not card.training_eligible and not allow_eval_only:
        blocking.append("source_not_training_eligible")
    if not supports_goal(card, source_use):
        warnings.append("source_adapter_not_yet_goal_specific")
    if allow_eval_only and card.eval_only and not override_reason:
        warnings.append("eval_only_override_missing_reason")
    if card.known_risks:
        warnings.extend(risk.value for risk in card.known_risks)
    return {
        "ok": not blocking,
        "source_id": card.id,
        "goal": source_use.value,
        "blocking_codes": blocking,
        "warnings": warnings,
        "requires_override_reason": bool(allow_eval_only and card.eval_only and not override_reason),
        "override_reason": override_reason,
    }


def recommend_sources(
    *,
    domain: str | None = None,
    goal: SourceUse | str | None = None,
    include_eval_only: bool = False,
) -> list[dict[str, Any]]:
    """Rank source cards for a domain/goal pair."""

    source_use = _coerce_use(goal) if goal else None
    recommendations: list[dict[str, Any]] = []
    for card in CATALOG:
        if not include_eval_only and card.eval_only and source_use in TRAINING_USES:
            continue
        score = 0
        reasons: list[str] = []
        if domain and card.domain == domain:
            score += 3
            reasons.append("domain match")
        if source_use and supports_goal(card, source_use):
            score += 3
            reasons.append("artifact match")
        if source_use and source_use in card.recommended_uses:
            score += 2
            reasons.append("recommended use")
        if not source_use and not card.eval_only:
            score += 1
            reasons.append("training eligible")
        if score > 0:
            recommendations.append({"score": score, "reasons": reasons, "source": card.to_dict()})
    return sorted(recommendations, key=lambda item: (-item["score"], item["source"]["id"]))


def prepare_source_manifest(
    card: SourceCard,
    *,
    goal: SourceUse | str,
    output_dir: str | Path | None = None,
    allow_eval_only: bool = False,
    override_reason: str | None = None,
) -> dict[str, Any]:
    """Create a source manifest for a downstream adapter or dry-run."""

    verdict = validate_source_use(
        card,
        goal,
        allow_eval_only=allow_eval_only,
        override_reason=override_reason,
    )
    manifest = {
        "schema_version": "bashgym.source_manifest.v1",
        "source": card.to_dict(),
        "goal": _coerce_use(goal).value,
        "use_verdict": verdict,
        "adapter": card.adapter,
        "next_artifacts": [artifact.value for artifact in card.artifact_types],
    }
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        manifest_path = output_path / "source_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)
    return manifest
