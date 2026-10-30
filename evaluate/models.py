"""Typed contracts for independent rectification evaluation."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Availability(StrEnum):
    """Describe whether an evaluation result is reliable and available."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class AnnotationLabel(BaseModel):
    """Describe one development-only annotated factual error."""

    model_config = ConfigDict(frozen=True)

    annotation_id: str
    error_text: str
    correction_text: str
    location: str | None = None
    error_type: str | None = None
    applicable: bool = True


class DetectedIssue(BaseModel):
    """Describe one issue detected by a rectification system."""

    model_config = ConfigDict(frozen=True)

    issue_id: str
    claim_text: str
    proposed_correction: str | None = None
    error_type: str | None = None


class PatchObservation(BaseModel):
    """Describe one proposed patch and its independent review status."""

    model_config = ConfigDict(frozen=True)

    patch_id: str
    original_text: str
    replacement_text: str
    applied: bool = False
    justified: bool | None = None
    evidence_ids: tuple[str, ...] = ()


class UsageObservation(BaseModel):
    """Store observed operational usage without estimating missing values."""

    model_config = ConfigDict(frozen=True)

    llm_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    latency_seconds: float = Field(default=0.0, ge=0.0)
    provider_cost_usd: float | None = Field(default=None, ge=0.0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)


class ScoreResult(BaseModel):
    """Represent a score with its numerator, denominator, and availability."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: float | None = Field(default=None, ge=0.0, le=1.0)
    numerator: int | float | None = Field(default=None, ge=0)
    denominator: int | float | None = Field(default=None, ge=0)
    availability: Availability = Availability.AVAILABLE
    note: str | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> ScoreResult:
        """Require unavailable scores to omit a numeric value.

        Returns:
            The validated score.

        Raises:
            ValueError: If availability and value conflict.
        """
        if self.availability is Availability.UNAVAILABLE and self.value is not None:
            raise ValueError("Unavailable scores cannot contain a value")
        return self


class PatchCounts(BaseModel):
    """Store deterministic patch and issue counts."""

    model_config = ConfigDict(frozen=True)

    proposed: int = Field(default=0, ge=0)
    applied: int = Field(default=0, ge=0)
    rejected: int = Field(default=0, ge=0)
    unresolved_issues: int = Field(default=0, ge=0)
    retrieval_failures: int = Field(default=0, ge=0)


class IntegrityResult(BaseModel):
    """Describe deterministic output-format checks."""

    model_config = ConfigDict(frozen=True)

    valid: bool
    violations: tuple[str, ...] = ()


class FactualityResult(BaseModel):
    """Represent independently evaluated source grounding."""

    model_config = ConfigDict(frozen=True)

    attribution: ScoreResult
    new_errors_introduced: int | None = Field(default=None, ge=0)
    method: str | None = None


class EvaluationReport(BaseModel):
    """Collect deterministic, reference, annotation, and operational metrics."""

    model_config = ConfigDict(frozen=True)

    preservation: ScoreResult
    character_edit_distance: int = Field(ge=0)
    changed_character_ratio: float = Field(ge=0.0)
    human_reference_similarity: ScoreResult
    annotation_detection_recall: ScoreResult
    correction_recall: ScoreResult
    proposed_edit_precision: ScoreResult
    applied_edit_precision: ScoreResult
    factuality: FactualityResult
    patch_counts: PatchCounts
    errors_detected_by_type: dict[str, int]
    integrity: IntegrityResult
    usage: UsageObservation
