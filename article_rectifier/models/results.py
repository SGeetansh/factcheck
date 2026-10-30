"""Verification, patch, and validation result contracts."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from article_rectifier.models.claims import ErrorType
from article_rectifier.models.edits import SurgicalEdit


class Verdict(StrEnum):
    """Classify source support for a factual claim."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"


class VerificationStatus(StrEnum):
    """Describe whether evidence permits a safe correction."""

    NO_CHANGE = "no_change"
    CORRECTION_AVAILABLE = "correction_available"
    UNRESOLVED = "unresolved"


class VerificationResult(BaseModel):
    """Record a grounded verdict for one fact-check request."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(min_length=1)
    verdict: Verdict
    evidence_ids: tuple[str, ...] = ()
    explanation: str = Field(min_length=1, max_length=500)
    error_types: tuple[ErrorType, ...] = ()
    proposed_factual_correction: str | None = None
    status: VerificationStatus

    @model_validator(mode="after")
    def validate_grounding(self) -> "VerificationResult":
        """Require evidence and correction fields for actionable results.

        Returns:
            The validated result.

        Raises:
            ValueError: If an actionable correction is not grounded.
        """
        if (
            self.verdict in {Verdict.SUPPORTED, Verdict.CONTRADICTED}
            and not self.evidence_ids
        ):
            raise ValueError("Supported or contradicted verdicts require evidence")
        if self.status is VerificationStatus.CORRECTION_AVAILABLE:
            if self.verdict is not Verdict.CONTRADICTED:
                raise ValueError("Only contradicted claims can have a correction")
            if not self.proposed_factual_correction:
                raise ValueError("Available corrections require corrected facts")
        elif self.proposed_factual_correction is not None:
            raise ValueError("Non-actionable verdicts cannot contain corrections")
        if (
            self.verdict is Verdict.SUPPORTED
            and self.status is not VerificationStatus.NO_CHANGE
        ):
            raise ValueError("Supported claims must use no-change status")
        if (
            self.verdict in {Verdict.UNSUPPORTED, Verdict.UNCERTAIN}
            and self.status is not VerificationStatus.UNRESOLVED
        ):
            raise ValueError("Insufficient evidence must remain unresolved")
        return self


class PatchStatus(StrEnum):
    """Describe the deterministic disposition of a proposed edit."""

    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    RETRYABLE = "retryable"
    UNRESOLVED = "unresolved"


class PatchResult(BaseModel):
    """Record the final status and resolved offsets of one patch."""

    model_config = ConfigDict(frozen=True)

    edit: SurgicalEdit
    status: PatchStatus
    reason: str | None = None
    final_start: int | None = Field(default=None, ge=0)
    final_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_disposition(self) -> "PatchResult":
        """Require reasons for failure and offsets for accepted edits.

        Returns:
            The validated patch result.

        Raises:
            ValueError: If required status details are absent.
        """
        if self.status is PatchStatus.ACCEPTED:
            if self.final_start is None or self.final_end is None:
                raise ValueError("Accepted patches require final offsets")
        elif self.status is not PatchStatus.PROPOSED and not self.reason:
            raise ValueError("Non-accepted patch dispositions require a reason")
        return self


class PatchAuditRecord(BaseModel):
    """Provide human-inspectable provenance for one accepted or rejected edit."""

    model_config = ConfigDict(frozen=True)

    patch_id: str
    sentence_id: str
    original_text: str
    replacement_text: str
    error_types: tuple[ErrorType, ...]
    evidence_ids: tuple[str, ...]
    verification_verdict: Verdict
    validation_status: PatchStatus
    reason: str | None = None
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
