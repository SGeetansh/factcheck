"""Constrained surgical-edit contracts."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from article_rectifier.models.claims import ErrorType


class EditOperation(StrEnum):
    """Enumerate supported bounded article edit operations."""

    REPLACE = "replace"
    INSERT = "insert"
    DELETE = "delete"


class SurgicalEdit(BaseModel):
    """Describe one evidence-backed edit within a permitted sentence."""

    model_config = ConfigDict(frozen=True)

    patch_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    sentence_id: str = Field(min_length=1)
    operation: EditOperation = EditOperation.REPLACE
    original_text: str
    replacement_text: str
    insertion_anchor: str | None = None
    insert_after_anchor: bool = False
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    error_type: ErrorType
    explanation: str = Field(min_length=1, max_length=500)
    global_start: int | None = Field(default=None, ge=0)
    global_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_operation(self) -> "SurgicalEdit":
        """Require text fields and offsets to agree with the operation.

        Returns:
            The validated edit.

        Raises:
            ValueError: If operation semantics or offsets are malformed.
        """
        if (self.global_start is None) != (self.global_end is None):
            raise ValueError("Patch offsets must be both set or both omitted")
        if self.operation is EditOperation.INSERT:
            if self.original_text or not self.insertion_anchor:
                raise ValueError("Insertions require an empty span and exact anchor")
        elif self.insertion_anchor is not None:
            raise ValueError("Only insertions can contain an insertion anchor")
        if self.operation is EditOperation.DELETE and self.replacement_text:
            raise ValueError("Deletions must use an empty replacement")
        if self.operation is EditOperation.REPLACE and (
            not self.original_text or not self.replacement_text
        ):
            raise ValueError("Replacements require non-empty old and new text")
        if self.original_text == self.replacement_text:
            raise ValueError("No-op edits are not permitted")
        return self


class PatchProposal(BaseModel):
    """Contain bounded edit proposals for one verified factual issue."""

    model_config = ConfigDict(frozen=True)

    edits: tuple[SurgicalEdit, ...] = ()

    @model_validator(mode="after")
    def validate_unique_patch_ids(self) -> "PatchProposal":
        """Reject duplicate patch identifiers.

        Returns:
            The validated patch proposal.

        Raises:
            ValueError: If patch IDs repeat.
        """
        identifiers = [edit.patch_id for edit in self.edits]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Patch IDs must be unique")
        return self
