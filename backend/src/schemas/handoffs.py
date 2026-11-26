from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated

from src.constants.workflow import Decision, HandoffStatus


class PairedDatasetImport(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: Annotated[str, Field(min_length=2, max_length=140)]
    csv: str = Field(min_length=5, max_length=600_000)


class PairedDatasetInfo(BaseModel):
    id: int
    name: str
    checksum: str
    count: int
    strips: int
    instruments: list[str]


class HandoffCreate(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    label: str = Field(min_length=2, max_length=140)
    base_run_id: int = Field(gt=0)
    dataset_ids: list[int] = Field(min_length=1, max_length=20)
    anchor_instrument: str = Field(min_length=1, max_length=64)
    previous_handoff_id: int | None = Field(default=None, gt=0)
    settings: dict[str, float] = Field(default_factory=dict, max_length=10)


class HandoffInfo(BaseModel):
    id: int
    label: str
    base_run_id: int
    dataset_ids: list[int]
    previous_handoff_id: int | None
    anchor_instrument: str
    created_at: str
    status: HandoffStatus
    settings: dict
    decision: str | None
    ready: bool | None
    superseded_by: int | None


class HandoffReviewCreate(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    decision: Decision
    note: str = Field(min_length=4, max_length=1000)


class HandoffReviewInfo(HandoffReviewCreate):
    handoff_id: int
    created_at: str
