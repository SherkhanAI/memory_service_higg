from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# --- /turns ---

class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["user", "assistant", "tool", "system"]
    content: str = ""
    name: str | None = None


class TurnIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(..., min_length=1, max_length=256)
    user_id: str | None = Field(None, max_length=256)
    messages: list[Message] = Field(..., min_length=1)
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnOut(BaseModel):
    id: str


# --- /recall ---

class RecallIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = Field(..., min_length=0, max_length=4096)
    session_id: str = Field(..., min_length=1, max_length=256)
    user_id: str | None = Field(None, max_length=256)
    max_tokens: int = Field(1024, gt=0, le=32_000)


class Citation(BaseModel):
    turn_id: str
    score: float
    snippet: str


class RecallOut(BaseModel):
    context: str
    citations: list[Citation] = Field(default_factory=list)


# --- /search ---

class SearchIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = Field(..., min_length=0, max_length=4096)
    session_id: str | None = Field(None, max_length=256)
    user_id: str | None = Field(None, max_length=256)
    limit: int = Field(10, gt=0, le=100)


class SearchHit(BaseModel):
    content: str
    score: float
    session_id: str
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchOut(BaseModel):
    results: list[SearchHit] = Field(default_factory=list)


# --- /users/{user_id}/memories ---

class MemoryRecord(BaseModel):
    id: str
    type: Literal["fact", "preference", "opinion", "event"]
    key: str
    value: str
    confidence: float
    source_session: str | None = None
    source_turn: str | None = None
    created_at: datetime
    updated_at: datetime
    supersedes: str | None = None
    active: bool


class MemoriesOut(BaseModel):
    memories: list[MemoryRecord] = Field(default_factory=list)
