"""Pipeline health checks and status helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class PipelineHealth:
    pipeline_name: str
    status: str
    checked_at: str
    details: str


def healthy(pipeline_name: str, details: str = "ok") -> PipelineHealth:
    return PipelineHealth(
        pipeline_name=pipeline_name,
        status="healthy",
        checked_at=datetime.utcnow().isoformat(),
        details=details,
    )


def degraded(pipeline_name: str, details: str) -> PipelineHealth:
    return PipelineHealth(
        pipeline_name=pipeline_name,
        status="degraded",
        checked_at=datetime.utcnow().isoformat(),
        details=details,
    )
