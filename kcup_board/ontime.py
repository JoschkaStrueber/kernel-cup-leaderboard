"""The on-time head of a branch, copied from scripts/ops/freeze.py (submission_at).

A test in the course repository checks that the two agree (design §9.3)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

ZERO_SHA = "0" * 40


@dataclass(frozen=True)
class Push:
    activity_id: int
    timestamp: str  # YYYY-MM-DDTHH:MM:SSZ, server time
    after: str


def _instant(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def on_time_head(pushes: list[Push], deadline: str) -> str | None:
    cutoff = _instant(deadline)
    on_time = [p for p in pushes if _instant(p.timestamp) <= cutoff]
    if not on_time:
        return None
    current = max(on_time, key=lambda p: (_instant(p.timestamp), p.activity_id))
    return None if current.after == ZERO_SHA else current.after
