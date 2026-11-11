"""team.toml: its fields, "never back" to Unassisted, and nickname clashes (design §3, §9.3).

The collector and the staff final re-run use these same functions."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime

NICKNAME = re.compile(r"[A-Za-z0-9_-]{3,24}")
DIVISIONS = ("unassisted", "assisted")
LANGUAGES = ("cute", "triton", "cuda")
PLACEHOLDER = "change-me"


@dataclass(frozen=True)
class TeamVersion:
    committed_at: str  # UTC, YYYY-MM-DDTHH:MM:SSZ
    text: str


@dataclass(frozen=True)
class Team:
    nickname: str | None
    division: str
    language: str | None
    publish_code: bool
    problems: tuple[str, ...]


def utc(timestamp: str) -> str:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_team(text: str) -> Team:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        return Team(None, "unassisted", None, False, (f"team.toml is not valid TOML: {error}",))
    problems = []
    nickname = data.get("nickname")
    if not isinstance(nickname, str) or not NICKNAME.fullmatch(nickname) or nickname.lower() == PLACEHOLDER:
        problems.append("set a nickname: 3 to 24 letters, digits, - or _")
        nickname = None
    division = data.get("division", "unassisted")
    if division not in DIVISIONS:
        problems.append(f"division must be one of {DIVISIONS}")
        division = "unassisted"
    language = data.get("language")
    if language not in LANGUAGES:
        problems.append(f"language must be one of {LANGUAGES}")
        language = None
    return Team(nickname, division, language, data.get("publish_code") is True, tuple(problems))


def resolve(history: list[TeamVersion]) -> Team:
    """The head version's fields, and Assisted if any version declared it."""
    ordered = sorted(history, key=lambda version: version.committed_at)
    head = parse_team(ordered[-1].text)
    if any(parse_team(version.text).division == "assisted" for version in ordered):
        head = Team(head.nickname, "assisted", head.language, head.publish_code, head.problems)
    return head


def display_names(histories: dict[int, list[TeamVersion]]) -> dict[int, str]:
    """Repository id -> shown name. The earliest first commit with a nickname keeps it. Each later claim gets
    the smallest suffix -k, k >= 2, that no nickname and no earlier suffix takes, compared case-insensitively,
    so no two entries show the same name."""
    claims = []
    for repo_id, history in histories.items():
        nickname = resolve(history).nickname
        if nickname is None:
            continue
        first = min(v.committed_at for v in history if parse_team(v.text).nickname
                    and parse_team(v.text).nickname.lower() == nickname.lower())
        claims.append((nickname.lower(), first, repo_id, nickname))
    names, kept, taken = {}, set(), {key for key, *_ in claims}
    for key, _first, repo_id, nickname in sorted(claims):
        if key not in kept:
            kept.add(key)
            names[repo_id] = nickname
            continue
        k = 2
        while f"{key}-{k}" in taken:
            k += 1
        taken.add(f"{key}-{k}")
        names[repo_id] = f"{nickname}-{k}"
    return names
