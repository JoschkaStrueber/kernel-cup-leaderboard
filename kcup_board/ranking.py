"""Final ranks and prizes from the re-run's three round scores per entry (design §10.3)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

PRIZE_PLACES = {"unassisted": 3, "assisted": 1}


@dataclass(frozen=True)
class Entry:
    key: str
    division: str
    rounds: tuple[float, ...]

    @property
    def median(self) -> float:
        return statistics.median(self.rounds)

    @property
    def spread(self) -> float:
        return (max(self.rounds) - min(self.rounds)) / self.median


def groups(entries: list[Entry], division: str) -> list[list[Entry]]:
    ordered = sorted((e for e in entries if e.division == division), key=lambda e: e.median, reverse=True)
    result: list[list[Entry]] = []
    for entry in ordered:
        if result:
            previous = result[-1][-1]
            gap = abs(previous.median - entry.median) / max(previous.median, entry.median)
            if gap == 0 or gap < max(previous.spread, entry.spread):
                result[-1].append(entry)
                continue
        result.append([entry])
    return result


def ranks(entries: list[Entry]) -> dict[str, int]:
    out = {}
    for division in PRIZE_PLACES:
        place = 1
        for group in groups(entries, division):
            for entry in group:
                out[entry.key] = place
            place += len(group)
    return out


def prize_winners(entries: list[Entry]) -> dict[str, list[str]]:
    rank = ranks(entries)
    return {division: [e.key for group in groups(entries, division) for e in group if rank[e.key] <= places]
            for division, places in PRIZE_PLACES.items()}
