"""The public leaderboard: leaderboard.json and one static page (design §9.5, §9.6).

board() is the only door to the public: it copies an allow-list of fields out of the collected
entries, so nothing else in a results JSON (node, host, an entry's SLURM job, file paths) gets out.
The reference rows (baseline()) do show the staff run's SLURM job ID, and nothing else of its environment. render()
reads board()'s output only and escapes every name. The page is HTML with inline CSS and SVG,
no scripts."""

from __future__ import annotations

import html
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path

from kcup_board import ranking

PUBLIC_FIELDS = ("name", "division", "language", "status", "frozen", "reasons", "score", "sol_percent",
                 "kernels_per_call", "submissions", "last_measured", "heatmap", "history", "rank", "prize")
DIVISIONS = {"unassisted": "Unassisted", "assisted": "Assisted"}
RANKED = ("provisional", "stale")
TOP = 5
BANNERS = {
    "open": "Provisional: self-measured on UC3. The final ranking comes from the staff re-run.",
    "freezing": "Freezing: the board shows the on-time heads from ten minutes after the freeze.",
    "frozen": "Frozen. The final ranking comes from the staff re-run.",
    "final": "Final ranking",
}
LANGUAGES = {"cute": "CuTe DSL", "triton": "Triton", "cuda": "CUDA C++"}
REASONS = {"no-result": "no eligible result", "harness-modified": "harness modified", "census": "census failed",
           "implausible": "above 105% of speed of light", "team-toml": "team.toml needs a fix",
           "writeup": "no write-up", "not-in-final": "not in the final ranking"}


# -- leaderboard.json ---------------------------------------------------------------------------


def _geomean(values: list[float]) -> float | None:
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def _comparator_score(results: dict, name: str) -> float | None:
    """A comparator's score in the entry's own job: the geometric mean of its speedups. A shape where it
    raised holds {"error": ...} instead of timings and is left out."""
    return _geomean([shape["impls"]["reference"]["median_s"] / shape["impls"][name]["median_s"]
                     for shape in results["shapes"] if "median_s" in shape["impls"].get(name, {})])


def baseline(name: str, impl: str, language: str, link: str | None, run: dict) -> dict:
    """A reference row (design §9.5): impl in the staff run of the starter, scored like an entry. Within a
    shape, % of speed of light is the ideal bytes over the time (the kit's score.sol_percent), so impl's is
    the submission's times t_submission / t_impl. The census covers the submission only, so only its row has
    kernels per call. Raises ValueError when a % of speed of light rounds to 0."""
    sols = [shape["sol_percent"] * shape["impls"]["submission"]["median_s"] / shape["impls"][impl]["median_s"]
            for shape in run["shapes"]]
    kernels = max(run["census"]["kernels_per_call"].values(), default=None) if impl == "submission" else None
    return {"name": name, "language": language, "link": link, "score": _comparator_score(run, impl),
            "sol_percent": _geomean(sols), "kernels_per_call": kernels, "last_measured": run["finished_at"],
            "job": run["environment"]["slurm_job_id"]}


def _public(entry: dict) -> dict:
    results = entry["results"]
    row = {field: entry.get(field) for field in ("name", "division", "language", "status", "frozen")}
    row.update(reasons=list(entry["reasons"]), submissions=len(entry["history"]),
               history=[[finished_at, score] for finished_at, score in entry["history"]], rank=None, prize=None,
               score=None, sol_percent=None, kernels_per_call=None, last_measured=None, heatmap=None)
    if results is not None:
        row.update(score=results["summary"]["score"], sol_percent=results["summary"]["sol_percent"],
                   kernels_per_call=max(results["census"]["kernels_per_call"].values(), default=None),
                   last_measured=results["finished_at"],
                   heatmap={f"{shape['n']},{shape['v']}": shape["sol_percent"] for shape in results["shapes"]})
    return row


def _apply_final(rows: dict[str, dict], final: dict) -> None:
    """final.json's ranks, prizes and re-run numbers replace the self-measured ones. An entry that is
    only in final.json (its branch could not be read today) still gets its row. An entry missing from
    final.json keeps only its history: its self-measured numbers would read as final ones."""
    for name, decided in final["entries"].items():
        row = rows.setdefault(name, {**dict.fromkeys(PUBLIC_FIELDS), "name": name, "frozen": True, "reasons": [],
                                     "submissions": 0, "history": []})
        row.update({field: decided[field] for field in ("division", "rank", "prize", "score", "sol_percent",
                                                         "heatmap")})
        row["status"] = "final" if decided["rank"] is not None else "unranked"
    for name in rows.keys() - final["entries"].keys():
        rows[name].update(status="unranked", reasons=["not-in-final"], rank=None, prize=False, score=None,
                          sol_percent=None, heatmap=None)


def _order(row: dict) -> tuple:
    return list(DIVISIONS).index(row["division"]), row["rank"] is None, row["rank"] or 0, row["name"].lower()


def board(entries: list[dict], *, now: str, final: dict | None, phase: str, baselines: list[dict] = ()) -> dict:
    """The allow-listed leaderboard.json, keyed by the shown name. Provisional ranks go by score within
    each division (identical scores tie, as in ranking.py). Final ranks come from final.json. The
    reference rows (collect.baselines()) are kept apart from the entries: never ranked, tied or counted."""
    named = [entry for entry in entries if entry["name"] is not None]
    nameless = [entry for entry in entries if entry["name"] is None]
    rows = {entry["name"]: _public(entry) for entry in named}
    ranked = [entry for entry in named if entry["status"] in RANKED and entry["results"] is not None]
    if final is None:
        places = ranking.ranks([ranking.Entry(e["name"], e["division"], (e["results"]["summary"]["score"],))
                                for e in ranked])
        for name, place in places.items():
            rows[name]["rank"] = place
    else:
        _apply_final(rows, final)
    reference = {}
    for name in ("starter", "quack"):
        scores = [s for s in (_comparator_score(e["results"], name) for e in ranked) if s is not None]
        reference[name] = statistics.median(scores) if scores else None
    unreadable = sum("collector-error" in entry["reasons"] for entry in nameless)
    return {"updated_at": now, "phase": phase, "reference": reference,
            "baselines": [dict(row) for row in baselines],
            "without_name": {"no_nickname": len(nameless) - unreadable, "unreadable": unreadable},
            "entries": {row["name"]: row for row in sorted(rows.values(), key=_order)}}


def write_site(site: Path, board: dict, prefix: str) -> None:
    """Raises ValueError, and writes nothing, if the board holds NaN or an infinity: JSON has neither."""
    data = json.dumps(board, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    index = render(board)
    site.mkdir(parents=True, exist_ok=True)
    (site / f"{prefix}index.html").write_text(index, encoding="utf-8")
    (site / f"{prefix}leaderboard.json").write_text(data, encoding="utf-8")


# -- formatting ---------------------------------------------------------------------------------


def _instant(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _when(text: str | None) -> str:
    return "–" if text is None else f"{_instant(text):%Y-%m-%d %H:%M} UTC"


def _score(value: float | None) -> str:
    return "–" if value is None else f"{value:.2f}×"


def _percent(value: float | None) -> str:
    return "–" if value is None else f"{value:.1f}%"


def _entries(count: int) -> str:
    return f"{count} entry" if count == 1 else f"{count} entries"


def _status(row: dict) -> str:
    """The banner says when the board is frozen, so the rows do not repeat it."""
    if row["status"] == "final":
        text = "Final, prize" if row["prize"] else "Final"
    elif row["status"] == "stale":
        text = "Stale: the kernel changed after this measurement"
    elif row["status"] == "unranked" and row["reasons"]:
        text = "Unranked: " + ", ".join(REASONS.get(reason, reason) for reason in row["reasons"])
    else:
        text = row["status"].capitalize()
    return html.escape(text)


# -- SVG ----------------------------------------------------------------------------------------


def _nice(top: float) -> tuple[float, float]:
    """A round axis maximum at or above top, and its step: about four ticks."""
    raw = top / 4
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    return math.ceil(top / step - 1e-9) * step, step


def _time_ticks(t0: float, t1: float) -> list[tuple[float, str]]:
    hour = 3600
    step = next((h * hour for h in (1, 2, 3, 6, 12, 24, 48, 96, 168) if (t1 - t0) / (h * hour) <= 4), 336 * hour)
    ticks = []
    t = math.ceil(t0 / step) * step
    while t <= t1:
        moment = datetime.fromtimestamp(t, UTC)
        label = f"{moment:%b} {moment.day}" if step >= 24 * hour else f"{moment:%b} {moment.day}, {moment:%H}:00"
        ticks.append((t, label))
        t += step
    return ticks


def _line_chart(series: list[tuple[str, int, list[tuple[float, float]]]], refs: list[tuple[str, float]], *,
                t0: float, t1: float, top: float, label: str) -> str:
    """series: (name, colour slot, points oldest first as (epoch seconds, score)). refs: horizontal
    dashed lines, labelled in the right margin."""
    width, height, left, upper, lower = 400, 240, 40, 12, 28
    right = 84 if refs else 14
    y_max, y_step = _nice(top)

    def x(t):
        return left + (t - t0) / (t1 - t0) * (width - left - right)

    def y(v):
        return upper + (1 - v / y_max) * (height - upper - lower)

    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label)}">']
    for k in range(int(round(y_max / y_step)) + 1):
        v = k * y_step
        out.append(f'<line class="{"axis" if k == 0 else "grid"}" x1="{left}" x2="{width - right}" '
                   f'y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text class="tick" x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:g}×</text>')
    for t, text in _time_ticks(t0, t1):
        out.append(f'<text class="tick" x="{x(t):.1f}" y="{height - 8}" text-anchor="middle">{text}</text>')
    for text, v in refs:
        out.append(f'<line class="ref" x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text class="tick" x="{width - right + 6}" y="{y(v) + 4:.1f}">{html.escape(text)} {v:.1f}×</text>')
    for name, slot, points in series:
        coords = [(x(t), y(v)) for t, v in points]
        if len(coords) > 1:
            path = " ".join(f"{'M' if k == 0 else 'L'}{px:.1f},{py:.1f}" for k, (px, py) in enumerate(coords))
            out.append(f'<path class="line s{slot}" d="{path}"/>')
        for (px, py), (t, v) in zip(coords, points):
            when = f"{datetime.fromtimestamp(t, UTC):%Y-%m-%d %H:%M} UTC"
            out.append(f'<g class="point s{slot}"><title>{html.escape(name)}, {when}: {_score(v)}</title>'
                       f'<circle class="hit" cx="{px:.1f}" cy="{py:.1f}" r="12"/>'
                       f'<circle class="dot" cx="{px:.1f}" cy="{py:.1f}" r="4"/></g>')
    out.append("</svg>")
    return "".join(out)


def _points(row: dict) -> list[tuple[float, float]]:
    return [(_instant(when).timestamp(), score) for when, score in reversed(row["history"])]


def _domain(rows: list[dict], now: str) -> tuple[float, float]:
    times = [t for row in rows for t, _ in _points(row)]
    t1 = max([_instant(now).timestamp(), *times])
    t0 = min(times, default=t1)
    return (t0, t1) if t1 - t0 >= 3600 else (t1 - 3600, t1)


HEAT_BINS = 7  # 0-15, 15-30, ..., 90 and more: one blue ramp, light to dark


def _heatmap(name: str, heatmap: dict | None) -> str:
    if not heatmap:
        return '<p class="note">No heatmap for this entry.</p>'
    cells = {tuple(int(part) for part in key.split(",")): value for key, value in heatmap.items()}
    ns, vs = sorted({n for n, _ in cells}), sorted({v for _, v in cells})
    left, upper, cell_w, cell_h, gap, key_w = 52, 22, 72, 36, 2, 30
    width = max(left + len(vs) * cell_w, left + HEAT_BINS * key_w) + 4  # the grid or the key below it
    grid_h = upper + len(ns) * cell_h
    height = grid_h + 40
    out = [f'<svg class="heat" viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="{html.escape(name)}: % of speed of light per shape">']
    for j, v in enumerate(vs):
        out.append(f'<text class="tick" x="{left + (j + 0.5) * cell_w:.0f}" y="{upper - 8}" '
                   f'text-anchor="middle">{v}</text>')
    for i, n in enumerate(ns):
        out.append(f'<text class="tick" x="{left - 8}" y="{upper + (i + 0.5) * cell_h + 4:.0f}" '
                   f'text-anchor="end">{n}</text>')
        for j, v in enumerate(vs):
            px, py = left + j * cell_w + gap / 2, upper + i * cell_h + gap / 2
            value = cells.get((n, v))
            if value is None:
                out.append(f'<rect class="empty" x="{px}" y="{py}" width="{cell_w - gap}" height="{cell_h - gap}" '
                           'rx="4"/>')
                continue
            bin_ = min(int(value // 15), HEAT_BINS - 1)
            out.append(f'<g class="cell h{bin_}"><title>N = {n}, V = {v}: {value:.1f}%</title>'
                       f'<rect x="{px}" y="{py}" width="{cell_w - gap}" height="{cell_h - gap}" rx="4"/>'
                       f'<text x="{px + (cell_w - gap) / 2:.0f}" y="{py + cell_h / 2 + 3:.0f}" '
                       f'text-anchor="middle">{value:.0f}%</text></g>')
    for k in range(HEAT_BINS):
        px = left + k * key_w
        out.append(f'<g class="cell h{k}"><rect x="{px}" y="{grid_h + 12}" width="{key_w - gap}" height="10" '
                   f'rx="2"/></g><text class="tick" x="{px}" y="{grid_h + 36}">'
                   f'{15 * k}{"+" if k == HEAT_BINS - 1 else ""}</text>')
    out.append("</svg>")
    return "".join(out)


# -- the page -----------------------------------------------------------------------------------


def _baseline_row(row: dict) -> str:
    name = html.escape(row["name"])
    cells = ("–", f'<a href="{html.escape(row["link"])}">{name}</a>' if row["link"] else name,
             _score(row["score"]), _percent(row["sol_percent"]),
             "–" if row["kernels_per_call"] is None else str(row["kernels_per_call"]),
             html.escape(row["language"]), "–",
             _when(row["last_measured"]),
             html.escape(f"Reference: measured by staff on {row['last_measured'][:10]}, job {row['job']}"))
    return '<tr class="baseline">' + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def _placed(rows: list[dict], baselines: list[dict]) -> list[tuple[bool, dict]]:
    """(is a reference row, the row) in table order. The reference rows go among the ranked rows by score,
    above the first one that scores less, so an entry with the same score stays above them. The unranked rows
    follow, in board()'s order."""
    references = sorted(baselines, key=lambda row: row["score"], reverse=True)
    out = []
    for row in rows:
        while references and (row["rank"] is None or row["score"] is None or references[0]["score"] > row["score"]):
            out.append((True, references.pop(0)))
        out.append((False, row))
    return out + [(True, row) for row in references]


def _table(rows: list[dict], baselines: list[dict]) -> str:
    none = '<p class="note">No entries yet.</p>'
    if not rows and not baselines:
        return none
    head = ("Rank", "Team", "Score", "% of speed of light", "Kernels per call", "Language", "Submissions",
            "Last measured", "Status")
    out = ['<div class="scroll"><table><thead><tr>', *(f"<th>{h}</th>" for h in head), "</tr></thead><tbody>"]
    for reference, row in _placed(rows, baselines):
        if reference:
            out.append(_baseline_row(row))
            continue
        muted = ' class="unranked"' if row["rank"] is None else ""
        name = html.escape(row["name"])
        cells = ("–" if row["rank"] is None else str(row["rank"]),
                 f'<a href="#team-{name}">{name}</a>',  # nicknames are [A-Za-z0-9_-]: the anchor outlives the order
                 _score(row["score"]), _percent(row["sol_percent"]),
                 "–" if row["kernels_per_call"] is None else str(row["kernels_per_call"]),
                 html.escape(LANGUAGES.get(row["language"], row["language"] or "–")),
                 str(row["submissions"]), _when(row["last_measured"]), _status(row))
        out.append(f"<tr{muted}>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out) + ("" if rows else none)


def _race(board: dict, by_division: dict[str, list[dict]]) -> str:
    leaders = {division: [row for row in rows if row["rank"] is not None and row["history"]][:TOP]
               for division, rows in by_division.items()}
    plotted = [row for rows in leaders.values() for row in rows]
    refs = [(text, board["reference"][key]) for key, text in (("starter", "Starter"), ("quack", "QuACK"))
            if board["reference"][key] is not None]
    t0, t1 = _domain(plotted, board["updated_at"])
    top = max([score for row in plotted for _, score in row["history"]] + [v for _, v in refs], default=1.0) * 1.05
    out = ['<figure class="race"><div class="panels">']
    for division, rows in leaders.items():
        out.append(f'<div class="panel"><h3>{DIVISIONS[division]}</h3>')
        if not rows:
            out.append('<p class="note">No ranked entries yet.</p></div>')
            continue
        out.append('<ul class="legend">')
        for slot, row in enumerate(rows, 1):
            out.append(f'<li><span class="key s{slot}"></span>{row["rank"]}. {html.escape(row["name"])} '
                       f'<span class="value">{_score(row["history"][0][1])}</span></li>')
        out.append("</ul>")
        series = [(row["name"], slot, _points(row)) for slot, row in enumerate(rows, 1)]
        out.append(_line_chart(series, refs, t0=t0, t1=t1, top=top,
                               label=f"{DIVISIONS[division]}: score over time of the top {len(rows)}"))
        out.append("</div>")
    out.append("</div><figcaption>Each point is one measured result. The dashed lines are the median of the "
               "starter's and QuACK's scores over the ranked entries' jobs.</figcaption></figure>")
    return "".join(out)


def _team(row: dict, now: str) -> str:
    name = html.escape(row["name"])
    language = html.escape(LANGUAGES.get(row["language"], row["language"] or "language not set"))
    out = [f'<article class="team" id="team-{name}"><h3>{name}</h3>'
           f'<p class="meta">{DIVISIONS[row["division"]]} division. {language}. {_status(row)}.</p>'
           '<div class="details">'
           f'<figure><figcaption>% of speed of light per shape. Rows are N, columns are V.</figcaption>'
           f'{_heatmap(row["name"], row["heatmap"])}</figure>'
           '<figure><figcaption>Score history (self-measured)</figcaption>']
    if row["history"]:
        t0, t1 = _domain([row], now)
        top = max(score for _, score in row["history"]) * 1.05
        out.append(_line_chart([(row["name"], 1, _points(row))], [], t0=t0, t1=t1, top=top,
                               label=f"{row['name']}: score history"))
        out.append('<details><summary>All measurements</summary><table class="history"><thead><tr>'
                   "<th>Measured</th><th>Score</th></tr></thead><tbody>")
        out.extend(f"<tr><td>{_when(when)}</td><td>{_score(score)}</td></tr>" for when, score in row["history"])
        out.append("</tbody></table></details>")
    else:
        out.append('<p class="note">No eligible measurement yet.</p>')
    out.append('</figure></div><p><a href="#top">Back to the tables</a></p></article>')
    return "".join(out)


def render(board: dict) -> str:
    phase = board["phase"]
    by_division = {division: [row for row in board["entries"].values() if row["division"] == division]
                   for division in DIVISIONS}
    missing = board["without_name"]
    lines = []
    if missing["no_nickname"]:
        lines.append(f"{_entries(missing['no_nickname'])} without a nickname yet.")
    if missing["unreadable"]:
        lines.append(f"{_entries(missing['unreadable'])} could not be read.")
    body = [f'<header id="top"><h1>Christmas Kernel Cup</h1>'
            f'<p class="banner {phase}">{html.escape(BANNERS[phase])}</p>'
            f'<p class="updated">Last update: {_when(board["updated_at"])}. Select a team for its heatmap '
            "and score history.</p></header>"]
    for division, rows in by_division.items():
        body.append(f"<section><h2>{DIVISIONS[division]}</h2>{_table(rows, board['baselines'])}</section>")
    if lines:
        body.append(f'<p class="note">{" ".join(lines)}</p>')
    title = "Self-measured scores over time" if phase == "final" else "Score over time"
    body.append(f"<section><h2>{title}</h2>{_race(board, by_division)}</section>")
    body.append('<section class="teams">'
                + "".join(_team(row, board["updated_at"]) for row in board["entries"].values()) + "</section>")
    body.append("<footer><p>Score: the geometric mean over the benchmark grid of the speedup over the eager "
                "PyTorch reference, both timed in the same job. % of speed of light: the share of the copy "
                "bandwidth measured in that job. Kernels per call: the most kernels one call launches on any "
                "shape. A stale entry changed its kernel after its newest eligible measurement, so the page "
                "shows the older score. The reference rows come from one staff run of the unchanged starter, which "
                "also timed PyTorch's bf16 F.cross_entropy. They are placed by score and never ranked. The "
                "harness counts kernels only for the submission, so the PyTorch row shows none.</p></footer>")
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Kernel Cup leaderboard</title><style>{CSS}</style></head>"
            f"<body><main>{''.join(body)}</main></body></html>\n")


# The reference palette of the dataviz skill: categorical slots 1-5 for the top five, the blue ramp
# for the heatmap (light to dark in light mode, dark to light in dark mode, so low values recede).
# Link text and every cell's label reach 4.5:1: the light accent is the ramp's step 500, one darker
# than series 1, and no ink reaches 4.5:1 on step 450, so the dark ramp takes step 400 there.
_LIGHT = {"page": "#f9f9f7", "surface": "#fcfcfb", "ink": "#0b0b0b", "ink-2": "#52514e", "muted": "#6f6e69",
          "grid": "#e1e0d9", "axis": "#c3c2b7", "border": "rgba(11,11,11,0.10)", "accent": "#256abf",
          "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100", "s5": "#e87ba4",
          **{f"h{k}": c for k, c in enumerate(("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
                                               "#0d366b"))},
          **{f"h{k}t": "#0b0b0b" if k < 4 else "#ffffff" for k in range(HEAT_BINS)}}
_DARK = {"page": "#0d0d0d", "surface": "#1a1a19", "ink": "#f0efec", "ink-2": "#c3c2b7", "muted": "#9a9891",
         "grid": "#2c2c2a", "axis": "#383835", "border": "rgba(255,255,255,0.10)", "accent": "#3987e5",
         "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "s4": "#c98500", "s5": "#d55181",
         **{f"h{k}": c for k, c in enumerate(("#0d366b", "#184f95", "#256abf", "#3987e5", "#5598e7", "#86b6ef",
                                              "#cde2fb"))},
         **{f"h{k}t": "#ffffff" if k < 3 else "#0b0b0b" for k in range(HEAT_BINS)}}


def _tokens(values: dict[str, str]) -> str:
    return "".join(f"--{key}:{value};" for key, value in values.items())


CSS = (
    f":root{{color-scheme:light;{_tokens(_LIGHT)}}}"
    f"@media (prefers-color-scheme:dark){{:root:not([data-theme=\"light\"]){{color-scheme:dark;{_tokens(_DARK)}}}}}"
    f":root[data-theme=\"dark\"]{{color-scheme:dark;{_tokens(_DARK)}}}"
    "*{box-sizing:border-box}"
    "body{margin:0;background:var(--page);color:var(--ink);"
    "font:15px/1.5 system-ui,-apple-system,\"Segoe UI\",sans-serif}"
    "main{max-width:1120px;margin:0 auto;padding:24px 16px 48px}"
    "h1{font-size:26px;margin:0 0 12px}h2{font-size:19px;margin:32px 0 10px}h3{font-size:16px;margin:0 0 8px}"
    "a{color:var(--accent)}"
    ".banner{margin:0 0 8px;padding:10px 14px;border-left:4px solid var(--accent);background:var(--surface);"
    "border-radius:4px;font-weight:600}"
    ".banner.final{border-left-color:var(--s3)}"
    ".updated,.note,figcaption,footer{color:var(--ink-2);font-size:14px}"
    ".scroll{overflow-x:auto;background:var(--surface);border:1px solid var(--border);border-radius:6px}"
    "table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}"
    "th,td{padding:7px 10px;text-align:left;white-space:nowrap;border-bottom:1px solid var(--grid)}"
    "th{font-size:13px;font-weight:600;color:var(--ink-2)}"
    "td:nth-child(n+3):nth-child(-n+5),th:nth-child(n+3):nth-child(-n+5),td:nth-child(7),th:nth-child(7)"
    "{text-align:right}"
    "td:last-child{white-space:normal;min-width:120px}"
    "tbody tr:last-child td{border-bottom:0}"
    "tr.unranked td,tr.baseline td{color:var(--ink-2)}"
    "tr.baseline td{font-style:italic;background:var(--page)}"
    ".panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}"
    ".panel,.team{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:14px}"
    "figure{margin:0}svg{display:block;width:100%;height:auto;max-width:480px}"
    "svg text{font:12px system-ui,-apple-system,\"Segoe UI\",sans-serif;fill:var(--ink)}"
    "svg .tick{fill:var(--muted)}"
    ".grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}"
    ".ref{stroke:var(--muted);stroke-width:1;stroke-dasharray:4 4}"
    ".line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}"
    ".dot{stroke:var(--surface);stroke-width:2}.hit{fill:transparent}"
    + "".join(f".s{k}.line{{stroke:var(--s{k})}}.s{k} .dot{{fill:var(--s{k})}}"
              f".key.s{k}{{border-top-color:var(--s{k})}}" for k in range(1, TOP + 1))
    + ".point:hover .dot{stroke:var(--ink)}"
    ".legend{list-style:none;margin:0 0 8px;padding:0;display:flex;flex-wrap:wrap;gap:4px 16px;font-size:14px}"
    ".key{display:inline-block;width:16px;border-top:2px solid;margin-right:6px;vertical-align:middle}"
    ".value{color:var(--ink-2);font-variant-numeric:tabular-nums}"
    ".race figcaption{margin-top:8px}"
    ".team{display:none;margin-top:16px}.team:target{display:block}"
    ".meta{margin:0 0 12px;color:var(--ink-2)}"
    ".details{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}"
    ".details figcaption{margin-bottom:6px}"
    ".heat{max-width:420px}.empty{fill:none;stroke:var(--grid)}"
    + "".join(f".h{k} rect{{fill:var(--h{k})}}.h{k} text{{fill:var(--h{k}t)}}" for k in range(HEAT_BINS))
    + ".cell:hover rect{stroke:var(--ink);stroke-width:1.5}"
    ".history{margin-top:6px;max-width:320px}summary{cursor:pointer;color:var(--accent);margin-top:8px}"
    "footer{margin-top:40px;border-top:1px solid var(--grid);padding-top:12px}"
)
