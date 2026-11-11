"""The stateless collector: every pair's kcup branch, read into the provisional board (design §9.3).

The decisions are pure functions. The reading goes through a client (kcup_board.github.Client).
Actions logs of this public repository are world-readable, so nothing here prints a repository
name, a pair ID, a path or a URL: counts, statuses and exception types only.

    python -m kcup_board.collect --config config.toml --site site --kit-versions kit_versions.json --schema schema.json

kit_versions.json (written by the course's publish script) maps every published kit tag to
{"harness": {path: blob id}, "writeup": {"blob": the template's blob id, ...}}."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jsonschema

from kcup_board import divisions, github, ontime, page

# The kit's kcup_harness.score.PLAUSIBLE_SOL_PERCENT. This repository has no kit, so a test in the
# course repository checks that the two agree.
PLAUSIBLE_SOL_PERCENT = 105.0
SETTLE = timedelta(minutes=10)  # freeze.py's settle time, beyond the activity endpoint's 60 s cache
HISTORY = 20  # eligible results per entry: the score history and the number of eligible submissions
MAX_READS = 60  # results files read per entry, newest first, to find them (the App's rate limit)
EARLIEST = datetime(2026, 11, 1, tzinfo=UTC)  # no results JSON is older than the kit
AHEAD = timedelta(hours=1)  # how far past the collector's clock a finished_at may lie
# Scores and speedups: anything outside (MIN_SPEEDUP, MAX_SPEEDUP) is an error. A larger one overflows the
# page's axes, and a smaller one rounds to 0 there, which has no logarithm.
MIN_SPEEDUP = 1e-6
MAX_SPEEDUP = 1e6
MAX_PERCENT = 1e4
MAX_SHAPES = 12  # the kit's benchmark grid, 3 N x 4 V; a test in the course repository checks the two agree
BRANCH = "kcup"
TEAM = "submission/team.toml"
WRITEUP = "submission/WRITEUP.md"
EXCLUDED_FROM_KERNEL = (TEAM, WRITEUP)
HARNESS_PREFIX = "kcup_harness/"
HARNESS_EXTRA = ("job.sh", "pytest.ini")
RESULTS = re.compile(r"^results/[^/]+\.json$")
PLACEHOLDER = "> Replace this"
BASELINE = "baselines/starter.json"  # next to config.toml: the staff run of the unchanged starter
# The reference rows (design §9.5): the label, the implementation in the staff run whose timings make
# the row, the language as the page shows it, and a link.
BASELINES = (("Starter (staff run)", "submission", "CuTe DSL",
              "https://github.com/LLM-Engineering-26-27/kernel-cup/blob/main/submission/kernel.py"),
             ("PyTorch bf16 F.cross_entropy (staff run)", "torch_bf16", "PyTorch", None))
CONFIG_KEYS = {"owner": str, "pattern": str, "freeze_at": str, "output_prefix": str, "accepted_kit_versions": list}


def read_config(path: Path) -> dict:
    """A collector config (design §9.3). It is refused before anything is read when a key has the wrong
    type, the pattern does not compile, freeze_at does not parse or no kit version is accepted."""
    config = tomllib.loads(path.read_text())
    for key, kind in CONFIG_KEYS.items():
        if not isinstance(config.get(key), kind):
            raise ValueError(f"{path.name}: {key} must be a {kind.__name__}")
    if not isinstance(config.get("cname", ""), str):
        raise ValueError(f"{path.name}: cname must be a str")
    versions = config["accepted_kit_versions"]
    if not versions or not all(isinstance(version, str) for version in versions):
        raise ValueError(f"{path.name}: accepted_kit_versions must name at least one kit tag")
    re.compile(config["pattern"])
    ontime._instant(config["freeze_at"])
    return config


def plausible(summary: dict) -> bool:
    return summary["sol_percent"] <= PLAUSIBLE_SOL_PERCENT


def _int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _drawable(results: dict, now: datetime) -> bool:
    """Every number and instant the page reads is in range, so no results JSON the schema accepts can
    take the page down: a score of 1e308 overflows its axes, one of 5e-324 has no logarithm there, a
    finished_at in year 1 stretches its time axis over two millennia, and n = 8192.0 is no heatmap row.
    At most the benchmark grid's shapes, each once, so a hand-made file cannot bloat the heatmap."""
    shapes = results["shapes"]
    try:
        finished = ontime._instant(results["finished_at"])
    except ValueError:
        return False
    summary = results["summary"]
    speedups = [shape["impls"]["reference"]["median_s"] / impl["median_s"] for shape in shapes
                for impl in shape["impls"].values() if "median_s" in impl]  # a comparator that raised has none
    return (EARLIEST <= finished <= now + AHEAD
            and MIN_SPEEDUP < summary["score"] < MAX_SPEEDUP and 0 <= summary["sol_percent"] < MAX_PERCENT
            and all(MIN_SPEEDUP < speedup < MAX_SPEEDUP for speedup in speedups)
            and all(_int(shape["n"]) and _int(shape["v"]) and 0 <= shape["sol_percent"] < MAX_PERCENT
                    for shape in shapes)
            and len({(shape["n"], shape["v"]) for shape in shapes}) == len(shapes) <= MAX_SHAPES
            and all(_int(count) for count in results["census"]["kernels_per_call"].values()))


def eligible(results, schema: dict, now: datetime) -> bool:
    """A full, committed H100 run that the schema accepts, with every number the page reads in range."""
    if not jsonschema.Draft202012Validator(schema).is_valid(results):
        return False
    return (results["mode"] == "full" and results["summary"]["full_grid"]
            and "H100" in (results["environment"]["gpu"] or "") and not results["kernel"]["uncommitted"]
            and _drawable(results, now))


def _blobs(files: dict) -> dict[str, str]:
    return {path: entry["blob"] for path, entry in files.items()}


def kernel_matches(results: dict, tree: dict[str, str]) -> bool:
    committed = {p: b for p, b in tree.items() if p.startswith("submission/") and p not in EXCLUDED_FROM_KERNEL}
    return _blobs(results["kernel"]["files"]) == committed


def harness_blobs(tree: dict[str, str]) -> dict[str, str]:
    return {p: b for p, b in tree.items() if p.startswith(HARNESS_PREFIX) or p in HARNESS_EXTRA}


def harness_version(blobs: dict[str, str], kit_versions: dict) -> str | None:
    return next((version for version, kit in kit_versions.items() if kit["harness"] == blobs), None)


def writeup_ok(text: str | None, template_blob: str, blob: str | None, division: str) -> bool:
    """Present, not the template, and no placeholder left. An Unassisted entry may keep the one
    under "## Assistants": that section is for the Assisted division."""
    if text is None or blob == template_blob:
        return False
    assistants = False
    for line in text.splitlines():
        if line.startswith("## "):
            assistants = line.startswith("## Assistants")
        elif line.startswith(PLACEHOLDER) and not (assistants and division == "unassisted"):
            return False
    return True


def team_history(versions: list[divisions.TeamVersion], head_text: str):
    """(the history divisions.display_names() reads, the entry's Team) from every committed version of
    team.toml and its text at the head. The nickname and language come from team.toml at the head
    (§9.3 step 3), never from commit dates, which can be forged: stamped no earlier than any version and
    listed last, the head's own text is the version divisions.resolve() reads. The versions still decide
    "never back" and the clash's first-commit dates. The staff final re-run calls this too, so final.json
    and the board show the same names."""
    if not versions:
        return [], divisions.parse_team("")
    history = [*versions, divisions.TeamVersion(max(version.committed_at for version in versions), head_text)]
    return history, divisions.resolve(history)


def status(*, tree: dict[str, str], results_newest_first: list[dict], kit_versions: dict,
           team: divisions.Team, frozen: bool, writeup: bool) -> dict:
    """results_newest_first holds the eligible results only, kit_versions the accepted versions only."""
    chosen = results_newest_first[0] if results_newest_first else None
    reasons = []
    if chosen is None:
        reasons.append("no-result")
    if harness_version(harness_blobs(tree), kit_versions) is None or (
            chosen is not None and harness_version(_blobs(chosen["harness"]["files"]), kit_versions) is None):
        reasons.append("harness-modified")
    if chosen is not None and not chosen["census"]["passed"]:
        reasons.append("census")
    if chosen is not None and not plausible(chosen["summary"]):
        reasons.append("implausible")
    if team.problems:
        reasons.append("team-toml")
    if frozen and not writeup:
        reasons.append("writeup")
    if reasons:
        state = "unranked"
    else:
        state = "provisional" if kernel_matches(chosen, tree) else "stale"
    return {"status": state, "frozen": frozen, "reasons": reasons, "results": chosen,
            "history": [(r["finished_at"], r["summary"]["score"]) for r in results_newest_first]}


def phase(config: dict, now: datetime, *, final: bool) -> str:
    """final: final/<output_prefix>final.json exists."""
    if final:
        return "final"
    freeze = ontime._instant(config["freeze_at"])
    if now <= freeze:
        return "open"
    return "freezing" if now < freeze + SETTLE else "frozen"


def _for_page(results: dict) -> dict:
    """What the page reads of a results JSON. The environment (node, host, SLURM job), the file hashes,
    the head SHA, the JSON's own team fields, the census calls and the shapes' messages (checks, a
    comparator's error) stay here: page.board()'s allow-list is the second line of defence, not the only one."""
    return {"kit_version": results["kit_version"], "finished_at": results["finished_at"],
            "summary": results["summary"],
            "census": {key: value for key, value in results["census"].items() if key != "calls"},
            "shapes": [{"n": shape["n"], "v": shape["v"], "sol_percent": shape["sol_percent"],
                        "impls": {name: {"median_s": impl["median_s"]} for name, impl in shape["impls"].items()
                                  if "median_s" in impl}}  # a comparator that raised holds only its error
                       for shape in results["shapes"]]}


def _no_constant(name: str):
    raise ValueError(f"{name} is not a number a results JSON may hold")


def _read_entry(client, owner: str, repo: str, kit_versions: dict, schema: dict, freeze_at: str | None,
                now: datetime):
    """(team.toml history, the entry without its shown name, unreadable results files), or None when
    there is no kcup branch. freeze_at is None until the freeze has settled, and then the head is the
    on-time one."""
    if freeze_at is None:
        head = client.branch_head(owner, repo, BRANCH)
    else:
        head = ontime.on_time_head(client.pushes(owner, repo, BRANCH), freeze_at)
    if head is None:
        return None
    tree = client.tree(owner, repo, head)
    versions = []
    for sha, date in client.commits_touching(owner, repo, TEAM, head):
        text = client.file(owner, repo, TEAM, sha)
        if text is not None:
            versions.append(divisions.TeamVersion(divisions.utc(date), text))
    head_text = (client.file(owner, repo, TEAM, head) or "") if versions else ""
    history, team = team_history(versions, head_text)
    results, unreadable = [], 0
    for path in sorted((p for p in tree if RESULTS.match(p)), reverse=True)[:MAX_READS]:
        try:
            parsed = json.loads(client.file(owner, repo, path, head) or "null", parse_constant=_no_constant)
        except ValueError:  # this file only: not base64, not UTF-8, not JSON, or NaN or an infinity in it
            unreadable += 1
            continue
        try:
            counts = eligible(parsed, schema, now)
        except OverflowError:  # this file only: a schema-valid integer no float holds, such as median_s = 10**400
            counts = False
        if counts:
            results.append(parsed)
            if len(results) == HISTORY:
                break
    frozen = freeze_at is not None
    writeup = False
    if frozen:
        try:
            text = client.file(owner, repo, WRITEUP, head)
        except ValueError:  # not UTF-8, or too large to read: no write-up that counts
            text = None
        writeup = all(writeup_ok(text, kit["writeup"]["blob"], tree.get(WRITEUP), team.division)
                      for kit in kit_versions.values())
    row = status(tree=tree, results_newest_first=results, kit_versions=kit_versions, team=team,
                 frozen=frozen, writeup=writeup)
    if row["results"] is not None:
        row["results"] = _for_page(row["results"])
    return history, {"division": team.division, "language": team.language, **row}, unreadable


def baselines(path: Path, config: dict, kit_versions: dict, schema: dict, now: datetime) -> list[dict]:
    """The reference rows (page.baseline()) from the staff run at path. The run passes what a ranked entry
    passes (eligible, a harness that config accepts, the census and the plausibility check) and names its
    SLURM job. A row needs its implementation timed on every shape and a % of speed of light the page can
    draw. Whatever fails is left out with a log line, so the board is still written (§9.4)."""
    accepted = {version: kit_versions[version] for version in config["accepted_kit_versions"]}
    try:
        run = json.loads(path.read_text(), parse_constant=_no_constant)
        ok = (eligible(run, schema, now) and harness_version(_blobs(run["harness"]["files"]), accepted) is not None
              and run["census"]["passed"] and plausible(run["summary"]) and run["environment"]["slurm_job_id"])
    except (OSError, ValueError, OverflowError) as error:
        print(f"baseline not shown: {type(error).__name__}", file=sys.stderr)
        return []
    if not ok:
        print("baseline not shown: the staff run fails a ranked entry's checks or has no job id", file=sys.stderr)
        return []
    rows = []
    for name, impl, language, link in BASELINES:
        row = None
        if all("median_s" in shape["impls"].get(impl, {}) for shape in run["shapes"]):
            try:
                row = page.baseline(name, impl, language, link, run)
            except ValueError:  # a % of speed of light that rounds to 0 has no logarithm
                pass
        if row is not None and 0 < row["sol_percent"] < MAX_PERCENT:
            rows.append(row)
        else:
            print(f"baseline not shown: {impl} is not timed on every shape or out of range", file=sys.stderr)
    return rows


def collect(client, config: dict, kit_versions: dict, schema: dict, now: datetime) -> list[dict]:
    """One entry per pair repository with a kcup branch: its shown name, division, language and status()."""
    owner = config["owner"]
    accepted = {version: kit_versions[version] for version in config["accepted_kit_versions"]}
    settled = phase(config, now, final=False) == "frozen"
    repos = sorted((r for r in client.installation_repositories()
                    if r["owner"]["login"] == owner and re.search(config["pattern"], r["name"])),
                   key=lambda r: r["id"])
    rows, histories, unreadable = [], {}, 0
    for repo in repos:
        try:
            read = _read_entry(client, owner, repo["name"], accepted, schema,
                               config["freeze_at"] if settled else None, now)
        except Exception as error:  # one broken entry must not take the board down (§9.4)
            print(f"entry not read: {type(error).__name__}", file=sys.stderr)
            rows.append((None, {"division": None, "language": None, "status": "unranked", "frozen": settled,
                                "reasons": ["collector-error"], "results": None, "history": []}))
            continue
        if read is None:
            continue
        history, row, skipped = read
        unreadable += skipped
        if history:
            histories[repo["id"]] = history
        rows.append((repo["id"], row))
    if unreadable:
        print(f"unreadable results files skipped: {unreadable}", file=sys.stderr)
    shown = divisions.display_names(histories)
    return [{"name": shown.get(repo_id), **row} for repo_id, row in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, action="append", required=True,
                        help="a collector config, given once for every config*.toml")
    parser.add_argument("--site", type=Path, required=True, help="the gh-pages checkout")
    parser.add_argument("--kit-versions", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    args = parser.parse_args(argv)
    client = github.Client(os.environ["KCUP_TOKEN"])
    kit_versions = json.loads(args.kit_versions.read_text())
    schema = json.loads(args.schema.read_text())
    now = datetime.now(UTC).replace(microsecond=0)
    failed = 0
    for path in args.config:
        config = read_config(path)
        prefix = config["output_prefix"]
        final_path = path.parent / "final" / f"{prefix}final.json"
        final = json.loads(final_path.read_text()) if final_path.is_file() else None
        entries = collect(client, config, kit_versions, schema, now)
        if entries and all(entry["reasons"] == ["collector-error"] for entry in entries):
            # Nothing could be read. This page stays as it is and keeps the time of its last update (§9.4),
            # and the other configs are still written.
            print(f"board {prefix or '(main)'}: every entry failed, the site is left as it was", file=sys.stderr)
            failed += 1
            continue
        board = page.board(entries, now=now.strftime("%Y-%m-%dT%H:%M:%SZ"), final=final,
                           phase=phase(config, now, final=final is not None),
                           baselines=baselines(path.parent / BASELINE, config, kit_versions, schema, now))
        page.write_site(args.site, board, prefix)
        if "cname" in config:  # GitHub Pages keeps a custom domain only while the root has this file
            (args.site / "CNAME").write_text(config["cname"] + "\n")
        counts = Counter(entry["status"] for entry in entries)
        print(f"board {prefix or '(main)'}: {len(entries)} entries, "
              + ", ".join(f"{counts[s]} {s}" for s in ("provisional", "stale", "unranked")))
    return 1 if failed == len(args.config) else 0


if __name__ == "__main__":
    raise SystemExit(main())
