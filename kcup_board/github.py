"""The GitHub REST reads the collector needs, over urllib alone (design §9.2, §9.3).

It mirrors scripts/ops/hosting/github.py in the course repository but stands alone, because
this repository is public and cannot import the course code. Its errors name the HTTP status
only, never a URL or a repository: Actions logs of a public repository are world-readable."""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.parse
import urllib.request

from kcup_board import ontime

API = "https://api.github.com"
PAGE_SIZE = 100
TIMEOUT = 30
NEXT_LINK = re.compile(r'<([^>]*)>\s*;\s*rel="next"')


class Client:
    def __init__(self, token: str) -> None:
        self.token = token

    def _get(self, url: str):
        """(payload, next page URL), or None on 404."""
        request = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github+json", "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "kernel-cup-leaderboard"})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                payload, link = json.loads(response.read()), response.headers.get("Link")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise RuntimeError(f"GitHub answered HTTP {error.code}") from None
        matched = NEXT_LINK.search(link or "")
        if matched and not matched[1].startswith(API + "/"):
            raise RuntimeError("GitHub's next page is not under the API root")
        return payload, matched[1] if matched else None

    def _one(self, endpoint: str, **query):
        answer = self._get(f"{API}{endpoint}" + (f"?{urllib.parse.urlencode(query)}" if query else ""))
        return None if answer is None else answer[0]

    def _pages(self, endpoint: str, **query) -> list:
        url, pages = f"{API}{endpoint}?{urllib.parse.urlencode({**query, 'per_page': PAGE_SIZE})}", []
        while url:
            answer = self._get(url)
            if answer is None:
                raise RuntimeError("GitHub answered HTTP 404")
            payload, url = answer
            pages.append(payload)
        return pages

    def installation_repositories(self) -> list[dict]:
        return [repo for page in self._pages("/installation/repositories") for repo in page["repositories"]]

    def branch_head(self, owner: str, repo: str, branch: str) -> str | None:
        branch_ = self._one(f"/repos/{owner}/{repo}/branches/{urllib.parse.quote(branch)}")
        return None if branch_ is None else branch_["commit"]["sha"]

    def tree(self, owner: str, repo: str, sha: str) -> dict[str, str]:
        """{path: blob id} of every file in the commit's tree."""
        tree = self._one(f"/repos/{owner}/{repo}/git/trees/{sha}", recursive=1)
        if tree is None:
            raise RuntimeError("GitHub answered HTTP 404")
        if tree["truncated"]:
            raise RuntimeError("the tree is truncated")
        return {entry["path"]: entry["sha"] for entry in tree["tree"] if entry["type"] == "blob"}

    def file(self, owner: str, repo: str, path: str, ref: str) -> str | None:
        """The file's text, or None on 404. A file that cannot be decoded (no base64 for it, as for a
        file over 1 MB, or not UTF-8) raises ValueError: its own problem, unlike an HTTP error."""
        content = self._one(f"/repos/{owner}/{repo}/contents/{urllib.parse.quote(path)}", ref=ref)
        if content is None:
            return None
        if not isinstance(content, dict) or content.get("encoding") != "base64":
            raise ValueError("the contents API returned no base64 file")
        return base64.b64decode(content["content"]).decode()

    def commits_touching(self, owner: str, repo: str, path: str, ref: str) -> list[tuple[str, str]]:
        """[(sha, committer date)] of the commits reachable from ref that change path."""
        pages = self._pages(f"/repos/{owner}/{repo}/commits", sha=ref, path=path)
        return [(c["sha"], c["commit"]["committer"]["date"]) for page in pages for c in page]

    def pushes(self, owner: str, repo: str, branch: str) -> list[ontime.Push]:
        """Every activity record of the branch (pushes, force-pushes, deletions), as freeze.py reads them."""
        pages = self._pages(f"/repos/{owner}/{repo}/activity", ref=f"refs/heads/{branch}")
        return [ontime.Push(r["id"], r["timestamp"], r["after"]) for page in pages for r in page]
