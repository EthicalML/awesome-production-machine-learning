#!/usr/bin/env python3
"""Check every GitHub link in README.md against the CONTRIBUTING.md criteria.

Reports entries whose upstream repository is archived, has had no push in over
12 months, or has moved to a new owner/name. Standard library only; uses the
GITHUB_TOKEN already available to Actions.

Writes a markdown report to the path given by --out (default: currency-report.md)
and prints the number of findings to stdout. Always exits 0 unless the check
itself fails — a listed project going archived is news for the maintainers, not
a broken build.
"""

import argparse
import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.request

GRAPHQL = "https://api.github.com/graphql"
BATCH = 50
STALE_MONTHS = 12

ENTRY = re.compile(r"^\*\s*\[[^\]]+\]\((https://github\.com/([\w.-]+)/([\w.-]+))/?\)", re.M)


def query(token, repos):
    """Return {owner/name: metadata or None} for one batch."""
    parts = []
    for i, repo in enumerate(repos):
        owner, name = repo.split("/", 1)
        parts.append(
            f'r{i}: repository(owner: "{owner}", name: "{name}") '
            "{ nameWithOwner isArchived pushedAt }"
        )
    body = json.dumps({"query": "{" + " ".join(parts) + "}"}).encode()
    req = urllib.request.Request(
        GRAPHQL,
        data=body,
        headers={
            "Authorization": f"bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "awesome-production-machine-learning-currency-check",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    data = payload.get("data") or {}
    return {repo: data.get(f"r{i}") for i, repo in enumerate(repos)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--out", default="currency-report.md")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("GITHUB_TOKEN is not set")

    text = open(args.readme, encoding="utf-8").read()
    repos = sorted({f"{m.group(2)}/{m.group(3)}" for m in ENTRY.finditer(text)})

    status = {}
    for i in range(0, len(repos), BATCH):
        batch = repos[i : i + BATCH]
        try:
            status.update(query(token, batch))
        except urllib.error.URLError as exc:
            sys.exit(f"GitHub API request failed: {exc}")

    cutoff = (
        datetime.date.today() - datetime.timedelta(days=365 * STALE_MONTHS // 12)
    ).isoformat()

    archived, stale, moved, missing = [], [], [], []
    for repo, meta in sorted(status.items()):
        if meta is None:
            missing.append(repo)
            continue
        pushed = meta["pushedAt"][:10]
        if meta["isArchived"]:
            archived.append((repo, pushed))
        elif pushed < cutoff:
            stale.append((repo, pushed))
        if meta["nameWithOwner"].lower() != repo.lower():
            moved.append((repo, meta["nameWithOwner"]))

    lines = [
        f"Checked {len(repos)} GitHub entries in `{args.readme}` on "
        f"{datetime.date.today().isoformat()}.",
        "",
        "Criteria from CONTRIBUTING.md: *tools should not be archived and must have "
        "been actively maintained within the last 12 months.*",
        "",
    ]

    def section(title, rows, render):
        lines.append(f"### {title} ({len(rows)})")
        lines.append("")
        if rows:
            lines.extend(render(r) for r in rows)
        else:
            lines.append("_None._")
        lines.append("")

    section(
        "Archived upstream",
        archived,
        lambda r: f"- `{r[0]}` — last push {r[1]}",
    )
    section(
        f"No push in over {STALE_MONTHS} months",
        stale,
        lambda r: f"- `{r[0]}` — last push {r[1]}",
    )
    section("Moved or renamed", moved, lambda r: f"- `{r[0]}` → `{r[1]}`")
    section("Unreachable (deleted or private)", missing, lambda r: f"- `{r}`")

    open(args.out, "w", encoding="utf-8").write("\n".join(lines))
    print(len(archived) + len(stale) + len(missing))


if __name__ == "__main__":
    main()
