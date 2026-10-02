#!/usr/bin/env python3
"""Check every GitHub link in README.md against the CONTRIBUTING.md criteria.

Reports entries whose upstream repository is archived, has had no push in over
12 months, or has moved to a new owner/name. Standard library only; uses the
GITHUB_TOKEN already available to Actions.

A repository is only reported as unreachable when GitHub answers NOT_FOUND for
it twice: once in its batch and again in a query of its own. Anything that
merely failed to answer — a rate limit, a timeout, a transient backend error —
is retried, and if it still has not answered it is listed as "could not be
checked" rather than counted as a finding. A question the API declined to
answer is not evidence that a project is gone.

Writes a markdown report to the path given by --out (default: currency-report.md)
and prints the number of findings to stdout. When GITHUB_OUTPUT is set, the
findings and could-not-check counts are also written there as step outputs.
Always exits 0 unless the check itself fails — a listed project going archived
is news for the maintainers, not a broken build.
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

GRAPHQL = "https://api.github.com/graphql"
BATCH = 50
RETRY_BATCH = 10
ATTEMPTS = 3
BACKOFF_SECONDS = 5
STALE_MONTHS = 12

ENTRY = re.compile(r"^\*\s*\[[^\]]+\]\((https://github\.com/([\w.-]+)/([\w.-]+))/?\)", re.M)

FIELDS = "{ nameWithOwner isArchived pushedAt }"


def warn(message):
    """Log to stderr, and annotate the run when this is an Actions job."""
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::warning::{message}", file=sys.stderr)
    else:
        print(f"warning: {message}", file=sys.stderr)


def post(token, repos):
    """Send one batched query. Returns (data, errors_by_alias)."""
    parts = []
    for i, repo in enumerate(repos):
        owner, name = repo.split("/", 1)
        parts.append(f'r{i}: repository(owner: "{owner}", name: "{name}") {FIELDS}')
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

    errors_by_alias = {}
    for error in payload.get("errors") or []:
        path = error.get("path") or []
        if path:
            errors_by_alias[path[0]] = error.get("type") or "ERROR"
        else:
            # An error with no path applies to the whole query, so no alias in
            # this batch can be trusted.
            for i in range(len(repos)):
                errors_by_alias.setdefault(f"r{i}", error.get("type") or "ERROR")
    return payload.get("data") or {}, errors_by_alias


def sweep(token, repos, resolved, not_found, size):
    """Query `repos`, filling `resolved` and `not_found`. Returns the unanswered."""
    unanswered = []
    for start in range(0, len(repos), size):
        batch = repos[start : start + size]
        try:
            data, errors = post(token, batch)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            warn(f"GitHub API request failed for {len(batch)} repositories: {exc}")
            unanswered.extend(batch)
            continue

        for i, repo in enumerate(batch):
            alias = f"r{i}"
            meta = data.get(alias)
            if meta:
                resolved[repo] = meta
            elif errors.get(alias) == "NOT_FOUND":
                not_found.append(repo)
            else:
                unanswered.append(repo)
    return unanswered


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

    resolved = {}
    candidates = []
    pending = sweep(token, repos, resolved, candidates, BATCH)

    # Retry whatever did not answer. Smaller batches, so one bad repository in a
    # batch of fifty cannot keep taking the other forty-nine down with it.
    for attempt in range(1, ATTEMPTS):
        if not pending:
            break
        time.sleep(BACKOFF_SECONDS * attempt)
        warn(f"retrying {len(pending)} repositories that did not answer (attempt {attempt + 1})")
        pending = sweep(token, sorted(pending), resolved, candidates, RETRY_BATCH)

    # Confirm every NOT_FOUND on its own before calling a project gone.
    missing, unchecked = [], sorted(pending)
    for repo in sorted(candidates):
        confirm_resolved, confirm_missing = {}, []
        still_pending = sweep(token, [repo], confirm_resolved, confirm_missing, 1)
        if confirm_resolved:
            resolved.update(confirm_resolved)
        elif confirm_missing:
            missing.append(repo)
        else:
            unchecked.extend(still_pending)

    unchecked = sorted(set(unchecked))
    if unchecked:
        warn(
            f"{len(unchecked)} repositories could not be checked and are reported "
            "separately rather than counted as findings"
        )

    cutoff = (
        datetime.date.today() - datetime.timedelta(days=365 * STALE_MONTHS // 12)
    ).isoformat()

    archived, stale, moved = [], [], []
    for repo, meta in sorted(resolved.items()):
        pushed = meta["pushedAt"][:10]
        if meta["isArchived"]:
            archived.append((repo, pushed))
        elif pushed < cutoff:
            stale.append((repo, pushed))
        if meta["nameWithOwner"].lower() != repo.lower():
            moved.append((repo, meta["nameWithOwner"]))

    lines = [
        f"Checked {len(resolved)} of {len(repos)} GitHub entries in `{args.readme}` on "
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
    section(
        "Unreachable (deleted or private)",
        missing,
        lambda r: f"- `{r}` — GitHub answered NOT_FOUND twice",
    )
    if unchecked:
        section(
            "Could not be checked",
            unchecked,
            lambda r: f"- `{r}`",
        )
        lines.append(
            "These are not findings. The API did not answer for them after "
            f"{ATTEMPTS} attempts, so nothing is known about them either way — "
            "they are listed so a failed run cannot look like a clean one."
        )
        lines.append("")

    open(args.out, "w", encoding="utf-8").write("\n".join(lines))

    findings = len(archived) + len(stale) + len(missing)
    if github_output := os.environ.get("GITHUB_OUTPUT"):
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"findings={findings}\n")
            handle.write(f"unchecked={len(unchecked)}\n")
    print(findings)


if __name__ == "__main__":
    main()
