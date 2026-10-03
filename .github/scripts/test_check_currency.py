#!/usr/bin/env python3
"""Self-test for check_currency.py. Standard library only, no network.

Run it directly:

    python3 .github/scripts/test_check_currency.py

It stubs the GitHub API so the cases that matter can be reproduced on demand:
a repository GitHub says is gone, one it simply fails to answer for, and one
that fails once and then answers. The distinction is the point of the script —
an unanswered question must never be reported as a deleted project.
"""

import importlib.util
import io
import json
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))

spec = importlib.util.spec_from_file_location("check_currency", os.path.join(HERE, "check_currency.py"))
check_currency = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_currency)

README = (
    "* [Healthy](https://github.com/octo/healthy)\n"
    "* [Flaky](https://github.com/octo/flaky)\n"
    "* [Deleted](https://github.com/octo/deleted)\n"
)

ALIAS = re.compile(r'(r\d+): repository\(owner: "([^"]+)", name: "([^"]+)"\)')


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_api(verdict):
    """Build a urlopen stub answering each alias according to `verdict(repo)`."""

    def urlopen(request, timeout=None):
        query = json.loads(request.data.decode())["query"]
        data, errors = {}, []
        for alias, owner, name in ALIAS.findall(query):
            repo = f"{owner}/{name}"
            answer = verdict(repo)
            if answer == "ok":
                data[alias] = {
                    "nameWithOwner": repo,
                    "isArchived": False,
                    "pushedAt": "2026-09-01T00:00:00Z",
                }
                continue
            data[alias] = None
            errors.append(
                {
                    "type": "NOT_FOUND" if answer == "not_found" else "SERVICE_UNAVAILABLE",
                    "path": [alias],
                    "message": answer,
                }
            )
        payload = {"data": data}
        if errors:
            payload["errors"] = errors
        return FakeResponse(payload)

    return urlopen


def _skip_backoff():
    """Make retry sleeps instant, tolerating a version that does not retry."""
    if hasattr(check_currency, "time"):
        check_currency.time.sleep = lambda _seconds: None


def run(verdict):
    """Run the script against a stub API. Returns (findings, report text)."""
    check_currency.urllib.request.urlopen = fake_api(verdict)
    _skip_backoff()

    workdir = tempfile.mkdtemp()
    readme = os.path.join(workdir, "README.md")
    report = os.path.join(workdir, "report.md")
    with open(readme, "w", encoding="utf-8") as handle:
        handle.write(README)

    os.environ["GITHUB_TOKEN"] = "stub"
    os.environ.pop("GITHUB_OUTPUT", None)
    os.environ.pop("GITHUB_ACTIONS", None)

    argv, stdout, stderr = sys.argv, sys.stdout, sys.stderr
    sys.argv = ["check_currency", "--readme", readme, "--out", report]
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        check_currency.main()
        findings = int(sys.stdout.getvalue().strip())
    finally:
        sys.argv, sys.stdout, sys.stderr = argv, stdout, stderr

    with open(report, encoding="utf-8") as handle:
        return findings, handle.read()


def section(report, title):
    """Return the body of one '### title' section."""
    body = report.split(f"### {title}", 1)[1]
    return body.split("###", 1)[0]


class CurrencyCheckTest(unittest.TestCase):
    def test_unanswered_repository_is_not_called_deleted(self):
        """The bug this guards: a transient error is not evidence of deletion."""
        findings, report = run(
            lambda repo: {"octo/flaky": "transient", "octo/deleted": "not_found"}.get(repo, "ok")
        )
        self.assertNotIn("octo/flaky", section(report, "Unreachable"))
        self.assertIn("octo/flaky", section(report, "Could not be checked"))
        self.assertEqual(findings, 1, "only the confirmed-missing repository is a finding")

    def test_repository_github_says_is_gone_is_reported(self):
        _findings, report = run(
            lambda repo: {"octo/flaky": "transient", "octo/deleted": "not_found"}.get(repo, "ok")
        )
        self.assertIn("octo/deleted", section(report, "Unreachable"))

    def test_one_failure_then_an_answer_resolves(self):
        seen = {"octo/flaky": 0}

        def verdict(repo):
            if repo == "octo/flaky":
                seen[repo] += 1
                return "transient" if seen[repo] == 1 else "ok"
            if repo == "octo/deleted":
                return "not_found"
            return "ok"

        _findings, report = run(verdict)
        self.assertNotIn("Could not be checked", report)
        self.assertNotIn("octo/flaky", section(report, "Unreachable"))

    def test_not_found_that_resolves_on_confirmation_is_dropped(self):
        seen = {"octo/deleted": 0}

        def verdict(repo):
            if repo == "octo/deleted":
                seen[repo] += 1
                return "not_found" if seen[repo] == 1 else "ok"
            return "ok"

        findings, report = run(verdict)
        self.assertIn("_None._", section(report, "Unreachable"))
        self.assertEqual(findings, 0)

    def test_a_query_wide_error_does_not_delete_the_whole_list(self):
        """An error with no path applies to every alias, so nothing is known."""

        def urlopen(request, timeout=None):
            query = json.loads(request.data.decode())["query"]
            data = {alias: None for alias, _owner, _name in ALIAS.findall(query)}
            return FakeResponse({"data": data, "errors": [{"message": "server error"}]})

        check_currency.urllib.request.urlopen = lambda *a, **k: urlopen(*a, **k)
        _skip_backoff()

        workdir = tempfile.mkdtemp()
        readme = os.path.join(workdir, "README.md")
        report_path = os.path.join(workdir, "report.md")
        with open(readme, "w", encoding="utf-8") as handle:
            handle.write(README)
        os.environ["GITHUB_TOKEN"] = "stub"
        os.environ.pop("GITHUB_OUTPUT", None)

        argv, stdout, stderr = sys.argv, sys.stdout, sys.stderr
        sys.argv = ["check_currency", "--readme", readme, "--out", report_path]
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        try:
            check_currency.main()
            findings = int(sys.stdout.getvalue().strip())
        finally:
            sys.argv, sys.stdout, sys.stderr = argv, stdout, stderr

        with open(report_path, encoding="utf-8") as handle:
            report = handle.read()
        self.assertEqual(findings, 0, "a failed run reports nothing rather than everything")
        self.assertIn("_None._", section(report, "Unreachable"))
        self.assertIn("octo/healthy", section(report, "Could not be checked"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
