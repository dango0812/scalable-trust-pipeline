"""Check commit messages and pull request titles against Conventional Commits.

Standard library only, so the commit-msg hook and CI need nothing but Python. The rules are
those of @commitlint/config-conventional, which this repository used before:

- header ``type(scope)!: subject`` with a known lower-case type and an optional lower-case scope
- subject present, not starting with a capital letter, not ending with a period
- header, body and footer lines at most 100 characters
- a blank line between the header and the body

Merge, revert, fixup and squash messages written by git are skipped, as commitlint does.

usage:
  commit_lint.py --file PATH        a commit message file (the commit-msg hook)
  commit_lint.py --message TEXT     one message, e.g. a pull request title
  commit_lint.py --range BASE HEAD  every commit in BASE..HEAD (CI)
"""

import re
import subprocess
import sys
from pathlib import Path

TYPES = (
    "build",
    "chore",
    "ci",
    "docs",
    "feat",
    "fix",
    "perf",
    "refactor",
    "revert",
    "style",
    "test",
)
MAX_LINE = 100
HEADER = re.compile(r"^(?P<type>[^(:!\s]+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?: (?P<subject>.*)$")
SKIPPED = re.compile(r"^(Merge |Revert |Automatic merge|Auto-merged |(amend|fixup|squash)!)")
SCISSORS = "# ------------------------ >8 ------------------------"


def clean(message: str) -> str:
    """Drop git's comment lines and everything below the scissors line, like git does."""
    lines = []
    for line in message.replace("\r\n", "\n").split("\n"):
        if line.startswith(SCISSORS):
            break
        if not line.startswith("#"):
            lines.append(line)
    return "\n".join(lines).strip("\n")


def problems(message: str) -> list[str]:
    """Return every rule the message breaks; an empty list means it is valid."""
    text = clean(message)
    if not text:
        return ["the message is empty"]
    lines = text.split("\n")
    header = lines[0]
    if SKIPPED.match(header):
        return []
    found = []
    match = HEADER.match(header)
    if match is None:
        found.append(f"header must look like 'type(scope): subject', got {header!r}")
    else:
        kind, scope, subject = match["type"], match["scope"], match["subject"]
        if kind not in TYPES:
            found.append(f"type {kind!r} is not one of {', '.join(TYPES)}")
        if scope is not None and (not scope or scope != scope.lower()):
            found.append(f"scope {scope!r} must be non-empty lower case")
        if not subject.strip():
            found.append("subject must not be empty")
        elif subject[0].isupper():
            found.append("subject must not start with a capital letter")
        elif subject.endswith("."):
            found.append("subject must not end with a period")
    if len(header) > MAX_LINE:
        found.append(f"header is {len(header)} characters; at most {MAX_LINE}")
    if len(lines) > 1 and lines[1]:
        found.append("leave a blank line between the header and the body")
    found.extend(
        f"line {number} is {len(line)} characters; at most {MAX_LINE}"
        for number, line in enumerate(lines[1:], start=2)
        if len(line) > MAX_LINE
    )
    return found


def commits(base: str, head: str) -> list[tuple[str, str]]:
    """The (sha, message) pairs of BASE..HEAD, oldest first."""
    shas = subprocess.run(
        ["git", "rev-list", "--reverse", f"{base}..{head}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return [
        (
            sha,
            subprocess.run(
                ["git", "log", "-1", "--format=%B", sha],
                check=True,
                capture_output=True,
                text=True,
            ).stdout,
        )
        for sha in shas
    ]


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "--file":
        checked = [("commit message", Path(argv[1]).read_text(encoding="utf-8"))]
    elif len(argv) == 2 and argv[0] == "--message":
        checked = [("message", argv[1])]
    elif len(argv) == 3 and argv[0] == "--range":
        checked = [(sha[:12], message) for sha, message in commits(argv[1], argv[2])]
    else:
        print(__doc__, file=sys.stderr)
        return 2
    failed = False
    for name, message in checked:
        for problem in problems(message):
            failed = True
            print(f"x {name}: {problem}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
