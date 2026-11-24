"""scripts/commit_lint.py: the Conventional Commits check used by the hook and by CI."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "commit_lint.py"


def lint(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False, cwd=cwd
    )


@pytest.mark.parametrize(
    "message",
    [
        "feat(schema): define core data schema and trust metadata structure",
        "fix: handle empty payloads",
        "refactor!: rename the lake layout",
        "docs: remove the contributing guide\n\nThe owner keeps conventions in checks.",
        "Merge remote-tracking branch 'origin/main' into feat/lake-contract-v2",
        'Revert "feat: add a thing"',
    ],
)
def test_valid_messages_pass(message: str) -> None:
    assert lint("--message", message).returncode == 0


@pytest.mark.parametrize(
    ("message", "problem"),
    [
        ("add a feature", "header must look like"),
        ("feature: add a thing", "type 'feature' is not one of"),
        ("feat(Schema): add a thing", "scope 'Schema' must be non-empty lower case"),
        ("feat: ", "subject must not be empty"),
        ("feat: Add a thing", "must not start with a capital letter"),
        ("feat: add a thing.", "must not end with a period"),
        ("feat: " + "a" * 100, "header is 106 characters"),
        ("feat: add a thing\nno blank line", "leave a blank line"),
        ("feat: add a thing\n\n" + "b" * 101, "line 3 is 101 characters"),
        ("", "the message is empty"),
    ],
)
def test_invalid_messages_fail_with_the_reason(message: str, problem: str) -> None:
    result = lint("--message", message)
    assert result.returncode == 1
    assert problem in result.stderr


def test_commit_message_files_drop_comments_and_the_scissors_part(tmp_path: Path) -> None:
    message = tmp_path / "COMMIT_EDITMSG"
    message.write_text(
        "fix: keep the reason\n"
        "# Please enter the commit message for your changes.\n"
        "# ------------------------ >8 ------------------------\n"
        "Not part of the message at all, and far longer than one hundred characters would allow.\n",
        encoding="utf-8",
    )
    assert lint("--file", str(message)).returncode == 0


def test_a_range_reports_each_bad_commit(tmp_path: Path) -> None:
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    commit_flags = ["-c", "commit.gpgsign=false", "commit", "-q", "--allow-empty", "--no-verify"]

    def commit(message: str) -> str:
        subprocess.run([*git, *commit_flags, "-m", message], check=True)
        head = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True)
        return head.stdout.decode().strip()

    base = commit("chore: start")
    commit("feat: add a thing")
    head = commit("Bad message")
    result = lint("--range", base, head, cwd=tmp_path)
    assert result.returncode == 1
    assert result.stderr.count("x ") == 1
    assert "header must look like" in result.stderr


def test_wrong_arguments_print_the_usage() -> None:
    result = lint("--nope")
    assert result.returncode == 2
    assert "usage" in result.stderr
