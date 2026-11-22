# Contributing

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you),
Node.js 24 (see `.nvmrc`) and [pnpm](https://pnpm.io/) 10.

```sh
uv sync        # Python dependencies into .venv, pinned by uv.lock
pnpm install   # commitlint and the husky Git hooks, pinned by pnpm-lock.yaml
```

## Checks

`sh scripts/check.sh` runs every gate CI runs, in the same order:

| Gate   | Command                      | Rule                                         |
| ------ | ---------------------------- | -------------------------------------------- |
| Lint   | `uv run ruff check`          | rule set in `pyproject.toml`                 |
| Format | `uv run ruff format --check` | fix with `uv run ruff format`                |
| Types  | `uv run mypy`                | strict mode, tests included                  |
| Tests  | `uv run pytest --cov`        | warnings are errors; 100% lines and branches |

## Workflow

1. Create a branch named `<type>/<kebab-case-description>`, for example `feat/gold-shards`.
   The types are the Conventional Commits types: build, chore, ci, docs, feat, fix, perf,
   refactor, revert, style, test.
2. Commit in feature-sized units using [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):
   `feat(schema): define core data schema`, then a blank line and `- ` bullets.
3. Open a pull request with the template: **Overview** (what was wrong or missing, and why)
   and **Changes** (one concrete change per bullet). Pull requests are squash-merged, so the
   title becomes the commit on `main` and must be a Conventional Commit too.
4. The code owner reviews it. It merges once the `ci` and `conventions` checks pass.

## What enforces this

| Where           | What it does                                                                    |
| --------------- | ------------------------------------------------------------------------------- |
| `pre-commit`    | refuses commits on `main` and badly named branches; lints staged Python files   |
| `commit-msg`    | commitlint                                                                      |
| `pre-push`      | refuses pushes to `main`; runs `scripts/check.sh`                               |
| CI              | static checks; tests on Linux and Windows with Python 3.12 and 3.14             |
| Pull request    | title, branch name and commit messages; review of newly added dependencies      |
| CodeQL          | security analysis of the Python package and of the workflow files, weekly too   |
| Dependabot      | weekly grouped updates for Python, Node tooling and GitHub Actions              |
| `main` ruleset  | pull request, approval and passing checks required; no force push, no deletion  |

Local hooks can be skipped with `--no-verify`; CI and the ruleset cannot, which is why every
local rule is repeated there.
