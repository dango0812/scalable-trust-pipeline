# scalable-trust-pipeline

A training-data pipeline for models that learn only from trustworthy data.
It starts with a single record and keeps the same structure all the way up to
billions or trillions of records.

## Development

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), and Node.js 20+ (Git hooks only).

```sh
uv sync       # Python dependencies into .venv, pinned by uv.lock
npm install   # installs the husky Git hooks
```

| Check  | Command                      |
| ------ | ---------------------------- |
| Lint   | `uv run ruff check`          |
| Format | `uv run ruff format --check` |
| Types  | `uv run mypy`                |
| Tests  | `uv run pytest`              |

## Workflow

- `main` changes only through pull requests. The local hooks refuse commits on `main` and
  pushes to it; the GitHub ruleset in [`.github/rulesets/main.json`](.github/rulesets/main.json)
  enforces the same on the server, since local hooks can be skipped with `--no-verify`.
  It requires a pull request and green CI but zero approvals, because GitHub does not let
  authors approve their own pull requests.
- Branch names: `<type>/<kebab-case-description>`, e.g. `feat/core-schema`, `ci/github-actions`.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
  (`feat(schema): ...`), checked by commitlint locally and in CI.
  Types: build, chore, ci, docs, feat, fix, perf, refactor, revert, style, test.
- Pull requests use the template: **Overview** (what was wrong or missing, and why) and
  **Changes** (one concrete change per bullet).
