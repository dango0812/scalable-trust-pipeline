#!/usr/bin/env sh
# Point git at the repository's hooks in .githooks/ (run once per clone).
# They block commits and pushes on main, check branch names and commit messages, lint
# staged Python files and run the full checks before a push. CI runs the same checks.
git config core.hooksPath .githooks && echo "git hooks: .githooks"
