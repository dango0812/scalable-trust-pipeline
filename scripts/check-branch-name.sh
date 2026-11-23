#!/usr/bin/env sh
# Validate a branch name against the repository convention:
#   <type>/<kebab-case-description>   e.g. feat/core-schema, ci/github-actions
# The types are the Conventional Commits types of scripts/commit_lint.py, so a branch
# and its commits share one vocabulary.
#
# Usage: scripts/check-branch-name.sh [branch]   (defaults to the current branch)

branch="${1:-$(git symbolic-ref --quiet --short HEAD 2>/dev/null)}"

# Detached HEAD (rebase, bisect, CI checkout): nothing to validate.
[ -z "$branch" ] && exit 0

case "$branch" in
  main | master)
    echo "x Commits to '$branch' are blocked. Work on a branch: git switch -c feat/<description>" >&2
    exit 1
    ;;
esac

if ! printf '%s\n' "$branch" |
  grep -Eq '^(build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)/[a-z0-9]+([.-][a-z0-9]+)*$'; then
  echo "x Branch '$branch' does not follow <type>/<kebab-case-description>." >&2
  echo "  types: build, chore, ci, docs, feat, fix, perf, refactor, revert, style, test" >&2
  exit 1
fi
