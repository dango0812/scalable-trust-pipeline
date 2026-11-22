// Conventional Commits, checked by the commit-msg hook and again in CI.
// The defaults of @commitlint/config-conventional are kept as-is:
//   types  build, chore, ci, docs, feat, fix, perf, refactor, revert, style, test
//   header, body and footer lines at most 100 characters; lower-case subject
// scripts/check-branch-name.sh reuses the same types for branch names.
export default {
  extends: ['@commitlint/config-conventional'],
};
