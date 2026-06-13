# Release and Documentation Workflow

Documentation publishing is fully automated through GitHub Actions.

## What Is Automated

- Docs build on pull requests (`docs-check.yml`).
- Docs deployment on push to:
  - `main`
  - `release/*`
  - tags matching `v*`
- Link integrity checks (`docs-linkcheck.yml`) on schedule and manual runs.

## Publishing Rules

- Push to `main`:
  - updates site root (latest docs),
  - publishes `versions/main/`,
  - refreshes docs for discovered `release/*` branches.
- Push to `release/*`:
  - publishes `versions/release-<branch-suffix>/`.
- Push tag `v*`:
  - publishes immutable `versions/<tag>/` snapshot.

No manual HTML generation or manual `gh-pages` commits are needed.

## If You Do Not Use Release Branches

You can run versioned docs with just:

- `main` (latest), and
- release tags (`v*`).

In that model, tags become your stable version history.

## Suggested Branch/Tag Pattern

```bash
# Optional release branch
git checkout -b release/v1.22.0
git push origin release/v1.22.0

# Immutable docs snapshot
git tag v1.22.0
git push origin v1.22.0
```

## Operational Checks

After merging docs changes:

1. Verify `docs-check` passed on the PR.
2. Verify `docs-deploy` completed on target branch/tag.
3. Open root docs and `versions/index.html` to confirm published output.

## Failure Recovery

If deployment fails:

1. Inspect workflow logs to identify build vs publish failure.
2. Re-run `docs-deploy` via `workflow_dispatch`.
3. Re-check `gh-pages` branch output and version index.
