# Docs

This directory contains the Sphinx-based documentation system for efficient-transformers.

## Documentation Architecture

The docs are organized into two layers:

1. **Primary docs (`docs/source/v2/`)**
   - New information architecture for onboarding, concepts, guides, and operations.
2. **Legacy docs (`docs/source/*.md|*.rst`)**
   - Existing pages retained for compatibility and deep historical references.

The root toctree is defined in `docs/index.rst`.

## Local Build

Install dependencies:

```bash
pip install -r docs/requirements.txt
```

Build HTML:

```bash
make -C docs html
```

Run linkcheck:

```bash
make -C docs linkcheck
```

Preview locally:

```bash
make -C docs serve
```

## Environment Variables

These are used by local builds and CI workflows:

- `DOC_VERSION`
- `DOCS_VERSIONS_INDEX_URL`
- `DOCS_LATEST_URL`
- `DOCS_AVAILABLE_VERSIONS`

## CI/CD Automation

- PRs run `.github/workflows/docs-check.yml`.
- Pushes to `main`, `release/*`, and tags `v*` are handled by `.github/workflows/docs-deploy.yml`.
- Link validation runs in `.github/workflows/docs-linkcheck.yml`.
