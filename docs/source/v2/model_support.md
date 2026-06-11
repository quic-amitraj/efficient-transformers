# Model Support

Efficient-transformers supports multiple model categories and keeps expanding per release.

## Supported Categories

- Causal LLMs
- Vision-language models
- Diffusion models
- Speech models
- Embedding and reranking models

## Source of Truth for Validated Models

Use [Validated models and tasks](../validate) as the release-tracked support matrix.

That page should be treated as the authoritative list when deciding production onboarding scope.

## Selecting a Model for Deployment

Before onboarding, verify:

1. Task path is supported (text, VLM, diffusion, speech, embeddings).
2. Example scripts exist for your model family.
3. Compile profile requirements fit your hardware plan.
4. Runtime constraints (prompt/context, memory, latency target) are realistic.

## Adding a New Model Family

Recommended process:

1. Implement or extend model architecture support.
2. Validate transformed vs baseline behavior.
3. Verify export, compile, and runtime stages.
4. Add an example in `examples/`.
5. Update docs (`validate` + usage pages).

## Release Discipline

If model support changes in a release, update:

- `docs/source/validate.md`
- release notes page (`docs/source/release_docs.md`)
- relevant examples and quickstart/guides
