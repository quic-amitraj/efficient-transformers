# FAQ

## Should I use `infer` or split export/compile/execute?

Use `infer` for most workflows. Split stages only when you need explicit artifact lifecycle control.

## Where are ONNX and QPC artifacts stored?

Default is `~/.cache/qeff_cache`. Override with `QEFF_HOME` (preferred) or `XDG_CACHE_HOME`.

## Do I need a release branch for versioned docs?

No. `main` and tags are enough for automated versioned publishing. `release/*` branches are optional and publish as `release-*` versions when present.

## Why do docs builds show autodoc warnings?

API pages import runtime modules. In docs-only or mismatched environments, optional/runtime dependencies can trigger warnings.

## Is manual HTML build/push required after merge?

No. Docs are built and deployed automatically by GitHub Actions on `main`, `release/*`, and `v*` tags.

## Can I use local model files instead of fetching from Hugging Face?

Yes. Pass local paths to `from_pretrained` in Python APIs.
