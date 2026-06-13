# Architecture and Execution Flow

This section explains how efficient-transformers converts a generic model checkpoint into a deployment-ready AI 100 runtime artifact.

## High-Level Components

- **Model adapters**: architecture-aware transforms for hardware-compatible graph behavior.
- **Export layer**: produces ONNX graphs aligned with runtime requirements.
- **Compile layer**: builds QPC artifacts for specific hardware/runtime profiles.
- **Runtime layer**: executes prompts using compiled artifacts and runtime options.

## End-to-End Lifecycle

1. **Load + transform**
   - Initializes supported model class and applies QEff transforms.
2. **Export to ONNX**
   - Generates runtime-oriented graph and metadata.
3. **Compile to QPC**
   - Produces deployment artifact bound to compile profile.
4. **Execute**
   - Runs prefill/decode paths using prompts and tokenizer/runtime controls.

## Artifact Layout

Default cache root:

- `~/.cache/qeff_cache`

Override options:

- `QEFF_HOME` (preferred)
- `XDG_CACHE_HOME` (fallback)

Typical artifact classes stored in cache:

- transformed model checkpoints,
- ONNX exports,
- compiled QPC outputs,
- compile metadata keyed by profile.

## Compile Profile Sensitivity

QPC artifacts are profile-specific. A change in any key compile parameter generally requires a new compile output.

Common profile dimensions:

- `num_cores`
- precision flags (for example `mxfp6`)
- prompt/context lengths
- device topology (`device_group`)
- QNN-enabled settings

## CLI vs Python Boundaries

Use CLI when you need:

- reproducible shell-driven workflows,
- simple CI jobs,
- low friction experimentation.

Use Python when you need:

- service-level lifecycle control,
- custom preprocessing/postprocessing,
- advanced orchestration and instrumentation.

## Execution Patterns by Modality

The same architecture supports multiple task families:

- Causal text generation
- Vision-language generation/understanding
- Speech sequence tasks
- Diffusion pipelines
- Embedding/reranking flows

See [Features](features.md) and [Model Support](model_support.md) for workload-level details.
