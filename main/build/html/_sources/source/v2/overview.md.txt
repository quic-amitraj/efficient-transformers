# Efficient-Transformers

Efficient-Transformers is a production-oriented toolkit for running and adapting Hugging Face models on Qualcomm Cloud AI 100.

It standardizes the full lifecycle from model onboarding to low-latency runtime execution.

## What You Get

- Model-family-aware transformations for AI 100 execution.
- ONNX export and QPC compile orchestration.
- Unified CLI (`infer`) and programmable Python APIs.
- Support across LLM, VLM, diffusion, speech, embedding, and reranking workloads.

## Intended Audience

- Platform engineers building AI 100 inference services.
- ML engineers porting new Hugging Face checkpoints.
- Release owners maintaining repeatable deployable artifacts.

## Execution Mental Model

For most workloads, use this sequence:

1. **Load + transform** model with QEff APIs.
2. **Export** transformed graph to ONNX.
3. **Compile** ONNX to QPC for your hardware profile.
4. **Execute** prompts on the compiled artifact.

`QEfficient.cloud.infer` automates this end to end and reuses cached artifacts when compile parameters match.

## Choose Your Starting Path

| Goal | Start Here |
| --- | --- |
| First successful run | [Quickstart](quickstart.md) |
| Environment setup | [Installation](installation.md) |
| CLI-first workflow | [CLI Guide](cli_guide.md) |
| Service integration | [Python API Guide](python_guide.md) |
| Understanding artifacts and compile profiles | [Architecture](architecture.md) |
| Debugging failures | [Troubleshooting](troubleshooting.md) |

## Documentation Layers

- `docs/source/v2/`: primary documentation (current architecture).
- `docs/source/*`: legacy pages retained for backward compatibility and deep historical references.

## Versioned Documentation

Published docs are versioned automatically by CI:

- `main` branch publishes latest docs at site root.
- `release/*` branches publish as `release-*` versions.
- `v*` tags publish immutable version snapshots.

See [Release Workflow](release_workflow.md) for the operational model.
