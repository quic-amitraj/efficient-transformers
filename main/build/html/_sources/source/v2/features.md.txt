# Features

Efficient-transformers focuses on practical throughput, latency, and deployment efficiency for AI 100.

## Capability Areas

- Model transformation for hardware-aware execution.
- Unified export/compile/execute orchestration.
- Cache and retention-oriented runtime paths.
- Precision and graph optimization controls.
- Multi-modality API coverage.

## Feature Groups

| Group | Typical Goal | Where to Start |
| --- | --- | --- |
| Compute context and long prompts | Improve throughput on large contexts | [Legacy supported features](../supported_features) |
| Speculative decoding variants | Reduce per-token latency | [Legacy supported features](../supported_features) |
| On-device sampling paths | Lower host-device overhead | [Legacy supported features](../supported_features) |
| Adapter and PEFT execution | Reuse base model efficiently | [Finetuning](finetuning.md) |
| Diffusion accelerations | Faster image/video generation paths | [Diffuser Classes](../diffuser_classes) |

## Modality Coverage

- Causal language models
- Image-text-to-text models
- Speech models
- Diffusion pipelines
- Embedding and reranking models

## Operational Guidance

1. Start from a working baseline (`infer` without optional acceleration toggles).
2. Enable one optimization family at a time.
3. Record compile profile + runtime args with each benchmark.
4. Keep per-model runbooks so deployment behavior is reproducible.
