# Quickstart

This page gives a minimal first success path for both CLI and Python APIs.

## Prerequisites

- Platform and SDK installation completed.
- Repository and Python environment set up.
- AI 100 device group ready.

See [Installation](installation.md) if any of these are not done.

## CLI: First End-to-End Run

`infer` is the recommended command because it can export, compile, and execute in one workflow.

```bash
python -m QEfficient.cloud.infer \
  --model_name gpt2 \
  --batch_size 1 \
  --prompt_len 32 \
  --ctx_len 128 \
  --num_cores 16 \
  --mxfp6 \
  --device_group [0] \
  --prompt "My name is" \
  --mos 1 \
  --aic_enable_depth_first
```

Expected behavior:

1. First run: model transform/export/compile, then execution.
2. Re-run with same compile profile: cached ONNX/QPC reused.
3. Changing core compile params creates a new compile artifact.

## CLI: Batch Prompt Input

Use pipe-separated prompts:

```bash
python -m QEfficient.cloud.infer \
  --model_name gpt2 \
  --batch_size 3 \
  --prompt_len 32 \
  --ctx_len 128 \
  --num_cores 16 \
  --device_group [0] \
  --prompt "Prompt A|Prompt B|Prompt C" \
  --mxfp6 --mos 1 --aic_enable_depth_first
```

Or pass a text file:

```bash
python -m QEfficient.cloud.infer \
  --model_name gpt2 \
  --batch_size 3 \
  --prompt_len 32 \
  --ctx_len 128 \
  --num_cores 16 \
  --device_group [0] \
  --prompts_txt_file_path examples/sample_prompts/prompts.txt \
  --mxfp6 --mos 1 --aic_enable_depth_first
```

## Optional: Split Stages Explicitly

Use these only when you need direct stage control:

1. Export: `python -m QEfficient.cloud.export ...`
2. Compile: `python -m QEfficient.cloud.compile ...`
3. Execute: `python -m QEfficient.cloud.execute ...`

## Python API: Minimal Flow

```python
from QEfficient import QEFFAutoModelForCausalLM
from transformers import AutoTokenizer

model_name = "gpt2"

model = QEFFAutoModelForCausalLM.from_pretrained(model_name)
model.compile(num_cores=16, mxfp6_matmul=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
out = model.generate(prompts=["Hello from AI 100"], tokenizer=tokenizer)
print(out)
```

## What to Read Next

- [Architecture](architecture.md) for artifact and execution flow details.
- [CLI Guide](cli_guide.md) for operational command patterns.
- [Python API Guide](python_guide.md) for service integration patterns.
