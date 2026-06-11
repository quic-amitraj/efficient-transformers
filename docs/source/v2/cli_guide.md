# CLI Guide

The CLI is the fastest path to reproducible model operations on AI 100.

## Primary Commands

- `QEfficient.cloud.infer`: recommended end-to-end command.
- `QEfficient.cloud.export`: explicit ONNX export.
- `QEfficient.cloud.compile`: explicit compile stage.
- `QEfficient.cloud.execute`: run precompiled QPC.
- `QEfficient.cloud.finetune`: fine-tune workflows on QAIC.

## Command Strategy

Use `infer` by default.

Split stages only when you need one of the following:

- to publish and reuse precompiled QPCs,
- to isolate compile failures from runtime failures,
- to benchmark export/compile/runtime independently.

## Common Argument Families

- Model selection: `--model_name`
- Compile profile: `--batch_size`, `--prompt_len`, `--ctx_len`, `--num_cores`, `--mxfp6`
- Runtime topology: `--device_group`
- Input payload: `--prompt`, `--prompts_txt_file_path`
- QNN controls: `--enable_qnn` and optional config path

## Recommended Baseline Run

```bash
python -m QEfficient.cloud.infer \
  --model_name gpt2 \
  --batch_size 1 \
  --prompt_len 32 \
  --ctx_len 128 \
  --num_cores 16 \
  --device_group [0] \
  --prompt "Hello world" \
  --mxfp6 --mos 1 --aic_enable_depth_first
```

## Reusing an Existing QPC

```bash
python -m QEfficient.cloud.execute \
  --model_name gpt2 \
  --qpc_path <path-to-qpc> \
  --prompt "Once upon a time" \
  --device_group [0]
```

## ZSH Note

In ZSH, pass `device_group` in quotes when needed:

```bash
'--device_group [0]'
```

## Practical CLI Best Practices

1. Keep compile arguments versioned in scripts.
2. Benchmark with a fixed prompt set for comparability.
3. Avoid changing multiple compile knobs in one experiment.
4. Preserve command + git SHA + artifact path in run logs.
