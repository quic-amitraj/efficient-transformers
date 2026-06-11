# Python API Guide

Use Python APIs when integrating QEff workflows into services and internal platforms.

## Minimal AutoClass Example

```python
from QEfficient import QEFFAutoModelForCausalLM
from transformers import AutoTokenizer

model_name = "gpt2"

# 1) Load + transform
model = QEFFAutoModelForCausalLM.from_pretrained(model_name)

# 2) Compile for your target profile
qpc_path = model.compile(
    num_cores=16,
    mxfp6_matmul=True,
)
print("QPC path:", qpc_path)

# 3) Execute generation
tokenizer = AutoTokenizer.from_pretrained(model_name)
output = model.generate(
    prompts=["My name is"],
    tokenizer=tokenizer,
)
print(output)
```

## Local Model Path Flow

```python
from QEfficient import QEFFAutoModelForCausalLM
from transformers import AutoTokenizer

local_repo = "~/.cache/huggingface/hub/models--gpt2/snapshots/<snapshot_id>"

model = QEFFAutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_repo)
model.compile(num_cores=16)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=local_repo)
result = model.generate(prompts=["Hello"], tokenizer=tokenizer)
print(result)
```

## Integration Patterns

- Initialize model/tokenizer once at service startup.
- Compile per deployment profile and cache artifacts.
- Route requests to profile-compatible QPCs.
- Log runtime params and latency counters with each request.

## API Reference Entry Points

- [QEff Auto Classes](../qeff_autoclasses)
- [Diffuser Classes](../diffuser_classes)
- [CLI API Reference](../cli_api)
