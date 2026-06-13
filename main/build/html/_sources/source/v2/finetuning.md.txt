# Finetuning Guide

Efficient-transformers includes QAIC-oriented fine-tuning workflows, including PEFT-based paths.

## Scope

Use finetuning when you need task/domain adaptation but still want deployment-ready inference flows in the same ecosystem.

## CLI Baseline Example

```bash
python -m QEfficient.cloud.finetune \
  --device qaic:0 \
  --use-peft \
  --output_dir ./qeff-finetuned \
  --num_epochs 2 \
  --context_length 256
```

## Recommended Workflow

1. Prepare/validate dataset and tokenization strategy.
2. Run fine-tuning (PEFT if parameter efficiency is required).
3. Evaluate quality metrics against baseline checkpoints.
4. Export and compile resulting model artifacts.
5. Run inference benchmarks on deployment profile.

## Post-Training Deployment Checklist

- Compile with production-equivalent parameters.
- Validate latency/throughput and memory behavior.
- Verify output quality on holdout prompts.
- Update docs/examples for reproducibility.

## Deep Reference

For full argument descriptions and dataset templates, see the legacy page:

- [Legacy finetune documentation](../finetune)
