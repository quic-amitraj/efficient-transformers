# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
GPU eager-mode comparison: blocking only vs blocking + skip-softmax.

In eager mode the Python 'continue' statement physically skips:
  - read_only_blockedV()   → zero V memory bandwidth for that KV block
  - update_running_softmax → exp(), rowsum, BMM2 never execute for skipped blocks

This measures real GPU speedup of skip-softmax before testing on AI100.

Note on GPU eager-mode:
  The skip condition uses .item() to bring the scalar bool to CPU, which
  causes a CPU-GPU synchronization per KV block. This sync overhead is present
  only in eager PyTorch mode; on AI100 AOT compilation predicated execution
  happens at the kernel level without sync overhead. Expect better gain on AI100.

Usage
-----
python run_skip_softmax_gpu_test.py
python run_skip_softmax_gpu_test.py --n-layers 4 --ctx-len 4096 --gen-len 50
"""

import argparse
import copy
import time

import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.transformers.cache_utils import QEffDynamicCache


# ---------------------------------------------------------------------------
# Sparsity counter
# ---------------------------------------------------------------------------

class SparsityCounter:
    """Counts KV blocks physically skipped by skip-softmax in eager mode."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.calls_to_softmax_update = 0   # blocks NOT skipped (reached update)
        self.total_calls             = 0   # placeholder, set externally

    @property
    def blocks_not_skipped(self):
        return self.calls_to_softmax_update


_COUNTER = SparsityCounter()


def _patch_for_counting():
    import QEfficient.blocking.blocked_attention_forwards as baf
    original = baf.update_running_softmax

    def _counting(current_max, attn_weights_block, current_denominator,
                  output, v_block, skip_kv=False, skip_future=None, block_max=None):
        _COUNTER.calls_to_softmax_update += 1
        return original(current_max, attn_weights_block, current_denominator,
                        output, v_block, skip_kv, skip_future, block_max)

    baf.update_running_softmax = _counting
    return original


def _restore_counting(original):
    import QEfficient.blocking.blocked_attention_forwards as baf
    baf.update_running_softmax = original


# ---------------------------------------------------------------------------
# Custom CUDA-compatible decode loop
# ---------------------------------------------------------------------------

def run_blocking_model_on_gpu(model, tokenizer, prompt, prompt_len,
                               ctx_len, gen_len, device):
    """
    Run the QEfficient blocking model directly on GPU using QEffDynamicCache.

    This bypasses ApiRunner (which creates CPU inputs) and drives the model
    directly with properly device-placed tensors.

    Returns: (generated_text, wall_time_sec, per_token_ms)
    """
    model.eval()

    # Tokenise and pad to prompt_len
    enc = tokenizer(
        [prompt],
        return_tensors="pt",
        padding="max_length",
        max_length=prompt_len,
        truncation=True,
    )
    input_ids      = enc["input_ids"].to(device)         # [1, prompt_len]
    attention_mask = enc["attention_mask"].to(device)    # [1, prompt_len]

    # Build position_ids
    # QEfficient expects position_ids of shape [1, prompt_len] for prefill
    # and [1, 1] for each decode step.
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0).to(device)

    # ── Prefill ──────────────────────────────────────────────────────────────
    past_key_values = QEffDynamicCache()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

    past_key_values = outputs.past_key_values

    # ── Decode loop ───────────────────────────────────────────────────────────
    next_token  = outputs.logits[:, -1, :].argmax(-1, keepdim=True)   # [1, 1]
    generated   = [next_token.item()]
    cur_pos     = prompt_len                                            # 0-indexed

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for step in range(gen_len - 1):
            cur_pos += 1
            pos_ids  = torch.tensor([[cur_pos]], device=device)        # [1, 1]

            outputs = model(
                input_ids=next_token,
                attention_mask=None,
                position_ids=pos_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(-1, keepdim=True)
            generated.append(next_token.item())

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    text = tokenizer.decode(generated, skip_special_tokens=True)
    total_sec = t1 - t0
    per_token_ms = total_sec / len(generated) * 1000
    return text, total_sec, per_token_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPU eager-mode comparison: blocking vs blocking+skip-softmax"
    )
    parser.add_argument("--model-name",  type=str,   default="Qwen/Qwen3-8B")
    parser.add_argument("--n-layers",    type=int,   default=4)
    parser.add_argument("--prompt",      type=str,
                        default="Tell me about the importance of attention mechanisms in transformers")
    parser.add_argument("--prompt-len",  type=int,   default=32)
    parser.add_argument("--ctx-len",     type=int,   default=4096)
    parser.add_argument("--gen-len",     type=int,   default=50)
    # NVIDIA calibrated TSF for Qwen3-8B
    # 50% sparsity: prefill=587,  decode=16.5
    # 70% sparsity: prefill=3293, decode=119
    parser.add_argument("--tsf-prefill", type=float, default=3293.0,
                        help="Prefill threshold scale factor (70%% sparsity default)")
    parser.add_argument("--tsf-decode",  type=float, default=119.0,
                        help="Decode threshold scale factor (70%% sparsity default)")
    args = parser.parse_args()

    # ── Device check ──────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("❌  CUDA not available. Run on a GPU machine.")
        return

    device   = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_gb   = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"\n{'='*62}")
    print(f"  Skip-Softmax GPU Eager Mode Comparison")
    print(f"  GPU      : {gpu_name}  ({gpu_gb:.1f} GB)")
    print(f"  Model    : {args.model_name}  ({args.n_layers} layers,  float16)")
    print(f"  ctx_len  : {args.ctx_len}   gen_len : {args.gen_len}")
    print(f"  TSF      : prefill={args.tsf_prefill}  decode={args.tsf_decode}")
    print(f"{'='*62}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ── Load base HF model on GPU in float16 ──────────────────────────────
    print("\nLoading base model on GPU (float16)...")
    base_hf = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        num_hidden_layers=args.n_layers,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map="cuda",
    )
    print(f"  GPU mem after load : {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    results = []

    # ── Variant 1: blocking only ───────────────────────────────────────────
    qaic_cfg_blocking = {"enable_blocking": True, "blocking_mode": "hqkv"}

    print("\nApplying blocking-only transform...")
    model_blocking = QEFFAutoModelForCausalLM(
        copy.deepcopy(base_hf),
        qaic_config=qaic_cfg_blocking,
    )
    model_blocking.transform(
        ctx_len=args.ctx_len,
        seq_len=args.prompt_len,
        batch_size=1,
        qaic_config=qaic_cfg_blocking,
    )

    print("Running blocking-only inference on GPU...")
    text1, t1, ms1 = run_blocking_model_on_gpu(
        model=model_blocking.model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        prompt_len=args.prompt_len,
        ctx_len=args.ctx_len,
        gen_len=args.gen_len,
        device=device,
    )

    print(f"\n  ── Blocking only ──────────────────────────────────")
    print(f"  Wall time     : {t1:.3f} s")
    print(f"  Per token     : {ms1:.2f} ms")
    print(f"  Tokens/sec    : {args.gen_len / t1:.2f}")
    print(f"  Generated     : {text1[:100]}{'...' if len(text1) > 100 else ''}")

    results.append({
        "label": "Blocking only (no skip-softmax)",
        "total_sec": t1, "per_token_ms": ms1,
        "tokens_per_sec": args.gen_len / t1,
        "skip_rate": 0.0, "generated": text1,
    })

    del model_blocking
    torch.cuda.empty_cache()

    # ── Variant 2: blocking + skip-softmax ────────────────────────────────
    qaic_cfg_skip = {
        "enable_blocking":                   True,
        "blocking_mode":                     "hqkv",
        "skip_softmax_scale_factor_prefill": args.tsf_prefill,
        "skip_softmax_scale_factor_decode":  args.tsf_decode,
    }

    print("\nApplying blocking + skip-softmax transform...")
    model_skip = QEFFAutoModelForCausalLM(
        copy.deepcopy(base_hf),
        qaic_config=qaic_cfg_skip,
    )
    model_skip.transform(
        ctx_len=args.ctx_len,
        seq_len=args.prompt_len,
        batch_size=1,
        qaic_config=qaic_cfg_skip,
    )

    print("Running blocking + skip-softmax inference on GPU...")
    _COUNTER.reset()
    original_urs = _patch_for_counting()

    text2, t2, ms2 = run_blocking_model_on_gpu(
        model=model_skip.model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        prompt_len=args.prompt_len,
        ctx_len=args.ctx_len,
        gen_len=args.gen_len,
        device=device,
    )
    _restore_counting(original_urs)

    print(f"\n  ── Blocking + skip-softmax ──────────────────────────")
    print(f"  Wall time     : {t2:.3f} s")
    print(f"  Per token     : {ms2:.2f} ms")
    print(f"  Tokens/sec    : {args.gen_len / t2:.2f}")
    print(f"  Softmax updates: {_COUNTER.calls_to_softmax_update}  "
          f"(lower = more blocks skipped via 'continue')")
    print(f"  Generated     : {text2[:100]}{'...' if len(text2) > 100 else ''}")

    results.append({
        "label": f"Blocking + skip-softmax (prefill={args.tsf_prefill}, decode={args.tsf_decode})",
        "total_sec": t2, "per_token_ms": ms2,
        "tokens_per_sec": args.gen_len / t2,
        "skip_rate": 0.0, "generated": text2,
    })

    del model_skip
    torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────────
    speedup = results[1]["tokens_per_sec"] / max(results[0]["tokens_per_sec"], 1e-9)

    print(f"\n{'='*62}")
    print(f"  SUMMARY  —  GPU Eager Mode  ({gpu_name})")
    print(f"{'='*62}")
    print(f"  {'Variant':48s}  {'tok/s':>7s}  {'ms/tok':>7s}")
    print(f"  {'─'*48}  {'─'*7}  {'─'*7}")
    for r in results:
        print(f"  {r['label'][:48]:48s}  "
              f"{r['tokens_per_sec']:7.2f}  "
              f"{r['per_token_ms']:7.2f}")

    print(f"\n  Speedup (skip-softmax vs blocking) : {speedup:.3f}x")
    if speedup > 1.05:
        print(f"  ✅ Skip-softmax shows gain on GPU eager mode.")
        print(f"     The 'continue' path physically skips V load + BMM2.")
    elif speedup > 0.97:
        print(f"  ⚠  Within noise floor — context may still be too short.")
        print(f"     Try --tsf-prefill 18471 --tsf-decode 852 (90%% sparsity).")
    else:
        print(f"  ⚠  CPU-GPU sync overhead per KV block dominates at this ctx_len.")
        print(f"     AI100 AOT avoids this sync — real gain will be visible there.")

    print(f"\n  Output match : "
          f"{'✅ identical' if text1 == text2 else '⚠  differ (expected at high sparsity)'}")
    print()


if __name__ == "__main__":
    main()
