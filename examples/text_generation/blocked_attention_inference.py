# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Blocked attention inference with optional skip softmax (BLASST).

Examples
--------
# Blocking only (no skip softmax)
python blocked_attention_inference.py --device-group 0,1,2,3

# Blocking + skip softmax, ~50% sparsity, separate prefill/decode thresholds
python blocked_attention_inference.py \
    --device-group 0,1,2,3 \
    --skip-softmax-scale-factor-prefill 587 \
    --skip-softmax-scale-factor-decode 16.5

# Compare against non-blocking baseline as well
python blocked_attention_inference.py \
    --device-group 0,1,2,3 \
    --skip-softmax-scale-factor-prefill 587 \
    --skip-softmax-scale-factor-decode 16.5 \
    --compare-non-blocking
"""

import argparse

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(
        description="Blocked attention inference with optional skip softmax",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Model / hardware ─────────────────────────────────────────────────────
    parser.add_argument(
        "--model-name", type=str, default="Qwen/Qwen3-8B",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--n-layers", type=int, default=1,
        help="Number of transformer layers to load (2 for fast iteration, -1 for full model)",
    )
    parser.add_argument("--prompt", type=str, default="My name is",
                        help="Input prompt")
    parser.add_argument("--prefill-seq-len", type=int, default=1,
                        help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=32768,
                        help="Context length — must be long enough to activate KV blocking")
    parser.add_argument("--generation-len", type=int, default=100,
                        help="Number of tokens to generate")
    parser.add_argument("--num-cores", type=int, default=16,
                        help="Number of NSP cores to compile for")
    parser.add_argument("--num-devices", type=int, default=8,
                        help="Number of AI100 devices for tensor-slicing")
    parser.add_argument(
        "--device-group",
        type=lambda s: [int(x) for x in s.strip("[]").split(",")],
        default=None,
        help="Device IDs (comma-separated) e.g. 0,1,2,3  or  [36,37,38,39]",
    )
    parser.add_argument(
        "--onnx-path", type=str, default=None,
        help="Path to a pre-exported ONNX file. Skips the JIT export step (saves 30+ min on large models). "
             "Use when the ONNX already exists from a prior run.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for compiled QPC files. Defaults to the framework default (testing_qeff/). "
             "Use separate directories when running two jobs in parallel to avoid cache collisions.",
    )

    # ── Blocking config ───────────────────────────────────────────────────────
    parser.add_argument(
        "--blocking-mode", type=str, default="hqkv",
        help="Blocking mode: kv | q | h | qkv | hqkv",
    )
    # Block size parameters — critical for skip-softmax gain.
    # Block size = past_seen_tokens / num_kv_blocks.
    # Larger blocks (fewer num_kv_blocks) = more compute saved per skip.
    # BLASST paper gains are measured at block_size ≥ 512 tokens.
    parser.add_argument(
        "--num-kv-blocks", type=int, default=None,
        help="Number of KV blocks. block_size = context_len / num_kv_blocks. "
             "Use 4-8 for large gains (512-1024 tokens per block). "
             "Default: auto-configured by the framework.",
    )
    parser.add_argument(
        "--num-q-blocks", type=int, default=None,
        help="Number of Q blocks (default 1 for decode).",
    )
    parser.add_argument(
        "--head-block-size", type=int, default=None,
        help="Number of attention heads per head block.",
    )
    parser.add_argument(
        "--compare-non-blocking", action="store_true",
        help="Also compile and run the non-blocked variant for comparison",
    )

    # ── Skip softmax (BLASST, arXiv:2512.12087) ───────────────────────────────
    # Threshold formula: λ = scale_factor / context_length
    # A KV block is skipped when:  block_max − running_max  <  ln(λ)
    #
    # Calibrated values for Qwen3-30B-A3B-Instruct-2507 (NVIDIA TRT-LLM):
    #   ~30% sparsity  prefill=105,   decode=2.3
    #   ~50% sparsity  prefill=587,   decode=16.5
    #   ~70% sparsity  prefill=3293,  decode=119
    #
    # Prefill and decode use different thresholds because their attention
    # sparsity patterns differ.  The split flags take priority over the
    # combined flag when both are provided.
    parser.add_argument(
        "--skip-softmax-scale-factor",
        type=float, default=None,
        help="Enable skip softmax with a single threshold for both prefill and decode. "
             "Use the split flags below for separate per-phase calibration.",
    )
    parser.add_argument(
        "--skip-softmax-scale-factor-prefill",
        type=float, default=None,
        help="Prefill threshold scale factor. Overrides --skip-softmax-scale-factor for prefill.",
    )
    parser.add_argument(
        "--skip-softmax-scale-factor-decode",
        type=float, default=None,
        help="Decode threshold scale factor. Overrides --skip-softmax-scale-factor for decode.",
    )

    args = parser.parse_args()

    # Determine the actual number of layers to load.
    # -1 means full model: do not pass num_hidden_layers so the config default is used.
    load_kwargs = {} if args.n_layers == -1 else {"num_hidden_layers": args.n_layers}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ── Optional non-blocking baseline ───────────────────────────────────────
    if args.compare_non_blocking:
        print("\n" + "=" * 60)
        print("  Running: non-blocking")
        print("=" * 60)
        model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
        qpc_path = model.compile(
            onnx_path=args.onnx_path,
            prefill_seq_len=args.prefill_seq_len,
            ctx_len=args.ctx_len,
            num_cores=args.num_cores,
            num_devices=args.num_devices,
            compile_dir=args.output_dir,
        )
        print(f"  Compiled : {qpc_path}")
        exec_info = model.generate(
            tokenizer=tokenizer,
            prompts=[args.prompt],
            device_id=args.device_group,
            generation_len=args.generation_len,
        )
        print(f"\n  Prompt    : {args.prompt}")
        print(f"  Generated : {exec_info.generated_texts[0]}")

    # ── Blocking (+ optional skip softmax) ───────────────────────────────────
    qaic_config = {"enable_blocking": True, "blocking_mode": args.blocking_mode}

    # Wire explicit block-size parameters if provided.
    # These override the auto-configurator and are critical for skip-softmax
    # experiments: larger blocks (smaller num_kv_blocks) mean more compute
    # saved per skipped block.
    if args.num_kv_blocks is not None:
        qaic_config["num_kv_blocks"] = args.num_kv_blocks
    if args.num_q_blocks is not None:
        qaic_config["num_q_blocks"] = args.num_q_blocks
    if args.head_block_size is not None:
        qaic_config["head_block_size"] = args.head_block_size

    # Wire skip-softmax threshold scale factors.
    # The combined field is the convenience shorthand; split prefill/decode
    # fields take priority in generic_blocked_attention_interface.
    if args.skip_softmax_scale_factor is not None:
        qaic_config["skip_softmax_scale_factor"] = args.skip_softmax_scale_factor
    if args.skip_softmax_scale_factor_prefill is not None:
        qaic_config["skip_softmax_scale_factor_prefill"] = args.skip_softmax_scale_factor_prefill
    if args.skip_softmax_scale_factor_decode is not None:
        qaic_config["skip_softmax_scale_factor_decode"] = args.skip_softmax_scale_factor_decode

    skip_active = (
        args.skip_softmax_scale_factor is not None
        or args.skip_softmax_scale_factor_prefill is not None
        or args.skip_softmax_scale_factor_decode is not None
    )
    run_label = f"blocking/{args.blocking_mode}" + (" + skip-softmax" if skip_active else "")

    print("\n" + "=" * 60)
    print(f"  Running: {run_label}")
    print("=" * 60)
    print(f"  qaic_config : {qaic_config}")

    model_blocked = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    qpc_path_blocked = model_blocked.compile(
        onnx_path=args.onnx_path,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        qaic_config=qaic_config,
        compile_dir=args.output_dir,
    )
    print(f"  Compiled : {qpc_path_blocked}")

    exec_info_blocked = model_blocked.generate(
        tokenizer=tokenizer,
        prompts=[args.prompt],
        device_id=args.device_group,
        generation_len=args.generation_len,
    )

    print(f"\n  Prompt    : {args.prompt}")
    print(f"  Generated : {exec_info_blocked.generated_texts[0]}")

    # ── Performance summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if args.compare_non_blocking:
        print("Performance — non-blocking:")
        print(exec_info)
        print()
    print(f"Performance — {run_label}:")
    print(exec_info_blocked)


if __name__ == "__main__":
    main()
