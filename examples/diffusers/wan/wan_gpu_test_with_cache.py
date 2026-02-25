# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
WAN GPU Test with First Block Cache

This script tests the First Block Cache implementation on GPU/CPU
using the QEffWanPipeline with device="cuda" (or "cpu").

Usage:
    python wan_gpu_test_with_cache.py
    python wan_gpu_test_with_cache.py --device cpu
    python wan_gpu_test_with_cache.py --no-cache
"""

import argparse
import time

import torch
from diffusers.utils import export_to_video

from QEfficient import QEffWanPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="WAN GPU test with First Block Cache")
    parser.add_argument(
        "--model",
        type=str,
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="Model ID or local path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable first block cache",
    )
    parser.add_argument(
        "--cache-threshold",
        type=float,
        default=0.1,
        help="Cache similarity threshold (default: 0.1)",
    )
    parser.add_argument(
        "--cache-warmup-steps",
        type=int,
        default=3,
        help="Number of warmup steps before caching (default: 3)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=10,
        help="Number of denoising steps (default: 10)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=96,
        help="Video height in pixels (default: 96 for fast testing)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=160,
        help="Video width in pixels (default: 160 for fast testing)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=9,
        help="Number of frames (default: 9 for fast testing)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cat playing in a sunny garden, high quality video",
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_gpu_test.mp4",
        help="Output video file path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    enable_cache = not args.no_cache

    print("\n" + "=" * 70)
    print("WAN GPU Test with First Block Cache")
    print("=" * 70)
    print(f"  Model:           {args.model}")
    print(f"  Device:          {args.device}")
    print(f"  Cache enabled:   {enable_cache}")
    if enable_cache:
        print(f"  Cache threshold: {args.cache_threshold}")
        print(f"  Cache warmup:    {args.cache_warmup_steps} steps")
    print(f"  Resolution:      {args.width}x{args.height}")
    print(f"  Frames:          {args.num_frames}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Prompt:          {args.prompt[:80]}...")
    print("=" * 70 + "\n")

    # ----------------------------------------------------------------
    # Load pipeline
    # ----------------------------------------------------------------
    print("Loading WAN 2.2 pipeline...")
    t0 = time.perf_counter()
    pipeline = QEffWanPipeline.from_pretrained(
        args.model,
        enable_first_cache=enable_cache,
    )
    print(f"Pipeline loaded in {time.perf_counter() - t0:.1f}s\n")

    # ----------------------------------------------------------------
    # Run inference on GPU
    # ----------------------------------------------------------------
    print(f"Running inference on {args.device}...")
    t_start = time.perf_counter()

    output = pipeline(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=1.0,       # No CFG for faster testing
        guidance_scale_2=1.0,
        generator=torch.manual_seed(args.seed),
        cache_threshold=args.cache_threshold if enable_cache else None,
        cache_warmup_steps=args.cache_warmup_steps if enable_cache else None,
        device=args.device,       # <-- GPU/CPU mode
    )

    t_total = time.perf_counter() - t_start

    # ----------------------------------------------------------------
    # Save output
    # ----------------------------------------------------------------
    frames = output.images[0]
    export_to_video(frames, args.output, fps=8)

    # ----------------------------------------------------------------
    # Print results
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Output saved to: {args.output}")
    print(f"  Total time:      {t_total:.2f}s")
    print()
    print(output)  # Uses QEffPipelineOutput.__repr__ for detailed metrics
    print("=" * 70)


if __name__ == "__main__":
    main()
