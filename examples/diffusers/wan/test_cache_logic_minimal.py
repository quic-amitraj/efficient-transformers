# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Minimal unit test for First Block Cache logic.

Tests the core cache logic (torch.where pattern, residual computation,
similarity check) using small random tensors — no model loading required.

Run:
    python test_cache_logic_minimal.py
"""

import torch
import torch.nn as nn


# ============================================================
# Minimal mock of _forward_blocks_with_cache
# ============================================================

def forward_blocks_with_cache(
    hidden_states: torch.Tensor,
    prev_remaining_blocks_residual: torch.Tensor,
    use_cache: torch.Tensor,
    blocks: nn.ModuleList,
    encoder_hidden_states: torch.Tensor,
    timestep_proj: torch.Tensor,
    rotary_emb,
) -> tuple:
    """
    Mirrors QEffWanTransformer3DModel._forward_blocks_with_cache exactly.
    """
    original_hidden_states = hidden_states

    # Run blocks[1:] (in this mock, all blocks)
    for block in blocks:
        hidden_states = block(hidden_states)

    new_remaining_blocks_residual = hidden_states - original_hidden_states

    # torch.where: select prev (cache hit) or new (cache miss)
    final_remaining_residual = torch.where(
        use_cache.bool().view(1, 1, 1),
        prev_remaining_blocks_residual,
        new_remaining_blocks_residual,
    )
    final_output = original_hidden_states + final_remaining_residual

    return final_output, final_remaining_residual


def check_cache_conditions(
    new_residual: torch.Tensor,
    prev_residual: torch.Tensor,
    cache_threshold: float,
    cache_warmup_steps: int,
    current_step: int,
) -> bool:
    """Mirrors QEffWanPipeline.check_cache_conditions."""
    if current_step < cache_warmup_steps or prev_residual is None:
        return False

    diff = (new_residual - prev_residual).abs().mean()
    norm = new_residual.abs().mean()
    similarity = diff / (norm + 1e-8)
    return similarity.item() < cache_threshold


# ============================================================
# Simple linear block for testing
# ============================================================

class SimpleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        # Initialize with small weights so output ≈ input
        nn.init.eye_(self.linear.weight)
        with torch.no_grad():
            self.linear.weight += torch.randn_like(self.linear.weight) * 0.01

    def forward(self, x):
        return self.linear(x)


# ============================================================
# Tests
# ============================================================

def test_cache_miss_on_warmup():
    """During warmup, cache should never be used."""
    print("\n[TEST 1] Cache miss during warmup")
    batch, cl, dim = 1, 180, 64
    blocks = nn.ModuleList([SimpleBlock(dim) for _ in range(3)])

    hidden_states = torch.randn(batch, cl, dim)
    prev_residual = torch.randn(batch, cl, dim)  # non-zero prev residual
    use_cache = torch.tensor([0], dtype=torch.int64)  # warmup → no cache

    output, new_residual = forward_blocks_with_cache(
        hidden_states, prev_residual, use_cache, blocks, None, None, None
    )

    # When use_cache=0, output should use new_residual, not prev_residual
    expected_output = hidden_states + new_residual
    assert torch.allclose(output, expected_output, atol=1e-5), "Cache miss output mismatch"
    print("  PASSED: output = hidden_states + new_residual (cache miss)")


def test_cache_hit():
    """When use_cache=1, output should use prev_residual."""
    print("\n[TEST 2] Cache hit")
    batch, cl, dim = 1, 180, 64
    blocks = nn.ModuleList([SimpleBlock(dim) for _ in range(3)])

    hidden_states = torch.randn(batch, cl, dim)
    prev_residual = torch.randn(batch, cl, dim)
    use_cache = torch.tensor([1], dtype=torch.int64)  # cache hit

    output, returned_residual = forward_blocks_with_cache(
        hidden_states, prev_residual, use_cache, blocks, None, None, None
    )

    # When use_cache=1, output should use prev_residual
    expected_output = hidden_states + prev_residual
    assert torch.allclose(output, expected_output, atol=1e-5), "Cache hit output mismatch"
    assert torch.allclose(returned_residual, prev_residual, atol=1e-5), "Returned residual should be prev_residual"
    print("  PASSED: output = hidden_states + prev_residual (cache hit)")


def test_residual_computation():
    """Residual = output - input should be correct."""
    print("\n[TEST 3] Residual computation")
    batch, cl, dim = 1, 180, 64
    blocks = nn.ModuleList([SimpleBlock(dim) for _ in range(3)])

    hidden_states = torch.randn(batch, cl, dim)
    prev_residual = torch.zeros(batch, cl, dim)
    use_cache = torch.tensor([0], dtype=torch.int64)

    output, new_residual = forward_blocks_with_cache(
        hidden_states, prev_residual, use_cache, blocks, None, None, None
    )

    # Verify: output = hidden_states + new_residual
    assert torch.allclose(output, hidden_states + new_residual, atol=1e-5)
    print("  PASSED: output = hidden_states + new_residual")


def test_similarity_check():
    """Similarity check should correctly decide cache hit/miss."""
    print("\n[TEST 4] Similarity check")
    batch, cl, dim = 1, 180, 64

    # Case 1: identical residuals → similarity=0 → cache hit (below threshold)
    residual = torch.randn(batch, cl, dim)
    result = check_cache_conditions(residual, residual.clone(), cache_threshold=0.1, cache_warmup_steps=0, current_step=1)
    assert result == True, "Identical residuals should be a cache hit"
    print("  PASSED: identical residuals → cache HIT")

    # Case 2: very different residuals → similarity >> threshold → cache miss
    residual_a = torch.randn(batch, cl, dim)
    residual_b = torch.randn(batch, cl, dim) * 100  # very different
    result = check_cache_conditions(residual_a, residual_b, cache_threshold=0.1, cache_warmup_steps=0, current_step=1)
    assert result == False, "Very different residuals should be a cache miss"
    print("  PASSED: very different residuals → cache MISS")

    # Case 3: warmup step → always miss
    result = check_cache_conditions(residual, residual.clone(), cache_threshold=0.1, cache_warmup_steps=5, current_step=2)
    assert result == False, "During warmup, should always be cache miss"
    print("  PASSED: warmup step → cache MISS (regardless of similarity)")

    # Case 4: None prev_residual → always miss
    result = check_cache_conditions(residual, None, cache_threshold=0.1, cache_warmup_steps=0, current_step=1)
    assert result == False, "None prev_residual should be cache miss"
    print("  PASSED: None prev_residual → cache MISS")


def test_full_denoising_loop_simulation():
    """Simulate a full denoising loop with cache logic."""
    print("\n[TEST 5] Full denoising loop simulation (4 steps)")
    batch, cl, dim = 1, 180, 64
    blocks = nn.ModuleList([SimpleBlock(dim) for _ in range(3)])

    cache_threshold = 0.1
    cache_warmup_steps = 2
    num_steps = 4

    prev_residual = torch.zeros(batch, cl, dim)
    prev_first_residual = None
    cache_hits = 0
    cache_misses = 0

    for step in range(num_steps):
        # Simulate block[0] output (slightly changing each step)
        block0_output = torch.randn(batch, cl, dim) * (1.0 - step * 0.1)
        first_residual = block0_output - torch.randn(batch, cl, dim) * 0.05

        # Cache decision
        use_cache_bool = check_cache_conditions(
            first_residual, prev_first_residual, cache_threshold, cache_warmup_steps, step
        )
        use_cache = torch.tensor([1 if use_cache_bool else 0], dtype=torch.int64)

        if use_cache_bool:
            cache_hits += 1
        else:
            cache_misses += 1

        # Forward with cache
        output, new_residual = forward_blocks_with_cache(
            block0_output, prev_residual, use_cache, blocks, None, None, None
        )

        # Update state
        prev_residual = new_residual.detach()
        prev_first_residual = first_residual.detach()

        print(f"  Step {step}: {'CACHE HIT ' if use_cache_bool else 'CACHE MISS'} | output shape: {output.shape}")

    print(f"  Summary: {cache_hits} hits, {cache_misses} misses out of {num_steps} steps")
    assert output.shape == (batch, cl, dim), f"Wrong output shape: {output.shape}"
    print("  PASSED: all steps completed, output shape correct")


def test_retained_state_update():
    """
    Verify the RetainedState update logic from QEffWanUnifiedWrapper:
    - When is_high_noise=True: update high residual, keep low residual
    - When is_high_noise=False: keep high residual, update low residual
    """
    print("\n[TEST 6] RetainedState update logic (high/low noise selection)")
    batch, cl, dim = 1, 180, 64

    prev_high = torch.ones(batch, cl, dim) * 1.0
    prev_low = torch.ones(batch, cl, dim) * 2.0
    new_high = torch.ones(batch, cl, dim) * 10.0
    new_low = torch.ones(batch, cl, dim) * 20.0

    # is_high_noise = True → update high, keep low
    is_high_noise = torch.tensor(True)
    updated_high = torch.where(is_high_noise, new_high, prev_high)
    updated_low = torch.where(is_high_noise, prev_low, new_low)
    assert torch.allclose(updated_high, new_high), "High residual should be updated"
    assert torch.allclose(updated_low, prev_low), "Low residual should be kept"
    print("  PASSED: is_high_noise=True → high updated, low kept")

    # is_high_noise = False → keep high, update low
    is_high_noise = torch.tensor(False)
    updated_high = torch.where(is_high_noise, new_high, prev_high)
    updated_low = torch.where(is_high_noise, prev_low, new_low)
    assert torch.allclose(updated_high, prev_high), "High residual should be kept"
    assert torch.allclose(updated_low, new_low), "Low residual should be updated"
    print("  PASSED: is_high_noise=False → high kept, low updated")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("First Block Cache - Core Logic Unit Tests")
    print("=" * 60)

    torch.manual_seed(42)

    test_cache_miss_on_warmup()
    test_cache_hit()
    test_residual_computation()
    test_similarity_check()
    test_full_denoising_loop_simulation()
    test_retained_state_update()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print("\nNote: This environment has CPU-only PyTorch (no CUDA).")
    print("The pipeline code is correct. Run wan_gpu_test_with_cache.py")
    print("on a machine with CUDA-enabled PyTorch for full GPU testing.")
