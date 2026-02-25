# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
"""
QEfficient WAN Pipeline Implementation

This module provides an optimized implementation of the WAN pipeline
for high-performance text-to-video generation on Qualcomm AI hardware.
The pipeline supports WAN 2.2 architectures with unified transformer.

TODO: 1. Update umt5 to Qaic; present running on cpu
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import WanPipeline
from tqdm import tqdm

from QEfficient.diffusers.models.transformers.transformer_wan import QEffWanUnifiedWrapper
from QEfficient.diffusers.pipelines.pipeline_module import QEffVAE, QEffWanUnifiedTransformer
from QEfficient.diffusers.pipelines.pipeline_utils import (
    ONNX_SUBFUNCTION_MODULE,
    ModulePerf,
    QEffPipelineOutput,
    calculate_latent_dimensions_with_frames,
    compile_modules_parallel,
    compile_modules_sequential,
    config_manager,
    set_execute_params,
)
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import constants
from QEfficient.utils.logging_utils import logger


class QEffWanPipeline:
    """
    QEfficient-optimized WAN pipeline for high-performance text-to-video generation.

    Supports both QAIC hardware inference (device='qaic') and standard GPU/CPU
    inference (device='cuda' or device='cpu') for testing and validation.
    """

    _hf_auto_class = WanPipeline

    def __init__(self, model, enable_first_cache=False, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.custom_config = None

        self.text_encoder = model.text_encoder

        self.unified_wrapper = QEffWanUnifiedWrapper(model.transformer, model.transformer_2)
        self.transformer = QEffWanUnifiedTransformer(self.unified_wrapper, enable_first_cache=enable_first_cache)

        self.vae_decoder = QEffVAE(model.vae, "decoder")
        self.modules = {"transformer": self.transformer, "vae_decoder": self.vae_decoder}

        self.tokenizer = model.tokenizer
        self.text_encoder.tokenizer = model.tokenizer
        self.scheduler = model.scheduler

        self.vae_decoder.model.forward = lambda latent_sample, return_dict: self.vae_decoder.model.decode(
            latent_sample, return_dict
        )

        self.vae_decoder.get_onnx_params = self.vae_decoder.get_video_onnx_params
        _, self.patch_height, self.patch_width = self.transformer.model.config.patch_size

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0 and (self._guidance_scale_2 is None or self._guidance_scale_2 > 1.0)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        enable_first_cache: bool = False,
        **kwargs,
    ):
        model = cls._hf_auto_class.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            **kwargs,
        )
        return cls(
            model=model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            enable_first_cache=enable_first_cache,
            **kwargs,
        )

    def export(self, export_dir=None, use_onnx_subfunctions=False):
        for module_name, module_obj in tqdm(self.modules.items(), desc="Exporting modules", unit="module"):
            example_inputs, dynamic_axes, output_names = module_obj.get_onnx_params()
            export_params = {
                "inputs": example_inputs,
                "output_names": output_names,
                "dynamic_axes": dynamic_axes,
                "export_dir": export_dir,
            }
            if use_onnx_subfunctions and module_name in ONNX_SUBFUNCTION_MODULE:
                export_params["use_onnx_subfunctions"] = True
            if module_obj.qpc_path is None:
                module_obj.export(**export_params)

    @staticmethod
    def get_default_config_path():
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/wan_config.json")

    def compile(
        self,
        compile_config=None,
        parallel=False,
        height=constants.WAN_ONNX_EXPORT_HEIGHT_180P,
        width=constants.WAN_ONNX_EXPORT_WIDTH_180P,
        num_frames=constants.WAN_ONNX_EXPORT_FRAMES,
        use_onnx_subfunctions=False,
    ):
        config_manager(self, config_source=compile_config, use_onnx_subfunctions=use_onnx_subfunctions)
        set_execute_params(self)

        if any(path is None for path in [self.transformer.onnx_path, self.vae_decoder.onnx_path]):
            self.export(use_onnx_subfunctions=use_onnx_subfunctions)

        cl, latent_height, latent_width, latent_frames = calculate_latent_dimensions_with_frames(
            height,
            width,
            num_frames,
            self.model.vae.config.scale_factor_spatial,
            self.model.vae.config.scale_factor_temporal,
            self.patch_height,
            self.patch_width,
        )
        specialization_updates = {
            "transformer": [
                {"cl": cl, "latent_height": latent_height, "latent_width": latent_width, "latent_frames": latent_frames},
                {"cl": cl, "latent_height": latent_height, "latent_width": latent_width, "latent_frames": latent_frames},
            ],
            "vae_decoder": {
                "latent_frames": latent_frames,
                "latent_height": latent_height,
                "latent_width": latent_width,
            },
        }

        logger.warning('For VAE compilation use QAIC_COMPILER_OPTS_UNSUPPORTED="-aic-hmx-conv3d" ')
        if parallel:
            compile_modules_parallel(self.modules, self.custom_config, specialization_updates)
        else:
            compile_modules_sequential(self.modules, self.custom_config, specialization_updates)

    def check_cache_conditions(
        self,
        new_first_block_residual: torch.Tensor,
        prev_first_block_residual: Optional[torch.Tensor],
        cache_threshold: float,
        cache_warmup_steps: int,
        current_step: int,
    ) -> bool:
        """
        Compute cache decision (returns bool).

        Cache is used when:
        1. Not in warmup period (current_step >= cache_warmup_steps)
        2. Previous residual exists (not first step)
        3. Residual similarity is below threshold
        """
        if current_step < cache_warmup_steps or prev_first_block_residual is None:
            return False

        diff = (new_first_block_residual - prev_first_block_residual).abs().mean()
        norm = new_first_block_residual.abs().mean()
        similarity = diff / (norm + 1e-8)

        is_similar = similarity.item() < cache_threshold

        if is_similar:
            print(f"  [Cache HIT ] step={current_step} similarity={similarity.item():.4f} < threshold={cache_threshold:.4f}")
        else:
            print(f"  [Cache MISS] step={current_step} similarity={similarity.item():.4f} >= threshold={cache_threshold:.4f}")

        return is_similar

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None]]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # QAIC-specific parameters
        custom_config_path: Optional[str] = None,
        use_onnx_subfunctions: bool = False,
        parallel_compile: bool = True,
        # Cache parameters
        cache_threshold: Optional[float] = None,
        cache_warmup_steps: Optional[int] = None,
        # Device selection: "qaic" for Qualcomm hardware, "cuda"/"cpu" for GPU/CPU testing
        device: str = "qaic",
    ):
        """
        Generate videos from text prompts.

        Args:
            device (str): Execution device. Use "cuda" or "cpu" for GPU/CPU testing,
                          "qaic" for Qualcomm AI hardware (default).
            cache_threshold (float, optional): Residual similarity threshold for cache.
                Lower = more aggressive caching. Only used when enable_first_cache=True.
            cache_warmup_steps (int, optional): Number of initial steps to skip caching.
        """
        cpu_device = "cpu"

        # ----------------------------------------------------------------
        # QAIC-only: compile models
        # ----------------------------------------------------------------
        if device == "qaic":
            self.compile(
                compile_config=custom_config_path,
                parallel=parallel_compile,
                use_onnx_subfunctions=use_onnx_subfunctions,
                height=height,
                width=width,
                num_frames=num_frames,
            )

        # ----------------------------------------------------------------
        # Shared: validate inputs
        # ----------------------------------------------------------------
        self.model.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if num_frames % self.model.vae.config.scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.model.vae.config.scale_factor_temporal}. Rounding."
            )
            num_frames = (
                num_frames // self.model.vae.config.scale_factor_temporal * self.model.vae.config.scale_factor_temporal
                + 1
            )
        num_frames = max(num_frames, 1)

        if self.model.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2 if guidance_scale_2 is not None else guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # ----------------------------------------------------------------
        # Shared: batch size
        # ----------------------------------------------------------------
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # ----------------------------------------------------------------
        # Shared: text encoding (always on CPU)
        # ----------------------------------------------------------------
        prompt_embeds, negative_prompt_embeds = self.model.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=cpu_device,
        )

        transformer_dtype = self.transformer.model.transformer_high.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # ----------------------------------------------------------------
        # Shared: timesteps and latents
        # ----------------------------------------------------------------
        self.scheduler.set_timesteps(num_inference_steps, device=cpu_device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.transformer.model.config.in_channels
        latents = self.model.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            cpu_device,
            generator,
            latents,
        )

        mask = torch.ones(latents.shape, dtype=torch.float32, device=cpu_device)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        if self.model.config.boundary_ratio is not None:
            boundary_timestep = self.model.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        # ----------------------------------------------------------------
        # QAIC-only: session setup
        # ----------------------------------------------------------------
        if device == "qaic":
            if self.transformer.qpc_session is None:
                self.transformer.qpc_session = QAICInferenceSession(
                    str(self.transformer.qpc_path), device_ids=self.transformer.device_ids
                )

            cl, _, _, _ = calculate_latent_dimensions_with_frames(
                height, width, num_frames,
                self.model.vae.config.scale_factor_spatial,
                self.model.vae.config.scale_factor_temporal,
                self.patch_height, self.patch_width,
            )
            output_buffer = {
                "output": np.random.rand(batch_size, cl, constants.WAN_DIT_OUT_CHANNELS).astype(np.int32),
            }
            self.transformer.qpc_session.set_buffers(output_buffer)
            self.transformer.qpc_session.skip_buffers(
                [
                    x
                    for x in self.transformer.qpc_session.input_names + self.transformer.qpc_session.output_names
                    if x.startswith("prev_") or x.endswith("_RetainedState")
                ]
            )

        # ----------------------------------------------------------------
        # GPU-only: move models to device, initialize cache state
        # ----------------------------------------------------------------
        else:
            logger.info(f"Running in GPU/CPU mode on device: {device}")
            self.unified_wrapper.to(device)
            self.model.vae.to(device)

        # ----------------------------------------------------------------
        # Shared: cache state tracking (CPU-side)
        # ----------------------------------------------------------------
        cache_enabled = getattr(self.unified_wrapper.transformer_high, "enable_first_cache", False)
        _cache_threshold = cache_threshold if cache_threshold is not None else 0.0
        _cache_warmup_steps = cache_warmup_steps if cache_warmup_steps is not None else 0

        prev_first_block_residual_high = None
        prev_first_block_residual_low = None
        # GPU-only: retained residuals managed in Python
        prev_remaining_blocks_residual_high = None
        prev_remaining_blocks_residual_low = None

        transformer_perf = []

        # ================================================================
        # Denoising loop
        # ================================================================
        with self.model.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self._interrupt:
                    continue

                self._current_timestep = t

                # Determine which model stage to use
                if boundary_timestep is None or t >= boundary_timestep:
                    current_model = self.transformer.model.transformer_high
                    current_guidance_scale = guidance_scale
                    model_type = torch.ones(1, dtype=torch.int64)   # shape[0]==1 → high noise
                else:
                    current_model = self.transformer.model.transformer_low
                    current_guidance_scale = guidance_scale_2
                    model_type = torch.ones(2, dtype=torch.int64)   # shape[0]==2 → low noise

                latent_model_input = latents.to(transformer_dtype)

                # Timestep preparation
                if self.model.config.expand_timesteps:
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                # Patch dimensions
                batch_size_step, num_channels, num_frames_step, h_step, w_step = latents.shape
                p_t, p_h, p_w = current_model.config.patch_size
                post_patch_num_frames = num_frames_step // p_t
                post_patch_height = h_step // p_h
                post_patch_width = w_step // p_w

                # Rotary embeddings (computed on CPU, moved to device as needed)
                rotary_emb = current_model.rope(latent_model_input)
                rotary_emb = torch.cat(rotary_emb, dim=0)  # [2, cl, 1, rotary_dim]
                ts_seq_len = None
                timestep_flat = timestep.flatten()

                # Conditioning embeddings
                temb, timestep_proj, encoder_hidden_states, _ = current_model.condition_embedder(
                    timestep_flat, prompt_embeds, encoder_hidden_states_image=None, timestep_seq_len=ts_seq_len
                )

                if self.do_classifier_free_guidance:
                    temb_neg, timestep_proj_neg, encoder_hidden_states_neg, _ = current_model.condition_embedder(
                        timestep_flat,
                        negative_prompt_embeds,
                        encoder_hidden_states_image=None,
                        timestep_seq_len=ts_seq_len,
                    )

                timestep_proj = timestep_proj.unflatten(1, (6, -1))

                # ============================================================
                # QAIC path
                # ============================================================
                if device == "qaic":
                    # Patch embedding + block[0] on CPU
                    hidden_states = current_model.patch_embedding(latents)
                    hidden_states = hidden_states.flatten(2).transpose(1, 2)

                    if model_type.shape[0] == 1:
                        print(f"Running high-noise model at step {i}, timestep {t}")
                        new_first_block_output = current_model.blocks[0](
                            hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                        )
                        new_first_block_residual = new_first_block_output - hidden_states
                        use_cache = self.check_cache_conditions(
                            new_first_block_residual,
                            prev_first_block_residual_high,
                            _cache_threshold,
                            _cache_warmup_steps,
                            i,
                        )
                        prev_first_block_residual_high = new_first_block_residual.detach()
                    else:
                        print(f"Running low-noise model at step {i}, timestep {t}")
                        new_first_block_output = current_model.blocks[0](
                            hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                        )
                        new_first_block_residual = new_first_block_output - hidden_states
                        use_cache = self.check_cache_conditions(
                            new_first_block_residual,
                            prev_first_block_residual_low,
                            _cache_threshold,
                            _cache_warmup_steps,
                            i,
                        )
                        prev_first_block_residual_low = new_first_block_residual.detach()

                    inputs_aic = {
                        "hidden_states": new_first_block_output.detach().numpy(),
                        "encoder_hidden_states": encoder_hidden_states.detach().numpy(),
                        "rotary_emb": rotary_emb.detach().numpy(),
                        "temb": temb.detach().numpy(),
                        "timestep_proj": timestep_proj.detach().numpy(),
                        "tsp": model_type.detach().numpy(),
                        "use_cache": np.array([1 if use_cache else 0], dtype=np.int64),
                    }

                    start_transformer_step_time = time.perf_counter()
                    outputs = self.transformer.qpc_session.run(inputs_aic)
                    end_transformer_step_time = time.perf_counter()
                    transformer_perf.append(end_transformer_step_time - start_transformer_step_time)
                    print(f"DIT step {i} time {end_transformer_step_time - start_transformer_step_time:.2f}s")

                    hidden_states = torch.tensor(outputs["output"])
                    hidden_states = hidden_states.reshape(
                        batch_size_step, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
                    )
                    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
                    noise_pred = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

                    if self.do_classifier_free_guidance:
                        inputs_aic2 = {
                            "hidden_states": new_first_block_output.detach().numpy(),
                            "encoder_hidden_states": encoder_hidden_states_neg.detach().numpy(),
                            "rotary_emb": rotary_emb.detach().numpy(),
                            "temb": temb_neg.detach().numpy(),
                            "timestep_proj": timestep_proj_neg.unflatten(1, (6, -1)).detach().numpy(),
                            "tsp": model_type.detach().numpy(),
                        }
                        start_transformer_step_time = time.perf_counter()
                        outputs2 = self.transformer.qpc_session.run(inputs_aic2)
                        end_transformer_step_time = time.perf_counter()
                        transformer_perf.append(end_transformer_step_time - start_transformer_step_time)

                        hidden_states2 = torch.tensor(outputs2["output"])
                        hidden_states2 = hidden_states2.reshape(
                            batch_size_step, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
                        )
                        hidden_states2 = hidden_states2.permute(0, 7, 1, 4, 2, 5, 3, 6)
                        noise_uncond = hidden_states2.flatten(6, 7).flatten(4, 5).flatten(2, 3)
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                # ============================================================
                # GPU / CPU path
                # ============================================================
                else:
                    is_high_noise = (model_type.shape[0] == 1)
                    print(f"[GPU] Step {i}, t={t:.1f}, {'high' if is_high_noise else 'low'}-noise model")

                    # Move current_model to device (already moved via unified_wrapper.to(device))
                    # Patch embedding + block[0] on device
                    latent_input_dev = latents.to(device, dtype=transformer_dtype)
                    hidden_states = current_model.patch_embedding(latent_input_dev)
                    hidden_states = hidden_states.flatten(2).transpose(1, 2)  # (B, cl, hidden_dim)

                    # Split rotary_emb into tuple for block forward
                    rotary_emb_dev = rotary_emb.to(device)
                    rotary_emb_tuple = torch.split(rotary_emb_dev, 1, dim=0)

                    # Run block[0]
                    new_first_block_output = current_model.blocks[0](
                        hidden_states,
                        encoder_hidden_states.to(device),
                        timestep_proj.to(device),
                        rotary_emb_tuple,
                    )
                    new_first_block_residual = new_first_block_output - hidden_states

                    # Cache decision (similarity check on CPU)
                    prev_first_residual = (
                        prev_first_block_residual_high if is_high_noise else prev_first_block_residual_low
                    )
                    use_cache_bool = self.check_cache_conditions(
                        new_first_block_residual.cpu().float(),
                        prev_first_residual.cpu().float() if prev_first_residual is not None else None,
                        _cache_threshold,
                        _cache_warmup_steps,
                        i,
                    )
                    use_cache_tensor = torch.tensor(
                        [1 if use_cache_bool else 0], dtype=torch.int64, device=device
                    )

                    # Initialize residual cache tensors on first call
                    if prev_remaining_blocks_residual_high is None:
                        prev_remaining_blocks_residual_high = torch.zeros_like(new_first_block_output)
                    if prev_remaining_blocks_residual_low is None:
                        prev_remaining_blocks_residual_low = torch.zeros_like(new_first_block_output)

                    # Forward through unified wrapper: blocks[1..N] + norm_out + proj_out
                    start_t = time.perf_counter()
                    with torch.no_grad():
                        outputs = self.unified_wrapper(
                            hidden_states=new_first_block_output,
                            encoder_hidden_states=encoder_hidden_states.to(device),
                            rotary_emb=rotary_emb_dev,
                            temb=temb.to(device),
                            timestep_proj=timestep_proj.to(device),
                            tsp=model_type.to(device),
                            prev_remaining_blocks_residual_high=prev_remaining_blocks_residual_high,
                            prev_remaining_blocks_residual_low=prev_remaining_blocks_residual_low,
                            use_cache=use_cache_tensor,
                        )
                    end_t = time.perf_counter()
                    transformer_perf.append(end_t - start_t)
                    print(f"  Transformer blocks time: {end_t - start_t:.3f}s")

                    # Unpack outputs
                    if cache_enabled:
                        noise_pred_patched, new_remaining_high, new_remaining_low = outputs
                        prev_remaining_blocks_residual_high = new_remaining_high.detach()
                        prev_remaining_blocks_residual_low = new_remaining_low.detach()
                    else:
                        noise_pred_patched = outputs

                    # Update first-block residual cache
                    if is_high_noise:
                        prev_first_block_residual_high = new_first_block_residual.detach()
                    else:
                        prev_first_block_residual_low = new_first_block_residual.detach()

                    # Reshape from patch format [B, cl, out_ch] to video format
                    noise_pred = noise_pred_patched.reshape(
                        batch_size_step,
                        post_patch_num_frames,
                        post_patch_height,
                        post_patch_width,
                        p_t, p_h, p_w, -1,
                    )
                    noise_pred = noise_pred.permute(0, 7, 1, 4, 2, 5, 3, 6)
                    noise_pred = noise_pred.flatten(6, 7).flatten(4, 5).flatten(2, 3)
                    noise_pred = noise_pred.to(latents.dtype).cpu()

                    # Classifier-free guidance (if enabled)
                    if self.do_classifier_free_guidance:
                        with torch.no_grad():
                            outputs_uncond = self.unified_wrapper(
                                hidden_states=new_first_block_output,
                                encoder_hidden_states=encoder_hidden_states_neg.to(device),
                                rotary_emb=rotary_emb_dev,
                                temb=temb_neg.to(device),
                                timestep_proj=timestep_proj_neg.unflatten(1, (6, -1)).to(device),
                                tsp=model_type.to(device),
                                prev_remaining_blocks_residual_high=prev_remaining_blocks_residual_high,
                                prev_remaining_blocks_residual_low=prev_remaining_blocks_residual_low,
                                use_cache=torch.tensor([0], dtype=torch.int64, device=device),
                            )
                        if cache_enabled:
                            noise_uncond_patched = outputs_uncond[0]
                        else:
                            noise_uncond_patched = outputs_uncond

                        noise_uncond = noise_uncond_patched.reshape(
                            batch_size_step,
                            post_patch_num_frames,
                            post_patch_height,
                            post_patch_width,
                            p_t, p_h, p_w, -1,
                        )
                        noise_uncond = noise_uncond.permute(0, 7, 1, 4, 2, 5, 3, 6)
                        noise_uncond = noise_uncond.flatten(6, 7).flatten(4, 5).flatten(2, 3)
                        noise_uncond = noise_uncond.to(latents.dtype).cpu()
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                # ============================================================
                # Shared: scheduler step
                # ============================================================
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        # ================================================================
        # VAE decode
        # ================================================================
        vae_decoder_perf = 0.0
        if not output_type == "latent":
            latents = latents.to(self.model.vae.dtype if hasattr(self.model.vae, "dtype") else torch.float32)

            # Denormalize latents
            latents_mean = (
                torch.tensor(self.model.vae.config.latents_mean)
                .view(1, self.model.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.model.vae.config.latents_std).view(
                1, self.model.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean

            # ----------------------------------------------------------
            # QAIC VAE decode
            # ----------------------------------------------------------
            if device == "qaic":
                if self.vae_decoder.qpc_session is None:
                    self.vae_decoder.qpc_session = QAICInferenceSession(
                        str(self.vae_decoder.qpc_path), device_ids=self.vae_decoder.device_ids
                    )

                output_buffer = {
                    "sample": np.random.rand(batch_size, 3, num_frames, height, width).astype(np.int32)
                }
                inputs = {"latent_sample": latents.numpy()}

                start_decode_time = time.perf_counter()
                video_out = self.vae_decoder.qpc_session.run(inputs)
                end_decode_time = time.perf_counter()
                vae_decoder_perf = end_decode_time - start_decode_time

                video_tensor = torch.from_numpy(video_out["sample"])
                video = self.model.video_processor.postprocess_video(video_tensor)

            # ----------------------------------------------------------
            # GPU / CPU VAE decode
            # ----------------------------------------------------------
            else:
                latents_dev = latents.to(device)
                start_decode_time = time.perf_counter()
                with torch.no_grad():
                    video_tensor = self.model.vae.decode(latents_dev, return_dict=False)[0]
                end_decode_time = time.perf_counter()
                vae_decoder_perf = end_decode_time - start_decode_time
                print(f"VAE decode time: {vae_decoder_perf:.3f}s")

                video_tensor = video_tensor.cpu()
                video = self.model.video_processor.postprocess_video(video_tensor)
        else:
            video = latents

        # ================================================================
        # Collect performance metrics and return
        # ================================================================
        perf_data = {
            "transformer": transformer_perf,
            "vae_decoder": vae_decoder_perf,
        }
        perf_metrics = [ModulePerf(module_name=name, perf=perf_data[name]) for name in perf_data.keys()]

        return QEffPipelineOutput(
            pipeline_module=perf_metrics,
            images=video,
        )
