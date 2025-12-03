# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pytest
import torch
from diffusers import FluxPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from QEfficient import QEFFFluxPipeline
from QEfficient.diffusers.pipelines.pipeline_utils import (
    ModulePerf,
    QEffPipelineOutput,
    config_manager,
    set_module_device_ids,
)
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils._utils import load_json
from tests.diffusers.diffusers_utils import DiffusersTestUtils, MADValidator

# Test Configuration for 256x256 resolution with 2 layers # update mad tolerance
INITIAL_TEST_CONFIG = load_json("tests/diffusers/flux_test_config.json")


def flux_pipeline_call_with_mad_validation(
    pipeline,
    pytorch_pipeline,
    height: int = 256,
    width: int = 256,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt: Union[str, List[str]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    true_cfg_scale: float = 1.0,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    custom_config_path: Optional[str] = None,
    parallel_compile: bool = False,
    mad_tolerances: Dict[str, float] = None,
):
    """
    Pipeline call function that replicates the exact flow of pipeline_flux.py.__call__()
    while adding comprehensive MAD validation at each step.

    This function follows the EXACT same structure as QEFFFluxPipeline.__call__()
    but adds MAD validation hooks throughout the process.
    """
    # Initialize MAD validator
    mad_validator = MADValidator(tolerances=mad_tolerances)

    device = "cpu"

    # Step 1: Load configuration, export and compile models if needed
    if custom_config_path is not None:
        config_manager(pipeline, config_source=custom_config_path)
        set_module_device_ids(pipeline)

    pipeline.compile(compile_config=custom_config_path, parallel=parallel_compile, height=height, width=width)

    # Set device IDs for all modules based on configuration
    set_module_device_ids(pipeline)

    # Validate all inputs
    pipeline.model.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    # Set pipeline attributes
    pipeline._guidance_scale = guidance_scale
    pipeline._interrupt = False

    # Step 2: Determine batch size from inputs
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # Step 3: Encode prompts with both text encoders
    has_neg_prompt = negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
    )
    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

    # Use pipeline's encode_prompt method
    (prompt_embeds, pooled_prompt_embeds, text_ids, text_encoder_perf) = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )

    # MAD Validation for Text Encoders
    print("🔍 Performing MAD validation for text encoders...")

    # CLIP Text Encoder MAD validation #TODO: discuss and optimise
    prompt_list = [prompt] if isinstance(prompt, str) else prompt
    text_inputs = pipeline.tokenizer(
        prompt_list,
        padding="max_length",
        max_length=pipeline.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )

    clip_pytorch_output = pytorch_pipeline.text_encoder(text_inputs.input_ids, output_hidden_states=False)
    clip_pt_pooled = clip_pytorch_output["pooler_output"].detach().cpu().numpy()

    # Get QAIC output (re-run to get raw output for comparison)
    if pipeline.text_encoder.qpc_session is not None:
        aic_text_input = {"input_ids": text_inputs.input_ids.numpy().astype(np.int64)}
        clip_qaic_output = pipeline.text_encoder.qpc_session.run(aic_text_input)
        clip_qaic_pooled = clip_qaic_output["pooler_output"]

        mad_validator.validate_module_mad(
            clip_pt_pooled, clip_qaic_pooled, module_name="clip_text_encoder"
        )  # make sure map module with config

    # T5 Text Encoder MAD validation
    prompt_2_list = prompt_2 or prompt_list
    prompt_2_list = [prompt_2_list] if isinstance(prompt_2_list, str) else prompt_2_list
    text_inputs_t5 = pipeline.text_encoder_2.tokenizer(
        prompt_2_list,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    t5_pytorch_output = pytorch_pipeline.text_encoder_2(text_inputs_t5.input_ids, output_hidden_states=False)
    t5_pt_hidden = t5_pytorch_output["last_hidden_state"].detach().cpu().numpy()
    # Get QAIC output (re-run to get raw output for comparison)
    if pipeline.text_encoder_2.qpc_session is not None:
        aic_text_input_t5 = {"input_ids": text_inputs_t5.input_ids.numpy().astype(np.int64)}
        t5_qaic_output = pipeline.text_encoder_2.qpc_session.run(aic_text_input_t5)
        t5_qaic_hidden = t5_qaic_output["last_hidden_state"]
        mad_validator.validate_module_mad(t5_pt_hidden, t5_qaic_hidden, "t5_text_encoder")

    # Encode negative prompts if using true classifier-free guidance
    if do_true_cfg:
        (
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            negative_text_ids,
        ) = pipeline.encode_prompt(
            prompt=negative_prompt,
            prompt_2=negative_prompt_2,
            prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

    # Step 4: Prepare timesteps for denoising
    timesteps, num_inference_steps = retrieve_timesteps(pipeline.scheduler, num_inference_steps, device, timesteps)
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)
    pipeline._num_timesteps = len(timesteps)

    # Step 5: Prepare initial latents
    num_channels_latents = pipeline.transformer.model.config.in_channels // 4
    latents, latent_image_ids = pipeline.model.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # Step 6: Initialize transformer inference session
    if pipeline.transformer.qpc_session is None:
        pipeline.transformer.qpc_session = QAICInferenceSession(
            str(pipeline.transformer.qpc_path), device_ids=pipeline.transformer.device_ids
        )

    # Calculate compressed latent dimension (cl) for transformer buffer allocation
    from QEfficient.diffusers.pipelines.pipeline_utils import calculate_compressed_latent_dimension

    cl, _, _ = calculate_compressed_latent_dimension(height, width, pipeline.model.vae_scale_factor)

    # Allocate output buffer for transformer
    output_buffer = {
        "output": np.random.rand(batch_size, cl, pipeline.transformer.model.config.in_channels).astype(np.float32),
    }
    pipeline.transformer.qpc_session.set_buffers(output_buffer)

    transformer_perf = []
    pipeline.scheduler.set_begin_index(0)

    # Step 7: Denoising loop (EXACTLY like pipeline with MAD validation added)
    with pipeline.model.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipeline._interrupt:
                continue

            # Prepare timestep embedding
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            temb = pipeline.transformer.model.time_text_embed(timestep, pooled_prompt_embeds)

            # Compute AdaLN embeddings for dual transformer blocks
            adaln_emb = []
            for block_idx in range(len(pipeline.transformer.model.transformer_blocks)):
                block = pipeline.transformer.model.transformer_blocks[block_idx]
                f1 = block.norm1.linear(block.norm1.silu(temb)).chunk(6, dim=1)
                f2 = block.norm1_context.linear(block.norm1_context.silu(temb)).chunk(6, dim=1)
                adaln_emb.append(torch.cat(list(f1) + list(f2)))
            adaln_dual_emb = torch.stack(adaln_emb)

            # Compute AdaLN embeddings for single transformer blocks
            adaln_emb = []
            for block_idx in range(len(pipeline.transformer.model.single_transformer_blocks)):
                block = pipeline.transformer.model.single_transformer_blocks[block_idx]
                f1 = block.norm.linear(block.norm.silu(temb)).chunk(3, dim=1)
                adaln_emb.append(torch.cat(list(f1)))
            adaln_single_emb = torch.stack(adaln_emb)

            # Compute output AdaLN embedding
            temp = pipeline.transformer.model.norm_out
            adaln_out = temp.linear(temp.silu(temb))

            # Normalize timestep to [0, 1] range
            timestep = timestep / 1000

            # Prepare all inputs for transformer inference
            inputs_aic = {
                "hidden_states": latents.detach().numpy(),
                "encoder_hidden_states": prompt_embeds.detach().numpy(),
                "pooled_projections": pooled_prompt_embeds.detach().numpy(),
                "timestep": timestep.detach().numpy(),
                "img_ids": latent_image_ids.detach().numpy(),
                "txt_ids": text_ids.detach().numpy(),
                "adaln_emb": adaln_dual_emb.detach().numpy(),
                "adaln_single_emb": adaln_single_emb.detach().numpy(),
                "adaln_out": adaln_out.detach().numpy(),
            }

            # MAD Validation for Transformer - PyTorch reference inference
            noise_pred_torch = pytorch_pipeline.transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                timestep=torch.tensor(timestep),
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                return_dict=False,
            )[0]

            # Run transformer inference and measure time
            start_transformer_step_time = time.time()
            outputs = pipeline.transformer.qpc_session.run(inputs_aic)
            end_transformer_step_time = time.time()
            transformer_perf.append(end_transformer_step_time - start_transformer_step_time)

            noise_pred = torch.from_numpy(outputs["output"])

            # Transformer MAD validation
            mad_validator.validate_module_mad(
                noise_pred_torch.detach().cpu().numpy(),
                outputs["output"],
                "transformer",
                f"step {i} (t={t.item():.1f})",
            )

            # Update latents using scheduler
            latents_dtype = latents.dtype
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # Handle dtype mismatch
            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

            # Execute callback if provided
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(pipeline, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # Update progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()

    # Step 8: Decode latents to images (EXACTLY like pipeline with MAD validation added)
    if output_type == "latent":
        image = latents
        vae_decode_perf = 0.0  # No VAE decoding for latent output
    else:
        # Unpack and denormalize latents
        latents = pipeline.model._unpack_latents(latents, height, width, pipeline.model.vae_scale_factor)

        # Denormalize latents
        latents = (latents / pipeline.vae_decode.model.scaling_factor) + pipeline.vae_decode.model.shift_factor
        # Initialize VAE decoder inference session
        if pipeline.vae_decode.qpc_session is None:
            pipeline.vae_decode.qpc_session = QAICInferenceSession(
                str(pipeline.vae_decode.qpc_path), device_ids=pipeline.vae_decode.device_ids
            )

        # Allocate output buffer for VAE decoder
        output_buffer = {"sample": np.random.rand(batch_size, 3, height, width).astype(np.float32)}
        pipeline.vae_decode.qpc_session.set_buffers(output_buffer)

        # MAD Validation for VAE
        # PyTorch reference inference
        image_torch = pytorch_pipeline.vae.decode(latents, return_dict=False)[0]

        # Run VAE decoder inference and measure time
        inputs = {"latent_sample": latents.numpy()}
        start_decode_time = time.time()
        image = pipeline.vae_decode.qpc_session.run(inputs)
        end_decode_time = time.time()
        vae_decode_perf = end_decode_time - start_decode_time

        # VAE MAD validation
        mad_validator.validate_module_mad(image_torch.detach().cpu().numpy(), image["sample"], "vae_decoder")

        # Post-process image
        image_tensor = torch.from_numpy(image["sample"])
        image = pipeline.model.image_processor.postprocess(image_tensor, output_type=output_type)

    # Build performance metrics
    perf_metrics = [
        ModulePerf(module_name="text_encoder", perf=text_encoder_perf[0]),
        ModulePerf(module_name="text_encoder_2", perf=text_encoder_perf[1]),
        ModulePerf(module_name="transformer", perf=transformer_perf),
        ModulePerf(module_name="vae_decoder", perf=vae_decode_perf),
    ]

    # formatting MAD summary
    mad_results = mad_validator.get_summary()

    # Return QEffPipelineOutput + MAD results
    if not return_dict:
        return (image, mad_results)

    return QEffPipelineOutput(
        pipeline_module=perf_metrics,
        images=image,
    ), mad_results


@pytest.fixture(scope="session")
def flux_pipeline():
    """Setup compiled Flux pipeline for testing"""
    config = INITIAL_TEST_CONFIG["model_setup"]

    pipeline = QEFFFluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        use_onnx_subfunctions=config["use_onnx_subfunctions"],
    )

    # Reduce to 2 layers for testing
    original_blocks = pipeline.transformer.model.transformer_blocks
    org_single_blocks = pipeline.transformer.model.single_transformer_blocks

    # pipeline.transformer.model.config.num_layers = config["num_transformer_layers"]
    # pipeline.transformer.model.config.num_single_layers = config["num_single_layers"]
    pipeline.transformer.model.config["num_layers"] = config["num_transformer_layers"]
    pipeline.transformer.model.config["num_single_layers"] = config["num_single_layers"]
    pipeline.transformer.model.transformer_blocks = torch.nn.ModuleList(
        [original_blocks[i] for i in range(0, pipeline.transformer.model.config["num_layers"])]
    )
    pipeline.transformer.model.single_transformer_blocks = torch.nn.ModuleList(
        [org_single_blocks[i] for i in range(0, pipeline.transformer.model.config["num_single_layers"])]
    )

    ### Pytorch pipeline
    pytorch_pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")
    # Reduce to 2 layers for testing
    original_blocks_pt = pytorch_pipeline.transformer.transformer_blocks
    org_single_blocks_pt = pytorch_pipeline.transformer.single_transformer_blocks
    pytorch_pipeline.transformer.config["num_layers"] = config["num_transformer_layers"]
    pytorch_pipeline.transformer.config["num_single_layers"] = config["num_single_layers"]
    pytorch_pipeline.transformer.transformer_blocks = torch.nn.ModuleList(
        [original_blocks_pt[i] for i in range(0, pytorch_pipeline.transformer.config["num_layers"])]
    )
    pytorch_pipeline.transformer.single_transformer_blocks = torch.nn.ModuleList(
        [org_single_blocks_pt[i] for i in range(0, pytorch_pipeline.transformer.config["num_single_layers"])]
    )
    return pipeline, pytorch_pipeline


@pytest.mark.diffusion_models
@pytest.mark.on_qaic
def test_flux_pipeline(flux_pipeline):
    """
    Comprehensive Flux pipeline test that follows the exact same flow as pipeline_flux.py:
    - 256x256 resolution - 2 transformer layers
    - MAD validation
    - Functional image generation test
    - Export/compilation checks
    - Returns QEffPipelineOutput with performance metrics
    """
    pipeline, pytorch_pipeline = flux_pipeline
    config = INITIAL_TEST_CONFIG

    # Print test header
    DiffusersTestUtils.print_test_header(
        f"FLUX PIPELINE TEST - {config['model_setup']['height']}x{config['model_setup']['width']} Resolution, {config['model_setup']['num_transformer_layers']} Layers",
        config,
    )

    # Test parameters
    test_prompt = config["functional_testing"]["test_prompt"]
    num_inference_steps = config["functional_testing"]["num_inference_steps"]
    guidance_scale = config["functional_testing"]["guidance_scale"]
    max_sequence_length = config["functional_testing"]["max_sequence_length"]
    custom_config_path = config["functional_testing"]["custom_config_path"]

    # Generate with MAD validation
    generator = torch.manual_seed(42)
    mad_results = {}
    start_time = time.time()

    try:
        # Run the pipeline with integrated MAD validation (follows exact pipeline flow)
        result, mad_data = flux_pipeline_call_with_mad_validation(
            pipeline,
            pytorch_pipeline,
            height=config["model_setup"]["height"],
            width=config["model_setup"]["width"],
            prompt=test_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            custom_config_path=custom_config_path,
            generator=generator,
            mad_tolerances=config["mad_validation"]["tolerances"],
            parallel_compile=True,
            return_dict=True,
        )

        execution_time = time.time() - start_time
        mad_results = mad_data

        # Validate image generation
        if config["functional_testing"]["validate_gen_img"]:
            assert result is not None, "Pipeline returned None"
            assert hasattr(result, "images"), "Result missing 'images' attribute"
            assert len(result.images) > 0, "No images generated"

            generated_image = result.images[0]
            expected_size = (config["model_setup"]["height"], config["model_setup"]["width"])
            # Validate image properties using utilities
            image_validation = DiffusersTestUtils.validate_image_generation(
                generated_image, expected_size, config["functional_testing"]["min_image_variance"]
            )

            print("\n✅ IMAGE VALIDATION PASSED")
            print(f"   - Size: {image_validation['size']}")
            print(f"   - Mode: {image_validation['mode']}")
            print(f"   - Variance: {image_validation['variance']:.2f}")
            print(f"   - Mean pixel value: {image_validation['mean_pixel_value']:.2f}")
            file_path = "test_flux_256x256_2layers.png"
            # Save test image
            generated_image.save(file_path)

            if os.path.exists(file_path):
                print(f"Image saved successfully at: {file_path}")
            else:
                print("Image was not saved.")

        if config["validation_checks"]["onnx_export"]:
            # Check if ONNX files exist (basic check)
            print("\n🔍 ONNX Export Validation:")
            for module_name in ["text_encoder", "text_encoder_2", "transformer", "vae_decode"]:
                module_obj = getattr(pipeline, module_name, None)
                if module_obj and hasattr(module_obj, "onnx_path") and module_obj.onnx_path:
                    DiffusersTestUtils.check_file_exists(str(module_obj.onnx_path), f"{module_name} ONNX")

        if config["validation_checks"]["compilation"]:
            # Check if QPC files exist (basic check)
            print("\n🔍 Compilation Validation:")
            for module_name in ["text_encoder", "text_encoder_2", "transformer", "vae_decode"]:
                module_obj = getattr(pipeline, module_name, None)
                if module_obj and hasattr(module_obj, "qpc_path") and module_obj.qpc_path:
                    DiffusersTestUtils.check_file_exists(str(module_obj.qpc_path), f"{module_name} QPC")

        # Print test summary using utilities
        DiffusersTestUtils.print_test_summary(
            success=True,
            execution_time=execution_time,
            image_count=len(result.images) if result and hasattr(result, "images") else 0,
            mad_results=mad_results,
        )

    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n❌ TEST FAILED: {e}")

        # Print failure summary
        DiffusersTestUtils.print_test_summary(
            success=False, execution_time=execution_time, image_count=0, mad_results=mad_results
        )

        raise


if __name__ == "__main__":
    # This allows running the test file directly for debugging
    pytest.main([__file__, "-v", "-s", "-m", "flux"])
# pytest tests/diffusers/test_flux.py -m flux -v -s --tb=short
