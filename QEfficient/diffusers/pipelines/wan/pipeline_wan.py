# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union
from venv import logger

import numpy as np
import torch
from diffusers import WanPipeline
from diffusers.models.transformers.transformer_wan import WanTransformerBlock
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from diffusers.video_processor import VideoProcessor

from QEfficient.diffusers.pipelines.pipeline_module import (
    QEFFWanUnifiedTransformer,
    QEFFWanUnifiedWrapper,
)
from QEfficient.diffusers.pipelines.pipeline_utils import (
    ModulePerf,
    QEffPipelineOutput,
    config_manager,
    set_module_device_ids,
)
from QEfficient.generation.cloud_infer import QAICInferenceSession


class QEFFWanPipeline(WanPipeline):
    r"""
    Pipeline for text-to-video generation using Wan.
    A QEfficient-optimized Wan pipeline, inheriting from `diffusers.WanPipeline`.

    This class integrates QEfficient components (e.g., optimized models for umt5 text encoders,
    wan transformer, and VAE) to enhance performance, particularly for deployment on Qualcomm AI hardware.
    It provides methods for text-to-video generation leveraging these optimized components.
    """
    _hf_auto_class = WanPipeline

    def __init__(self, model, use_onnx_function=False, *args, **kwargs):
        # Required by diffusers for serialization and device management
        self.model = model
        self.kwargs = kwargs
        self.custom_config = None
        self.use_onnx_function = use_onnx_function
        self.text_encoder = model.text_encoder   ##TODO : update with  Qeff umt5
        self.unified_wrapper  = QEFFWanUnifiedWrapper(model.transformer, model.transformer_2, 875 ) # 875 ==>  boundary_ratio * 1000 (self.scheduler.config.num_train_timesteps= 1000)
        self.transformer = QEFFWanUnifiedTransformer(self.unified_wrapper)
        self.vae_decode = model.vae  ##TODO: QEffVAE(model, "decoder")

        # All modules of WanPipeline stored in a dictionary for easy access and iteration
        self.modules = {
            "transformer": self.transformer
        }
        self.tokenizer = model.tokenizer
        self.text_encoder.tokenizer = model.tokenizer
        self.scheduler = model.scheduler

        self.register_modules(
            text_encoder=self.text_encoder,
            scheduler=self.scheduler,
            tokenizer=self.tokenizer,
            transformer=self.transformer ,
            vae=self.vae_decode,
        )

        # boundary_ratio = self.kwargs.get("boundary_ratio", None)  # TODO: for wan 5 B
        # expand_timesteps = self.kwargs.get("expand_timesteps", True) ##TODO : for wan 5B
        boundary_ratio = 0.875
        expand_timesteps = self.kwargs.get("expand_timesteps", False)
        self.register_to_config(boundary_ratio=boundary_ratio)
        self.register_to_config(expand_timesteps=expand_timesteps)
        self.vae_scale_factor_temporal = self.vae_decode.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae_decode.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        _, self.patch_height, self.patch_width = self.transformer.model.config.patch_size


    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        use_onnx_function: bool = False,
        **kwargs,
    ):
        """
        Instantiate a QEffFluxTransformer2DModel from pretrained Diffusers models.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                The path to the pretrained model or its name.
            **kwargs (additional keyword arguments):
                Additional arguments that can be passed to the underlying `FluxPipeline.from_pretrained`
                method.
        """
        model = cls._hf_auto_class.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float32,
            **kwargs,
        )
        model.to("cpu")
        return cls(
            model=model,
            use_onnx_function=use_onnx_function,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **kwargs,
        )


    @property
    def components(self):
        return {
            "text_encoder": self.text_encoder,
            "transformer": self.transformer,
            "vae": self.vae_decode,
            "tokenizer": self.tokenizer,
            "scheduler": self.scheduler,
        }

    def configure_height_width_cl_latents_hw(self, height, width):
        self.height = height
        self.width = width
        self.cl, self.latent_height, self.latent_width = self.calculate_compressed_latent_dimension(height, width)

    def calculate_compressed_latent_dimension(self, height, width):
        """
        Calculate the compressed latent dimension.
        Args:
            height (int): Target image height in pixels
            width (int): Target image width in pixels
        Returns:
            int: Compressed latent dimension (cl) for transformer input buffer allocation
        """
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        cl = (latent_height // self.patch_height * latent_width // self.patch_width) * 21 # for 81 frames, latent frames are 21
        return cl, latent_height, latent_width


    def export(self, height: int = 192, width:int = 320, export_dir: Optional[str] = None) -> str:
        """
        Exports the model to ``ONNX`` format using ``torch.onnx.export``.

        ``Optional`` Args:
           :height (int) : export height
           :width (int) : export width
           :export_dir (str, optional): The directory path to store ONNX-graph.

        Returns:
            :str: Path of the generated ``ONNX`` graph.
        """
        export_cl, export_latent_height, export_latent_width = self.calculate_compressed_latent_dimension(height, width)
        for module_name, module_obj in self.modules.items():
            example_inputs, dynamic_axes, output_names = (
                module_obj.get_onnx_config(batch_size=1, seq_length=512, cl=export_cl, latent_height=export_latent_height, latent_width=export_latent_width)
            )
            export_kwargs = {}
            if "transformer" in module_name and self.use_onnx_function:
                export_kwargs = {
                    "export_modules_as_functions": {WanTransformerBlock}
                }
            start_time= time.time()
            module_obj.export(
                inputs=example_inputs,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                export_dir=export_dir,
                export_kwargs=export_kwargs,
            )
            end_time= time.time()
            print(f"{module_name} export took {end_time - start_time:.2f} seconds")


    def get_default_config_path():
        """
        Returns the default configuration file path for Flux pipeline.

        Returns:
            str: Path to the default flux_config.json file.
        """
        return os.path.join(os.path.dirname(__file__), "flux_config.json")

    def compile(
        self,
        compile_config: Optional[str] = None,
    ) -> str:
        """
        Compiles the ONNX graphs of the different model components for deployment on Qualcomm AI hardware.

        This method takes the ONNX paths of the text encoders, transformer, and VAE decoder,
        and compiles them into an optimized format for inference using JSON-based configuration.

        Args:
            compile_config (`str`, *optional*):
                Path to JSON configuration file. If None, uses the default configuration.
        """
        # Check if ONNX export is needed
        if any(
            path is None
            for path in [
                # self.text_encoder.onnx_path,
                self.transformer.onnx_path,
                # self.vae_decode.onnx_path,
            ]
        ):
            self.export(self.height, self.width)

        # Initialize configuration manager (JSON-only approach)
        if self.custom_config is None:
            config_manager(self, config_source=compile_config)

        for module_name, module_obj in self.modules.items():
            # Get specialization values directly from config
            module_config = self.custom_config["modules"]
            specializations = [module_config[module_name]["specializations"]]

            # Get compilation parameters from configuration
            compile_kwargs = module_config[module_name]["compilation"]
            # Handling dynamic values which depends on latent height and width
            if "transformer" in module_name:
                for i in range(len(specializations[0])):
                    specializations[0][i]["cl"] = self.cl
                    specializations[0][i]["latent_height"] = self.latent_height
                    specializations[0][i]["latent_width"] = self.latent_width

            elif module_name == "vae_decoder":
                specializations[0]["latent_height"] = self.latent_height
                specializations[0]["latent_width"] = self.latent_width
            # Compile the module
            module_obj.compile(specializations=specializations, **compile_kwargs)


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
        custom_config_path: Optional[str] = None,

    ):
        r"""The call function to the pipeline for generation. """
        device = "cpu"
        # Set height, width
        self.configure_height_width_cl_latents_hw(height, width)

        ##TODO : remove after UMT5 is enabled on QAIC
        # max_sequence_length = 226

        # 1. Check inputs. Raise error if not correct
        if custom_config_path is not None:
            config_manager(self, custom_config_path)
            set_module_device_ids(self)

        # Calling compile with custom config
        self.compile(compile_config=custom_config_path)

        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )
        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False


        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        start_encoder_time = time.time() #TODO will update once UMT5 enabled on QAIC
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        end_encoder_time = time.time()
        text_encoder_perf = end_encoder_time - start_encoder_time

        transformer_dtype = self.transformer.model.transformer_high.dtype # if self.transformer is not None else self.transformer_2.dtype update it to self.transformer_2.model.dtype for 14 B
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = (
            self.transformer.model.config.in_channels
        )

        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        # self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        # 6. Denoising loop
        ###### AIC related changes of transformers ######
        if self.transformer.qpc_session is None:
            self.transformer.qpc_session = QAICInferenceSession(str(self.transformer.qpc_path), device_ids=self.transformer.device_ids)

        output_buffer = {
            "output": np.random.rand(
                batch_size, self.cl, 64
                ##TODO: check for wan 5B : 6240, 192 #self.transformer.model.config.joint_attention_dim , self.transformer.model.config.in_channels
            ).astype(np.int32),
        }
        self.transformer.qpc_session.set_buffers(output_buffer)
        transformer_perf = []

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if boundary_timestep is None or t >= boundary_timestep:
                    # wan2.1 or high-noise stage in wan2.2
                    current_model = self.transformer.model.transformer_high
                    current_guidance_scale = guidance_scale
                    model_type = torch.ones(1, dtype=torch.int64)
                else:
                    ## TODO : not available for wan 5B
                    # low-noise stage in wan2.2
                    current_model = self.transformer.model.transformer_low
                    current_guidance_scale = guidance_scale_2
                    model_type = torch.ones(2, dtype=torch.int64)


                latent_model_input = latents.to(transformer_dtype)
                if self.config.expand_timesteps:
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                batch_size, num_channels, num_frames, height, width = latents.shape
                p_t, p_h, p_w = current_model.config.patch_size
                post_patch_num_frames = num_frames // p_t
                post_patch_height = height // p_h
                post_patch_width = width // p_w

                rotary_emb = current_model.rope(latent_model_input)
                rotary_emb = torch.cat(rotary_emb, dim=0)
                ts_seq_len = None
                timestep = timestep.flatten()

                temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = current_model.condition_embedder(
                    timestep, prompt_embeds, encoder_hidden_states_image=None, timestep_seq_len=ts_seq_len
                )
                if self.do_classifier_free_guidance:
                    temb, timestep_proj, encoder_hidden_states_neg, encoder_hidden_states_image = current_model.condition_embedder(
                        timestep, negative_prompt_embeds, encoder_hidden_states_image=None, timestep_seq_len=ts_seq_len
                    )
                # timestep_proj = timestep_proj.unflatten(2, (6, -1)) # for 5 B new_app.py ##TODO: cross check once
                timestep_proj = timestep_proj.unflatten(1, (6, -1))
                inputs_aic = {
                    "hidden_states": latents.detach().numpy(),
                    "encoder_hidden_states": encoder_hidden_states.detach().numpy(),
                    "rotary_emb": rotary_emb.detach().numpy(),
                    "temb": temb.detach().numpy(),
                    "timestep_proj": timestep_proj.detach().numpy(),
                    "tsp": model_type.detach().numpy()
                }
                if self.do_classifier_free_guidance:
                    inputs_aic2 = {
                        "hidden_states": latents.detach().numpy(),
                        "encoder_hidden_states": encoder_hidden_states_neg.detach().numpy(),
                        "rotary_emb": rotary_emb.detach().numpy(),
                        "temb": temb.detach().numpy(),
                        "timestep_proj": timestep_proj.detach().numpy()
                    }

                with current_model.cache_context("cond"):
                    ########### pytorch
                    # noise_pred_torch = current_model(
                    #     hidden_states=latent_model_input,
                    #     # timestep=timestep,
                    #     encoder_hidden_states=encoder_hidden_states,
                    #     rotary_emb=rotary_emb,
                    #     temb=temb,
                    #     timestep_proj=timestep_proj,
                    #     attention_kwargs=attention_kwargs,
                    #     return_dict=False,
                    # )[0]

                    start_transformer_step_time = time.time()
                    outputs = self.transformer.qpc_session.run(inputs_aic)
                    end_transfromer_step_time = time.time()
                    transformer_perf.append(end_transfromer_step_time - start_transformer_step_time)
                    print(f"DIT {i} time {end_transfromer_step_time - start_transformer_step_time:.2f} seconds")

                    # noise_pred = torch.from_numpy(outputs["output"])
                    hidden_states = torch.tensor(outputs["output"])

                    hidden_states = hidden_states.reshape(
                        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
                    )

                    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
                    noise_pred = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

                if self.do_classifier_free_guidance: # for Wan lighting CFG is False
                    with current_model.cache_context("uncond"):
                        ############ pytorch
                        # noise_uncond_pytorch = current_model(
                        #     hidden_states=latent_model_input,
                        #     timestep=timestep,
                        #     encoder_hidden_states=negative_prompt_embeds,
                        #     attention_kwargs=attention_kwargs,
                        #     return_dict=False,
                        # )[0]
                        start_transformer_step_time = time.time()
                        outputs = self.transformer.qpc_session.run(inputs_aic2)
                        end_transfromer_step_time = time.time()
                        transformer_perf.append(end_transfromer_step_time - start_transformer_step_time)

                        # noise_uncond = torch.from_numpy(outputs["output"])
                        hidden_states = torch.tensor(outputs["output"])

                        hidden_states = hidden_states.reshape(
                            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
                        )

                        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
                        noise_uncond = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                # Execute callback if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                # if XLA_AVAILABLE:
                #     xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            # self.vae_decode.model
            latents = latents.to(self.vae_decode.dtype)
            latents_mean = (
                torch.tensor(self.vae_decode.config.latents_mean)
                .view(1, self.vae_decode.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae_decode.config.latents_std).view(1, self.vae_decode.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            start_decode_time = time.time()
            video = self.model.vae.decode(latents, return_dict=False)[0] #TODO: to enable aic with qpc self.vae_decode(latents, return_dict=False)[0]
            end_decode_time = time.time()
            vae_decode_perf = end_decode_time - start_decode_time
            video = self.video_processor.postprocess_video(video.detach())
        else:
            video = latents

        # Offload all models
        # self.maybe_free_model_hooks()
        # Collect performance data in a dict
        perf_data = {
            "umt5_cpu" : text_encoder_perf,
            "transformer_qaic": transformer_perf,
            "vae_decoder_cpu": vae_decode_perf,
        }

        # Build performance metrics dynamically
        perf_metrics = [ModulePerf(module_name=name, perf=perf_data[name]) for name in perf_data.keys()]

        return QEffPipelineOutput(
            pipeline_module=perf_metrics,
            images=video,
        )