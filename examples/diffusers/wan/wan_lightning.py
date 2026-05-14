# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import safetensors.torch
import torch
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_wan_lora_to_diffusers
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download

from QEfficient import QEffWanPipeline

# Load the pipeline
pipeline = QEffWanPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers")

# # Download the LoRAs
# high_noise_lora_path = hf_hub_download(
#     repo_id="lightx2v/Wan2.2-Lightning",
#     filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors",
# )
# low_noise_lora_path = hf_hub_download(
#     repo_id="lightx2v/Wan2.2-Lightning",
#     filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
# )


# # LoRA conversion
# def load_wan_lora(path: str):
#     return _convert_non_diffusers_wan_lora_to_diffusers(safetensors.torch.load_file(path))


# # Load into the transformers
# pipeline.transformer.model.transformer_high.load_lora_adapter(
#     load_wan_lora(high_noise_lora_path), adapter_name="high_noise"
# )
# pipeline.transformer.model.transformer_high.set_adapters(["high_noise"], weights=[1.0])
# pipeline.transformer.model.transformer_low.load_lora_adapter(
#     load_wan_lora(low_noise_lora_path), adapter_name="low_noise"
# )
# pipeline.transformer.model.transformer_low.set_adapters(["low_noise"], weights=[1.0])

# prompt = "In a warmly lit living room, an elderly man with gray hair sits in a wooden armchair adorned with a blue cushion. He wears a gray cardigan over a white shirt, engrossed in reading a book. As he turns the pages, he subtly adjusts his posture, ensuring his glasses stay in place. He then removes his glasses, holding them in his hand, and turns his head to the right, maintaining his grip on the book. The soft glow of a bedside lamp bathes the scene, creating a calm and serene atmosphere, with gentle shadows enhancing the intimate setting."
USE_MAGCACHE = False


prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=81,
    guidance_scale=4.0,
    guidance_scale_2=3.0,
    num_inference_steps=40,
    generator=torch.Generator().manual_seed(42),
    height=192,
    width=320,
    use_onnx_subfunctions=True,
    parallel_compile=True,
    use_magcache=USE_MAGCACHE,
    magcache_thresh=0.05,
    magcache_K=2,
    magcache_retention_ratio=0.4,
    magcache_verbose=True,
)
frames = output.images[0]
export_to_video(frames, "output_t2v.mp4", fps=16)
print(output)
