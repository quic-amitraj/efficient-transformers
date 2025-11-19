# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import torch

from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_wan_lora_to_diffusers
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download
import safetensors.torch
from QEfficient import QEFFWanPipeline

# for 480-P
# height = 480
# width = 832

# height = 192 # for 180-P
# width = 320

# Load the pipeline
# To enable sub function pass use_onnx_function=True
pipeline = QEFFWanPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers", use_onnx_function=False, height=480, width=832)

# Download the LoRAs
high_noise_lora_path = hf_hub_download(
    repo_id="lightx2v/Wan2.2-Lightning",
    filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors"
)
low_noise_lora_path = hf_hub_download(
    repo_id="lightx2v/Wan2.2-Lightning",
    filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
)

# LoRA conversion
def load_wan_lora(path: str):
    return _convert_non_diffusers_wan_lora_to_diffusers(
        safetensors.torch.load_file(path)
    )

# Load into the transformers
pipeline.transformer.model.load_lora_adapter(load_wan_lora(high_noise_lora_path), adapter_name="high_noise")
pipeline.transformer.model.set_adapters(["high_noise"], weights=[1.0])

pipeline.transformer_2.model.load_lora_adapter(load_wan_lora(low_noise_lora_path), adapter_name="low_noise")
pipeline.transformer_2.model.set_adapters(["low_noise"], weights=[1.0])

# ### for 2 layer model
# pipeline.transformer.model.config.num_layers = 2 # for  2 layers
# pipeline.transformer_2.model.config.num_layers = 2 # for 2 layers
# original_blocks = pipeline.transformer.model.blocks
# org_blocks = pipeline.transformer_2.model.blocks
# pipeline.transformer.model.blocks = torch.nn.ModuleList([original_blocks[i] for i in range(0,pipeline.transformer.model.config.num_layers)])
# pipeline.transformer_2.model.blocks = torch.nn.ModuleList([org_blocks[i] for i in range(0,pipeline.transformer_2.model.config.num_layers)])

pipeline = pipeline.to("cpu")

# prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
prompt = "In a warmly lit living room, an elderly man with gray hair sits in a wooden armchair adorned with a blue cushion. He wears a gray cardigan over a white shirt, engrossed in reading a book. As he turns the pages, he subtly adjusts his posture, ensuring his glasses stay in place. He then removes his glasses, holding them in his hand, and turns his head to the right, maintaining his grip on the book. The soft glow of a bedside lamp bathes the scene, creating a calm and serene atmosphere, with gentle shadows enhancing the intimate setting."
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

output = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=61,
    guidance_scale=1.0,
    guidance_scale_2=1.0,
    num_inference_steps=4,
    generator=torch.manual_seed(0),
    custom_config_path = "examples/diffusers/wan/wan_config.json"
    )
frames = output.images[0]
export_to_video(frames, "out_t2v_480p_sub_fn_elder_man.mp4", fps=12)
print(output)

## To run with HQKV blocking for 480 P
# ATTENTION_BLOCKING_MODE=qkv head_block_size=16 num_kv_blocks=16 num_q_blocks=2  python -W ignore examples/diffusers/wan/wan_lightning.py

## To run with Sub function
# QAIC_COMPILER_OPTS_UNSUPPORTED="-loader-inline-all=0" ATTENTION_BLOCKING_MODE=qkv head_block_size=16 num_kv_blocks=16 num_q_blocks=2  python -W ignore examples/diffusers/wan/wan_lightning.py