# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from QEfficient import QEFFStableDiffusion3Pipeline
import torch

pipeline = QEFFStableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large-turbo")

original_blocks = pipeline.transformer.model.transformer_blocks
pipeline.transformer.model.transformer_blocks = torch.nn.ModuleList([original_blocks[0]])

# Update num_layers to reflect the change
pipeline.transformer.model.config.num_layers = 1

model = pipeline.transformer.model
# 5. Prepare example inputs
print("Preparing example inputs...")
# These shapes are typical for SD3
# batch_size = 1
# patch_size = model.config.patch_size # Typically 2
# num_channels = model.config.num_channels # E.g., 16 for SD3 latent output
# image_size = 64 # Example, for a 512x512 output image (512/patch_size=2=256, but SD3 latent might be smaller)
# latent_resolution = 64 # Example latent resolution (e.g., for 512x512 image / 8)
# sequence_length = (latent_resolution // patch_size) * (latent_resolution // patch_size)
# hidden_size = model.config.hidden_size # The dimension of each token

# hidden_states = torch.randn(
#     batch_size, sequence_length, hidden_size,
#     dtype=model.dtype, device=model.device
# )
# device='cpu'
# text_seq_len = 77
# text_embed_dim = 2048 # Common for SD3 (combination of multiple CLIP embeddings)
# encoder_hidden_states = torch.randn(
#     batch_size, text_seq_len, text_embed_dim,
#     dtype=model.dtype, device=device
# )


# pooled_projections = torch.randn(
#     batch_size, text_embed_dim, # Can also be something like model.config.pooled_projections_dim
#     dtype=model.dtype, device=device
# )

# timestep = torch.tensor([999], dtype=torch.long, device=device)

# attention_mask = None 
# output = model(
#         hidden_states=hidden_states,
#         encoder_hidden_states=encoder_hidden_states,
#         pooled_projections=pooled_projections,
#         timestep=timestep,
#         attention_mask=attention_mask,
    # )
pipeline.compile(num_devices_text_encoder=1, num_devices_transformer=4, num_devices_vae_decoder=1)

# NOTE: guidance_scale <=1 is not supported
image = pipeline("A girl laughing", num_inference_steps=28, guidance_scale=2.0).images[0]
image.save("girl_laughing_turbo.png")
