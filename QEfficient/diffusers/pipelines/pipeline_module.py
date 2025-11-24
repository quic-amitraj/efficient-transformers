# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy

import torch
import torch.nn as nn

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, SplitTensorsTransform
from QEfficient.diffusers.models.pytorch_transforms import (
    AttentionTransform,
    CustomOpsTransform,
    NormalizationTransform,
)
from QEfficient.transformers.models.pytorch_transforms import T5ModelTransform, UMT5ModelTransform
from QEfficient.utils import constants


class QEffTextEncoder(QEFFBaseModel):
    _pytorch_transforms = [CustomOpsTransform, T5ModelTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]
    """
    QEffTextEncoder is a wrapper class for text encoder models that provides ONNX export and compilation capabilities.

    This class extends QEFFBaseModel to handle text encoder models (like T5EncoderModel) with specific
    transformations and optimizations for efficient inference on Qualcomm AI hardware.
    """

    def __init__(self, model: nn.modules):
        super().__init__(model)
        self.model = copy.deepcopy(model)

    def get_onnx_config(self):
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE

        example_inputs = {
            "input_ids": torch.zeros((bs, self.model.config.max_position_embeddings), dtype=torch.int64),
        }

        dynamic_axes = {"input_ids": {0: "batch_size", 1: "seq_len"}}
        output_names = ["last_hidden_state", "pooler_output"]

        if self.model.__class__.__name__ == "T5EncoderModel":
            output_names = ["last_hidden_state"]
        else:
            example_inputs["output_hidden_states"] = False

        return example_inputs, dynamic_axes, output_names

    @property
    def get_model_config(self) -> dict:
        """
        Get the model configuration as a dictionary.

        Returns
        -------
        dict
            The configuration dictionary of the underlying HuggingFace model.
        """
        return self.model.config.__dict__

    def export(
        self,
        inputs,
        output_names,
        dynamic_axes,
        export_dir=None,
        export_kwargs=None,
    ):
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            export_kwargs=export_kwargs,
        )

    def compile(self, specializations, **compiler_options):
        self._compile(specializations=specializations, **compiler_options)

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname


class QEffUmt5TextEncoder(QEFFBaseModel):
    _pytorch_transforms = [CustomOpsTransform, UMT5ModelTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]
    """
    QEffTextEncoder is a wrapper class for text encoder models that provides ONNX export and compilation capabilities.

    This class extends QEFFBaseModel to handle text encoder models (like T5EncoderModel) with specific
    transformations and optimizations for efficient inference on Qualcomm AI hardware.
    """

    def __init__(self, model: nn.modules):
        super().__init__(model)
        self.model = copy.deepcopy(model)

    def get_onnx_config(self, batch_size=1, seq_length=512, **kwargs):
        bs = batch_size
        example_inputs = {
            "input_ids": torch.zeros((bs, seq_length), dtype=torch.int64),
            "attention_mask": torch.zeros((bs, seq_length), dtype=torch.int64),  # TODO: cross check
        }

        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_length"},
            "attention_mask": {0: "batch_size", 1: "seq_length"},
        }  # TODO: cross check
        output_names = ["pooler_output", "last_hidden_state"]

        if self.model.__class__.__name__ == "QEffUMT5EncoderModel":
            output_names = ["last_hidden_state"]
        else:
            example_inputs["output_hidden_states"] = (True,)
        return example_inputs, dynamic_axes, output_names

    def get_scale_factors(self):
        wo_scaling_factors = []
        fp16_max = torch.finfo(torch.float16).max
        encoder = self.model.encoder
        print(f"Total encoder blocks: {len(encoder.block)}")

        for i, block in enumerate(encoder.block):
            print(f"Number of layers in block: {len(block.layer)}")
            for layer_idx, layer in enumerate(block.layer):
                layer_type = layer.__class__.__name__
                print(f"Layer {layer_idx}: {layer_type}")
                if hasattr(layer, "DenseReluDense"):
                    dense_relu_dense = layer.DenseReluDense
                    dense_type = dense_relu_dense.__class__.__name__
                    # Check for layer norm
                    if hasattr(layer, "layer_norm"):
                        g = layer.layer_norm.weight
                        root_n = torch.sqrt(torch.tensor(g.shape[0], dtype=torch.float32))
                        # Check for different types of dense layers
                        if hasattr(dense_relu_dense, "wo"):
                            wow = dense_relu_dense.wo.weight
                            # Calculate scaling based on wi weights
                            if hasattr(dense_relu_dense, "wi_0") and hasattr(dense_relu_dense, "wi_1"):
                                wiw0 = dense_relu_dense.wi_0.weight
                                wiw1 = dense_relu_dense.wi_1.weight
                                gw0 = g.unsqueeze(1) * wiw0.T
                                max_row0 = root_n * gw0.norm(dim=0)
                                gw1 = g.unsqueeze(1) * wiw1.T
                                max_row1 = root_n * gw1.norm(dim=0)
                                max_row = max_row0 * max_row1
                            elif hasattr(dense_relu_dense, "wi"):
                                wiw = dense_relu_dense.wi.weight
                                gw = g.unsqueeze(1) * wiw.T
                                max_row = root_n * gw.norm(dim=0)
                            else:
                                continue
                            # Calculate wo scaling factor
                            wo_max = max_row.norm() * wow.T.norm(dim=0).max()
                            wo_scaling_factor = torch.ceil(wo_max / fp16_max)
                            wo_sf_value = int(wo_scaling_factor.item())
                            wo_scaling_factors.append(wo_sf_value)
                        else:
                            print("No wo weight found!")
                elif hasattr(layer, "SelfAttention"):
                    print("SelfAttention (skipping for now)")
                else:
                    print("Unknown layer structure")
        return wo_scaling_factors

    def export(
        self,
        inputs,
        output_names,
        dynamic_axes,
        export_dir=None,
        export_kwargs=None,
    ):
        wo_sfs = [float(x) for x in self.get_scale_factors()]
        # print(f">>>>>>>>>>>> wo_sfs : {wo_sfs}") #[1829.0, 10745.0, 17357.0, 19643.0, 21625.0, 22191.0, 25395.0, 25954.0, 34890.0, 35209.0, 43846.0, 51801.0, 51082.0, 54136.0, 50560.0, 51044.0, 49073.0, 47763.0, 42468.0, 40632.0, 37141.0, 33225.0, 27185.0, 12966.0]
        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            prev_sf = 1
            num_blocks = len(self.model.encoder.block)
            if num_blocks != len(wo_sfs):
                raise ValueError(f"Model has {num_blocks} blocks but {len(wo_sfs)} scaling factors provided!")
            for i in range(num_blocks):
                wosf = wo_sfs[i]
                self.model.encoder.block[i].layer[0].SelfAttention.o.weight *= 1 / wosf
                self.model.encoder.block[i].layer[0].scaling_factor *= prev_sf / wosf
                self.model.encoder.block[i].layer[1].DenseReluDense.wo.weight *= 1 / wosf
                prev_sf = wosf
                if (i + 1) % 6 == 0 or i == num_blocks - 1:
                    print(f"Scaled blocks 0-{i}")

        print(f"Applied scaling factors to all {num_blocks} blocks")
        return self._export(inputs, output_names, dynamic_axes, export_dir, export_kwargs)

    def get_specializations(
        self,
        batch_size: int,
        seq_len: int,
    ):
        specializations = [
            {"batch_size": batch_size, "seq_length": seq_len},
        ]

        return specializations

    def compile(self, specializations, **compiler_options):
        self._compile(specializations=specializations, **compiler_options)

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

    @property
    def get_model_config(self) -> dict:
        return self.model.model.vision_model.config.__dict__


class QEffUNet(QEFFBaseModel):
    _pytorch_transforms = [CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    """
    QEffUNet is a wrapper class for UNet models that provides ONNX export and compilation capabilities.

    This class extends QEFFBaseModel to handle UNet models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. It is commonly used in diffusion models for image
    generation tasks.
    """

    def __init__(self, model: nn.modules):
        super().__init__(model.unet)
        self.model = model.unet

    def export(
        self,
        inputs,
        output_names,
        dynamic_axes,
        export_dir=None,
        export_kwargs=None,
    ):
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            export_kwargs=export_kwargs,
        )

    @property
    def get_model_config(self) -> dict:
        """
        Get the model configuration as a dictionary.

        Returns
        -------
        dict
            The configuration dictionary of the underlying HuggingFace model.
        """
        return self.model.config.__dict__

    def compile(self, specializations, **compiler_options):
        self._compile(specializations=specializations, **compiler_options)

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname


class QEffVAE(QEFFBaseModel):
    _pytorch_transforms = [CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    """
    QEffVAE is a wrapper class for Variational Autoencoder (VAE) models that provides ONNX export and compilation capabilities.

    This class extends QEFFBaseModel to handle VAE models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. VAE models are commonly used in diffusion pipelines
    for encoding images to latent space and decoding latent representations back to images.
    """

    def __init__(self, model: nn.modules, type: str):
        super().__init__(model.vae)
        self.model = copy.deepcopy(model.vae)
        self.type = type

    def get_onnx_config(self, latent_height=32, latent_width=32):
        # VAE decode
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        example_inputs = {
            "latent_sample": torch.randn(bs, 16, latent_height, latent_width),
            "return_dict": False,
        }

        output_names = ["sample"]

        dynamic_axes = {
            "latent_sample": {0: "batch_size", 1: "channels", 2: "latent_height", 3: "latent_width"},
        }
        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs,
        output_names,
        dynamic_axes,
        export_dir=None,
        export_kwargs=None,
    ):
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            export_kwargs=export_kwargs,
        )

    def compile(self, specializations, **compiler_options):
        self._compile(specializations=specializations, **compiler_options)

    @property
    def get_model_config(self) -> dict:
        """
        Get the model configuration as a dictionary.

        Returns
        -------
        dict
            The configuration dictionary of the underlying HuggingFace model.
        """
        return self.model.config.__dict__

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname


class QEffSafetyChecker(QEFFBaseModel):
    _pytorch_transforms = [CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    """
    QEffSafetyChecker is a wrapper class for safety checker models that provides ONNX export and compilation capabilities.

    This class extends QEFFBaseModel to handle safety checker models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. Safety checker models are commonly used in diffusion pipelines
    to filter out potentially harmful or inappropriate generated content.
    """

    def __init__(self, model: nn.modules):
        super().__init__(model.vae)
        self.model = model.safety_checker

    def export(
        self,
        inputs,
        output_names,
        dynamic_axes,
        export_dir=None,
        export_kwargs=None,
    ):
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            export_kwargs=export_kwargs,
        )

    def compile(self, specializations, **compiler_options):
        self._compile(specializations=specializations, **compiler_options)

    @property
    def get_model_config(self) -> dict:
        """
        Get the model configuration as a dictionary.

        Returns
        -------
        dict
            The configuration dictionary of the underlying HuggingFace model.
        """
        return self.model.config.__dict__

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname


class QEffFluxTransformerModel(QEFFBaseModel):
    _pytorch_transforms = [AttentionTransform, CustomOpsTransform, NormalizationTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]
    """
    QEffFluxTransformerModel is a wrapper class for Flux Transformer2D models that provides ONNX export and compilation capabilities.

    This class extends QEFFBaseModel to handle Flux Transformer2D models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. It is designed for the newer Flux transformer architecture
    that uses transformer-based diffusion models instead of traditional UNet architectures.
    """

    def __init__(self, model: nn.modules, use_onnx_function):
        super().__init__(model)
        # Ensure the model and all its submodules are on CPU to avoid meta device issues
        self.model = model.to("cpu")

    def get_onnx_config(self, batch_size=1, seq_length=256, cl=4096):
        example_inputs = {
            "hidden_states": torch.randn(batch_size, cl, self.model.config.in_channels, dtype=torch.float32),
            "encoder_hidden_states": torch.randn(
                batch_size, seq_length, self.model.config.joint_attention_dim, dtype=torch.float32
            ),
            "pooled_projections": torch.randn(batch_size, self.model.config.pooled_projection_dim, dtype=torch.float32),
            "timestep": torch.tensor([1.0], dtype=torch.float32),
            "img_ids": torch.randn(cl, 3, dtype=torch.float32),
            "txt_ids": torch.randn(seq_length, 3, dtype=torch.float32),
            "adaln_emb": torch.randn(
                self.model.config.num_layers, 12, 3072, dtype=torch.float32
            ),  # num_layers, #chunks, # Adalan_hidden_dim
            "adaln_single_emb": torch.randn(self.model.config.num_single_layers, 3, 3072, dtype=torch.float32),
            "adaln_out": torch.randn(batch_size, 6144, dtype=torch.float32),
        }

        output_names = ["output"]

        dynamic_axes = {
            "hidden_states": {0: "batch_size", 1: "cl"},
            "encoder_hidden_states": {0: "batch_size", 1: "seq_len"},
            "pooled_projections": {0: "batch_size"},
            "timestep": {0: "steps"},
            "img_ids": {0: "cl"},
        }

        return example_inputs, dynamic_axes, output_names

    @property
    def get_model_config(self) -> dict:
        """
        Get the model configuration as a dictionary.

        Returns
        -------
        dict
            The configuration dictionary of the underlying HuggingFace model.
        """
        return self.model.config.__dict__

    def export(
        self,
        inputs,
        output_names,
        dynamic_axes,
        export_dir=None,
        export_kwargs=None,
    ):
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            export_kwargs=export_kwargs,
        )

    def get_specializations(self, batch_size: int, seq_len: int, cl: int):
        specializations = [
            {
                "batch_size": batch_size,
                "stats-batchsize": batch_size,
                "num_layers": self.model.config.num_layers,
                "num_single_layers": self.model.config.num_single_layers,
                "seq_len": seq_len,
                "cl": cl,
                "steps": 1,
            }
        ]

        return specializations

    def compile(self, specializations, **compiler_options):
        self._compile(specializations=specializations, **compiler_options)

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname


class QEffWanTransformerModel(QEFFBaseModel):
    _pytorch_transforms = [AttentionTransform, CustomOpsTransform, NormalizationTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]
    """
    QEffWanTransformerModel is a wrapper class for WanTransformer 3DModel models that provides ONNX export and compilation capabilities.
    This class extends QEFFBaseModel to handle Wan Transformer3DModel models with specific transformations and optimizations for efficient inference on Qualcomm AI hardware.
    """

    def __init__(self, model: nn.modules, use_onnx_function):
        super().__init__(model)
        # Ensure the model and all its submodules are on CPU to avoid meta device issues
        self.model = model.to("cpu")

    def get_onnx_config(
        self, batch_size=1, seq_length=512, **kwargs
    ):  # cl = 3840, # TODO update generic for Wan 2.2 5 B (6240), 14 B
        # cl=3840, latent_height=24, latent_width=40
        cl = kwargs.get("cl", 3840)
        latent_height = kwargs.get("latent_height", 3840)
        latent_width = kwargs.get("latent_width", 3840)
        num_frames = kwargs.get("num_frames", 21)

        example_inputs = {
            "hidden_states": torch.randn(
                batch_size, self.model.config.in_channels, num_frames, latent_height, latent_width, dtype=torch.float32
            ),  # TODO check self.model.config.num_frames - wan 5B  #1, 48, 16, 30, 52
            "encoder_hidden_states": torch.randn(
                batch_size, seq_length, 5120, dtype=torch.float32
            ),  # BS, seq len , text dim #TODO: check why 5120, not like wan 5B - text_dim : 4096
            "rotary_emb": torch.randn(2, cl, 1, 128, dtype=torch.float32),
            "temb": torch.randn(1, 5120, dtype=torch.float32),  # TODO: wan 5b - 1, cl, 3072
            "timestep_proj": torch.randn(1, 6, 5120, dtype=torch.float32),  # TODO  wan 5b - 1, cl, 6, 3072
            "timestep": torch.tensor(
                [1], dtype=torch.float32
            ),  # Adding timestep as a 1D array # error: assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array
        }

        output_names = ["output"]

        dynamic_axes = {
            "hidden_states": {
                0: "batch_size",
                1: "num_channels",
                2: "num_frames",
                3: "latent_height",
                4: "latent_width",
            },
            "timestep": {0: "steps"},
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
            "rotary_emb": {1: "cl"},
        }

        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs,
        output_names,
        dynamic_axes,
        export_dir=None,
        export_kwargs=None,
    ):
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            export_kwargs=export_kwargs,
        )

    def get_specializations(
        self,
        batch_size: int,
        seq_len: int,
        latent_height: int,
        latent_width: int,
        cl: int,
    ):
        specializations = [
            {
                "batch_size": batch_size,
                "num_channels": self.model.in_channels,  # TODO check for wan 5B=48
                "num_frames": "16",
                "latent_height": latent_height,
                "latent_width": latent_width,
                "sequence_length": seq_len,
                "steps": 1,
                "cl": cl,
            }
        ]

        return specializations

    def compile(self, specializations, **compiler_options):
        self._compile(specializations=specializations, **compiler_options)

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

    @property
    def get_model_config(self) -> dict:
        """
        Get the model configuration as a dictionary.

        Returns
        -------
        dict
            The configuration dictionary of the underlying HuggingFace model.
        """
        return self.model.config.__dict__


class QEffAutoencoderKLWan(QEFFBaseModel):
    _pytorch_transforms = [AttentionTransform, CustomOpsTransform, NormalizationTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    """
    QEffWanTransformerModel is a wrapper class for WanTransformer 3DModel models that provides ONNX export and compilation capabilities.
    This class extends QEFFBaseModel to handle Wan Transformer3DModel models with specific transformations and optimizations for efficient inference on Qualcomm AI hardware.
    """

    def __init__(self, model: nn.modules):
        super().__init__(model)
        self.model = copy.deepcopy(model)
        # self.type = type

    def get_onnx_config(self):
        # VAE decode
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        example_inputs = {
            "sample": torch.randn(bs, 16, 16, 24, 40),
            # "return_dict": False,
        }

        output_names = ["video"]

        dynamic_axes = {
            "sample": {0: "batch_size", 1: "num_channels", 2: "num_frames", 3: "latent_height", 4: "latent_width"},
        }

        return example_inputs, dynamic_axes, output_names

    def export(self, inputs, output_names, dynamic_axes, export_dir=None):
        return self._export(inputs, output_names, dynamic_axes, export_dir)

    def get_specializations(
        self,
        batch_size: int,
    ):
        sepcializations = [
            {
                "batch_size": batch_size,
                "num_channels": 16,
                "num_frames": 16,
                "latent_height": 24,
                "latent_width": 40,
            }
        ]
        return sepcializations

    def compile(
        self,
        compile_dir,
        compile_only,
        specializations,
        convert_to_fp16,
        mxfp6_matmul,
        mdp_ts_num_devices,
        aic_num_cores,
        custom_io,
        **compiler_options,
    ) -> str:
        return self._compile(
            compile_dir=compile_dir,
            compile_only=compile_only,
            specializations=specializations,
            convert_to_fp16=convert_to_fp16,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=mdp_ts_num_devices,
            aic_num_cores=aic_num_cores,
            custom_io=custom_io,
            **compiler_options,
        )

    @property
    def model_hash(self) -> str:
        # Compute the hash with: model_config, continuous_batching, transforms
        mhash = hashlib.sha256()
        mhash.update(to_hashable(dict(self.model.config)))
        mhash.update(to_hashable(self._transform_names()))
        mhash.update(to_hashable(self.type))
        mhash = mhash.hexdigest()[:16]
        return mhash

    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

    @property
    def get_model_config(self) -> dict:
        return self.model.model.vision_model.config.__dict__
