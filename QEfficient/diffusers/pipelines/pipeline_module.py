# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_wan import WanTransformerBlock

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, SplitTensorsTransform
from QEfficient.diffusers.models.pytorch_transforms import (
    AttentionTransform,
    CustomOpsTransform,
    NormalizationTransform,
)
from QEfficient.diffusers.models.transformers.transformer_flux import (
    QEffFluxSingleTransformerBlock,
    QEffFluxTransformerBlock,
)
from QEfficient.transformers.models.pytorch_transforms import (
    T5ModelTransform,
)
from QEfficient.utils import constants


class QEffTextEncoder(QEFFBaseModel):
    """
    Wrapper for text encoder models with ONNX export and QAIC compilation capabilities.

    This class handles text encoder models (CLIP, T5) with specific transformations and
    optimizations for efficient inference on Qualcomm AI hardware. It applies custom
    PyTorch and ONNX transformations to prepare models for deployment.

    Attributes:
        model (nn.Module): The wrapped text encoder model (deep copy of original)
        _pytorch_transforms (List): PyTorch transformations applied before ONNX export
        _onnx_transforms (List): ONNX transformations applied after export
    """

    _pytorch_transforms = [CustomOpsTransform, T5ModelTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    @property
    def get_model_config(self) -> Dict:
        """
        Get the model configuration as a dictionary.

        Returns:
            Dict: The configuration dictionary of the underlying text encoder model
        """
        return self.model.config.__dict__

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize the text encoder wrapper.

        Args:
            model (nn.Module): The text encoder model to wrap (CLIP or T5)
        """
        super().__init__(model)
        self.model = model

    def get_onnx_params(self) -> Tuple[Dict, Dict, List[str]]:
        """
        Generate ONNX export configuration for the text encoder.

        Creates example inputs, dynamic axes specifications, and output names
        tailored to the specific text encoder type (CLIP vs T5).

        Returns:
            Tuple containing:
                - example_inputs (Dict): Sample inputs for ONNX export
                - dynamic_axes (Dict): Specification of dynamic dimensions
                - output_names (List[str]): Names of model outputs
        """
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE

        # Create example input with max sequence length
        example_inputs = {
            "input_ids": torch.zeros((bs, self.model.config.max_position_embeddings), dtype=torch.int64),
        }

        # Define which dimensions can vary at runtime
        dynamic_axes = {"input_ids": {0: "batch_size", 1: "seq_len"}}

        # T5 only outputs hidden states, CLIP outputs both hidden states and pooled output
        if self.model.__class__.__name__ == "T5EncoderModel":
            output_names = ["last_hidden_state"]
        else:
            output_names = ["last_hidden_state", "pooler_output"]
            example_inputs["output_hidden_states"] = False

        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        export_kwargs: Dict = None,
    ) -> str:
        """
        Export the text encoder model to ONNX format.

        Args:
            inputs (Dict): Example inputs for ONNX export
            output_names (List[str]): Names of model outputs
            dynamic_axes (Dict): Specification of dynamic dimensions
            export_dir (str, optional): Directory to save ONNX model
            export_kwargs (Dict, optional): Additional export arguments

        Returns:
            str: Path to the exported ONNX model
        """
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            export_kwargs=export_kwargs,
        )

    def compile(self, specializations: List[Dict], **compiler_options) -> None:
        """
        Compile the ONNX model for Qualcomm AI hardware.

        Args:
            specializations (List[Dict]): Model specialization configurations
            **compiler_options: Additional compiler options (e.g., num_cores, aic_num_of_activations)
        """
        self._compile(specializations=specializations, **compiler_options)


class QEffUNet(QEFFBaseModel):
    """
    Wrapper for UNet models with ONNX export and QAIC compilation capabilities.

    This class handles UNet models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. UNet is commonly used in
    diffusion models for image generation tasks.

    Attributes:
        model (nn.Module): The wrapped UNet model
        _pytorch_transforms (List): PyTorch transformations applied before ONNX export
        _onnx_transforms (List): ONNX transformations applied after export
    """

    _pytorch_transforms = [CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    @property
    def get_model_config(self) -> Dict:
        """
        Get the model configuration as a dictionary.

        Returns:
            Dict: The configuration dictionary of the underlying UNet model
        """
        return self.model.config.__dict__

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize the UNet wrapper.

        Args:
            model (nn.Module): The pipeline model containing the UNet
        """
        super().__init__(model.unet)
        self.model = model.unet

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        export_kwargs: Dict = None,
    ) -> str:
        """
        Export the UNet model to ONNX format.

        Args:
            inputs (Dict): Example inputs for ONNX export
            output_names (List[str]): Names of model outputs
            dynamic_axes (Dict): Specification of dynamic dimensions
            export_dir (str, optional): Directory to save ONNX model
            export_kwargs (Dict, optional): Additional export arguments

        Returns:
            str: Path to the exported ONNX model
        """
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            export_kwargs=export_kwargs,
        )

    def compile(self, specializations: List[Dict], **compiler_options) -> None:
        """
        Compile the ONNX model for Qualcomm AI hardware.

        Args:
            specializations (List[Dict]): Model specialization configurations
            **compiler_options: Additional compiler options
        """
        self._compile(specializations=specializations, **compiler_options)


class QEffVAE(QEFFBaseModel):
    """
    Wrapper for Variational Autoencoder (VAE) models with ONNX export and QAIC compilation.

    This class handles VAE models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. VAE models are used in diffusion
    pipelines for encoding images to latent space and decoding latents back to images.

    Attributes:
        model (nn.Module): The wrapped VAE model (deep copy of original)
        type (str): VAE operation type ("encoder" or "decoder")
        _pytorch_transforms (List): PyTorch transformations applied before ONNX export
        _onnx_transforms (List): ONNX transformations applied after export
    """

    _pytorch_transforms = [CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    @property
    def get_model_config(self) -> Dict:
        """
        Get the model configuration as a dictionary.

        Returns:
            Dict: The configuration dictionary of the underlying VAE model
        """
        return self.model.config.__dict__

    def __init__(self, model: nn.Module, type: str) -> None:
        """
        Initialize the VAE wrapper.

        Args:
            model (nn.Module): The pipeline model containing the VAE
            type (str): VAE operation type ("encoder" or "decoder")
        """
        super().__init__(model)
        self.model = model

        # To have different hashing for encoder/decoder
        self.model.config["type"] = type

    def get_onnx_params(self, latent_height: int = 32, latent_width: int = 32) -> Tuple[Dict, Dict, List[str]]:
        """
        Generate ONNX export configuration for the VAE decoder.

        Args:
            latent_height (int): Height of latent representation (default: 32)
            latent_width (int): Width of latent representation (default: 32)

        Returns:
            Tuple containing:
                - example_inputs (Dict): Sample inputs for ONNX export
                - dynamic_axes (Dict): Specification of dynamic dimensions
                - output_names (List[str]): Names of model outputs
        """
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE

        # VAE decoder takes latent representation as input
        example_inputs = {
            "latent_sample": torch.randn(bs, 16, latent_height, latent_width),
            "return_dict": False,
        }

        output_names = ["sample"]

        # All dimensions except channels can be dynamic
        dynamic_axes = {
            "latent_sample": {0: "batch_size", 1: "channels", 2: "latent_height", 3: "latent_width"},
        }

        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        export_kwargs: Dict = None,
    ) -> str:
        """
        Export the VAE model to ONNX format.

        Args:
            inputs (Dict): Example inputs for ONNX export
            output_names (List[str]): Names of model outputs
            dynamic_axes (Dict): Specification of dynamic dimensions
            export_dir (str, optional): Directory to save ONNX model
            export_kwargs (Dict, optional): Additional export arguments

        Returns:
            str: Path to the exported ONNX model
        """
        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            export_kwargs=export_kwargs,
        )

    def compile(self, specializations: List[Dict], **compiler_options) -> None:
        """
        Compile the ONNX model for Qualcomm AI hardware.

        Args:
            specializations (List[Dict]): Model specialization configurations
            **compiler_options: Additional compiler options
        """
        self._compile(specializations=specializations, **compiler_options)


class QEffFluxTransformerModel(QEFFBaseModel):
    """
    Wrapper for Flux Transformer2D models with ONNX export and QAIC compilation capabilities.

    This class handles Flux Transformer2D models with specific transformations and optimizations
    for efficient inference on Qualcomm AI hardware. Flux uses a transformer-based diffusion
    architecture instead of traditional UNet, with dual transformer blocks and adaptive layer
    normalization (AdaLN) for conditioning.

    Attributes:
        model (nn.Module): The wrapped Flux transformer model
        _pytorch_transforms (List): PyTorch transformations applied before ONNX export
        _onnx_transforms (List): ONNX transformations applied after export
    """

    _pytorch_transforms = [AttentionTransform, NormalizationTransform, CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    @property
    def get_model_config(self) -> Dict:
        """
        Get the model configuration as a dictionary.

        Returns:
            Dict: The configuration dictionary of the underlying Flux transformer model
        """
        return self.model.config.__dict__

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize the Flux transformer wrapper.

        Args:
            model (nn.Module): The Flux transformer model to wrap
        """
        super().__init__(model)

    def get_onnx_params(
        self,
        batch_size: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
        seq_length: int = constants.FLUX_ONNX_EXPORT_SEQ_LENGTH,
        cl: int = constants.FLUX_ONNX_EXPORT_COMPRESSED_LATENT_DIM,
    ) -> Tuple[Dict, Dict, List[str]]:
        """
        Generate ONNX export configuration for the Flux transformer.

        Creates example inputs for all Flux-specific inputs including hidden states,
        text embeddings, timestep conditioning, and AdaLN embeddings.

        Args:
            batch_size (int): Batch size for example inputs (default: FLUX_ONNX_EXPORT_BATCH_SIZE)
            seq_length (int): Text sequence length (default: FLUX_ONNX_EXPORT_SEQ_LENGTH)
            cl (int): Compressed latent dimension (default: FLUX_ONNX_EXPORT_COMPRESSED_LATENT_DIM)

        Returns:
            Tuple containing:
                - example_inputs (Dict): Sample inputs for ONNX export
                - dynamic_axes (Dict): Specification of dynamic dimensions
                - output_names (List[str]): Names of model outputs
        """
        example_inputs = {
            # Latent representation of the image
            "hidden_states": torch.randn(batch_size, cl, self.model.config.in_channels, dtype=torch.float32),
            "encoder_hidden_states": torch.randn(
                batch_size, seq_length, self.model.config.joint_attention_dim, dtype=torch.float32
            ),
            "pooled_projections": torch.randn(batch_size, self.model.config.pooled_projection_dim, dtype=torch.float32),
            "timestep": torch.tensor([1.0], dtype=torch.float32),
            "img_ids": torch.randn(cl, 3, dtype=torch.float32),
            "txt_ids": torch.randn(seq_length, 3, dtype=torch.float32),
            # AdaLN embeddings for dual transformer blocks
            # Shape: [num_layers, FLUX_ADALN_DUAL_BLOCK_CHUNKS, FLUX_ADALN_HIDDEN_DIM]
            "adaln_emb": torch.randn(
                self.model.config["num_layers"],
                constants.FLUX_ADALN_DUAL_BLOCK_CHUNKS,
                constants.FLUX_ADALN_HIDDEN_DIM,
                dtype=torch.float32,
            ),
            # AdaLN embeddings for single transformer blocks
            # Shape: [num_single_layers, FLUX_ADALN_SINGLE_BLOCK_CHUNKS, FLUX_ADALN_HIDDEN_DIM]
            "adaln_single_emb": torch.randn(
                self.model.config["num_single_layers"],
                constants.FLUX_ADALN_SINGLE_BLOCK_CHUNKS,
                constants.FLUX_ADALN_HIDDEN_DIM,
                dtype=torch.float32,
            ),
            # Output AdaLN embedding
            # Shape: [batch_size, FLUX_ADALN_OUTPUT_DIM] for final projection
            "adaln_out": torch.randn(batch_size, constants.FLUX_ADALN_OUTPUT_DIM, dtype=torch.float32),
        }

        output_names = ["output"]

        # Define dynamic dimensions for runtime flexibility
        dynamic_axes = {
            "hidden_states": {0: "batch_size", 1: "cl"},
            "encoder_hidden_states": {0: "batch_size", 1: "seq_len"},
            "pooled_projections": {0: "batch_size"},
            "timestep": {0: "steps"},
            "img_ids": {0: "cl"},
        }

        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        export_kwargs: Dict = None,
        use_onnx_subfunctions: bool = False,
    ) -> str:
        """
        Export the Flux transformer model to ONNX format.

        Args:
            inputs (Dict): Example inputs for ONNX export
            output_names (List[str]): Names of model outputs
            dynamic_axes (Dict): Specification of dynamic dimensions
            export_dir (str, optional): Directory to save ONNX model
            export_kwargs (Dict, optional): Additional export arguments (e.g., export_modules_as_functions)

        Returns:
            str: Path to the exported ONNX model
        """

        if use_onnx_subfunctions:
            export_kwargs = {"export_modules_as_functions": {QEffFluxTransformerBlock, QEffFluxSingleTransformerBlock}}

        # Sort _use_default_values in config to ensure consistent hash generation during export
        self.model.config["_use_default_values"].sort()

        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            export_kwargs=export_kwargs,
            offload_pt_weights=False,  # As weights are needed with AdaLN changes
        )

    def compile(self, specializations: List[Dict], **compiler_options) -> None:
        """
        Compile the ONNX model for Qualcomm AI hardware.

        Args:
            specializations (List[Dict]): Model specialization configurations
            **compiler_options: Additional compiler options (e.g., num_cores, aic_num_of_activations)
        """
        self._compile(specializations=specializations, **compiler_options)

class QEffWanUnifiedWrapper(nn.Module):
    def __init__(self, transformer_high, transformer_low, boundary_timestep):
        super().__init__()
        self.transformer_high = transformer_high
        self.transformer_low = transformer_low
        self.boundary_timestep = torch.tensor(boundary_timestep)
        self.config = transformer_high.config

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        rotary_emb,
        temb,
        timestep_proj,
        tsp,
        attention_kwargs=None,
        return_dict=False,
    ):
        # Condition based on timestep value
        # timestep shape: [batch_size]
        is_high_noise = tsp.shape[0] == torch.tensor(1)
        # Run both models

        high_hs = hidden_states.detach()
        ehs = encoder_hidden_states.detach()
        rhs = rotary_emb.detach()
        ths = temb.detach()
        projhs = timestep_proj.detach()

        noise_pred_high = self.transformer_high(
            hidden_states=high_hs,
            encoder_hidden_states=ehs,
            rotary_emb = rhs,
            temb = ths,
            timestep_proj = projhs,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]

        noise_pred_low = self.transformer_low(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            rotary_emb = rotary_emb,
            temb = temb,
            timestep_proj = timestep_proj,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]

        # Select based on timestep condition
        noise_pred = torch.where(
            is_high_noise,
            noise_pred_high,
            noise_pred_low
        )
        return noise_pred

class QEffWanUnifiedTransformer(QEFFBaseModel):
    """ TODO update this """
    _pytorch_transforms = [AttentionTransform, CustomOpsTransform, NormalizationTransform ]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(self, unified_transformer):
        """
        Initialize the Wan unified transformer.

        Args:
            model (nn.Module): Wan unified transformer model
        """
        super().__init__(unified_transformer)
        self.model =  unified_transformer

    @property
    def get_model_config(self) -> Dict:
        """
        Get the model configuration as a dictionary.

        Returns:
            Dict: The configuration dictionary of the underlying Wan transformer model
        """
        return self.model.config.__dict__


    def get_onnx_params(self, batch_size=1, seq_length=512, cl=3840, latent_height=24, latent_width=40, latent_frames=21):
         #cl = 3840, # TODO update generic for Wan 2.2 5 B (6240), 14 B
        example_inputs = {
            ## TODO : chekc AttributeError: 'QEFFWanUnifiedWrapper' object has no attribute 'config' #self.model.config.in_channels, self.model.config.out_channels (self.model.transformer_high.config)
            "hidden_states": torch.randn(batch_size, 16, latent_frames, latent_height, latent_width ,dtype=torch.float32), #TODO check self.model.config.num_frames - wan 5B  #1, 48, 16, 30, 52
            "encoder_hidden_states": torch.randn(batch_size, seq_length , 5120, dtype=torch.float32), # BS, seq len , text dim #TODO: check why 5120, not like wan 5B - text_dim : 4096
            "rotary_emb": torch.randn(2, cl, 1, 128 , dtype=torch.float32), #TODO update wtih CL
            "temb": torch.randn(1, 5120, dtype=torch.float32), #TODO: wan 5b - 1, cl, 3072
            "timestep_proj": torch.randn(1, 6, 5120, dtype=torch.float32), #TODO  wan 5b - 1, cl, 6, 3072
            "tsp": torch.ones(1, dtype=torch.int64) # will be using a parameter to decide based on length
        }

        output_names = ["output"]

        dynamic_axes={
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
            "tsp": {0: "model_type"}
        }

        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        export_kwargs: Dict = None,
        use_onnx_subfunctions: bool = False,
    ) -> str:
        """ TODO docs"""
        if use_onnx_subfunctions:
            export_kwargs = {"export_modules_as_functions": {WanTransformerBlock}}

        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            export_kwargs=export_kwargs,
            offload_pt_weights=False,  # As weights are needed with AdaLN changes
        )

    def compile(self, specializations, **compiler_options):
        self._compile(specializations=specializations, **compiler_options)


    @property
    def model_name(self) -> str:
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname


class QEffAutoencoderKLWan(QEFFBaseModel):
    _pytorch_transforms = [AttentionTransform, CustomOpsTransform, NormalizationTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    """
    QEffWanTransformerModel is a wrapper class for WanTransformer 3DModel models that provides ONNX export and compilation capabilities.
    This class extends QEFFBaseModel to handle Wan Transformer3DModel models with specific transformations and optimizations for efficient inference on Qualcomm AI hardware.
    """

    def __init__(self, model: nn.modules):
        super().__init__(model)
        self.model = model

    def get_onnx_params(self, batch_size=1, num_frames=21,latent_height=24, latent_width=40, **kwargs):
        example_inputs = {
            "sample": torch.randn(batch_size, 16, num_frames, latent_height, latent_width),
        }

        output_names = ["video"]

        dynamic_axes = {
            "sample": {0: "batch_size", 1: "num_channels", 2: "num_frames", 3: "latent_height", 4: "latent_width"},
        }

        return example_inputs, dynamic_axes, output_names

    def export(
        self,
        inputs: Dict,
        output_names: List[str],
        dynamic_axes: Dict,
        export_dir: str = None,
        export_kwargs: Dict = None,
        use_onnx_subfunctions: bool = False,
    ) -> str:
        """ TODO docs"""
        print(export_dir)
        # Sort _use_default_values in config to ensure consistent hash generation during export
        self.model.config["_use_default_values"].sort()

        return self._export(
            example_inputs=inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
            export_kwargs=export_kwargs,
            offload_pt_weights=True,
        )

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