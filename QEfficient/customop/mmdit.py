import onnxscript
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
import torch.nn.functional as F

# from onnxscript import int_literal
from diffusers.models.attention_processor import Attention
from diffusers.models.activations import GELU
from diffusers.models.attention import FeedForward, JointTransformerBlock
from diffusers.models.normalization import AdaLayerNormZero

CUSTOM_OPSET = onnxscript.values.Opset(domain="com.qualcomm.cloud", version=1)
# Import the ONNX Script opset for version 13
ops = getattr(onnxscript, "opset" + str(13))


# @onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
# def SD35AdaLayerNormZeroX(
#     hidden_states: onnxscript.FLOAT,
#     emb: onnxscript.FLOAT,
#     linear_weight: onnxscript.FLOAT,
#     linear_bias: onnxscript.FLOAT,
#     norm_epsilon: float,
# ):
#     # 1. emb = self.linear(self.silu(emb))
#     silu_emb = ops.Mul(emb, ops.Sigmoid(emb))
#     linear_out = ops.MatMul(silu_emb, ops.Transpose(linear_weight, perm=[1, 0]))
#     linear_out = ops.Add(linear_out, linear_bias)

#     # 2. Chunk `linear_out` into 9
#     # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2
#     # Determine chunk size dynamically, assuming equal chunks.
#     output_dim_linear = ops.Shape(linear_out)[-1]
#     chunk_size = ops.Cast(output_dim_linear / 9, to=6)  # Cast to Int64

#     split_sizes = ops.Constant(value_ints=[chunk_size] * 9)  # A tuple of 9 chunk_size values
#     split_outputs = ops.Split(linear_out, split_size=split_sizes, axis=1)

#     shift_msa = split_outputs[0]
#     scale_msa = split_outputs[1]
#     gate_msa = split_outputs[2]
#     shift_mlp = split_outputs[3]
#     scale_mlp = split_outputs[4]
#     gate_mlp = split_outputs[5]
#     shift_msa2 = split_outputs[6]
#     scale_msa2 = split_outputs[7]
#     gate_msa2 = split_outputs[8]

#     # 3. norm_hidden_states = self.norm(hidden_states)
#     norm_hidden_states = ops.LayerNormalization(
#         hidden_states,
#         scale=ops.Constant(value_float=1.0, output_dtype=1),  # float
#         bias=ops.Constant(value_float=0.0, output_dtype=1),  # float
#         epsilon=norm_epsilon,
#     )

#     # 4. hidden_states = norm_hidden_states * (1 + scale_msa[:, None]) + shift_msa[:, None]
#     # This `hidden_states` becomes the first output of the function.
#     output_hidden_states = ops.Add(
#         ops.Mul(norm_hidden_states, ops.Add(ops.Constant(value_float=1.0), ops.Unsqueeze(scale_msa, axes=[1]))),
#         ops.Unsqueeze(shift_msa, axes=[1]),
#     )

#     # 5. norm_hidden_states2 = norm_hidden_states * (1 + scale_msa2[:, None]) + shift_msa2[:, None]
#     output_norm_hidden_states2 = ops.Add(
#         ops.Mul(norm_hidden_states, ops.Add(ops.Constant(value_float=1.0), ops.Unsqueeze(scale_msa2, axes=[1]))),
#         ops.Unsqueeze(shift_msa2, axes=[1]),
#     )

#     # Return signature of SD35AdaLayerNormZeroX's forward is:
#     # Tuple[torch.Tensor, ...]: hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2
#     return output_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, output_norm_hidden_states2, gate_msa2


class AdaLayerNormZeroFunc(torch.autograd.Function):
    @staticmethod
    # ctx must still be the first argument after cls
    def forward(
        x: torch.Tensor,
        emb: torch.Tensor,
        linear_weight: torch.Tensor,
        linear_bias: torch.Tensor,
        norm_epsilon: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # PyTorch forward logic (identical to previous version)
        silu_emb = F.silu(emb)
        linear_out = F.linear(silu_emb, linear_weight, linear_bias)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = linear_out.chunk(6, dim=1)

        norm_x = F.layer_norm(x, (x.shape[-1],), None, None, eps=norm_epsilon)

        scaled_shifted_x = norm_x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

        return scaled_shifted_x, gate_msa, shift_mlp, scale_mlp, gate_mlp

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # This can be left as `pass` if no backward context is needed
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        # For inference-only, raise NotImplementedError or return None for all inputs.
        raise NotImplementedError("AdaLayerNormZeroFunc backward not implemented for inference-only.")

    @staticmethod
    def symbolic(
        g: torch.Graph,
        x: torch.Value,
        emb: torch.Value,
        linear_weight: torch.Value,
        linear_bias: torch.Value,
        norm_epsilon: float,  # Pass as Python float if it's a constant
    ) -> Tuple[torch.Value, ...]:
        # Call the corresponding ONNXScript function
        # g.onnxscript_op automatically handles packing/unpacking inputs/outputs
        result = g.onnxscript_op(
            AdaLayerNormZeroOnnx,  # Your ONNXScript function
            x,
            emb,
            linear_weight,
            linear_bias,
            norm_epsilon,
        )
        return result


@onnxscript.script(CUSTOM_OPSET)
def AdaLayerNormZeroOnnx(
    x: onnxscript.FLOAT,
    emb: onnxscript.FLOAT,
    linear_weight: onnxscript.FLOAT,  # Weight for self.linear
    linear_bias: onnxscript.FLOAT,  # Bias for self.linear
    norm_epsilon: float,
):
    # 1. `emb = self.linear(self.silu(emb))`
    silu_emb = ops.Mul(emb, ops.Sigmoid(emb))  # Equivalent to nn.SiLU()

    linear_out = ops.MatMul(silu_emb, ops.Transpose(linear_weight, perm=[1, 0]))
    linear_out = ops.Add(linear_out, linear_bias)

    # 2. `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)`
    # The linear_out has a shape of [..., 6 * embedding_dim]
    output_dim_linear = ops.Shape(linear_out)[-1]
    
    # Calculate chunk_size dynamically (as an ONNX symbolic tensor)
    # Cast output_dim_linear to a float for division, then back to INT64 for chunk_size
    # (Since output_dim_linear is a symbolic tensor, it's already an integer value type usually)
    chunk_size = ops.Cast(ops.Div(output_dim_linear, ops.Constant(value_int=6)), to=6) # 6 is ONNX INT64

    # Create a 1D tensor of `chunk_size` repeated 6 times for ops.Split's split_size input.
    num_chunks = ops.Constant(value_int=6) # Scalar ONNX integer for number of chunks
    
    # Reshape `chunk_size` (which is a scalar tensor) to a 1-element 1D tensor
    chunk_size_1d = ops.Reshape(chunk_size, ops.Constant(value_ints=[1]))
    
    # Tile `chunk_size_1d` 6 times to get the required split_size tensor
    # The `reps` argument for ops.Tile needs to be a 1D tensor.
    reps_1d = ops.Reshape(num_chunks, ops.Constant(value_ints=[1]))
    split_sizes_tensor = ops.Tile(chunk_size_1d, reps_1d) # split_sizes_tensor has a shape of [6]
    split_outputs = ops.Split(linear_out, split_sizes_tensor, axis=1)

    shift_msa = split_outputs[0]
    scale_msa = split_outputs[1]
    gate_msa = split_outputs[2]
    shift_mlp = split_outputs[3]
    scale_mlp = split_outputs[4]
    gate_mlp = split_outputs[5]

    # 3. `x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]`
    # norm_x = ops.LayerNormalization(
    #     x,
    #     scale=ops.Constant(value_float=1.0),  # float type (ONNX FLOAT)
    #     bias=ops.Constant(value_float=0.0),  # float type (ONNX FLOAT)
    #     epsilon=norm_epsilon,
    # )
    # Get the `embedding_dim` (the last dimension of `x`) to create correct scale/bias shapes
    x_shape = ops.Shape(x)
    embedding_dim = x_shape[-1] 

    # Create 1D constant tensors for scale (all ones) and bias (all zeros)
    # The `shape` argument to ops.ConstantOfShape needs to be a 1D tensor representing the output shape.
    scale_shape_tensor = ops.Reshape(embedding_dim, ops.Constant(value_ints=[1])) # Reshape scalar `embedding_dim` to a 1D tensor `[embedding_dim]`

    scale_tensor = ops.ConstantOfShape(
        scale_shape_tensor,
        value=1.0  # Value to fill with
    )
    bias_tensor = ops.ConstantOfShape(
        scale_shape_tensor, # Same shape as scale
        value=0.0 
    )

    norm_x = ops.LayerNormalization(
        x,           # Input tensor (X)
        scale_tensor, # Scale input (S) - a tensor of ones
        bias_tensor,  # Bias input (B) - a tensor of zeros
        epsilon=norm_epsilon # Epsilon is an attribute
    )

    # Apply the scaling and shifting: `norm_x * (1 + scale_msa[:, None]) + shift_msa[:, None]`
    # Use ops.Unsqueeze for `[:, None]`
    scaled_shifted_x = ops.Add(
        ops.Mul(norm_x, ops.Add(ops.Constant(value_float=1.0), ops.Unsqueeze(scale_msa, axes=[1]))),
        ops.Unsqueeze(shift_msa, axes=[1]),
    )

    return scaled_shifted_x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroAIC(nn.Module):
    """
    AdaLayerNormZero module that works by replacing the current module with
    compiler-known custom-op via AdaLayerNormZeroFunc.
    """

    def __init__(self, original_module: AdaLayerNormZero):
        super().__init__()
        # Store necessary sub-modules from the original for their forward pass
        # (e.g., self.emb as it performs computation)
        self.emb_module = original_module.emb  # Can be None

        # Extract parameters needed for AdaLayerNormZeroFunc.apply()
        # These are usually the learnable weights/biases and fixed hyperparameters.
        self.linear_weight = original_module.linear.weight
        # If bias is False, original_module.linear.bias will be None.
        # F.linear in AdaLayerNormZeroFunc.forward handles None bias,
        # and symbolic conversion should handle torch.Value from None.
        self.linear_bias = original_module.linear.bias

        # Norm epsilon
        self.norm_epsilon = original_module.norm.eps

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,  # External emb input
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Replicate the conditional `emb` calculation from the original module's forward.
        # This part ensures that the correct 'emb' tensor is passed to the autograd.Function.
        if self.emb_module is not None:
            # If `self.emb_module` exists, `timestep` and `class_labels` (and hidden_dtype)
            # are expected to be provided to generate `emb`.
            if timestep is None:
                raise ValueError("timestep must be provided when original module has CombinedTimestepLabelEmbeddings.")
            emb_for_linear = self.emb_module(timestep, class_labels, hidden_dtype=hidden_dtype)
        else:
            # If `self.emb_module` does not exist, `emb` should be passed as a direct input.
            if emb is None:
                raise ValueError(
                    "External 'emb' tensor must be provided when original module has no CombinedTimestepLabelEmbeddings."
                )
            emb_for_linear = emb

        return AdaLayerNormZeroFunc.apply(
            x,
            emb_for_linear,  # The `emb` tensor that is the input to the linear layer
            self.linear_weight,
            self.linear_bias,
            self.norm_epsilon,
        )


@onnxscript.script(CUSTOM_OPSET)
# _ApproximateTanh
def GELUOnnx(
    hidden_states: onnxscript.FLOAT,
    proj_weight: onnxscript.FLOAT,
    proj_bias: onnxscript.FLOAT,
):
    """
    ONNXScript equivalent of GELU with approximate="tanh" activation.
    Corresponds to:
    hidden_states = nn.Linear(in_dim, out_dim)(hidden_states)
    return 0.5 * hidden_states * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / math.pi)) * (hidden_states + 0.044715 * torch.pow(hidden_states, 3))))
    """
    projected_states = ops.MatMul(hidden_states, ops.Transpose(proj_weight, perm=[1, 0]))
    projected_states = ops.Add(projected_states, proj_bias)

    x = projected_states
    x_cubed = ops.Pow(x, ops.Constant(value_float=3.0))
    term_x_plus_044715_x_cubed = ops.Add(x, ops.Mul(ops.Constant(value_float=0.044715), x_cubed))
    sqrt_2_div_pi = ops.Constant(value_float=math.sqrt(2.0 / math.pi))
    argument_for_tanh = ops.Mul(sqrt_2_div_pi, term_x_plus_044715_x_cubed)
    tanh_val = ops.Tanh(argument_for_tanh)
    one_plus_tanh_val = ops.Add(ops.Constant(value_float=1.0), tanh_val)
    final_gelu_output = ops.Mul(ops.Mul(ops.Constant(value_float=0.5), x), one_plus_tanh_val)

    return final_gelu_output


class GELUFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,  # ctx is still required
        hidden_states: torch.Tensor,
        proj_weight: torch.Tensor,
        proj_bias: torch.Tensor,
        approximate_type: str,  # "none" or "tanh"
    ) -> torch.Tensor:
        # 1. Apply the Linear Projection (self.proj(hidden_states))
        projected_states = F.linear(hidden_states, proj_weight, proj_bias)

        # 2. Apply the GELU activation using F.gelu
        # Handle MPS specific logic if it's relevant for your deployment
        # For general PyTorch/ONNX export, we mostly rely on F.gelu's CPU/CUDA behavior.
        if projected_states.device.type == "mps" and projected_states.dtype == torch.float16:
            # Replicate mps: gelu is not implemented for float16
            output = F.gelu(projected_states.to(dtype=torch.float32), approximate=approximate_type).to(
                dtype=projected_states.dtype
            )
        else:
            output = F.gelu(projected_states, approximate=approximate_type)

        return output

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass  # Inference-only, no ctx needed

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("GELUFunc backward not implemented for inference-only.")

    @staticmethod
    def symbolic(
        g: torch.Graph,
        hidden_states: torch.Value,
        proj_weight: torch.Value,
        proj_bias: torch.Value,
        approximate_type: str,  # Passed as Python string
    ) -> torch.Value:
        # Call the corresponding ONNXScript function based on approximate_type
        if approximate_type == "tanh":
            result = g.onnxscript_op(
                GELUOnnx,
                hidden_states,
                proj_weight,
                proj_bias,
            )
        return result


class GELUAIC(nn.Module):
    """
    GELU module that works by replacing the current module with
    compiler-known custom-op via GELUFunc.
    """

    def __init__(self, original_module: GELU):
        super().__init__()
        # Extract parameters from the original GELU module's `proj` linear layer
        self.proj_weight = original_module.proj.weight
        # Handle bias being None if original_module.proj.bias was False
        self.proj_bias = original_module.proj.bias if original_module.proj.bias is not None else None

        # Extract the approximate type string
        self.approximate_type = original_module.approximate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Ensure bias tensor is handled correctly (e.g., pass a zero tensor if None)
        # The ONNXScript function expects a bias tensor even if it's all zeros.
        proj_bias_to_pass = self.proj_bias
        if proj_bias_to_pass is None:
            # Create a zero tensor for bias on the same device and dtype as input
            # Shape should match output features of the linear layer (proj_weight.shape[0])
            proj_bias_to_pass = torch.zeros(
                self.proj_weight.shape[0], device=hidden_states.device, dtype=hidden_states.dtype
            )

        return GELUFunc.apply(
            hidden_states,
            self.proj_weight,
            proj_bias_to_pass,
            self.approximate_type,
        )


# ff_context #ff
@onnxscript.script(CUSTOM_OPSET)
def FeedForwardOnnx(
    hidden_states: onnxscript.FLOAT,
    dim: int,
    dim_out: int,  # `dim_out` from FeedForward init (output dimension of the block)
    mult: int,  # `mult` from FeedForward init (used to calculate inner_dim)
    dropout_ratio: float,  # `dropout` for nn.Dropout
    final_dropout: bool,  # `final_dropout` bool
    act_fn_proj_weight: onnxscript.FLOAT,
    act_fn_proj_bias: onnxscript.FLOAT,
    project_out_weight: onnxscript.FLOAT,
    project_out_bias: onnxscript.FLOAT,
):
    # Calculate inner_dim as in PyTorch FeedForward.__init__
    # inner_dim = int(dim * mult)
    inner_dim_val = ops.Cast(ops.Mul(dim, mult), to=6)  # 6 is ONNX INT64
    # 1. Apply act_fn (which is GELUOnnx here)
    ff_output = GELUOnnx(
        hidden_states,
        act_fn_proj_weight,
        act_fn_proj_bias,
    )

    # 2. Apply first Dropout

    ff_output = ops.Dropout(ff_output, ratio=dropout_ratio)

    # 3. Apply project out (final Linear layer)

    ff_output = ops.MatMul(ff_output, ops.Transpose(project_out_weight, perm=[1, 0]))
    ff_output = ops.Add(ff_output, project_out_bias)

    # 4. Apply final Dropout (if final_dropout is True)
    if final_dropout:
        ff_output = ops.Dropout(ff_output, ratio=dropout_ratio)

    return ff_output


class FeedForwardFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        hidden_states: torch.Tensor,
        # Parameters derived from FeedForward __init__
        dim: int,
        dim_out: int,
        mult: int,
        dropout_ratio: float,
        final_dropout: bool,
        # Only parameters for 'gelu-approximate' act_fn
        act_fn_proj_weight: torch.Tensor,
        act_fn_proj_bias: torch.Tensor,  # Passed if bias=True
        # Parameters for final project out layer
        project_out_weight: torch.Tensor,
        project_out_bias: torch.Tensor,  # Passed if bias=True
    ) -> torch.Tensor:
        # Replicate PyTorch forward logic for 'gelu-approximate' only

        # 1. Apply act_fn (GELU with approximate="tanh")
        # Corresponds to: `hidden_states = GELU(dim, inner_dim, approximate="tanh", bias=bias)(hidden_states)`
        # GELU's internal linear projection
        projected_gelu_input = F.linear(hidden_states, act_fn_proj_weight, act_fn_proj_bias)
        # Apply GELU activation with approximate="tanh"
        ff_output = F.gelu(projected_gelu_input, approximate="tanh")

        # 2. Apply first Dropout
        ff_output = F.dropout(ff_output, p=dropout_ratio, training=False)  # For inference, training=False

        # 3. Apply project out (final Linear layer)
        ff_output = F.linear(ff_output, project_out_weight, project_out_bias)

        # 4. Apply final Dropout (if final_dropout is True)
        if final_dropout:
            ff_output = F.dropout(ff_output, p=dropout_ratio, training=False)  # For inference, training=False

        return ff_output

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass  # Inference-only, no ctx needed

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("FeedForwardFunc backward not implemented for inference-only.")

    @staticmethod
    def symbolic(
        g: torch.Graph,
        hidden_states: torch.Value,
        dim: int,
        dim_out: int,
        mult: int,
        dropout_ratio: float,
        final_dropout: bool,
        act_fn_proj_weight: torch.Value,
        act_fn_proj_bias: torch.Value,
        project_out_weight: torch.Value,
        project_out_bias: torch.Value,
    ) -> torch.Value:
        # Call the corresponding ONNXScript FeedForwardOnnx function
        # Note: If FeedForwardOnnx itself has branches, ensure it's configured
        # to take the 'gelu-approximate' path by constant inputs during symbolic export.
        # Alternatively, if FeedForwardOnnx is also simplified to only call GELUOnnx_ApproximateTanh directly,
        # that simplifies things further.
        result = g.onnxscript_op(
            FeedForwardOnnx,
            hidden_states,
            dim=dim,
            dim_out=dim_out,
            mult=mult,
            dropout_ratio=dropout_ratio,
            final_dropout=final_dropout,
            # No activation_fn_type param needed here if FeedForwardOnnx is simplified
            act_fn_proj_weight=act_fn_proj_weight,
            act_fn_proj_bias=act_fn_proj_bias,
            project_out_weight=project_out_weight,
            project_out_bias=project_out_bias,
        )
        return result


class FeedForwardAIC(nn.Module):
    """
    FeedForward module that works by replacing the current module with
    compiler-known custom-op via FeedForwardFunc.
    This version is specialized for 'gelu-approximate' activation.
    """

    def __init__(self, original_module: "FeedForward"):  # Use 'FeedForward' string for forward reference
        super().__init__()
        # Store essential configuration parameters
        self.dim = original_module.dim
        self.dim_out = original_module.dim_out if original_module.dim_out is not None else original_module.dim
        self.mult = original_module.mult
        self.dropout_ratio = original_module.dropout
        self.final_dropout = original_module.final_dropout
        self.bias = original_module.bias  # Indicates if original linear layers had bias

        # # Validate that the original module actually uses 'gelu-approximate'
        # if not (isinstance(original_module.net[0], GELU) and hasattr(original_module.net[0], 'approximate') and original_module.net[0].approximate == "tanh"):
        #     raise ValueError(
        #         "FeedForwardAIC (gelu-approximate only) can only wrap FeedForward modules "
        #         "with activation_fn='gelu-approximate'. Found: "
        #         f"type={type(original_module.net[0])}, approximate={getattr(original_module.net[0], 'approximate', 'N/A')}"
        #     )
        # Extract weights and biases from the original FeedForward's `net`
        # Assumed structure: [act_fn (GELU), Dropout, Linear, (Dropout)]

        # act_fn is original_module.net[0]
        # It's a GELU module, which has an internal 'proj' Linear layer
        self.act_fn_proj_weight = original_module.net[0].proj.weight
        # Handle bias being None if original_module.bias was False
        self.act_fn_proj_bias = original_module.net[0].proj.bias if original_module.bias else None

        # project_out is original_module.net[2] (skipping the first Dropout)
        self.project_out_weight = original_module.net[2].weight
        self.project_out_bias = original_module.net[2].bias if original_module.bias else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,  # Absorb any extra arguments from upstream callers
        **kwargs,
    ) -> torch.Tensor:
        # Ensure dummy zero tensors for bias have correct device and dtype
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Provide a zero tensor for bias if it's None (i.e., if original_module.bias was False)
        # This makes the `apply` call consistent. The ONNXScript side will handle it.
        act_fn_proj_bias_to_pass = self.act_fn_proj_bias
        if act_fn_proj_bias_to_pass is None:
            act_fn_proj_bias_to_pass = torch.zeros(
                self.act_fn_proj_weight.shape[0], device=device, dtype=dtype
            )  # Use proper shape

        project_out_bias_to_pass = self.project_out_bias
        if project_out_bias_to_pass is None:
            project_out_bias_to_pass = torch.zeros(
                self.project_out_weight.shape[0], device=device, dtype=dtype
            )  # Use proper shape
        return FeedForwardFunc.apply(
            hidden_states,
            self.dim,
            self.dim_out,
            self.mult,
            self.dropout_ratio,
            self.final_dropout,
            self.act_fn_proj_weight,
            act_fn_proj_bias_to_pass,
            self.project_out_weight,
            project_out_bias_to_pass,
        )


# qk_norm == "rms_norm"
# recheck this with the customRMSNorm we have earlier


# @onnxscript.script(CUSTOM_OPSET)
# def RMSNormOnnx(
#     hidden_states: onnxscript.FLOAT,  # Input tensor
#     weight: onnxscript.FLOAT,  # Corresponds to self.weight (nn.Parameter)
#     # Pass an empty tensor or zero tensor if elementwise_affine is False
#     eps: float,  # Corresponds to self.eps
#     elementwise_affine: bool,  # Corresponds to elementwise_affine in __init__
# ):
#     """
#     ONNXScript equivalent of RMSNorm.
#     Handles dtype conversions and conditional weight application.
#     """

#     # 1. variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)

#     hidden_states_fp32 = ops.Cast(hidden_states, to=1)  # 1 is ONNX FLOAT (float32)

#     variance = ops.ReduceMean(ops.Pow(hidden_states_fp32, 2), axes=[-1], keepdims=1)

#     # 2. hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

#     variance_with_eps = ops.Add(variance, ops.Constant(value_float=eps))
#     rsqrt_val = ops.Reciprocal(ops.Sqrt(variance_with_eps))

#     hidden_states_normalized = ops.Mul(hidden_states, rsqrt_val)

#     # 3. Conditional weight application: if self.weight is not None: hidden_states = hidden_states * self.weight
#     # This `if` corresponds to `elementwise_affine` boolean
#     if elementwise_affine:
#         hidden_states_to_weight_dtype = ops.Cast(hidden_states_normalized, to=ops.DTYPE_MAP[weight.dtype])  # type: ignore

#         output = ops.Mul(hidden_states_to_weight_dtype, weight)
#     else:
#         output = hidden_states_normalized

#     return output

@onnxscript.script(onnxscript.values.Opset(domain="com.qti.aisw.onnx", version=1))
def CustomRMSNorm(hidden_states: onnxscript.FLOAT, weight: onnxscript.FLOAT, epsilon: float):
    weight = ops.Cast(weight, to=1)
    variance = ops.ReduceMean(ops.Pow(hidden_states, 2), axes=[-1], keepdims=1)
    epsilon = ops.Expand(epsilon, ops.Shape(variance))
    hidden_states = hidden_states * ops.Reciprocal(ops.Sqrt(variance + epsilon))
    return weight * hidden_states


class CustomRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
        return weight * hidden_states

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, hidden_states: torch.Value, weight: torch.Value, epsilon: torch.Value) -> torch.Value:
        return g.onnxscript_op(CustomRMSNorm, hidden_states, weight, epsilon_f=epsilon).setTypeAs(hidden_states)

class CustomRMSNormAIC(nn.Module):
    """
    RMSNorm module that works by replacing the current module with compiler known custom-op.
    """

    def __init__(self, hidden_size, eps=1e-05):
        super(CustomRMSNormAIC, self).__init__()
        self.variance_epsilon = eps
        self.eps = eps  # Added to support GemmaRMSNorm
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        return CustomRMSNormFunc.apply(
            hidden_states, self.weight, self.variance_epsilon if hasattr(self, "variance_epsilon") else self.eps
        )
        
@onnxscript.script(CUSTOM_OPSET)
def JointAttnProcessor2_0Onnx(
    hidden_states: onnxscript.FLOAT,
    encoder_hidden_states: onnxscript.FLOAT,  # This can conceptually be an empty tensor for None
    attention_mask: onnxscript.FLOAT,
    # Parameters from `Attention` module
    attn_heads: int,
    attn_head_dim: int,
    attn_scale_qk: bool,
    attn_scale_val: float,  # self.scale
    attn_query_dim: int,  # self.query_dim (input dim to to_q, to_out[0])
    attn_inner_dim: int,  # output dim of to_q
    attn_inner_kv_dim: int,  # output dim of to_k, to_v
    # Weights and Biases for Attention Projections (`attn.to_q`, `attn.to_k`, `attn.to_v`)
    to_q_weight: onnxscript.FLOAT,
    to_q_bias: onnxscript.FLOAT,
    to_k_weight: onnxscript.FLOAT,
    to_k_bias: onnxscript.FLOAT,
    to_v_weight: onnxscript.FLOAT,
    to_v_bias: onnxscript.FLOAT,
    # RMSNorm parameters (`attn.norm_q`, `attn.norm_k`, `attn.norm_added_q`, `attn.norm_added_k`)
    # Pass weight and epsilon for each
    norm_q_weight: onnxscript.FLOAT,
    norm_q_eps: float,
    norm_q_elementwise_affine: bool,  # From RMSNorm init
    norm_k_weight: onnxscript.FLOAT,
    norm_k_eps: float,
    norm_k_elementwise_affine: bool,
    # Weights and Biases for Cross-Attention Projections (`attn.add_q_proj`, `attn.add_k_proj`, `attn.add_v_proj`)
    add_q_proj_weight: onnxscript.FLOAT,
    add_q_proj_bias: onnxscript.FLOAT,
    add_k_proj_weight: onnxscript.FLOAT,
    add_k_proj_bias: onnxscript.FLOAT,
    add_v_proj_weight: onnxscript.FLOAT,
    add_v_proj_bias: onnxscript.FLOAT,
    norm_added_q_weight: onnxscript.FLOAT,
    norm_added_q_eps: float,
    norm_added_q_elementwise_affine: bool,
    norm_added_k_weight: onnxscript.FLOAT,
    norm_added_k_eps: float,
    norm_added_k_elementwise_affine: bool,
    # Weights and Biases for Output Projections (`attn.to_out[0]`, `attn.to_out[1]`)
    to_out_0_weight: onnxscript.FLOAT,
    to_out_0_bias: onnxscript.FLOAT,
    to_out_1_dropout_p: float,  # Dropout ratio for self.to_out[1]
    # Other flags
    attn_context_pre_only: bool,  # From attn.context_pre_only
    attn_added_kv_proj_dim: int,  # From attn.added_kv_proj_dim (to determine if add_q_proj etc. exist)
    to_add_out_weight: onnxscript.FLOAT,  # For attn.to_add_out
    to_add_out_bias: onnxscript.FLOAT,
):
    residual = hidden_states
    batch_size = ops.Shape(hidden_states)[0]

    # Convert Python ints to 1-element ONNX tensors
    # These are needed for ops.Concat as it expects tensors.
    attn_heads_tensor = ops.Constant(value_int=attn_heads)
    attn_head_dim_tensor = ops.Constant(value_int=attn_head_dim)

    # Check if encoder_hidden_states is "None" (represented by empty tensor)
    # This conditional handling for ONNX `If` is tricky. I'll use Python `if` for tracing.
    # A true ONNX graph might trace both paths if this is a dynamic input.
    encoder_is_not_none = ops.Cast(ops.Size(encoder_hidden_states) > 0, to=9)  # to BOOL

    # --- Sample Projections (Query, Key, Value from hidden_states) ---
    query = ops.Add(ops.MatMul(hidden_states, ops.Transpose(to_q_weight, perm=[1, 0])), to_q_bias)
    key = ops.Add(ops.MatMul(hidden_states, ops.Transpose(to_k_weight, perm=[1, 0])), to_k_bias)
    value = ops.Add(ops.MatMul(hidden_states, ops.Transpose(to_v_weight, perm=[1, 0])), to_v_bias)

    # Reshape for multi-head attention and transpose
    # (batch_size, seq_len, inner_dim) -> (batch_size, seq_len, heads, head_dim) -> (batch_size, heads, seq_len, head_dim)
    seq_len = ops.Shape(hidden_states)[1]
    # Create the shape tensor for Reshape by concatenating 1-element tensors
    # Each element in the list for ops.Concat MUST be an ONNX tensor.
    # We reshape scalar tensors (like batch_size, seq_len, attn_heads_tensor, attn_head_dim_tensor)
    # into 1-element 1D tensors before concatenating them.
    s0 = ops.Reshape(batch_size, ops.Constant(value_ints=[1]))
    s1 = ops.Reshape(seq_len, ops.Constant(value_ints=[1]))
    s2 = ops.Reshape(attn_heads_tensor, ops.Constant(value_ints=[1]))
    s3 = ops.Reshape(attn_head_dim_tensor, ops.Constant(value_ints=[1]))

    reshape_shape = ops.Concat(
        s0, s1, s2, s3, 
        axis=0
    )

    query = ops.Transpose(ops.Reshape(query, reshape_shape), perm=[0, 2, 1, 3])
    key = ops.Transpose(ops.Reshape(key, reshape_shape), perm=[0, 2, 1, 3])
    value = ops.Transpose(ops.Reshape(value, reshape_shape), perm=[0, 2, 1, 3])
    

    query = CustomRMSNorm(query, norm_q_weight, norm_q_eps)
    key = CustomRMSNorm(key, norm_k_weight, norm_k_eps)
    # --- Context Projections (from encoder_hidden_states) ---
    # This block is conditional on `encoder_hidden_states is not None`
    # We will compute both paths and use `ops.If` or a conditional switch later if truly dynamic.
    # For tracing, it will trace with a non-empty encoder_hidden_states if provided.

    # Placeholder for conditional output to ensure full graph is traced if encoder_is_not_none can be dynamic.
    encoder_hidden_states_query_proj_out = ops.Constant(value_float=0.0)
    encoder_hidden_states_key_proj_out = ops.Constant(value_float=0.0)
    encoder_hidden_states_value_proj_out = ops.Constant(value_float=0.0)

    if encoder_is_not_none:  # `if encoder_hidden_states is not None` branch
        encoder_hidden_states_query_proj = ops.Add(
            ops.MatMul(encoder_hidden_states, ops.Transpose(add_q_proj_weight, perm=[1, 0])), add_q_proj_bias
        )
        encoder_hidden_states_key_proj = ops.Add(
            ops.MatMul(encoder_hidden_states, ops.Transpose(add_k_proj_weight, perm=[1, 0])), add_k_proj_bias
        )
        encoder_hidden_states_value_proj = ops.Add(
            ops.MatMul(encoder_hidden_states, ops.Transpose(add_v_proj_weight, perm=[1, 0])), add_v_proj_bias
        )

        # Reshape and transpose for multi-head attention
        enc_seq_len = ops.Shape(encoder_hidden_states)[1]
        es0 = ops.Reshape(batch_size, ops.Constant(value_ints=[1]))
        es1 = ops.Reshape(enc_seq_len, ops.Constant(value_ints=[1]))
        es2 = ops.Reshape(attn_heads_tensor, ops.Constant(value_ints=[1]))
        es3 = ops.Reshape(attn_head_dim_tensor, ops.Constant(value_ints=[1]))

        reshape_enc_shape = ops.Concat(
            es0, es1, es2, es3, # Changed to separate arguments
            axis=0
        )
        encoder_hidden_states_query_proj = ops.Transpose(ops.Reshape(encoder_hidden_states_query_proj, reshape_enc_shape), perm=[0, 2, 1, 3])
        encoder_hidden_states_key_proj = ops.Transpose(ops.Reshape(encoder_hidden_states_key_proj, reshape_enc_shape), perm=[0, 2, 1, 3])
        encoder_hidden_states_value_proj = ops.Transpose(ops.Reshape(encoder_hidden_states_value_proj, reshape_enc_shape), perm=[0, 2, 1, 3])


        # Apply RMSNorm if enabled (norm_added_q, norm_added_k)
        encoder_hidden_states_query_proj = CustomRMSNorm(
            encoder_hidden_states_query_proj, norm_added_q_weight, norm_added_q_eps
        )
        encoder_hidden_states_key_proj = CustomRMSNorm(
            encoder_hidden_states_key_proj, norm_added_k_weight, norm_added_k_eps
        )

        # Concatenate query, key, value from sample and context
        query = ops.Concat(query, encoder_hidden_states_query_proj, axis=2)  # Concat along sequence length (dim=2)
        key = ops.Concat(key, encoder_hidden_states_key_proj, axis=2)
        value = ops.Concat(value, encoder_hidden_states_value_proj, axis=2)
        # --- Scaled Dot-Product Attention ---
    # `dropout_p=0.0, is_causal=False` are fixed.
    hidden_states_attn = ops.ScaledDotProductAttention(query, key, value, attention_mask, is_causal=False)

    # Reshape output back to (batch_size, seq_len, total_heads * head_dim)
    # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    combined_head_dim = ops.Mul(attn_heads_tensor, attn_head_dim_tensor)
    
    fo0 = ops.Reshape(batch_size, ops.Constant(value_ints=[1]))
    fo1 = ops.Reshape(ops.Constant(value_int=-1), ops.Constant(value_ints=[1]))
    fo2 = ops.Reshape(combined_head_dim, ops.Constant(value_ints=[1]))

    final_output_reshape_shape = ops.Concat(
        fo0, fo1, fo2, # Changed to separate arguments
        axis=0
    )
    
    hidden_states_attn = ops.Transpose(hidden_states_attn, perm=[0, 2, 1, 3])
    hidden_states_attn = ops.Reshape(hidden_states_attn, final_output_reshape_shape)

    
    final_hidden_states = ops.Constant(value_float=0.0)
    final_encoder_hidden_states = ops.Constant(value_float=0.0)

    if encoder_is_not_none:  # If cross-attention was performed, split the output
        sample_output_len = ops.Shape(residual)[1]  # Length of the original 'sample' sequence
        total_output_len = ops.Shape(hidden_states_attn)[1]
        s_end_0 = ops.Reshape(ops.Shape(hidden_states_attn)[0], ops.Constant(value_ints=[1]))
        s_end_1 = ops.Reshape(sample_output_len, ops.Constant(value_ints=[1]))
        
        # Slice `hidden_states_attn` into two parts
        final_hidden_states = ops.Slice(
            hidden_states_attn,
            starts=ops.Constant(value_ints=[0, 0]),
            ends=ops.Concat(s_end_0, s_end_1, axis=0),
            axes=ops.Constant(value_ints=[0, 1]),
        )
        s2_start_0 = ops.Reshape(ops.Constant(value_int=0), ops.Constant(value_ints=[1]))
        s2_start_1 = ops.Reshape(sample_output_len, ops.Constant(value_ints=[1]))

        s2_end_0 = ops.Reshape(ops.Shape(hidden_states_attn)[0], ops.Constant(value_ints=[1]))
        s2_end_1 = ops.Reshape(total_output_len, ops.Constant(value_ints=[1]))

        final_encoder_hidden_states = ops.Slice(
            hidden_states_attn,
            starts=ops.Concat(s2_start_0, s2_start_1, axis=0), # Changed to separate arguments
            ends=ops.Concat(s2_end_0, s2_end_1, axis=0), 
            axes=ops.Constant(value_ints=[0, 1]),
        )
    else:  # If no cross-attention, the attention output is just for `hidden_states`
        final_hidden_states = hidden_states_attn
        final_encoder_hidden_states = (
            encoder_hidden_states  # encoder_hidden_states remains what it was (e.g., empty tensor)
        )

    # --- Post-Attention Processing ---
    # Apply attn.to_add_out if not context_pre_only and encoder_hidden_states was present
    # context_pre_only = false in config
    final_encoder_hidden_states = ops.Add(
            ops.MatMul(final_encoder_hidden_states, ops.Transpose(to_add_out_weight, perm=[1, 0])), to_add_out_bias
        )

    # Apply attn.to_out[0] (Linear proj)
    final_hidden_states = ops.Add(
        ops.MatMul(final_hidden_states, ops.Transpose(to_out_0_weight, perm=[1, 0])), to_out_0_bias
    )

    # Apply attn.to_out[1] (Dropout)
    final_hidden_states = ops.Dropout(final_hidden_states, ratio=to_out_1_dropout_p)
    # Return based on whether encoder_hidden_states was provided in the input
    # The output signature must be consistent for ONNX `If` operators.
    # So both outputs are always returned.
    return final_hidden_states, final_encoder_hidden_states


class JointAttnProcessor2_0Func(torch.autograd.Function):
    @staticmethod
    def forward(
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,  # Will be a dummy zero tensor if original was None
        attention_mask: torch.Tensor,  # Will be a dummy zero tensor if original was None
        # Parameters from `Attention` module (passed as separate arguments)
        attn_heads: int,
        attn_head_dim: int,
        attn_scale_qk: bool,
        attn_scale_val: float,
        attn_query_dim: int,
        attn_inner_dim: int,
        attn_inner_kv_dim: int,
        # Weights and Biases for Attention Projections (`attn.to_q`, `attn.to_k`, `attn.to_v`)
        to_q_weight: torch.Tensor,
        to_q_bias: torch.Tensor,
        to_k_weight: torch.Tensor,
        to_k_bias: torch.Tensor,
        to_v_weight: torch.Tensor,
        to_v_bias: torch.Tensor,
        # RMSNorm parameters (`attn.norm_q`, `attn.norm_k`, `attn.norm_added_q`, `attn.norm_added_k`)
        # Use a consistent structure for weights and epsilons
        norm_q_weight: torch.Tensor,
        norm_q_eps: float,
        norm_k_weight: torch.Tensor,
        norm_k_eps: float,
        # Weights and Biases for Cross-Attention Projections (`attn.add_q_proj`, etc.)
        add_q_proj_weight: torch.Tensor,
        add_q_proj_bias: torch.Tensor,
        add_k_proj_weight: torch.Tensor,
        add_k_proj_bias: torch.Tensor,
        add_v_proj_weight: torch.Tensor,
        add_v_proj_bias: torch.Tensor,
        norm_added_q_weight: torch.Tensor,
        norm_added_q_eps: float,
        norm_added_k_weight: torch.Tensor,
        norm_added_k_eps: float,
        # Weights and Biases for Output Projections (`attn.to_out[0]`, `attn.to_out[1]`)
        to_out_0_weight: torch.Tensor,
        to_out_0_bias: torch.Tensor,
        to_out_1_dropout_p: float,
        # Other flags
        attn_context_pre_only: bool,
        attn_added_kv_proj_dim: int,
        to_add_out_weight: torch.Tensor,
        to_add_out_bias: torch.Tensor,
        # --- Internal flags for the autograd.Function to replicate exact behavior ---
        # Indicates if `encoder_hidden_states` was originally None.
        # This will allow proper `if encoder_hidden_states is not None` logic.
        _original_encoder_hidden_states_was_none: bool,
        _original_attention_mask_was_none: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # Returns two tensors as per JointAttnProcessor2_0Onnx
        # Replicate PyTorch forward logic for JointAttnProcessor2_0
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        # Use the internal flag to control conditional logic for PyTorch execution
        # (This differs from ONNXScript where a dummy tensor would be passed)
        actual_encoder_hidden_states = None
        if not _original_encoder_hidden_states_was_none:
            actual_encoder_hidden_states = encoder_hidden_states  # Use the passed non-None tensor

        actual_attention_mask = None
        if not _original_attention_mask_was_none:
            actual_attention_mask = attention_mask  # Use the passed non-None tensor

        # --- Sample Projections ---
        query = F.linear(hidden_states, to_q_weight, to_q_bias)
        key = F.linear(hidden_states, to_k_weight, to_k_bias)
        value = F.linear(hidden_states, to_v_weight, to_v_bias)

        # Reshape for multi-head attention and transpose
        query = query.view(batch_size, -1, attn_heads, attn_head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn_heads, attn_head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn_heads, attn_head_dim).transpose(1, 2)
        # Apply RMSNorm if enabled (norm_q, norm_k)
        # Note: RMSNorm here is the PyTorch module's forward, not the ONNXScript function.
        # We need an instance of RMSNorm or a functional version
        query = CustomRMSNormFunc.apply(query, norm_q_weight, norm_q_eps)
        key = CustomRMSNormFunc.apply(key, norm_k_weight, norm_k_eps)

        # --- Context Projections (conditional on actual_encoder_hidden_states) ---
        if actual_encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = F.linear(
                actual_encoder_hidden_states, add_q_proj_weight, add_q_proj_bias
            )
            encoder_hidden_states_key_proj = F.linear(actual_encoder_hidden_states, add_k_proj_weight, add_k_proj_bias)
            encoder_hidden_states_value_proj = F.linear(
                actual_encoder_hidden_states, add_v_proj_weight, add_v_proj_bias
            )

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn_heads, attn_head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn_heads, attn_head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn_heads, attn_head_dim
            ).transpose(1, 2)

            encoder_hidden_states_query_proj = CustomRMSNormFunc.apply(
                encoder_hidden_states_query_proj, norm_added_q_weight, norm_added_q_eps, 
            )
            encoder_hidden_states_key_proj = CustomRMSNormFunc.apply(
                encoder_hidden_states_key_proj, norm_added_k_weight, norm_added_k_eps, 
            )
            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        # --- Scaled Dot-Product Attention ---
        # Ensure attention_mask is passed only if it's not None
        sdp_attention_mask = actual_attention_mask if actual_attention_mask is not None else None

        hidden_states_attn = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False, attn_mask=sdp_attention_mask
        )

        hidden_states_attn = hidden_states_attn.transpose(1, 2).reshape(batch_size, -1, attn_heads * attn_head_dim)
        hidden_states_attn = hidden_states_attn.to(query.dtype)  # Convert back to original query dtype

        # --- Split attention outputs if actual_encoder_hidden_states was present ---
        # The output of SDPA contains both hidden_states and encoder_hidden_states concatenated
        final_hidden_states = hidden_states_attn
        final_encoder_hidden_states = None

        if actual_encoder_hidden_states is not None:
            # Need to split the combined output
            split_size = residual.shape[1]  # Length of the original 'sample' sequence
            final_hidden_states, final_encoder_hidden_states = (
                hidden_states_attn[:, :split_size],
                hidden_states_attn[:, split_size:],
            )

            # Apply attn.to_add_out if not context_pre_only
            if not attn_context_pre_only:
                final_encoder_hidden_states = F.linear(final_encoder_hidden_states, to_add_out_weight, to_add_out_bias)

        # --- Post-Attention Processing ---
        # Apply attn.to_out[0] (Linear proj)
        final_hidden_states = F.linear(final_hidden_states, to_out_0_weight, to_out_0_bias)
        # Apply attn.to_out[1] (Dropout)
        final_hidden_states = F.dropout(final_hidden_states, p=to_out_1_dropout_p, training=False)

        # Ensure encoder_hidden_states is returned as a tensor, not None,
        # to match the consistent output signature of the ONNXScript function.
        if final_encoder_hidden_states is None:
            # Return an empty tensor or a zero tensor if it was originally None
            # Matching the input's shape for a dummy tensor.
            final_encoder_hidden_states = torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype)

        return final_hidden_states, final_encoder_hidden_states

    @staticmethod
    def symbolic(
        g: torch.Graph,
        hidden_states: torch.Value,
        encoder_hidden_states: torch.Value,
        attention_mask: torch.Value,
        attn_heads: int,
        attn_head_dim: int,
        attn_scale_qk: bool,
        attn_scale_val: float,
        attn_query_dim: int,
        attn_inner_dim: int,
        attn_inner_kv_dim: int,
        to_q_weight: torch.Value,
        to_q_bias: torch.Value,
        to_k_weight: torch.Value,
        to_k_bias: torch.Value,
        to_v_weight: torch.Value,
        to_v_bias: torch.Value,
        norm_q_weight: torch.Value,
        norm_q_eps: float,
        norm_k_weight: torch.Value,
        norm_k_eps: float,
        add_q_proj_weight: torch.Value,
        add_q_proj_bias: torch.Value,
        add_k_proj_weight: torch.Value,
        add_k_proj_bias: torch.Value,
        add_v_proj_weight: torch.Value,
        add_v_proj_bias: torch.Value,
        norm_added_q_weight: torch.Value,
        norm_added_q_eps: float,
        norm_added_k_weight: torch.Value,
        norm_added_k_eps: float,
        to_out_0_weight: torch.Value,
        to_out_0_bias: torch.Value,
        to_out_1_dropout_p: float,
        attn_context_pre_only: bool,
        attn_added_kv_proj_dim: int,
        to_add_out_weight: torch.Value,
        to_add_out_bias: torch.Value,
        _original_encoder_hidden_states_was_none: bool,  # From ctx
        _original_attention_mask_was_none: bool,  # From ctx
    ) -> Tuple[torch.Value, torch.Value]:
        # Pass all relevant parameters to the ONNXScript function.
        # The ONNXScript function will handle the conditional logic based on
        # whether encoder_hidden_states/attention_mask are zero-sized/empty tensors.

        result = g.onnxscript_op(
            JointAttnProcessor2_0Onnx,
            hidden_states,
            encoder_hidden_states,  # Pass the (potentially dummy) tensor
            attention_mask,  # Pass the (potentially dummy) tensor
            attn_heads_i=attn_heads,
            attn_head_dim_i=attn_head_dim,
            attn_scale_qk_i=attn_scale_qk,
            attn_scale_val_f=attn_scale_val,
            attn_query_dim_i=attn_query_dim,
            attn_inner_dim_i=attn_inner_dim,
            attn_inner_kv_dim_i=attn_inner_kv_dim,
            to_q_weight=to_q_weight,
            to_q_bias=to_q_bias,
            to_k_weight=to_k_weight,
            to_k_bias=to_k_bias,
            to_v_weight=to_v_weight,
            to_v_bias=to_v_bias,
            norm_q_weight=norm_q_weight,
            norm_q_eps_f=norm_q_eps,
            norm_k_weight=norm_k_weight,
            norm_k_eps_f=norm_k_eps,
            add_q_proj_weight=add_q_proj_weight,
            add_q_proj_bias=add_q_proj_bias,
            add_k_proj_weight=add_k_proj_weight,
            add_k_proj_bias=add_k_proj_bias,
            add_v_proj_weight=add_v_proj_weight,
            add_v_proj_bias=add_v_proj_bias,
            norm_added_q_weight=norm_added_q_weight,
            norm_added_q_eps_f=norm_added_q_eps,
            norm_added_k_weight=norm_added_k_weight,
            norm_added_k_eps_f=norm_added_k_eps,
            to_out_0_weight=to_out_0_weight,
            to_out_0_bias=to_out_0_bias,
            to_out_1_dropout_p_f=to_out_1_dropout_p,
            attn_context_pre_only_i=attn_context_pre_only,
            attn_added_kv_proj_dim_i=attn_added_kv_proj_dim,
            to_add_out_weight=to_add_out_weight,
            to_add_out_bias=to_add_out_bias,
        )
        return result


class JointAttnProcessor2_0AIC(nn.Module):
    """
    JointAttnProcessor2_0 module that works by replacing the current processor instance
    with compiler-known custom-op via JointAttnProcessor2_0Func.
    Inherits from the original JointAttnProcessor2_0 to maintain its API.
    """

    def __init__(self, original_attn_module: Attention):
        # Call the parent's constructor. This sets up the `hasattr(F, "scaled_dot_product_attention")` check.
        super().__init__()

        # Store a reference to the `Attention` module to extract its parameters
        self.attn = original_attn_module

        # Extract all parameters needed by JointAttnProcessor2_0Func.apply()
        self._extract_parameters(original_attn_module)

    def _extract_parameters(self, attn_module: Attention):
        # This helper method stores all necessary parameters from the Attention module
        # It's separated for clarity and to keep __init__ clean.

        self.attn_heads = attn_module.heads
        self.attn_head_dim = attn_module.dim_head
        self.attn_scale_qk = attn_module.scale_qk
        self.attn_scale_val = attn_module.scale
        self.attn_query_dim = attn_module.query_dim
        self.attn_inner_dim = attn_module.inner_dim
        self.attn_inner_kv_dim = attn_module.inner_kv_dim

        # Helper to get parameter or a dummy zero tensor if it's None (e.g., bias=False or norm=None)
        # We define it here to access self.attn_head_dim etc. during extraction.
        def get_param_or_dummy_zero(param, default_shape_dim=None):
            if isinstance(param, nn.Parameter):  # It's a parameter, use its data
                return param
            if param is None:  # It's None, create a dummy zero tensor
                if default_shape_dim is not None:
                    # For biases of Linear, shape is (out_features,)
                    if isinstance(default_shape_dim, int):
                        return torch.zeros(default_shape_dim)
                    # For weights of RMSNorm, shape is (dim,)
                    if isinstance(default_shape_dim, tuple):
                        return torch.zeros(default_shape_dim)
                return torch.tensor(0.0)  # Fallback scalar zero if shape not specific
            return param  # If it's a non-nn.Parameter object, return as is (e.g., RMSNorm module itself)

        # Projection weights and biases
        self.to_q_weight = attn_module.to_q.weight
        self.to_q_bias = get_param_or_dummy_zero(attn_module.to_q.bias, attn_module.to_q.out_features)
        self.to_k_weight = attn_module.to_k.weight
        self.to_k_bias = get_param_or_dummy_zero(attn_module.to_k.bias, attn_module.to_k.out_features)
        self.to_v_weight = attn_module.to_v.weight
        self.to_v_bias = get_param_or_dummy_zero(attn_module.to_v.bias, attn_module.to_v.out_features)
        # RMSNorm parameters (for norm_q, norm_k, norm_added_q, norm_added_k)
        # Assumed norm objects have .weight, .eps, .elementwise_affine
        self.norm_q_weight = get_param_or_dummy_zero(attn_module.norm_q.weight, (self.attn_head_dim,))
        self.norm_q_eps = attn_module.norm_q.eps if hasattr(attn_module.norm_q, "eps") else 1e-6
        self.norm_q_elementwise_affine = (
            attn_module.norm_q.elementwise_affine if hasattr(attn_module.norm_q, "elementwise_affine") else False
        )

        self.norm_k_weight = get_param_or_dummy_zero(attn_module.norm_k.weight, (self.attn_head_dim,))
        self.norm_k_eps = attn_module.norm_k.eps if hasattr(attn_module.norm_k, "eps") else 1e-6
        self.norm_k_elementwise_affine = (
            attn_module.norm_k.elementwise_affine if hasattr(attn_module.norm_k, "elementwise_affine") else False
        )

        # Cross-attention projection weights and biases (add_q_proj etc.)
        self.add_q_proj_weight = get_param_or_dummy_zero(
            attn_module.add_q_proj.weight, attn_module.add_q_proj.out_features
        )
        self.add_q_proj_bias = get_param_or_dummy_zero(attn_module.add_q_proj.bias, attn_module.add_q_proj.out_features)
        self.add_k_proj_weight = get_param_or_dummy_zero(
            attn_module.add_k_proj.weight, attn_module.add_k_proj.out_features
        )
        self.add_k_proj_bias = get_param_or_dummy_zero(attn_module.add_k_proj.bias, attn_module.add_k_proj.out_features)
        self.add_v_proj_weight = get_param_or_dummy_zero(
            attn_module.add_v_proj.weight, attn_module.add_v_proj.out_features
        )
        self.add_v_proj_bias = get_param_or_dummy_zero(attn_module.add_v_proj.bias, attn_module.add_v_proj.out_features)
        # Norms for added projections
        self.norm_added_q_weight = get_param_or_dummy_zero(attn_module.norm_added_q.weight, (self.attn_head_dim,))
        self.norm_added_q_eps = attn_module.norm_added_q.eps if hasattr(attn_module.norm_added_q, "eps") else 1e-6
        self.norm_added_q_elementwise_affine = (
            attn_module.norm_added_q.elementwise_affine
            if hasattr(attn_module.norm_added_q, "elementwise_affine")
            else False
        )

        self.norm_added_k_weight = get_param_or_dummy_zero(attn_module.norm_added_k.weight, (self.attn_head_dim,))
        self.norm_added_k_eps = attn_module.norm_added_k.eps if hasattr(attn_module.norm_added_k, "eps") else 1e-6
        self.norm_added_k_elementwise_affine = (
            attn_module.norm_added_k.elementwise_affine
            if hasattr(attn_module.norm_added_k, "elementwise_affine")
            else False
        )

        # Output projection weights and biases (from ModuleList)
        self.to_out_0_weight = attn_module.to_out[0].weight
        self.to_out_0_bias = get_param_or_dummy_zero(attn_module.to_out[0].bias, attn_module.to_out[0].out_features)
        self.to_out_1_dropout_p = attn_module.to_out[1].p  # Dropout probability
        # Other flags from Attention
        self.attn_context_pre_only = attn_module.context_pre_only
        self.attn_added_kv_proj_dim = attn_module.added_kv_proj_dim

        self.to_add_out_weight = get_param_or_dummy_zero(
            attn_module.to_add_out.weight, attn_module.to_add_out.out_features
        )
        self.to_add_out_bias = get_param_or_dummy_zero(attn_module.to_add_out.bias, attn_module.to_add_out.out_features)

    # Override the __call__ method of the parent class
    def __call__(
        self,
        attn_context: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Determine if encoder_hidden_states and attention_mask were originally None
        # These flags are passed to the autograd.Function to guide symbolic export.
        _original_encoder_hidden_states_was_none = (encoder_hidden_states is None) or (
            encoder_hidden_states.numel() == 0
        )
        _original_attention_mask_was_none = (attention_mask is None) or (attention_mask.numel() == 0)

        # Prepare dummy tensors for optional inputs if they are None,
        # to maintain consistent tensor types for `apply` call and ONNX.
        device = hidden_states.device
        dtype = hidden_states.dtype

        encoder_hidden_states_to_pass = encoder_hidden_states
        if _original_encoder_hidden_states_was_none:
            encoder_hidden_states_to_pass = torch.empty(0, device=device, dtype=dtype)

        attention_mask_to_pass = attention_mask
        if _original_attention_mask_was_none:
            # Create a zero-sized tensor or a tensor of appropriate shape for masking
            # For ONNX, an empty tensor is often interpreted as "None" for ops like SDPA.
            # If your ONNXScript function expects a specific mask shape (e.g., (1,1,Q,K)),
            # you might need to create a mask filled with specific values here.
            attention_mask_to_pass = torch.empty(0, device=device, dtype=dtype)

        # Pass all relevant parameters to the autograd.Function
        return JointAttnProcessor2_0Func.apply(
            hidden_states,
            encoder_hidden_states_to_pass,
            attention_mask_to_pass,  # Pass the (potentially dummy) mask
            self.attn_heads,
            self.attn_head_dim,
            self.attn_scale_qk,
            self.attn_scale_val,
            self.attn_query_dim,
            self.attn_inner_dim,
            self.attn_inner_kv_dim,
            self.to_q_weight,
            self.to_q_bias,
            self.to_k_weight,
            self.to_k_bias,
            self.to_v_weight,
            self.to_v_bias,
            self.norm_q_weight,
            self.norm_q_eps,
            # self.norm_q_elementwise_affine,
            self.norm_k_weight,
            self.norm_k_eps,
            # self.norm_k_elementwise_affine,
            self.add_q_proj_weight,
            self.add_q_proj_bias,
            self.add_k_proj_weight,
            self.add_k_proj_bias,
            self.add_v_proj_weight,
            self.add_v_proj_bias,
            self.norm_added_q_weight,
            self.norm_added_q_eps,
            # self.norm_added_q_elementwise_affine,
            self.norm_added_k_weight,
            self.norm_added_k_eps,
            # self.norm_added_k_elementwise_affine,
            self.to_out_0_weight,
            self.to_out_0_bias,
            self.to_out_1_dropout_p,
            self.attn_context_pre_only,
            self.attn_added_kv_proj_dim,
            self.to_add_out_weight,
            self.to_add_out_bias,
            _original_encoder_hidden_states_was_none,  # Passed as flag to autograd.Function
            _original_attention_mask_was_none,  # Passed as flag to autograd.Function
        )


# MMDit block
@onnxscript.script(CUSTOM_OPSET)
def JointTransformerBlockOnnx(
    hidden_states: onnxscript.FLOAT,
    encoder_hidden_states: onnxscript.FLOAT,
    temb: onnxscript.FLOAT,
    # Weights and parameters for norm1 (AdaLayerNormZeroOnnx)
    norm1_linear_weight: onnxscript.FLOAT,
    norm1_linear_bias: onnxscript.FLOAT,
    norm1_epsilon: float,
    # Weights and parameters for norm1_context (AdaLayerNormZeroOnnx)
    norm1_context_linear_weight: onnxscript.FLOAT,
    norm1_context_linear_bias: onnxscript.FLOAT,
    norm1_context_epsilon: float,
    # Weights and parameters for attn (JointAttnProcessor2_0Onnx)
    attn_heads: int,
    attn_head_dim: int,
    attn_scale_qk: bool,
    attn_scale_val: float,
    attn_query_dim: int,
    attn_inner_dim: int,
    attn_inner_kv_dim: int,
    attn_to_q_weight: onnxscript.FLOAT,
    attn_to_q_bias: onnxscript.FLOAT,
    attn_to_k_weight: onnxscript.FLOAT,
    attn_to_k_bias: onnxscript.FLOAT,
    attn_to_v_weight: onnxscript.FLOAT,
    attn_to_v_bias: onnxscript.FLOAT,
    attn_norm_q_weight: onnxscript.FLOAT,
    attn_norm_q_eps: float,
    attn_norm_q_elementwise_affine: bool,
    attn_norm_k_weight: onnxscript.FLOAT,
    attn_norm_k_eps: float,
    attn_norm_k_elementwise_affine: bool,
    attn_add_q_proj_weight: onnxscript.FLOAT,
    attn_add_q_proj_bias: onnxscript.FLOAT,
    attn_add_k_proj_weight: onnxscript.FLOAT,
    attn_add_k_proj_bias: onnxscript.FLOAT,
    attn_add_v_proj_weight: onnxscript.FLOAT,
    attn_add_v_proj_bias: onnxscript.FLOAT,
    attn_norm_added_q_weight: onnxscript.FLOAT,
    attn_norm_added_q_eps: float,
    attn_norm_added_q_elementwise_affine: bool,
    attn_norm_added_k_weight: onnxscript.FLOAT,
    attn_norm_added_k_eps: float,
    attn_norm_added_k_elementwise_affine: bool,
    attn_to_out_0_weight: onnxscript.FLOAT,
    attn_to_out_0_bias: onnxscript.FLOAT,
    attn_to_out_1_dropout_p: float,
    attn_context_pre_only_flag: bool,  # This needs to be set to False for this export
    attn_added_kv_proj_dim: int,
    attn_to_add_out_weight: onnxscript.FLOAT,
    attn_to_add_out_bias: onnxscript.FLOAT,
    # Weights and parameters for norm2 (RMSNormOnnx)
    norm2_weight: onnxscript.FLOAT,
    norm2_eps: float,
    # Weights and parameters for ff (FeedForwardOnnx)
    ff_dim: int,
    ff_dim_out: int,
    ff_mult: int,
    ff_dropout_ratio: float,
    ff_final_dropout: bool,
    ff_bias: bool,  # Assuming this is the bias for both act_fn_proj and project_out
    ff_act_fn_proj_weight: onnxscript.FLOAT,
    ff_act_fn_proj_bias: onnxscript.FLOAT,  # Passed if bias=True
    ff_project_out_weight: onnxscript.FLOAT,
    ff_project_out_bias: onnxscript.FLOAT,  # Passed if bias=True
    # Weights and parameters for norm2_context (RMSNormOnnx)
    norm2_context_weight: onnxscript.FLOAT,
    norm2_context_eps: float,
    # Weights and parameters for ff_context (FeedForwardOnnx)
    ff_context_dim: int,
    ff_context_dim_out: int,
    ff_context_mult: int,
    ff_context_dropout_ratio: float,
    ff_context_final_dropout: bool,
    ff_context_bias: bool,
    ff_context_act_fn_proj_weight: onnxscript.FLOAT,
    ff_context_act_fn_proj_bias: onnxscript.FLOAT,
    ff_context_project_out_weight: onnxscript.FLOAT,
    ff_context_project_out_bias: onnxscript.FLOAT,
):
    # Fixed conditions: use_dual_attention = False, context_pre_only = False, _chunk_size = None

    # ----------------------------------------------------------------------
    # 1. Norm1 (Fixed to AdaLayerNormZero as use_dual_attention is False)
    # ----------------------------------------------------------------------
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = AdaLayerNormZeroOnnx(
        hidden_states, temb, norm1_linear_weight, norm1_linear_bias, norm1_epsilon
    )

    # ----------------------------------------------------------------------
    # 2. Norm1_context (Fixed to AdaLayerNormZero as context_pre_only is False)
    # ----------------------------------------------------------------------
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = AdaLayerNormZeroOnnx(
        encoder_hidden_states, temb, norm1_context_linear_weight, norm1_context_linear_bias, norm1_context_epsilon
    )

    # ----------------------------------------------------------------------
    # 3. Attention
    # ----------------------------------------------------------------------
    # JointAttnProcessor2_0Onnx returns (hidden_states_out, encoder_hidden_states_out)
    attn_output, context_attn_output = JointAttnProcessor2_0Onnx(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        attention_mask=ops.Constant(value_float=0.0),  # Assuming attention_mask is handled externally or not dynamic
        attn_heads=attn_heads,
        attn_head_dim=attn_head_dim,
        attn_scale_qk=attn_scale_qk,
        attn_scale_val=attn_scale_val,
        attn_query_dim=attn_query_dim,
        attn_inner_dim=attn_inner_dim,
        attn_inner_kv_dim=attn_inner_kv_dim,
        to_q_weight=attn_to_q_weight,
        to_q_bias=attn_to_q_bias,
        to_k_weight=attn_to_k_weight,
        to_k_bias=attn_to_k_bias,
        to_v_weight=attn_to_v_weight,
        to_v_bias=attn_to_v_bias,
        norm_q_weight=attn_norm_q_weight,
        norm_q_eps=attn_norm_q_eps,
        # norm_q_elementwise_affine=attn_norm_q_elementwise_affine,
        norm_k_weight=attn_norm_k_weight,
        norm_k_eps=attn_norm_k_eps,
        # norm_k_elementwise_affine=attn_norm_k_elementwise_affine,
        add_q_proj_weight=attn_add_q_proj_weight,
        add_q_proj_bias=attn_add_q_proj_bias,
        add_k_proj_weight=attn_add_k_proj_weight,
        add_k_proj_bias=attn_add_k_proj_bias,
        add_v_proj_weight=attn_add_v_proj_weight,
        add_v_proj_bias=attn_add_v_proj_bias,
        norm_added_q_weight=attn_norm_added_q_weight,
        norm_added_q_eps=attn_norm_added_q_eps,
        # norm_added_q_elementwise_affine=attn_norm_added_q_elementwise_affine,
        norm_added_k_weight=attn_norm_added_k_weight,
        norm_added_k_eps=attn_norm_added_k_eps,
        # norm_added_k_elementwise_affine=attn_norm_added_k_elementwise_affine,
        to_out_0_weight=attn_to_out_0_weight,
        to_out_0_bias=attn_to_out_0_bias,
        to_out_1_dropout_p=attn_to_out_1_dropout_p,
        attn_context_pre_only_flag=attn_context_pre_only_flag,
        attn_added_kv_proj_dim=attn_added_kv_proj_dim,
        to_add_out_weight=attn_to_add_out_weight,
        to_add_out_bias=attn_to_add_out_bias,
    )
    # Process attention outputs for the `hidden_states`.
    attn_output = ops.Mul(ops.Unsqueeze(gate_msa, axes=[1]), attn_output)
    hidden_states = ops.Add(hidden_states, attn_output)

    # Note: `if self.use_dual_attention` block is skipped (fixed to False)

    # ----------------------------------------------------------------------
    # 4. MLP for hidden_states
    # ----------------------------------------------------------------------
    norm_hidden_states = CustomRMSNorm(hidden_states, norm2_weight, norm2_eps)
    norm_hidden_states = ops.Add(
        ops.Mul(norm_hidden_states, ops.Add(ops.Constant(value_float=1.0), ops.Unsqueeze(scale_mlp, axes=[1]))),
        ops.Unsqueeze(shift_mlp, axes=[1]),
    )

    # Note: `if self._chunk_size is not None` block is skipped (fixed to None)
    ff_output = FeedForward(
        norm_hidden_states,
        ff_dim,
        ff_dim_out,
        ff_mult,
        ff_dropout_ratio,
        ff_final_dropout,
        ff_bias,
        ff_act_fn_proj_weight,
        ff_act_fn_proj_bias,
        ff_project_out_weight,
        ff_project_out_bias,
    )
    ff_output = ops.Mul(ops.Unsqueeze(gate_mlp, axes=[1]), ff_output)
    hidden_states = ops.Add(hidden_states, ff_output)
    # ----------------------------------------------------------------------
    # 5. Process attention outputs for the `encoder_hidden_states`.
    # ----------------------------------------------------------------------
    # Note: `if self.context_pre_only` block is skipped (fixed to False)

    context_attn_output = ops.Mul(ops.Unsqueeze(c_gate_msa, axes=[1]), context_attn_output)
    encoder_hidden_states = ops.Add(encoder_hidden_states, context_attn_output)

    norm_encoder_hidden_states = CustomRMSNorm(
        encoder_hidden_states, norm2_context_weight, norm2_context_eps, 
    )
    norm_encoder_hidden_states = ops.Add(
        ops.Mul(
            norm_encoder_hidden_states, ops.Add(ops.Constant(value_float=1.0), ops.Unsqueeze(c_scale_mlp, axes=[1]))
        ),
        ops.Unsqueeze(c_shift_mlp, axes=[1]),
    )

    # Note: `if self._chunk_size is not None` block is skipped (fixed to None)
    context_ff_output = FeedForward(
        norm_encoder_hidden_states,
        ff_context_dim,
        ff_context_dim_out,
        ff_context_mult,
        ff_context_dropout_ratio,
        ff_context_final_dropout,
        ff_context_bias,
        ff_context_act_fn_proj_weight,
        ff_context_act_fn_proj_bias,
        ff_context_project_out_weight,
        ff_context_project_out_bias,
    )
    encoder_hidden_states = ops.Add(
        encoder_hidden_states, ops.Mul(ops.Unsqueeze(c_gate_mlp, axes=[1]), context_ff_output)
    )

    return encoder_hidden_states, hidden_states


class JointTransformerBlockFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        # Fixed parameters (passed as Python literals during export)
        _use_dual_attention: bool,  # False
        _context_pre_only: bool,  # False
        _chunk_size: Optional[int],  # None
        _chunk_dim: int,  # Dummy, as _chunk_size is None
        # --- Parameters for norm1 (AdaLayerNormZero) ---
        norm1_linear_weight: torch.Tensor,
        norm1_linear_bias: torch.Tensor,
        norm1_epsilon: float,
        # --- Parameters for norm1_context (AdaLayerNormZero) ---
        norm1_context_linear_weight: torch.Tensor,
        norm1_context_linear_bias: torch.Tensor,
        norm1_context_epsilon: float,
        # --- Parameters for attn (JointAttnProcessor2_0) ---
        attn_heads: int,
        attn_head_dim: int,
        attn_scale_qk: bool,
        attn_scale_val: float,
        attn_query_dim: int,
        attn_inner_dim: int,
        attn_inner_kv_dim: int,
        attn_to_q_weight: torch.Tensor,
        attn_to_q_bias: torch.Tensor,
        attn_to_k_weight: torch.Tensor,
        attn_to_k_bias: torch.Tensor,
        attn_to_v_weight: torch.Tensor,
        attn_to_v_bias: torch.Tensor,
        attn_norm_q_weight: torch.Tensor,
        attn_norm_q_eps: float,
        # attn_norm_q_elementwise_affine: bool,
        attn_norm_k_weight: torch.Tensor,
        attn_norm_k_eps: float,
        # attn_norm_k_elementwise_affine: bool,
        attn_add_q_proj_weight: torch.Tensor,
        attn_add_q_proj_bias: torch.Tensor,
        attn_add_k_proj_weight: torch.Tensor,
        attn_add_k_proj_bias: torch.Tensor,
        attn_add_v_proj_weight: torch.Tensor,
        attn_add_v_proj_bias: torch.Tensor,
        attn_norm_added_q_weight: torch.Tensor,
        attn_norm_added_q_eps: float,
        # attn_norm_added_q_elementwise_affine: bool,
        attn_norm_added_k_weight: torch.Tensor,
        attn_norm_added_k_eps: float,
        # attn_norm_added_k_elementwise_affine: bool,
        attn_to_out_0_weight: torch.Tensor,
        attn_to_out_0_bias: torch.Tensor,
        attn_to_out_1_dropout_p: float,
        attn_context_pre_only_flag: bool,  # This is the internal flag for JointAttnProcessor2_0
        attn_added_kv_proj_dim: int,
        attn_to_add_out_weight: torch.Tensor,
        attn_to_add_out_bias: torch.Tensor,
        # --- Parameters for norm2 (RMSNorm) ---
        norm2_weight: torch.Tensor,
        norm2_eps: float,
         # --- Parameters for ff (FeedForward) ---
        ff_dim: int, ff_dim_out: int, ff_mult: int, ff_dropout_ratio: float, ff_final_dropout: bool,
        ff_act_fn_proj_weight: torch.Tensor, ff_act_fn_proj_bias: torch.Tensor,
        ff_project_out_weight: torch.Tensor, ff_project_out_bias: torch.Tensor,
        ff_activation_fn_type: int, # The type for FeedForwardFunc
        
        # --- Parameters for norm2_context (RMSNorm) ---
        norm2_context_weight: torch.Tensor,
        norm2_context_eps: float,
        # --- Parameters for ff_context (FeedForward) ---
        ff_context_dim: int,
        ff_context_dim_out: int,
        ff_context_mult: int,
        ff_context_dropout_ratio: float,
        ff_context_final_dropout: bool,
        ff_context_act_fn_proj_weight: torch.Tensor,
        ff_context_act_fn_proj_bias: torch.Tensor,
        ff_context_project_out_weight: torch.Tensor,
        ff_context_project_out_bias: torch.Tensor,
        ff_context_activation_fn_type: int,  # The type for FeedForwardFunc
        # Flags for `JointAttnProcessor2_0Func` to handle optional tensor inputs
        _attn_original_encoder_hidden_states_was_none: bool,
        _attn_original_attention_mask_was_none: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # --- Replicate PyTorch forward logic for fixed conditions ---
        # use_dual_attention = False, context_pre_only = False, _chunk_size = None

        # 1. Norm1 (AdaLayerNormZero)
        # Assuming AdaLayerNormZeroFunc.apply exists and works (from previous steps)
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = AdaLayerNormZeroFunc.apply(
            hidden_states, temb, norm1_linear_weight, norm1_linear_bias, norm1_epsilon
        )

        # 2. Norm1_context (AdaLayerNormZero)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = AdaLayerNormZeroFunc.apply(
            encoder_hidden_states, temb, norm1_context_linear_weight, norm1_context_linear_bias, norm1_context_epsilon
        )

        # 3. Attention (JointAttnProcessor2_0)
        # Prepare dummy attention_mask for processor if it was None
        attention_mask_for_attn = None
        if not _attn_original_attention_mask_was_none:
            attention_mask_for_attn = torch.empty(0)  # Dummy empty tensor for now

        attn_output, context_attn_output = JointAttnProcessor2_0Func.apply(
            norm_hidden_states,
            norm_encoder_hidden_states,  # Pass non-None to processor
            attention_mask_for_attn,  # Pass non-None to processor (potentially empty)
            attn_heads,
            attn_head_dim,
            attn_scale_qk,
            attn_scale_val,
            attn_query_dim,
            attn_inner_dim,
            attn_inner_kv_dim,
            attn_to_q_weight,
            attn_to_q_bias,
            attn_to_k_weight,
            attn_to_k_bias,
            attn_to_v_weight,
            attn_to_v_bias,
            attn_norm_q_weight,
            attn_norm_q_eps,
            # attn_norm_q_elementwise_affine,
            attn_norm_k_weight,
            attn_norm_k_eps,
            # attn_norm_k_elementwise_affine,
            attn_add_q_proj_weight,
            attn_add_q_proj_bias,
            attn_add_k_proj_weight,
            attn_add_k_proj_bias,
            attn_add_v_proj_weight,
            attn_add_v_proj_bias,
            attn_norm_added_q_weight,
            attn_norm_added_q_eps,
            # attn_norm_added_q_elementwise_affine,
            attn_norm_added_k_weight,
            attn_norm_added_k_eps,
            # attn_norm_added_k_elementwise_affine,
            attn_to_out_0_weight,
            attn_to_out_0_bias,
            attn_to_out_1_dropout_p,
            attn_context_pre_only_flag,
            attn_added_kv_proj_dim,
            attn_to_add_out_weight,
            attn_to_add_out_bias,
            False,  # _original_encoder_hidden_states_was_none - always false for this case
            _attn_original_attention_mask_was_none,
        )
        # Process attention outputs for the `hidden_states`.
        # Assuming original `hidden_states` (input to this forward) is what is added to.
        current_hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        # Note: `if self.use_dual_attention` block is skipped.

        # 4. MLP for hidden_states
        # Assuming RMSNormFunc.apply exists
        norm_hidden_states_mlp = CustomRMSNormFunc.apply(
            current_hidden_states, norm2_weight, norm2_eps
        )
        norm_hidden_states_mlp = norm_hidden_states_mlp * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)

        # Note: `if self._chunk_size is not None` block is skipped.
        # Assuming FeedForwardFunc.apply exists
        ff_output = FeedForwardFunc.apply(
            norm_hidden_states_mlp,
            ff_dim,
            ff_dim_out,
            ff_mult,
            ff_dropout_ratio,
            ff_final_dropout,
            ff_activation_fn_type,
            ff_act_fn_proj_weight,
            ff_act_fn_proj_bias,
            ff_project_out_weight,
            ff_project_out_bias,
        )
        current_hidden_states = current_hidden_states + gate_mlp.unsqueeze(1) * ff_output

        # 5. Process attention outputs for the `encoder_hidden_states`.
        # Note: `if self.context_pre_only` block is skipped.
        current_encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output

        norm_encoder_hidden_states_mlp = CustomRMSNormFunc.apply(
            current_encoder_hidden_states, norm2_context_weight, norm2_context_eps
        )
        norm_encoder_hidden_states_mlp = norm_encoder_hidden_states_mlp * (
            1 + c_scale_mlp.unsqueeze(1)
        ) + c_shift_mlp.unsqueeze(1)

        # Note: `if self._chunk_size is not None` block is skipped.
        context_ff_output = FeedForwardFunc.apply(
            norm_encoder_hidden_states_mlp,
            ff_context_dim,
            ff_context_dim_out,
            ff_context_mult,
            ff_context_dropout_ratio,
            ff_context_final_dropout,
            ff_context_activation_fn_type,
            ff_context_act_fn_proj_weight,
            ff_context_act_fn_proj_bias,
            ff_context_project_out_weight,
            ff_context_project_out_bias,
        )
        current_encoder_hidden_states = current_encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return current_encoder_hidden_states, current_hidden_states

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.Graph,
        hidden_states: torch.Value,
        encoder_hidden_states: torch.Value,
        temb: torch.Value,
        # Fixed parameters (passed as Python literals from ctx)
        _use_dual_attention: bool,
        _context_pre_only: bool,
        _chunk_size: Optional[int],
        _chunk_dim: int,
        # Parameters for norm1 (AdaLayerNormZero)
        norm1_linear_weight: torch.Value,
        norm1_linear_bias: torch.Value,
        norm1_epsilon: float,
        # Parameters for norm1_context (AdaLayerNormZero)
        norm1_context_linear_weight: torch.Value,
        norm1_context_linear_bias: torch.Value,
        norm1_context_epsilon: float,
        # Parameters for attn (JointAttnProcessor2_0)
        attn_heads: int,
        attn_head_dim: int,
        attn_scale_qk: bool,
        attn_scale_val: float,
        attn_query_dim: int,
        attn_inner_dim: int,
        attn_inner_kv_dim: int,
        attn_to_q_weight: torch.Value,
        attn_to_q_bias: torch.Value,
        attn_to_k_weight: torch.Value,
        attn_to_k_bias: torch.Value,
        attn_to_v_weight: torch.Value,
        attn_to_v_bias: torch.Value,
        attn_norm_q_weight: torch.Value,
        attn_norm_q_eps: float,
        attn_norm_q_elementwise_affine: bool,
        attn_norm_k_weight: torch.Value,
        attn_norm_k_eps: float,
        attn_norm_k_elementwise_affine: bool,
        attn_add_q_proj_weight: torch.Value,
        attn_add_q_proj_bias: torch.Value,
        attn_add_k_proj_weight: torch.Value,
        attn_add_k_proj_bias: torch.Value,
        attn_add_v_proj_weight: torch.Value,
        attn_add_v_proj_bias: torch.Value,
        attn_norm_added_q_weight: torch.Value,
        attn_norm_added_q_eps: float,
        attn_norm_added_q_elementwise_affine: bool,
        attn_norm_added_k_weight: torch.Value,
        attn_norm_added_k_eps: float,
        attn_norm_added_k_elementwise_affine: bool,
        attn_to_out_0_weight: torch.Value,
        attn_to_out_0_bias: torch.Value,
        attn_to_out_1_dropout_p: float,
        attn_context_pre_only_flag: bool,
        attn_added_kv_proj_dim: int,
        attn_to_add_out_weight: torch.Value,
        attn_to_add_out_bias: torch.Value,
        # Parameters for norm2 (RMSNorm)
        norm2_weight: torch.Value,
        norm2_eps: float,
        norm2_elementwise_affine: bool,
        # Parameters for ff (FeedForward)
        ff_dim: int,
        ff_dim_out: int,
        ff_mult: int,
        ff_dropout_ratio: float,
        ff_final_dropout: bool,
        ff_act_fn_proj_weight: torch.Value,
        ff_act_fn_proj_bias: torch.Value,
        ff_project_out_weight: torch.Value,
        ff_project_out_bias: torch.Value,
        ff_activation_fn_type: int,
        # Parameters for norm2_context (RMSNorm)
        norm2_context_weight: torch.Value,
        norm2_context_eps: float,
        norm2_context_elementwise_affine: bool,
        # Parameters for ff_context (FeedForward)
        ff_context_dim: int,
        ff_context_dim_out: int,
        ff_context_mult: int,
        ff_context_dropout_ratio: float,
        ff_context_final_dropout: bool,
        ff_context_act_fn_proj_weight: torch.Value,
        ff_context_act_fn_proj_bias: torch.Value,
        ff_context_project_out_weight: torch.Value,
        ff_context_project_out_bias: torch.Value,
        ff_context_activation_fn_type: int,
        _attn_original_encoder_hidden_states_was_none: bool,
        _attn_original_attention_mask_was_none: bool,
    ) -> Tuple[torch.Value, torch.Value]:
        # Pass all parameters directly to the ONNXScript function
        result = g.onnxscript_op(
            JointTransformerBlockOnnx,
            hidden_states,
            encoder_hidden_states,
            temb,
            # No need to pass _use_dual_attention, _context_pre_only, _chunk_size, _chunk_dim
            # as separate inputs to JointTransformerBlockOnnx because it's already hardcoded
            # to that specific path in its ONNXScript definition.
            # If JointTransformerBlockOnnx were generic, these would be passed.
            norm1_linear_weight,
            norm1_linear_bias,
            norm1_epsilon,
            norm1_context_linear_weight,
            norm1_context_linear_bias,
            norm1_context_epsilon,
            attn_heads_i=attn_heads,
            attn_head_dim_i=attn_head_dim,
            attn_scale_qk_i=attn_scale_qk,
            attn_scale_val_f=attn_scale_val,
            attn_query_dim_i=attn_query_dim,
            attn_inner_dim_i=attn_inner_dim,
            attn_inner_kv_dim_i=attn_inner_kv_dim,
            attn_to_q_weight=attn_to_q_weight,
            attn_to_q_bias=attn_to_q_bias,
            attn_to_k_weight=attn_to_k_weight,
            attn_to_k_bias=attn_to_k_bias,
            attn_to_v_weight=attn_to_v_weight,
            attn_to_v_bias=attn_to_v_bias,
            attn_norm_q_weight=attn_norm_q_weight,
            attn_norm_q_eps_f=attn_norm_q_eps,
            attn_norm_q_elementwise_affine_i=attn_norm_q_elementwise_affine,
            attn_norm_k_weight=attn_norm_k_weight,
            attn_norm_k_eps_f=attn_norm_k_eps,
            attn_norm_k_elementwise_affine_i=attn_norm_k_elementwise_affine,
            attn_add_q_proj_weight=attn_add_q_proj_weight,
            attn_add_q_proj_bias=attn_add_q_proj_bias,
            attn_add_k_proj_weight=attn_add_k_proj_weight,
            attn_add_k_proj_bias=attn_add_k_proj_bias,
            attn_add_v_proj_weight=attn_add_v_proj_weight,
            attn_add_v_proj_bias=attn_add_v_proj_bias,
            attn_norm_added_q_weight=attn_norm_added_q_weight,
            attn_norm_added_q_eps_f=attn_norm_added_q_eps,
            attn_norm_added_q_elementwise_affine_i=attn_norm_added_q_elementwise_affine,
            attn_norm_added_k_weight=attn_norm_added_k_weight,
            attn_norm_added_k_eps_f=attn_norm_added_k_eps,
            attn_norm_added_k_elementwise_affine_i=attn_norm_added_k_elementwise_affine,
            attn_to_out_0_weight=attn_to_out_0_weight,
            attn_to_out_0_bias=attn_to_out_0_bias,
            attn_to_out_1_dropout_p_f=attn_to_out_1_dropout_p,
            attn_context_pre_only_flag_i=attn_context_pre_only_flag,
            attn_added_kv_proj_dim_i=attn_added_kv_proj_dim,
            attn_to_add_out_weight=attn_to_add_out_weight,
            attn_to_add_out_bias=attn_to_add_out_bias,
            norm2_weight=norm2_weight,
            norm2_eps_f=norm2_eps,
            norm2_elementwise_affine_i=norm2_elementwise_affine,
            ff_dim_i=ff_dim,
            ff_dim_out_i=ff_dim_out,
            ff_mult_i=ff_mult,
            ff_dropout_ratio_f=ff_dropout_ratio,
            ff_final_dropout_i=ff_final_dropout,
            ff_act_fn_proj_weight=ff_act_fn_proj_weight,
            ff_act_fn_proj_bias=ff_act_fn_proj_bias,
            ff_project_out_weight=ff_project_out_weight,
            ff_project_out_bias=ff_project_out_bias,
            ff_activation_fn_type_i=ff_activation_fn_type,
            norm2_context_weight=norm2_context_weight,
            norm2_context_eps_f=norm2_context_eps,
            norm2_context_elementwise_affine_i=norm2_context_elementwise_affine,
            ff_context_dim_i=ff_context_dim,
            ff_context_dim_out_i=ff_context_dim_out,
            ff_context_mult_i=ff_context_mult,
            ff_context_dropout_ratio_f=ff_context_dropout_ratio,
            ff_context_final_dropout_i=ff_context_final_dropout,
            ff_context_act_fn_proj_weight=ff_context_act_fn_proj_weight,
            ff_context_act_fn_proj_bias=ff_context_act_fn_proj_bias,
            ff_context_project_out_weight=ff_context_project_out_weight,
            ff_context_project_out_bias=ff_context_project_out_bias,
            ff_context_activation_fn_type_i=ff_context_activation_fn_type,
        )
        return result


# Helper for safely getting params or dummy zeros
def _get_param_or_dummy_zero(param, default_tensor_if_none: Optional[torch.Tensor] = None):
    if isinstance(param, nn.Parameter):
        return param
    if param is None:
        if default_tensor_if_none is not None:
            return default_tensor_if_none
        return torch.tensor(0.0)  # Scalar zero for non-tensor defaults
    return param


def _get_activation_fn_type_from_module(act_fn_module: nn.Module) -> int:
    # Based on the original FeedForward's activation_fn logic
    # Mapping: 0: "gelu", 1: "gelu-approximate"
    if isinstance(act_fn_module, GELU):
        if hasattr(act_fn_module, "approximate") and act_fn_module.approximate == "tanh":
            return 1  # "gelu-approximate"
        else:
            return 0  # "gelu"
    else:
        raise ValueError(f"Unsupported activation function module type: {type(act_fn_module)}")


class JointTransformerBlockAIC(nn.Module):
    """
    JointTransformerBlock module that works by replacing the current module with
    compiler-known custom-op via JointTransformerBlockFunc.
    This version is hardcoded for specific conditions:
    - use_dual_attention = False
    - context_pre_only = False
    - _chunk_size = None
    """

    def __init__(self, original_module: JointTransformerBlock):
        super().__init__()
        # Store original fixed parameters
        self._use_dual_attention = original_module.use_dual_attention  # Should be False
        self._context_pre_only = original_module.context_pre_only  # Should be False
        self._chunk_size = original_module._chunk_size  # Should be None
        self._chunk_dim = original_module._chunk_dim  # Should be 0

        if self._use_dual_attention or self._context_pre_only or self._chunk_size is not None:
            raise ValueError(
                "JointTransformerBlockAIC is specialized for "
                "use_dual_attention=False, context_pre_only=False, _chunk_size=None. "
                "Found different values."
            )

        # --- Extract parameters for norm1 (AdaLayerNormZero) ---
        self.norm1_linear_weight = original_module.norm1.linear.weight
        self.norm1_linear_bias = _get_param_or_dummy_zero(
            original_module.norm1.linear.bias, torch.zeros(original_module.norm1.linear.out_features)
        )
        self.norm1_epsilon = original_module.norm1.norm.eps
        # --- Extract parameters for norm1_context (AdaLayerNormZero) ---
        self.norm1_context_linear_weight = original_module.norm1_context.linear.weight
        self.norm1_context_linear_bias = _get_param_or_dummy_zero(
            original_module.norm1_context.linear.bias, torch.zeros(original_module.norm1_context.linear.out_features)
        )
        self.norm1_context_epsilon = original_module.norm1_context.norm.eps

        # --- Extract parameters for attn (Attention / JointAttnProcessor2_0) ---
        # Assuming original_module.attn is an Attention instance.
        # It's better if it's already wrapped in AttentionAIC, but we can extract raw params.
        self.attn_heads = original_module.attn.heads
        self.attn_head_dim = original_module.attn.dim_head
        self.attn_scale_qk = original_module.attn.scale_qk
        self.attn_scale_val = original_module.attn.scale
        self.attn_query_dim = original_module.attn.query_dim
        self.attn_inner_dim = original_module.attn.inner_dim
        self.attn_inner_kv_dim = original_module.attn.inner_kv_dim

        self.attn_to_q_weight = original_module.attn.to_q.weight
        self.attn_to_q_bias = _get_param_or_dummy_zero(
            original_module.attn.to_q.bias, torch.zeros(original_module.attn.to_q.out_features)
        )
        self.attn_to_k_weight = original_module.attn.to_k.weight
        self.attn_to_k_bias = _get_param_or_dummy_zero(
            original_module.attn.to_k.bias, torch.zeros(original_module.attn.to_k.out_features)
        )
        self.attn_to_v_weight = original_module.attn.to_v.weight
        self.attn_to_v_bias = _get_param_or_dummy_zero(
            original_module.attn.to_v.bias, torch.zeros(original_module.attn.to_v.out_features)
        )

        self.attn_norm_q_weight = _get_param_or_dummy_zero(
            original_module.attn.norm_q.weight,
            torch.zeros(
                original_module.attn.norm_q.weight.shape
                if hasattr(original_module.attn.norm_q, "weight") and original_module.attn.norm_q.weight is not None
                else original_module.attn.dim_head
            ),
        )
        self.attn_norm_q_eps = original_module.attn.norm_q.eps if hasattr(original_module.attn.norm_q, "eps") else 1e-6
        self.attn_norm_q_elementwise_affine = (
            original_module.attn.norm_q.elementwise_affine
            if hasattr(original_module.attn.norm_q, "elementwise_affine")
            else True
        )

        self.attn_norm_k_weight = _get_param_or_dummy_zero(
            original_module.attn.norm_k.weight,
            torch.zeros(
                original_module.attn.norm_k.weight.shape
                if hasattr(original_module.attn.norm_k, "weight") and original_module.attn.norm_k.weight is not None
                else original_module.attn.dim_head
            ),
        )
        self.attn_norm_k_eps = original_module.attn.norm_k.eps if hasattr(original_module.attn.norm_k, "eps") else 1e-6
        self.attn_norm_k_elementwise_affine = (
            original_module.attn.norm_k.elementwise_affine
            if hasattr(original_module.attn.norm_k, "elementwise_affine")
            else True
        )

        self.attn_add_q_proj_weight = _get_param_or_dummy_zero(
            original_module.attn.add_q_proj.weight, torch.zeros(original_module.attn.add_q_proj.out_features)
        )
        self.attn_add_q_proj_bias = _get_param_or_dummy_zero(
            original_module.attn.add_q_proj.bias, torch.zeros(original_module.attn.add_q_proj.out_features)
        )
        self.attn_add_k_proj_weight = _get_param_or_dummy_zero(
            original_module.attn.add_k_proj.weight, torch.zeros(original_module.attn.add_k_proj.out_features)
        )
        self.attn_add_k_proj_bias = _get_param_or_dummy_zero(
            original_module.attn.add_k_proj.bias, torch.zeros(original_module.attn.add_k_proj.out_features)
        )
        self.attn_add_v_proj_weight = _get_param_or_dummy_zero(
            original_module.attn.add_v_proj.weight, torch.zeros(original_module.attn.add_v_proj.out_features)
        )
        self.attn_add_v_proj_bias = _get_param_or_dummy_zero(
            original_module.attn.add_v_proj.bias, torch.zeros(original_module.attn.add_v_proj.out_features)
        )

        self.attn_norm_added_q_weight = _get_param_or_dummy_zero(
            original_module.attn.norm_added_q.weight,
            torch.zeros(
                original_module.attn.norm_added_q.weight.shape
                if hasattr(original_module.attn.norm_added_q, "weight")
                and original_module.attn.norm_added_q.weight is not None
                else original_module.attn.dim_head
            ),
        )
        self.attn_norm_added_q_eps = (
            original_module.attn.norm_added_q.eps if hasattr(original_module.attn.norm_added_q, "eps") else 1e-6
        )
        self.attn_norm_added_q_elementwise_affine = (
            original_module.attn.norm_added_q.elementwise_affine
            if hasattr(original_module.attn.norm_added_q, "elementwise_affine")
            else True
        )

        self.attn_norm_added_k_weight = _get_param_or_dummy_zero(
            original_module.attn.norm_added_k.weight,
            torch.zeros(
                original_module.attn.norm_added_k.weight.shape
                if hasattr(original_module.attn.norm_added_k, "weight")
                and original_module.attn.norm_added_k.weight is not None
                else original_module.attn.dim_head
            ),
        )
        self.attn_norm_added_k_eps = (
            original_module.attn.norm_added_k.eps if hasattr(original_module.attn.norm_added_k, "eps") else 1e-6
        )
        self.attn_norm_added_k_elementwise_affine = (
            original_module.attn.norm_added_k.elementwise_affine
            if hasattr(original_module.attn.norm_added_k, "elementwise_affine")
            else True
        )

        self.attn_to_out_0_weight = original_module.attn.to_out[0].weight
        self.attn_to_out_0_bias = _get_param_or_dummy_zero(
            original_module.attn.to_out[0].bias, torch.zeros(original_module.attn.to_out[0].out_features)
        )
        self.attn_to_out_1_dropout_p = original_module.attn.to_out[1].p
        self.attn_context_pre_only_flag = original_module.attn.context_pre_only
        self.attn_added_kv_proj_dim = original_module.attn.added_kv_proj_dim
        self.attn_to_add_out_weight = _get_param_or_dummy_zero(
            original_module.attn.to_add_out.weight, torch.zeros(original_module.attn.to_add_out.out_features)
        )
        self.attn_to_add_out_bias = _get_param_or_dummy_zero(
            original_module.attn.to_add_out.bias, torch.zeros(original_module.attn.to_add_out.out_features)
        )

        # --- Extract parameters for norm2 (RMSNorm) ---
        self.norm2_weight = _get_param_or_dummy_zero(
            original_module.norm2.weight,
            torch.zeros(
                original_module.norm2.weight.shape
                if hasattr(original_module.norm2, "weight") and original_module.norm2.weight is not None
                else original_module.norm2.dim
            ),
        )
        self.norm2_eps = original_module.norm2.eps if hasattr(original_module.norm2, "eps") else 1e-6
        self.norm2_elementwise_affine = (
            original_module.norm2.elementwise_affine if hasattr(original_module.norm2, "elementwise_affine") else True
        )

        # --- Extract parameters for ff (FeedForward) ---
        self.ff_dim = original_module.ff.dim
        self.ff_dim_out = original_module.ff.dim_out
        self.ff_mult = original_module.ff.mult
        self.ff_dropout_ratio = original_module.ff.dropout
        self.ff_final_dropout = original_module.ff.final_dropout
        self.ff_act_fn_proj_weight = original_module.ff.net[0].proj.weight
        self.ff_act_fn_proj_bias = _get_param_or_dummy_zero(
            original_module.ff.net[0].proj.bias, torch.zeros(original_module.ff.net[0].proj.out_features)
        )
        self.ff_project_out_weight = original_module.ff.net[2].weight
        self.ff_project_out_bias = _get_param_or_dummy_zero(
            original_module.ff.net[2].bias, torch.zeros(original_module.ff.net[2].out_features)
        )
        self.ff_activation_fn_type = _get_activation_fn_type_from_module(original_module.ff.net[0])

        # --- Extract parameters for norm2_context (RMSNorm) ---
        self.norm2_context_weight = _get_param_or_dummy_zero(
            original_module.norm2_context.weight,
            torch.zeros(
                original_module.norm2_context.weight.shape
                if hasattr(original_module.norm2_context, "weight") and original_module.norm2_context.weight is not None
                else original_module.norm2_context.dim
            ),
        )
        self.norm2_context_eps = (
            original_module.norm2_context.eps if hasattr(original_module.norm2_context, "eps") else 1e-6
        )
        self.norm2_context_elementwise_affine = (
            original_module.norm2_context.elementwise_affine
            if hasattr(original_module.norm2_context, "elementwise_affine")
            else True
        )

        # --- Extract parameters for ff_context (FeedForward) ---
        self.ff_context_dim = original_module.ff_context.dim
        self.ff_context_dim_out = original_module.ff_context.dim_out
        self.ff_context_mult = original_module.ff_context.mult
        self.ff_context_dropout_ratio = original_module.ff_context.dropout
        self.ff_context_final_dropout = original_module.ff_context.final_dropout
        self.ff_context_act_fn_proj_weight = original_module.ff_context.net[0].proj.weight
        self.ff_context_act_fn_proj_bias = _get_param_or_dummy_zero(
            original_module.ff_context.net[0].proj.bias,
            torch.zeros(original_module.ff_context.net[0].proj.out_features),
        )
        self.ff_context_project_out_weight = original_module.ff_context.net[2].weight
        self.ff_context_project_out_bias = _get_param_or_dummy_zero(
            original_module.ff_context.net[2].bias, torch.zeros(original_module.ff_context.net[2].out_features)
        )
        self.ff_context_activation_fn_type = _get_activation_fn_type_from_module(original_module.ff_context.net[0])


def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # --- Handle optional attention_mask for processor if it was None ---
    # JointAttnProcessor2_0Onnx expects attention_mask, even if zero-sized.
    # If it's dynamically generated or None, pass an empty tensor.
    # This wrapper expects attention_mask not to be an input to JTB.
    # So we assume it's always an empty tensor to the processor.
    attention_mask_for_attn = torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype)
    _attn_original_attention_mask_was_none = True

    # JointAttnProcessor2_0 always expects encoder_hidden_states (not None).
    # We also pass it as a regular tensor here.
    _attn_original_encoder_hidden_states_was_none = False

    return JointTransformerBlockFunc.apply(
        hidden_states,
        encoder_hidden_states,
        temb,
        self._use_dual_attention,
        self._context_pre_only,
        self._chunk_size,
        self._chunk_dim,
        self.norm1_linear_weight,
        self.norm1_linear_bias,
        self.norm1_epsilon,
        self.norm1_context_linear_weight,
        self.norm1_context_linear_bias,
        self.norm1_context_epsilon,
        self.attn_heads,
        self.attn_head_dim,
        self.attn_scale_qk,
        self.attn_scale_val,
        self.attn_query_dim,
        self.attn_inner_dim,
        self.attn_inner_kv_dim,
        self.attn_to_q_weight,
        self.attn_to_q_bias,
        self.attn_to_k_weight,
        self.attn_to_k_bias,
        self.attn_to_v_weight,
        self.attn_to_v_bias,
        self.attn_norm_q_weight,
        self.attn_norm_q_eps,
        self.attn_norm_q_elementwise_affine,
        self.attn_norm_k_weight,
        self.attn_norm_k_eps,
        self.attn_norm_k_elementwise_affine,
        self.attn_add_q_proj_weight,
        self.attn_add_q_proj_bias,
        self.attn_add_k_proj_weight,
        self.attn_add_k_proj_bias,
        self.attn_add_v_proj_weight,
        self.attn_add_v_proj_bias,
        self.attn_norm_added_q_weight,
        self.attn_norm_added_q_eps,
        self.attn_norm_added_q_elementwise_affine,
        self.attn_norm_added_k_weight,
        self.attn_norm_added_k_eps,
        self.attn_norm_added_k_elementwise_affine,
        self.attn_to_out_0_weight,
        self.attn_to_out_0_bias,
        self.attn_to_out_1_dropout_p,
        self.attn_context_pre_only_flag,
        self.attn_added_kv_proj_dim,
        self.attn_to_add_out_weight,
        self.attn_to_add_out_bias,
        self.norm2_weight,
        self.norm2_eps,
        self.norm2_elementwise_affine,
        self.ff_dim,
        self.ff_dim_out,
        self.ff_mult,
        self.ff_dropout_ratio,
        self.ff_final_dropout,
        self.ff_act_fn_proj_weight,
        self.ff_act_fn_proj_bias,
        self.ff_project_out_weight,
        self.ff_project_out_bias,
        self.ff_activation_fn_type,
        self.norm2_context_weight,
        self.norm2_context_eps,
        self.norm2_context_elementwise_affine,
        self.ff_context_dim,
        self.ff_context_dim_out,
        self.ff_context_mult,
        self.ff_context_dropout_ratio,
        self.ff_context_final_dropout,
        self.ff_context_act_fn_proj_weight,
        self.ff_context_act_fn_proj_bias,
        self.ff_context_project_out_weight,
        self.ff_context_project_out_bias,
        self.ff_context_activation_fn_type,
        _attn_original_encoder_hidden_states_was_none,
        _attn_original_attention_mask_was_none,
    )
