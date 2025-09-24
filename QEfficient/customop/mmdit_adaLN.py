import onnxscript
import torch
import torch.nn as nn
from typing import Optional, Tuple
import torch.nn.functional as F

from diffusers.models.normalization import AdaLayerNormZero

from torch._dynamo.comptime import comptime

CUSTOM_OPSET = onnxscript.values.Opset(domain="com.qualcomm.cloud", version=1)
ops = getattr(onnxscript, "opset" + str(17))

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
    def symbolic(
        g: torch.Graph,
        x: torch.Value,
        emb: torch.Value,
        linear_weight: torch.Value,
        linear_bias: torch.Value,
        norm_epsilon: torch.Value,  
    ) -> Tuple[torch.Value, ...]:
        # Call the corresponding ONNXScript function
        # g.onnxscript_op automatically handles packing/unpacking inputs/outputs
        # import pdb; pdb.set_trace()
        scaled_shifted_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = g.onnxscript_op(
            AdaLayerNormZeroOnnx,  # Your ONNXScript function
            x,
            emb,
            linear_weight,
            linear_bias,
            norm_epsilon_f=norm_epsilon, 
            outputs=5
        )
        return scaled_shifted_x, gate_msa, shift_mlp, scale_mlp, gate_mlp

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

    print(f"Using ONNX opset version: {ops.__name__}")
    linear_out = ops.MatMul(silu_emb, ops.Transpose(linear_weight, perm=[1, 0]))
    linear_out = ops.Add(linear_out, linear_bias)

    # 2. `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)`
    # The linear_out has a shape of [..., 6 * embedding_dim]
    output_dim_linear = ops.Shape(linear_out)[-1]
    
    # Calculate chunk_size dynamically (as an ONNX symbolic tensor)
    # Cast output_dim_linear to a float for division, then back to INT64 for chunk_size
    # (Since output_dim_linear is a symbolic tensor, it's already an integer value type usually)
    chunk_size = ops.Div(output_dim_linear, ops.Constant(value_int=6))

    # Create a 1D tensor of `chunk_size` repeated 6 times for ops.Split's split_size input.
    num_chunks = ops.Constant(value_int=6) # Scalar ONNX integer for number of chunks
    
    # Reshape `chunk_size` (which is a scalar tensor) to a 1-element 1D tensor
    chunk_size_1d = ops.Reshape(chunk_size, ops.Constant(value_ints=[1]))
    
    # Tile `chunk_size_1d` 6 times to get the required split_size tensor
    # The `reps` argument for ops.Tile needs to be a 1D tensor.
    reps_1d = ops.Reshape(num_chunks, ops.Constant(value_ints=[1]))
    split_sizes_tensor = ops.Tile(chunk_size_1d, reps_1d) # split_sizes_tensor has a shape of [6]
    split_outputs_1, split_outputs_2, split_outputs_3, split_outputs_4, split_outputs_5, split_outputs_6 = ops.Split(linear_out, split_sizes_tensor, axis=1)

    shift_msa = split_outputs_1
    scale_msa = split_outputs_2
    gate_msa = split_outputs_3
    shift_mlp = split_outputs_4
    scale_mlp = split_outputs_5
    gate_mlp = split_outputs_6

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
    one_scalar = ops.Constant(value_floats=[1.0])
    zero_scaler = ops.Constant(value_floats=[0.0])
    # Expand it to the desired shape
    scale_tensor = ops.Expand(one_scalar, scale_shape_tensor)
    bias_tensor = ops.Expand(zero_scaler, scale_shape_tensor)
    # scale_tensor = ops.ConstantOfShape(scale_shape_tensor, value=1.0)
    # bias_tensor = ops.ConstantOfShape(scale_shape_tensor, value=0.0)

    norm_x = ops.LayerNormalization(
        x,           # Input tensor (X)
        scale_tensor, # Scale input (S) - a tensor of ones
        bias_tensor,  # Bias input (B) - a tensor of zeros
        epsilon=norm_epsilon # Epsilon is an attribute
    )

    # Apply the scaling and shifting: `norm_x * (1 + scale_msa[:, None]) + shift_msa[:, None]`
    # Use ops.Unsqueeze for `[:, None]`
    scaled_shifted_x = ops.Add(
        ops.Mul(norm_x[0], ops.Add(ops.Constant(value_float=1.0 ), ops.Unsqueeze(scale_msa, axes=[1]))),
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