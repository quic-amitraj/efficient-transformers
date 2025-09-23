import torch
import torch.nn.functional as F
from typing import Tuple
import torch.nn as nn
import onnxscript
from onnx import TensorProto
CUSTOM_OPSET = onnxscript.values.Opset(domain="com.qualcomm.cloud", version=1)
ops = getattr(onnxscript, "opset" + str(13))

@onnxscript.script(CUSTOM_OPSET)
def CustomLayerNormOnnx(
    x: onnxscript.FLOAT,          # current_hidden_states
    epsilon: float, 
    dim:int,
    # norm2_eps (epsilon value) - passed as primitive float for attribute
    # Assuming bias is not provided or implicitly handled as zero
    # If bias is explicitly provided as a tensor, add bias: onnxscript.FLOAT here
) -> onnxscript.FLOAT:
    """
    ONNXScript equivalent of CustomLayerNorm(x, weight, eps) assuming no bias.
    This mimics a standard LayerNormalization but only applies scale (weight)
    and uses epsilon, omitting bias (beta).
    """
    
    # LayerNormalization has inputs: X, Scale, B_optional
    # And attribute: epsilon
    
    # If your ONNX runtime supports LayerNormalization with an optional bias,
    # you can pass ops.Cast(ops.Constant(value_int=0), to=torch.float32) for a zero bias.
    # However, some ONNX operators might implicitly assume no bias if it's omitted.
    
    # Let's assume you're using a version of ONNX that either:
    # A) Has LayerNormalization that takes a 3rd input (Bias), so we pass a zero bias.
    # B) Has a custom operator that truly implements LayerNorm without bias.
    # C) We implement it manually.

    
    shape_tensor_for_expand = ops.Cast(ops.Constant(value_int=dim), to=TensorProto.FLOAT)
    shape_tensor_for_expand = ops.Unsqueeze(shape_tensor_for_expand, axes=[0]) # Make it a 1D tensor like [dim]

    # Create dummy scale (weight) tensor with all ones.
    scale = ops.Cast(ops.Constant(value_float=1.0), to=TensorProto.FLOAT)
    scale = ops.Expand(scale, shape_tensor_for_expand)
    # Create a zero bias tensor with the same shape as weight, and same data type as input x
    zero_bias = ops.Cast(ops.Constant(value_float=0.0), to=TensorProto.FLOAT)
    zero_bias = ops.Expand(zero_bias, shape_tensor_for_expand) # Expand scalar 0 to shape of weight

    normalized_output = ops.LayerNormalization(
        x,
        scale,
        zero_bias, # Pass zero bias
        epsilon=epsilon 
    )
    
    # If LayerNormalization without bias is truly the case,
    # and your custom LayerNorm ONNX op definition doesn't take bias,
    # then you'd call it without the third input.
    # E.g., `ops.YourCustomLayerNorm(x, weight, epsilon=epsilon)`
    
    return normalized_output


class CustomLayerNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        hidden_states: torch.Tensor,
        eps: float,  
        dim: int,   # The epsilon value
    ) -> torch.Tensor:
        # F.layer_norm with elementwise_affine=False means no weight/bias params
        # To call F.layer_norm without explicit weight/bias, pass None.
        # It correctly infers elementwise_affine=False if weight and bias are None.
        norm_hidden_states = F.layer_norm(
            input=hidden_states,
            normalized_shape=(dim,),
            weight=None,  # Explicitly None for elementwise_affine=False
            bias=None,    # Explicitly None for elementwise_affine=False
            eps=eps
        )
        return norm_hidden_states
    
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError("LayerNormNoAffineFunc backward not implemented for inference-only.")

    @staticmethod
    def symbolic(
        g: torch.Graph,
        hidden_states: torch.Value,           # This will be a Python primitive (int)
        eps: torch.Value,         # This will be a Python primitive (float)
        dim: torch.Value,
    ) -> torch.Value:
        # Call the ONNXScript function
        result = g.onnxscript_op(
            CustomLayerNormOnnx,
            hidden_states,
            epsilon_f=eps, # Pass eps as an attribute
            dim_i=dim
        )
        return result
    
class LayerNormNoAffineAIC(nn.Module):
    """
    Wrapper for nn.LayerNorm(..., elementwise_affine=False, ...) using LayerNormNoAffineFunc.
    """
    def __init__(self, original_layernorm_module: nn.LayerNorm):
        super().__init__()
        # Store the necessary attributes from the original LayerNorm module
        self.dim = original_layernorm_module.normalized_shape[0] # Assuming (dim,)
        self.eps = original_layernorm_module.eps

        # Assert that it truly is elementwise_affine=False
        assert not original_layernorm_module.elementwise_affine, \
            "LayerNormNoAffineAIC expects original_layernorm_module.elementwise_affine to be False"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Call the autograd.Function
        return CustomLayerNormFunc.apply(
            hidden_states,
            self.eps,
            self.dim,
        )

