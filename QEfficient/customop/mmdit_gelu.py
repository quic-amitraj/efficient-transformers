
import onnxscript
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Any
import torch.nn.functional as F

from diffusers.models.activations import GELU

CUSTOM_OPSET = onnxscript.values.Opset(domain="com.qualcomm.cloud", version=1)
# Import the ONNX Script opset for version 13
ops = getattr(onnxscript, "opset" + str(13))


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

