# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import onnxscript
import torch
from torch import nn

from QEfficient.utils import constants

ops = getattr(onnxscript, "opset" + str(constants.ONNX_EXPORT_OPSET))


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


class GemmaCustomRMSNormAIC(CustomRMSNormAIC):
    """
    Modify the init function to add +1 to the weights
    """

    def __qeff_init__(self):
        with torch.no_grad():
            self.weight.copy_(self.weight + 1.0)

@onnxscript.script(onnxscript.values.Opset(domain="com.qti.aisw.onnx", version=1))
def CustomLayerNorm(hidden_states: onnxscript.FLOAT, weight: onnxscript.FLOAT, bias: onnxscript.FLOAT, epsilon: float=1e-06):
    weight = ops.Cast(weight, to=1)
    bias = ops.Cast(bias, to=1)
    
    # Calculate mean and variance
    mean = ops.ReduceMean(hidden_states, axes=[-1], keepdims=1)
    variance = ops.ReduceMean(ops.Pow(ops.Sub(hidden_states, mean), 2), axes=[-1], keepdims=1)
    
    # Normalize
    epsilon_tensor = ops.Expand(epsilon, ops.Shape(variance))
    normalized = ops.Div(ops.Sub(hidden_states, mean), ops.Sqrt(ops.Add(variance, epsilon_tensor)))
    
    # Apply weight and bias
    return ops.Add(ops.Mul(weight, normalized), bias)


class CustomLayerNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(hidden_states: torch.Tensor, weight: torch.Tensor=None, bias: torch.Tensor=None, epsilon: float= 1e-06):
        # Calculate mean and variance
        mean = hidden_states.mean(-1, keepdim=True)
        variance = ((hidden_states - mean) ** 2).mean(-1, keepdim=True)
        
        # Normalize
        normalized = (hidden_states - mean) / torch.sqrt(variance + epsilon)
        
        # Apply weight and bias
        return weight * normalized + bias

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, hidden_states: torch.Value, weight: torch.Value, bias: torch.Value, epsilon: torch.Value) -> torch.Value:
        return g.onnxscript_op(CustomLayerNorm, hidden_states, weight, bias, epsilon_f=epsilon).setTypeAs(hidden_states)


class CustomLayerNormAIC(nn.Module):
    """
    LayerNorm module that works by replacing the current module with compiler known custom-op.
    """

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super(CustomLayerNormAIC, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
            self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, hidden_states):
        if self.elementwise_affine:
            return CustomLayerNormFunc.apply(hidden_states, self.weight, self.bias, self.eps)
        else:
            # If no elementwise affine, use ones for weight and zeros for bias
            weight = torch.ones_like(hidden_states[..., :self.normalized_shape[-1]])
            bias = torch.zeros_like(hidden_states[..., :self.normalized_shape[-1]])
            return CustomLayerNormFunc.apply(hidden_states, weight, bias, self.eps)
