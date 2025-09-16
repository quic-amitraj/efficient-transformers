import onnxscript
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention import FeedForward

CUSTOM_OPSET = onnxscript.values.Opset(domain="com.qualcomm.cloud", version=1)
ops = getattr(onnxscript, "opset" + str(13))

from QEfficient.customop.mmdit_gelu import GELUOnnx
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
        # If FeedForwardOnnx itself has branches, ensure it's configured
        # to take the 'gelu-approximate' path by constant inputs during symbolic export.
        # Alternatively, if FeedForwardOnnx is also simplified to only call GELUOnnx_ApproximateTanh directly,
        # that simplifies things further.
        result = g.onnxscript_op(
            FeedForwardOnnx,
            hidden_states,
            dim=dim,
            dim_out=dim_out,
            mult_i=mult,
            dropout_ratio_f=dropout_ratio,
            final_dropout_f=final_dropout,
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
