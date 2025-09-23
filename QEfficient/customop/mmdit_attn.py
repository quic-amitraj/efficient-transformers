import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import onnxscript
from diffusers.models.attention_processor import Attention
from QEfficient.customop.mmdit_attn_processor import JointAttnProcessor2_0Func, JointAttnProcessor2_0Onnx

CUSTOM_OPSET = onnxscript.values.Opset(domain="com.qualcomm.cloud", version=1)
# Import the ONNX Script opset for version 13
ops = getattr(onnxscript, "opset" + str(13))

@onnxscript.script(CUSTOM_OPSET)
def AttentionOnnx(
    hidden_states:  onnxscript.FLOAT,
    encoder_hidden_states_to_pass: onnxscript.FLOAT,
    attention_mask_to_pass: onnxscript.FLOAT,
    attn_heads: int,
    attn_head_dim: int,
    attn_scale: float,
    attn_query_dim: int,
    attn_inner_dim: int,
    attn_inner_kv_dim: int,
    to_q_weight: onnxscript.FLOAT,
    to_q_bias: onnxscript.FLOAT,
    to_k_weight: onnxscript.FLOAT,
    to_k_bias: onnxscript.FLOAT,
    to_v_weight: onnxscript.FLOAT,
    to_v_bias: onnxscript.FLOAT,
    norm_q_weight: onnxscript.FLOAT,
    norm_q_eps: float,
    norm_k_weight: onnxscript.FLOAT,
    norm_k_eps: float,
    add_q_proj_weight: onnxscript.FLOAT,
    add_q_proj_bias: onnxscript.FLOAT,
    add_k_proj_weight: onnxscript.FLOAT,
    add_k_proj_bias: onnxscript.FLOAT,
    add_v_proj_weight: onnxscript.FLOAT,
    add_v_proj_bias: onnxscript.FLOAT,
    norm_added_q_weight: onnxscript.FLOAT,
    norm_added_q_eps: float,
    norm_added_k_weight: onnxscript.FLOAT,
    norm_added_k_eps: float,
    to_out_0_weight: onnxscript.FLOAT,
    to_out_0_bias: onnxscript.FLOAT,
    to_out_1_dropout_p: float,
    to_add_out_weight: onnxscript.FLOAT,
    to_add_out_bias: onnxscript.FLOAT,
    attn_upcast_attention: bool,
    attn_upcast_softmax: bool,
    _attn_original_encoder_hidden_states_was_none: bool,
    _attn_original_attention_mask_was_none: bool,
):
    attn_output, context_attn_output = JointAttnProcessor2_0Onnx(
        hidden_states,
        encoder_hidden_states_to_pass, # Prepared dummy or actual
        attention_mask_to_pass,        # Prepared dummy or actual
        attn_heads,
        attn_head_dim,
        attn_scale,
        attn_query_dim,
        attn_inner_dim,
        attn_inner_kv_dim,
        to_q_weight,
        to_q_bias,
        to_k_weight,
        to_k_bias,
        to_v_weight,
        to_v_bias,    
        norm_q_weight,
        norm_q_eps,
        norm_k_weight,
        norm_k_eps,
        add_q_proj_weight,
        add_q_proj_bias,
        add_k_proj_weight,
        add_k_proj_bias,
        add_v_proj_weight,
        add_v_proj_bias,
        norm_added_q_weight,
        norm_added_q_eps,
        norm_added_k_weight,
        norm_added_k_eps,
        to_out_0_weight,
        to_out_0_bias,
        to_out_1_dropout_p,
        to_add_out_weight,
        to_add_out_bias,
        attn_upcast_attention,
        attn_upcast_softmax,
        _attn_original_encoder_hidden_states_was_none,
        _attn_original_attention_mask_was_none,
    )
    
    return attn_output, context_attn_output

# This class will house the autograd.Function for the Attention block
class AttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        hidden_states: torch.Tensor,
        encoder_hidden_states_to_pass: torch.Tensor, # Prepared dummy or actual
        attention_mask_to_pass: torch.Tensor,        # Prepared dummy or actual
        # All parameters from JointAttnProcessor2_0Func (assuming 'self' from Attention)
        attn_heads: int,
        attn_head_dim: int,
        attn_scale: float,
        attn_query_dim: int,
        attn_inner_dim: int,
        attn_inner_kv_dim: int,
        to_q_weight: torch.Tensor,
        to_q_bias: torch.Tensor,
        to_k_weight: torch.Tensor,
        to_k_bias: torch.Tensor,
        to_v_weight: torch.Tensor,
        to_v_bias: torch.Tensor,
        norm_q_weight: torch.Tensor,
        norm_q_eps: float,
        norm_k_weight: torch.Tensor,
        norm_k_eps: float,
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
        to_out_0_weight: torch.Tensor,
        to_out_0_bias: torch.Tensor,
        to_out_1_dropout_p: float,
        to_add_out_weight: torch.Tensor,
        to_add_out_bias: torch.Tensor,
        attn_upcast_attention: bool,
        attn_upcast_softmax: bool,
        _attn_original_encoder_hidden_states_was_none: bool,
        _attn_original_attention_mask_was_none: bool,
    ) -> torch.Tensor:

        # Call the lower-level processor's apply method
        # This calls the JointAttnProcessor2_0Func's forward method
        # Ensure the arguments exactly match its signature
        attn_output, context_attn_output = JointAttnProcessor2_0Func.apply(
            hidden_states,
            encoder_hidden_states_to_pass,
            attention_mask_to_pass,
            attn_heads,
            attn_head_dim,
            attn_scale,
            attn_query_dim,
            attn_inner_dim,
            attn_inner_kv_dim,
            to_q_weight,
            to_q_bias,
            to_k_weight,
            to_k_bias,
            to_v_weight,
            to_v_bias,
            norm_q_weight,
            norm_q_eps,
            norm_k_weight,
            norm_k_eps,
            add_q_proj_weight,
            add_q_proj_bias,
            add_k_proj_weight,
            add_k_proj_bias,
            add_v_proj_weight,
            add_v_proj_bias,
            norm_added_q_weight,
            norm_added_q_eps,
            norm_added_k_weight,
            norm_added_k_eps,
            to_out_0_weight,
            to_out_0_bias,
            to_out_1_dropout_p,
            to_add_out_weight,
            to_add_out_bias,
            attn_upcast_attention,
            attn_upcast_softmax,
            _attn_original_encoder_hidden_states_was_none,
            _attn_original_attention_mask_was_none,
        )
        return attn_output, context_attn_output

    @staticmethod
    def setup_context(ctx, inputs, outputs):
            pass

    @staticmethod
    def symbolic(
        g: torch.Graph,
        hidden_states: torch.Value,
        encoder_hidden_states_to_pass: torch.Value,
        attention_mask_to_pass: torch.Value,
        attn_heads: torch.Value,
        attn_head_dim: torch.Value,
        attn_scale: torch.Value,
        attn_query_dim: torch.Value,
        attn_inner_dim: torch.Value,
        attn_inner_kv_dim: torch.Value,
        to_q_weight: torch.Value,
        to_q_bias: torch.Value,
        to_k_weight: torch.Value,
        to_k_bias: torch.Value,
        to_v_weight: torch.Value,
        to_v_bias: torch.Value,
        norm_q_weight: torch.Value,
        norm_q_eps: torch.Value, # Changed from float to torch.Value for ONNX export
        norm_k_weight: torch.Value,
        norm_k_eps: torch.Value, # Changed from float to torch.Value for ONNX export
        add_q_proj_weight: torch.Value,
        add_q_proj_bias: torch.Value,
        add_k_proj_weight: torch.Value,
        add_k_proj_bias: torch.Value,
        add_v_proj_weight: torch.Value,
        add_v_proj_bias: torch.Value,
        norm_added_q_weight: torch.Value,
        norm_added_q_eps: torch.Value, # Changed from float to torch.Value for ONNX export
        norm_added_k_weight: torch.Value,
        norm_added_k_eps: torch.Value, # Changed from float to torch.Value for ONNX export
        to_out_0_weight: torch.Value,
        to_out_0_bias: torch.Value,
        to_out_1_dropout_p: torch.Value, # Changed from float to torch.Value for ONNX export
        to_add_out_weight: torch.Value,
        to_add_out_bias: torch.Value,
        attn_upcast_attention: bool,
        attn_upcast_softmax: bool,
        _attn_original_encoder_hidden_states_was_none: bool,
        _attn_original_attention_mask_was_none: bool,
        # Add elementwise_affine parameters here if they are part of your real signature
    ) -> torch.Value:
        # Here we call the symbolic method of the underlying processor (JointAttnProcessor2_0Func)
        # This will construct the ONNX graph for the attention operation.
        attn_output, context_attn_output = g.onnxscript_op(AttentionOnnx,
            hidden_states,
            encoder_hidden_states_to_pass,
            attention_mask_to_pass,
            attn_heads_i = attn_heads, # Pass the torch.Value here
            attn_head_dim_i = attn_head_dim, # Pass the torch.Value here
            attn_scale_f = attn_scale, # Pass the torch.Value here
            attn_query_dim_i = attn_query_dim, # Pass the torch.Value here
            attn_inner_dim_i = attn_inner_dim, # Pass the torch.Value here
            attn_inner_kv_dim_i = attn_inner_kv_dim, # Pass the torch.Value here
            to_q_weight=to_q_weight,
            to_q_bias=to_q_bias,
            to_k_weight=to_k_weight,
            to_k_bias=to_k_bias,
            to_v_weight=to_v_weight,
            to_v_bias=to_v_bias,
            norm_q_weight=norm_q_weight,
            norm_q_eps_f=norm_q_eps, # Pass the torch.Value here
            norm_k_weight=norm_k_weight,
            norm_k_eps_f=norm_k_eps, # Pass the torch.Value here
            add_q_proj_weight=add_q_proj_weight,
            add_q_proj_bias=add_q_proj_bias,
            add_k_proj_weight=add_k_proj_weight,
            add_k_proj_bias=add_k_proj_bias,
            add_v_proj_weight=add_v_proj_weight,
            add_v_proj_bias=add_v_proj_bias,
            norm_added_q_weight=norm_added_q_weight,
            norm_added_q_eps_f=norm_added_q_eps, # Pass the torch.Value here
            norm_added_k_weight=norm_added_k_weight,
            norm_added_k_eps_f=norm_added_k_eps, # Pass the torch.Value here
            to_out_0_weight=to_out_0_weight,
            to_out_0_bias=to_out_0_bias,
            to_out_1_dropout_p_f=to_out_1_dropout_p, # Pass the torch.Value here
            to_add_out_weight=to_add_out_weight,
            to_add_out_bias=to_add_out_bias,
            attn_upcast_attention=attn_upcast_attention,
            attn_upcast_softmax=attn_upcast_softmax,
            _attn_original_encoder_hidden_states_was_none=_attn_original_encoder_hidden_states_was_none,
            _attn_original_attention_mask_was_none=_attn_original_attention_mask_was_none,
            outputs=2
        )
        return attn_output, context_attn_output
    
def _get_param_or_dummy_zero(param, default_tensor_if_none: Optional[torch.Tensor] = None):
    if isinstance(param, nn.Parameter):
        return param
    if param is None:
        if default_tensor_if_none is not None:
            return default_tensor_if_none
        return torch.tensor(0.0)  # Scalar zero for non-tensor defaults
    return param
class AttentionAIC(nn.Module):
    """
    Dummy AttentionAIC module that just extracts parameters and calls AttentionFuncAIC.apply.
    This replaces the original Attention module in the model.
    """
    def __init__(self, original_module: Attention):
        super().__init__()
        self.attn_heads = original_module.heads
        self.attn_head_dim = original_module.inner_dim // original_module.heads
        self.attn_scale = original_module.scale
        self.attn_query_dim = original_module.query_dim
        self.attn_inner_dim = original_module.inner_dim
        self.attn_inner_kv_dim = original_module.inner_kv_dim
        self.to_q_weight = original_module.to_q.weight
        self.to_q_bias = _get_param_or_dummy_zero(
            original_module.to_q.bias, torch.zeros(original_module.to_q.out_features)
        )
        self.to_k_weight = original_module.to_k.weight
        self.to_k_bias = _get_param_or_dummy_zero(
            original_module.to_k.bias, torch.zeros(original_module.to_k.out_features)
        )
        self.to_v_weight = original_module.to_v.weight
        self.to_v_bias = _get_param_or_dummy_zero(
            original_module.to_v.bias, torch.zeros(original_module.to_v.out_features)
        )
        self.norm_q_weight = _get_param_or_dummy_zero(
            original_module.norm_q.weight,
            torch.zeros(
                original_module.norm_q.weight.shape
                if hasattr(original_module.norm_q, "weight") and original_module.norm_q.weight is not None
                else original_module.dim_head
            ),
        )
        self.norm_q_eps = original_module.norm_q.eps if hasattr(original_module.norm_q, "eps") else 1e-6
        self.norm_k_weight = _get_param_or_dummy_zero(
            original_module.norm_k.weight,
            torch.zeros(
                original_module.norm_k.weight.shape
                if hasattr(original_module.norm_k, "weight") and original_module.norm_k.weight is not None
                else original_module.dim_head
            ),
        )
        self.norm_k_eps = original_module.norm_k.eps if hasattr(original_module.norm_k, "eps") else 1e-6
        self.add_q_proj_weight = _get_param_or_dummy_zero(
            original_module.add_q_proj.weight, torch.zeros(original_module.add_q_proj.out_features)
        )
        self.add_q_proj_bias =  _get_param_or_dummy_zero(
            original_module.add_q_proj.bias, torch.zeros(original_module.add_q_proj.out_features)
        )
        self.add_k_proj_weight =  _get_param_or_dummy_zero(
            original_module.add_k_proj.weight, torch.zeros(original_module.add_k_proj.out_features)
        )
        self.add_k_proj_bias = _get_param_or_dummy_zero(
            original_module.add_k_proj.bias, torch.zeros(original_module.add_k_proj.out_features)
        )
        self.add_v_proj_weight = _get_param_or_dummy_zero(
            original_module.add_v_proj.weight, torch.zeros(original_module.add_v_proj.out_features)
        )
        self.add_v_proj_bias = _get_param_or_dummy_zero(
            original_module.add_v_proj.bias, torch.zeros(original_module.add_v_proj.out_features)
        )
        self.norm_added_q_weight = _get_param_or_dummy_zero(
            original_module.norm_added_q.weight,
            torch.zeros(
                original_module.norm_added_q.weight.shape
                if hasattr(original_module.norm_added_q, "weight")
                and original_module.norm_added_q.weight is not None
                else original_module.dim_head
            ),
        )
        self.norm_added_q_eps =  (
            original_module.norm_added_q.eps if hasattr(original_module.norm_added_q, "eps") else 1e-6
        )
        self.norm_added_k_weight = _get_param_or_dummy_zero(
            original_module.norm_added_k.weight,
            torch.zeros(
                original_module.norm_added_k.weight.shape
                if hasattr(original_module.norm_added_k, "weight")
                and original_module.norm_added_k.weight is not None
                else original_module.dim_head
            ),
        )
        self.norm_added_k_eps =  (
            original_module.norm_added_k.eps if hasattr(original_module.norm_added_k, "eps") else 1e-6
        )
        self.to_out_0_weight = original_module.to_out[0].weight
        self.to_out_0_bias = _get_param_or_dummy_zero(
            original_module.to_out[0].bias, torch.zeros(original_module.to_out[0].out_features)
        )
        self.to_out_1_dropout_p = getattr(original_module.to_out[1], "p", 0.0) if getattr(original_module, "to_out", None) and len(original_module.to_out) > 1 else 0.0
        self.to_add_out_weight =  _get_param_or_dummy_zero(
            getattr(getattr(original_module, "to_add_out", None), "weight", None),
            torch.zeros(original_module.to_out[0].out_features),
        )

        self.to_add_out_bias = _get_param_or_dummy_zero(
            getattr(getattr(original_module, "to_add_out", None), "bias", None),
            torch.zeros(original_module.to_out[0].out_features),
        )
        self.attn_upcast_attention = original_module.upcast_attention
        self.attn_upcast_softmax = original_module.upcast_softmax
        self._attn_original_encoder_hidden_states_was_none = False
        self._attn_original_attention_mask_was_none = True

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor],
        # attention_mask: Optional[torch.FloatTensor],
       
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        attention_mask_for_attn = torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Convert float parameters to torch.Tensor for the autograd.Function call
        # norm_q_eps_tensor = torch.tensor(self.norm_q_eps, dtype=dtype, device=device)
        # norm_k_eps_tensor = torch.tensor(self.norm_k_eps, dtype=dtype, device=device)
        # norm_added_q_eps_tensor = torch.tensor(self.norm_added_q_eps, dtype=dtype, device=device)
        # norm_added_k_eps_tensor = torch.tensor(self.norm_added_k_eps, dtype=dtype, device=device)
        # to_out_1_dropout_p_tensor = torch.tensor(self.to_out_1_dropout_p, dtype=dtype, device=device)

        # Call the higher-level AttentionFuncAIC.apply
        return AttentionFunc.apply(
            hidden_states,
            encoder_hidden_states,
            attention_mask_for_attn,
            self.attn_heads,
            self.attn_head_dim,
            self.attn_scale,
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
            self.norm_k_weight,
            self.norm_k_eps,
            self.add_q_proj_weight,
            self.add_q_proj_bias,
            self.add_k_proj_weight,
            self.add_k_proj_bias,
            self.add_v_proj_weight,
            self.add_v_proj_bias,
            self.norm_added_q_weight,
            self.norm_added_q_eps,
            self.norm_added_k_weight,
            self.norm_added_k_eps,
            self.to_out_0_weight,
            self.to_out_0_bias,
            self.to_out_1_dropout_p,
            self.to_add_out_weight,
            self.to_add_out_bias,
            self.attn_upcast_attention,
            self.attn_upcast_softmax,
            self._attn_original_encoder_hidden_states_was_none,
            self._attn_original_attention_mask_was_none,
        )
