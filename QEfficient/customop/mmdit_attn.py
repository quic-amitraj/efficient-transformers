import torch
import torch.nn as nn
import inspect
from typing import Optional, Tuple, Dict, Any
import onnx
import torch
import torch.nn as nn
import inspect
from typing import Optional, Tuple, Dict, Any
from diffusers.models.attention_processor import Attention
from QEfficient.customop.mmdit import JointAttnProcessor2_0Func

# This class will house the autograd.Function for the Attention block
class AttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        # Inputs from the Attention module's forward
        hidden_states: torch.Tensor,
        encoder_hidden_states_to_pass: torch.Tensor, # Prepared dummy or actual
        attention_mask_to_pass: torch.Tensor,        # Prepared dummy or actual
        # Flags
        _original_encoder_hidden_states_was_none: bool,
        _original_attention_mask_was_none: bool,
        original_input_onnx_dtype_code: int,
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
        attn_added_kv_proj_dim: int,
        to_add_out_weight: torch.Tensor,
        to_add_out_bias: torch.Tensor,
        attn_upcast_attention: bool,
        attn_upcast_softmax: bool,
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
            attn_added_kv_proj_dim,
            to_add_out_weight,
            to_add_out_bias,
            attn_upcast_attention,
            attn_upcast_softmax,
            _original_encoder_hidden_states_was_none,
            _original_attention_mask_was_none,
            original_input_onnx_dtype_code,
        )
        return attn_output

    @staticmethod
    def symbolic(
        g: torch.Graph,
        hidden_states: torch.Value,
        encoder_hidden_states_to_pass: torch.Value,
        attention_mask_to_pass: torch.Value,
        _original_encoder_hidden_states_was_none: bool,
        _original_attention_mask_was_none: bool,
        original_input_onnx_dtype_code: int,
        attn_heads: int,
        attn_head_dim: int,
        attn_scale: float,
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
        attn_added_kv_proj_dim: int,
        to_add_out_weight: torch.Value,
        to_add_out_bias: torch.Value,
        attn_upcast_attention: bool,
        attn_upcast_softmax: bool,
        # Add elementwise_affine parameters here if they are part of your real signature
    ) -> torch.Value:
        # Here we call the symbolic method of the underlying processor (JointAttnProcessor2_0Func)
        # This will construct the ONNX graph for the attention operation.
        attn_output, context_attn_output = JointAttnProcessor2_0Func.symbolic(
            g, # The graph builder
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
            norm_q_eps, # Pass the torch.Value here
            norm_k_weight,
            norm_k_eps, # Pass the torch.Value here
            add_q_proj_weight,
            add_q_proj_bias,
            add_k_proj_weight,
            add_k_proj_bias,
            add_v_proj_weight,
            add_v_proj_bias,
            norm_added_q_weight,
            norm_added_q_eps, # Pass the torch.Value here
            norm_added_k_weight,
            norm_added_k_eps, # Pass the torch.Value here
            to_out_0_weight,
            to_out_0_bias,
            to_out_1_dropout_p, # Pass the torch.Value here
            attn_added_kv_proj_dim,
            to_add_out_weight,
            to_add_out_bias,
            attn_upcast_attention,
            attn_upcast_softmax,
            _original_encoder_hidden_states_was_none,
            _original_attention_mask_was_none,
            original_input_onnx_dtype_code,
        )
        return attn_output
    

class AttentionAIC(nn.Module):
    """
    Dummy AttentionAIC module that just extracts parameters and calls AttentionFuncAIC.apply.
    This replaces the original Attention module in the model.
    """
    def __init__(self, original_module: Attention):
        super().__init__()
        self.attn = original_module
        self._extract_parameters(original_module)

    def _extract_parameters(self, attn_module: Attention):
        def get_param_or_dummy_zero(param, default_shape_dim=None, device=None, dtype=None):
            if isinstance(param, nn.Parameter): return param
            if param is None:
                if device is None: device = torch.device("cpu")
                if dtype is None: dtype = torch.float32
                if default_shape_dim is not None:
                    if isinstance(default_shape_dim, int):
                        return nn.Parameter(torch.zeros(default_shape_dim, device=device, dtype=dtype))
                    if isinstance(default_shape_dim, tuple):
                        return nn.Parameter(torch.zeros(default_shape_dim, device=device, dtype=dtype))
                return nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype)) 
            return param

        device = attn_module.to_q.weight.device
        dtype = attn_module.to_q.weight.dtype

        # Store all parameters that AttentionFuncAIC expects
        self.attn_heads = attn_module.heads
        self.attn_head_dim = attn_module.inner_dim // attn_module.heads
        self.attn_scale = attn_module.scale
        self.attn_query_dim = attn_module.query_dim
        self.attn_inner_dim = attn_module.inner_dim
        self.attn_inner_kv_dim = attn_module.inner_kv_dim
        self.attn_upcast_attention = attn_module.upcast_attention
        self.attn_upcast_softmax = attn_module.upcast_softmax
        self.attn_added_kv_proj_dim = attn_module.added_kv_proj_dim

        self.to_q_weight = attn_module.to_q.weight
        self.to_q_bias = get_param_or_dummy_zero(attn_module.to_q.bias, attn_module.to_q.out_features, device, dtype)
        self.to_k_weight = attn_module.to_k.weight
        self.to_k_bias = get_param_or_dummy_zero(attn_module.to_k.bias, attn_module.to_k.out_features, device, dtype)
        self.to_v_weight = attn_module.to_v.weight
        self.to_v_bias = get_param_or_dummy_zero(attn_module.to_v.bias, attn_module.to_v.out_features, device, dtype)

        self.norm_q_weight = get_param_or_dummy_zero(attn_module.norm_q.weight, (self.attn_head_dim,), device, dtype)
        self.norm_q_eps = attn_module.norm_q.eps if hasattr(attn_module.norm_q, "eps") else 1e-6
        self.norm_q_elementwise_affine = attn_module.norm_q.elementwise_affine if hasattr(attn_module.norm_q, "elementwise_affine") else False
        self.norm_k_weight = get_param_or_dummy_zero(attn_module.norm_k.weight, (self.attn_head_dim,), device, dtype)
        self.norm_k_eps = attn_module.norm_k.eps if hasattr(attn_module.norm_k, "eps") else 1e-6
        self.norm_k_elementwise_affine = attn_module.norm_k.elementwise_affine if hasattr(attn_module.norm_k, "elementwise_affine") else False

        self.add_q_proj_weight = get_param_or_dummy_zero(attn_module.add_q_proj.weight, attn_module.add_q_proj.out_features, device, dtype)
        self.add_q_proj_bias = get_param_or_dummy_zero(attn_module.add_q_proj.bias, attn_module.add_q_proj.out_features, device, dtype)
        self.add_k_proj_weight = get_param_or_dummy_zero(attn_module.add_k_proj.weight, attn_module.add_k_proj.out_features, device, dtype)
        self.add_k_proj_bias = get_param_or_dummy_zero(attn_module.add_k_proj.bias, attn_module.add_k_proj.out_features, device, dtype)
        self.add_v_proj_weight = get_param_or_dummy_zero(attn_module.add_v_proj.weight, attn_module.add_v_proj.out_features, device, dtype)
        self.add_v_proj_bias = get_param_or_dummy_zero(attn_module.add_v_proj.bias, attn_module.add_v_proj.out_features, device, dtype)

        self.norm_added_q_weight = get_param_or_dummy_zero(attn_module.norm_added_q.weight, (self.attn_head_dim,), device, dtype)
        self.norm_added_q_eps = attn_module.norm_added_q.eps if hasattr(attn_module.norm_added_q, "eps") else 1e-6
        self.norm_added_q_elementwise_affine = attn_module.norm_added_q.elementwise_affine if hasattr(attn_module.norm_added_q, "elementwise_affine") else False
        self.norm_added_k_weight = get_param_or_dummy_zero(attn_module.norm_added_k.weight, (self.attn_head_dim,), device, dtype)
        self.norm_added_k_eps = attn_module.norm_added_k.eps if hasattr(attn_module.norm_added_k, "eps") else 1e-6
        self.norm_added_k_elementwise_affine = attn_module.norm_added_k.elementwise_affine if hasattr(attn_module.norm_added_k, "elementwise_affine") else False

        self.to_out_0_weight = attn_module.to_out[0].weight
        self.to_out_0_bias = get_param_or_dummy_zero(attn_module.to_out[0].bias, attn_module.to_out[0].out_features, device, dtype)
        self.to_out_1_dropout_p = attn_module.to_out[1].p

        self.to_add_out_weight = None#get_param_or_dummy_zero(attn_module.to_add_out.weight, attn_module.to_add_out.out_features, device, dtype)
       
        self.to_add_out_bias = None#get_param_or_dummy_zero(attn_module.to_add_out.bias, attn_module.to_add_out.out_features, device, dtype)
        
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor],
        attn_heads:int,
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
        attn_added_kv_proj_dim: int,
        to_add_out_weight: torch.Tensor,
        to_add_out_bias: torch.Tensor,
        attn_upcast_attention: bool,
        attn_upcast_softmax: bool,
        *args,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        _original_encoder_hidden_states_was_none = (encoder_hidden_states is None) or (
            encoder_hidden_states.numel() == 0
        )
        _original_attention_mask_was_none = (attention_mask is None) or (
            attention_mask.numel() == 0
        )

        device = hidden_states.device
        dtype = hidden_states.dtype
        _original_input_onnx_dtype_code = (
            onnx.TensorProto.FLOAT if dtype == torch.float32 else onnx.TensorProto.UNDEFINED
        )

        encoder_hidden_states_to_pass = encoder_hidden_states
        if _original_encoder_hidden_states_was_none:
            encoder_hidden_states_to_pass = torch.empty(0, device=device, dtype=dtype)

        attention_mask_to_pass = attention_mask
        if _original_attention_mask_was_none:
            attention_mask_to_pass = torch.empty(0, device=device, dtype=dtype)
        
        # Convert float parameters to torch.Tensor for the autograd.Function call
        norm_q_eps_tensor = torch.tensor(self.norm_q_eps, dtype=dtype, device=device)
        norm_k_eps_tensor = torch.tensor(self.norm_k_eps, dtype=dtype, device=device)
        norm_added_q_eps_tensor = torch.tensor(self.norm_added_q_eps, dtype=dtype, device=device)
        norm_added_k_eps_tensor = torch.tensor(self.norm_added_k_eps, dtype=dtype, device=device)
        to_out_1_dropout_p_tensor = torch.tensor(self.to_out_1_dropout_p, dtype=dtype, device=device)

        # Call the higher-level AttentionFuncAIC.apply
        return AttentionFunc.apply(
            hidden_states,
            encoder_hidden_states_to_pass,
            attention_mask_to_pass,
            self.attn_heads,
            self.attn_head_dim,
            self.attn_scale,
            self.attn_query_dim,
            self.attn_inner_dim,
            self.attn_inner_kv_dim,
            self.attn_added_kv_proj_dim,
            self.attn_upcast_attention,
            self.attn_upcast_softmax,
            self.to_q_weight,
            self.to_q_bias,
            self.to_k_weight,
            self.to_k_bias,
            self.to_v_weight,
            self.to_v_bias,
            self.norm_q_weight,
            norm_q_eps_tensor,
            self.norm_q_elementwise_affine,
            self.norm_k_weight,
            norm_k_eps_tensor,
            self.norm_k_elementwise_affine,
            self.add_q_proj_weight,
            self.add_q_proj_bias,
            self.add_k_proj_weight,
            self.add_k_proj_bias,
            self.add_v_proj_weight,
            self.add_v_proj_bias,
            self.norm_added_q_weight,
            norm_added_q_eps_tensor,
            self.norm_added_q_elementwise_affine,
            self.norm_added_k_weight,
            norm_added_k_eps_tensor,
            self.norm_added_k_elementwise_affine,
            self.to_out_0_weight,
            self.to_out_0_bias,
            to_out_1_dropout_p_tensor,
            self.to_add_out_weight,
            self.to_add_out_bias,
            _original_encoder_hidden_states_was_none,
            _original_attention_mask_was_none,
            _original_input_onnx_dtype_code,
        )