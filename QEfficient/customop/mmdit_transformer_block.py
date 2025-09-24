
import onnxscript
import torch
import torch.nn as nn
from typing import Optional, Tuple, Any
from onnx import TensorProto
from diffusers.models.activations import GELU
import torch.nn.functional as F

from diffusers.models.attention import JointTransformerBlock
from QEfficient.customop.mmdit_attention import AttentionFunc, AttentionOnnx
from QEfficient.customop.mmdit_feedforward import FeedForwardOnnx, FeedForwardFunc
from QEfficient.customop.mmdit_adaLN import AdaLayerNormZeroOnnx, AdaLayerNormZeroFunc
# from QEfficient.customop.mmdit_attn_processor import JointAttnProcessor2_0Onnx
from QEfficient.customop.rms_norm import CustomRMSNorm, CustomRMSNormFunc
from QEfficient.customop.mmdit_layernorm import CustomLayerNormFunc, CustomLayerNormOnnx
CUSTOM_OPSET = onnxscript.values.Opset(domain="com.qualcomm.cloud", version=1)
# Import the ONNX Script opset for version 13
ops = getattr(onnxscript, "opset" + str(13))

# map PyTorch dtype to ONNX dtype code
dtype_map = {
    torch.float32: TensorProto.FLOAT,
    torch.float16: TensorProto.FLOAT16,
    torch.int32: TensorProto.INT32,
    torch.int64: TensorProto.INT64,
}
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
    attn_scale: float,
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
    attn_norm_k_weight: onnxscript.FLOAT,
    attn_norm_k_eps: float,
    attn_add_q_proj_weight: onnxscript.FLOAT,
    attn_add_q_proj_bias: onnxscript.FLOAT,
    attn_add_k_proj_weight: onnxscript.FLOAT,
    attn_add_k_proj_bias: onnxscript.FLOAT,
    attn_add_v_proj_weight: onnxscript.FLOAT,
    attn_add_v_proj_bias: onnxscript.FLOAT,
    attn_norm_added_q_weight: onnxscript.FLOAT,
    attn_norm_added_q_eps: float,
    attn_norm_added_k_weight: onnxscript.FLOAT,
    attn_norm_added_k_eps: float,
    attn_to_out_0_weight: onnxscript.FLOAT,
    attn_to_out_0_bias: onnxscript.FLOAT,
    attn_to_out_1_dropout_p: float,
    # attn_context_pre_only_flag: bool,  # This needs to be set to False for this export
    attn_to_add_out_weight: onnxscript.FLOAT,
    attn_to_add_out_bias: onnxscript.FLOAT,
    # Weights and parameters for norm2 (RMSNormOnnx)
    norm2_weight: onnxscript.FLOAT,
    norm2_eps: float,
    # Weights and parameters for ff (FeedForwardOnnx)
    ff_dim: int,
    ff_dropout_ratio: float,
    ff_act_fn_proj_weight: onnxscript.FLOAT,
    ff_act_fn_proj_bias: onnxscript.FLOAT,  # Passed if bias=True
    ff_project_out_weight: onnxscript.FLOAT,
    ff_project_out_bias: onnxscript.FLOAT,  # Passed if bias=True
    # Weights and parameters for norm2_context (RMSNormOnnx)
    norm2_context_weight: onnxscript.FLOAT,
    norm2_context_eps: float,
    # Weights and parameters for ff_context (FeedForwardOnnx)
    ff_context_dim: int,
    ff_context_dropout_ratio: float,
    ff_context_act_fn_proj_weight: onnxscript.FLOAT,
    ff_context_act_fn_proj_bias: onnxscript.FLOAT,
    ff_context_project_out_weight: onnxscript.FLOAT,
    ff_context_project_out_bias: onnxscript.FLOAT,
    attn_upcast_attention: bool,
    attn_upcast_softmax: bool,
    _attn_original_encoder_hidden_states_was_none: bool,
    _attn_original_attention_mask_was_none: bool,
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
    attn_output, context_attn_output = AttentionOnnx(
        hidden_states=norm_hidden_states,
        encoder_hidden_states_to_pass=norm_encoder_hidden_states,
        attention_mask_to_pass=ops.Constant(value_float=0.0),  # Assuming attention_mask is handled externally or not dynamic
        attn_heads=attn_heads,
        attn_head_dim=attn_head_dim,
        attn_scale=attn_scale,
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
        norm_k_weight=attn_norm_k_weight,
        norm_k_eps=attn_norm_k_eps,
        add_q_proj_weight=attn_add_q_proj_weight,
        add_q_proj_bias=attn_add_q_proj_bias,
        add_k_proj_weight=attn_add_k_proj_weight,
        add_k_proj_bias=attn_add_k_proj_bias,
        add_v_proj_weight=attn_add_v_proj_weight,
        add_v_proj_bias=attn_add_v_proj_bias,
        norm_added_q_weight=attn_norm_added_q_weight,
        norm_added_q_eps=attn_norm_added_q_eps,
        norm_added_k_weight=attn_norm_added_k_weight,
        norm_added_k_eps=attn_norm_added_k_eps,
        to_out_0_weight=attn_to_out_0_weight,
        to_out_0_bias=attn_to_out_0_bias,
        to_out_1_dropout_p=attn_to_out_1_dropout_p,
        to_add_out_weight=attn_to_add_out_weight,
        to_add_out_bias=attn_to_add_out_bias,
        attn_upcast_attention=attn_upcast_attention, 
        attn_upcast_softmax=attn_upcast_softmax,  
        _attn_original_attention_mask_was_none=_attn_original_attention_mask_was_none,
        _attn_original_encoder_hidden_states_was_none=_attn_original_encoder_hidden_states_was_none,
        
    )
    # Process attention outputs for the `hidden_states`.
    attn_output = ops.Mul(ops.Unsqueeze(gate_msa, axes=[1]), attn_output)
    current_hidden_states = ops.Add(hidden_states, attn_output)

    # Note: `if self.use_dual_attention` block is skipped (fixed to False)

    # ----------------------------------------------------------------------
    # 4. MLP for hidden_states
    # ----------------------------------------------------------------------
    # Use CustomLayerNorm instead of CustomRMSNorm to match F.layer_norm in forward method
    norm_hidden_states_mlp = CustomLayerNormOnnx(current_hidden_states, norm2_eps, ff_dim)
    norm_hidden_states_mlp = ops.Add(
        ops.Mul(norm_hidden_states_mlp, ops.Add(ops.Constant(value_float=1.0), ops.Unsqueeze(scale_mlp, axes=[1]))),
        ops.Unsqueeze(shift_mlp, axes=[1]),
    )

    # Note: `if self._chunk_size is not None` block is skipped (fixed to None)
    ff_output = FeedForwardOnnx(
        norm_hidden_states_mlp,
        ff_dropout_ratio,
        ff_act_fn_proj_weight,
        ff_act_fn_proj_bias,
        ff_project_out_weight,
        ff_project_out_bias,
    )
    ff_output = ops.Mul(ops.Unsqueeze(gate_mlp, axes=[1]), ff_output)
    current_hidden_states = ops.Add(current_hidden_states, ff_output)
    
    # ----------------------------------------------------------------------
    # 5. Process attention outputs for the `encoder_hidden_states`.
    # ----------------------------------------------------------------------
    # Note: `if self.context_pre_only` block is skipped (fixed to False)

    context_attn_output = ops.Mul(ops.Unsqueeze(c_gate_msa, axes=[1]), context_attn_output)
    current_encoder_hidden_states = ops.Add(encoder_hidden_states, context_attn_output)

    # Use CustomLayerNorm instead of CustomRMSNorm to match F.layer_norm in forward method

    norm_encoder_hidden_states_mlp = CustomLayerNormOnnx(
        current_encoder_hidden_states, norm2_context_eps, ff_dim
    )
    norm_encoder_hidden_states_mlp = ops.Add(
        ops.Mul(
            norm_encoder_hidden_states_mlp, ops.Add(ops.Constant(value_float=1.0), ops.Unsqueeze(c_scale_mlp, axes=[1]))
        ),
        ops.Unsqueeze(c_shift_mlp, axes=[1]),
    )

    # Note: `if self._chunk_size is not None` block is skipped (fixed to None)
    context_ff_output = FeedForwardOnnx(
        norm_encoder_hidden_states_mlp,
        ff_context_dropout_ratio,
        ff_context_act_fn_proj_weight,
        ff_context_act_fn_proj_bias,
        ff_context_project_out_weight,
        ff_context_project_out_bias,
    )
    current_encoder_hidden_states = ops.Add(
        current_encoder_hidden_states, ops.Mul(ops.Unsqueeze(c_gate_mlp, axes=[1]), context_ff_output)
    )

    return current_encoder_hidden_states, current_hidden_states


class JointTransformerBlockFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
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
        attn_scale: float,
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
        attn_norm_added_k_weight: torch.Tensor,
        attn_norm_added_k_eps: float,
        attn_to_out_0_weight: torch.Tensor,
        attn_to_out_0_bias: torch.Tensor,
        attn_to_out_1_dropout_p: float,
        attn_to_add_out_weight: torch.Tensor,
        attn_to_add_out_bias: torch.Tensor,
        # --- Parameters for norm2 (RMSNorm) ---
        norm2_weight: torch.Tensor,
        norm2_eps: float,
         # --- Parameters for ff (FeedForward) ---
        ff_dim: int,
        ff_dropout_ratio: float, 
        ff_act_fn_proj_weight: torch.Tensor, ff_act_fn_proj_bias: torch.Tensor,
        ff_project_out_weight: torch.Tensor, ff_project_out_bias: torch.Tensor,
        
        # --- Parameters for norm2_context (RMSNorm) ---
        norm2_context_weight: torch.Tensor,
        norm2_context_eps: float,
        # --- Parameters for ff_context (FeedForward) ---
        ff_context_dim: int,
        ff_context_dropout_ratio: float,
        ff_context_act_fn_proj_weight: torch.Tensor,
        ff_context_act_fn_proj_bias: torch.Tensor,
        ff_context_project_out_weight: torch.Tensor,
        ff_context_project_out_bias: torch.Tensor,
        # Flags for `JointAttnProcessor2_0Func` to handle optional tensor inputs
        attn_upcast_attention: bool,
        attn_upcast_softmax: bool,
        _attn_original_encoder_hidden_states_was_none: bool,
        _attn_original_attention_mask_was_none: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # use_dual_attention = False, context_pre_only = False, _chunk_size = None

        # 1. Norm1 (AdaLayerNormZero)
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = AdaLayerNormZeroFunc.apply(
            hidden_states, temb, norm1_linear_weight, norm1_linear_bias, norm1_epsilon
        )

        # 2. Norm1_context (AdaLayerNormZero)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = AdaLayerNormZeroFunc.apply(
            encoder_hidden_states, temb, norm1_context_linear_weight, norm1_context_linear_bias, norm1_context_epsilon
        )

        # 3. Attention (AttentionFunc-JointAttnProcessor2_0)
        # Prepare dummy attention_mask for processor if it was None
        attention_mask_for_attn = None
        if not _attn_original_attention_mask_was_none:
            attention_mask_for_attn = torch.empty(0)  # Dummy empty tensor for now

        attn_output, context_attn_output = AttentionFunc.apply(
            norm_hidden_states,
            norm_encoder_hidden_states,  # Pass non-None to processor
            attention_mask_for_attn,  # Pass non-None to processor
            attn_heads,
            attn_head_dim,
            attn_scale,
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
            attn_norm_k_weight,
            attn_norm_k_eps,
            attn_add_q_proj_weight,
            attn_add_q_proj_bias,
            attn_add_k_proj_weight,
            attn_add_k_proj_bias,
            attn_add_v_proj_weight,
            attn_add_v_proj_bias,
            attn_norm_added_q_weight,
            attn_norm_added_q_eps,
            attn_norm_added_k_weight,
            attn_norm_added_k_eps,
            attn_to_out_0_weight,
            attn_to_out_0_bias,
            attn_to_out_1_dropout_p,
            attn_to_add_out_weight,
            attn_to_add_out_bias,
            attn_upcast_attention,
            attn_upcast_softmax,
            _attn_original_encoder_hidden_states_was_none,  # _original_encoder_hidden_states_was_none - always false for this case
            _attn_original_attention_mask_was_none,
        )
        # Process attention outputs for the `hidden_states`.
        # Assuming original `hidden_states` (input to this forward) is what is added to.
        current_hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        # Note: `if self.use_dual_attention` block is skipped.

        # 4. MLP for hidden_states
        # Use the correct dimension for hidden states normalization
        # norm_hidden_states_mlp = F.layer_norm(input=current_hidden_states, normalized_shape=(ff_dim,), eps=norm2_eps)
        norm_hidden_states_mlp= CustomLayerNormFunc.apply(
            current_hidden_states, eps=norm2_eps, dim=ff_dim,
        )
        
        norm_hidden_states_mlp = norm_hidden_states_mlp * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)

        # Note: `if self._chunk_size is not None` block is skipped.
        # Assuming FeedForwardFunc.apply exists
        ff_output = FeedForwardFunc.apply(
            norm_hidden_states_mlp,
            ff_dropout_ratio,
            ff_act_fn_proj_weight,
            ff_act_fn_proj_bias,
            ff_project_out_weight,
            ff_project_out_bias,
        )
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        current_hidden_states=current_hidden_states+ff_output
        # 5. Process attention outputs for the `encoder_hidden_states`.
        # Note: `if self.context_pre_only` block is skipped.
        
        
        current_encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output

        
        # norm_encoder_hidden_states_mlp=F.layer_norm(input=current_encoder_hidden_states,  normalized_shape=(ff_context_dim,), eps=norm2_context_eps)

        norm_encoder_hidden_states_mlp= CustomLayerNormFunc.apply(
            current_encoder_hidden_states, eps=norm2_context_eps, dim=ff_context_dim
        )
        norm_encoder_hidden_states_mlp = norm_encoder_hidden_states_mlp * (
            1 + c_scale_mlp.unsqueeze(1)
        ) + c_shift_mlp.unsqueeze(1)

        # Note: `if self._chunk_size is not None` block is skipped.
        context_ff_output = FeedForwardFunc.apply(
            norm_encoder_hidden_states_mlp,
            ff_context_dropout_ratio,
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
        attn_scale: torch.Value,
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
        attn_norm_q_eps: torch.Value,
        attn_norm_k_weight: torch.Value,
        attn_norm_k_eps: torch.Value,
        attn_add_q_proj_weight: torch.Value,
        attn_add_q_proj_bias: torch.Value,
        attn_add_k_proj_weight: torch.Value,
        attn_add_k_proj_bias: torch.Value,
        attn_add_v_proj_weight: torch.Value,
        attn_add_v_proj_bias: torch.Value,
        attn_norm_added_q_weight: torch.Value,
        attn_norm_added_q_eps: torch.Value,
        attn_norm_added_k_weight: torch.Value,
        attn_norm_added_k_eps: torch.Value,
        attn_to_out_0_weight: torch.Value,
        attn_to_out_0_bias: torch.Value,
        attn_to_out_1_dropout_p: torch.Value,
        attn_to_add_out_weight: torch.Value,
        attn_to_add_out_bias: torch.Value,
        # Parameters for norm2 (RMSNorm)
        norm2_weight: torch.Value,
        norm2_eps: torch.Value,
        # Parameters for ff (FeedForward)
        ff_dim: torch.Value,
        ff_dropout_ratio: torch.Value,
        ff_act_fn_proj_weight: torch.Value,
        ff_act_fn_proj_bias: torch.Value,
        ff_project_out_weight: torch.Value,
        ff_project_out_bias: torch.Value,   
        # Parameters for norm2_context (RMSNorm)
        norm2_context_weight: torch.Value,
        norm2_context_eps: torch.Value,
        # Parameters for ff_context (FeedForward)
        ff_context_dim: torch.Value,
        ff_context_dropout_ratio: torch.Value,
        ff_context_act_fn_proj_weight: torch.Value, 
        ff_context_act_fn_proj_bias: torch.Value,
        ff_context_project_out_weight: torch.Value,
        ff_context_project_out_bias: torch.Value,
        attn_upcast_attention: bool,
        attn_upcast_softmax: bool,
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
            norm1_context_linear_weight,
            norm1_context_linear_bias,
            norm1_context_epsilon_f=norm1_context_epsilon,
            norm1_epsilon_f=norm1_epsilon,
            attn_heads_i=attn_heads,
            attn_head_dim_i=attn_head_dim,
            attn_scale_f=attn_scale,
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
            attn_norm_k_weight=attn_norm_k_weight,
            attn_norm_k_eps_f=attn_norm_k_eps,
            attn_add_q_proj_weight=attn_add_q_proj_weight,
            attn_add_q_proj_bias=attn_add_q_proj_bias,
            attn_add_k_proj_weight=attn_add_k_proj_weight,
            attn_add_k_proj_bias=attn_add_k_proj_bias,
            attn_add_v_proj_weight=attn_add_v_proj_weight,
            attn_add_v_proj_bias=attn_add_v_proj_bias,
            attn_norm_added_q_weight=attn_norm_added_q_weight,
            attn_norm_added_q_eps_f=attn_norm_added_q_eps,
            attn_norm_added_k_weight=attn_norm_added_k_weight,
            attn_norm_added_k_eps_f=attn_norm_added_k_eps,
            attn_to_out_0_weight=attn_to_out_0_weight,
            attn_to_out_0_bias=attn_to_out_0_bias,
            attn_to_out_1_dropout_p_f=attn_to_out_1_dropout_p,
            attn_to_add_out_weight=attn_to_add_out_weight,
            attn_to_add_out_bias=attn_to_add_out_bias,
            norm2_weight=norm2_weight,
            norm2_eps_f=norm2_eps,
            ff_dim_i=ff_dim,
            ff_dropout_ratio_f=ff_dropout_ratio,
            ff_act_fn_proj_weight=ff_act_fn_proj_weight,
            ff_act_fn_proj_bias=ff_act_fn_proj_bias,
            ff_project_out_weight=ff_project_out_weight,
            ff_project_out_bias=ff_project_out_bias,
            norm2_context_weight=norm2_context_weight,
            norm2_context_eps_f=norm2_context_eps,
            ff_context_dim_i=ff_context_dim,
            ff_context_dropout_ratio_f=ff_context_dropout_ratio,
            ff_context_act_fn_proj_weight=ff_context_act_fn_proj_weight,
            ff_context_act_fn_proj_bias=ff_context_act_fn_proj_bias,
            ff_context_project_out_weight=ff_context_project_out_weight,
            ff_context_project_out_bias=ff_context_project_out_bias,
            attn_upcast_attention_i=attn_upcast_attention,
            attn_upcast_softmax_i=attn_upcast_softmax,
            _attn_original_encoder_hidden_states_was_none_i=_attn_original_encoder_hidden_states_was_none,
            _attn_original_attention_mask_was_none_i=_attn_original_attention_mask_was_none,
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
# Robust helper for safely extracting parameters or creating defaults
def _get_param_or_default(param_or_attr, default_val=0.0, is_weight=False, expected_shape=None):
    if isinstance(param_or_attr, nn.Parameter):
        return param_or_attr
    elif isinstance(param_or_attr, torch.Tensor):
        return param_or_attr
    elif param_or_attr is None:
        if is_weight: # Default for weights is often 1.0 (identity) for affine, or None if truly optional
            if expected_shape is not None:
                return torch.ones(expected_shape)
            # Cannot create tensor without shape if it's a weight, returning None implies it's handled by caller
            return None
        # Default for bias/other is often 0.0, or None if truly optional
        if expected_shape is not None:
            return torch.zeros(expected_shape)
        return torch.tensor(float(default_val))
    # For scalar types (int, float, bool)
    return param_or_attr

def _extract_param(obj, path, is_weight=False, expected_shape=None):
    current_obj = obj
    parts = path.split('.')
    for part in parts:
        try:
            # Handle list/tuple indices (e.g., 'to_out.0.weight')
            if part.isdigit():
                current_obj = current_obj[int(part)]
            else:
                current_obj = getattr(current_obj, part)
        except (AttributeError, TypeError, IndexError):
            # If a part of the path is missing (e.g., optional module not present),
            # or it's an unexpected type, return default
            return _get_param_or_default(None, is_weight=is_weight, expected_shape=expected_shape)
    
    # If it's a scalar (int, float, bool), return directly
    if not isinstance(current_obj, (nn.Parameter, torch.Tensor)):
        return current_obj
    
    # For tensors/parameters, apply the _get_param_or_default logic
    return _get_param_or_default(current_obj, is_weight=is_weight, expected_shape=expected_shape)

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
        # --- Extract parameters for norm1 (AdaLayerNormZero) ---
        self.norm1_linear_weight = _extract_param(original_module, 'norm1.linear.weight', is_weight=True)
       
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
        self.attn_head_dim = original_module.attn.inner_dim // original_module.attn.heads
        self.attn_scale = original_module.attn.scale
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
        self.attn_to_out_1_dropout_p = getattr(original_module.attn.to_out[1], "p", 0.0) if getattr(original_module.attn, "to_out", None) and len(original_module.attn.to_out) > 1 else 0.0
        self.attn_context_pre_only_flag = original_module.attn.context_pre_only
        
        self.attn_to_add_out_weight = _get_param_or_dummy_zero(
            getattr(getattr(original_module.attn, "to_add_out", None), "weight", None)
        )

        self.attn_to_add_out_bias = _get_param_or_dummy_zero(
            # original_module.attn.to_add_out.bias, torch.zeros(original_module.attn.to_add_out.out_features)
            getattr(getattr(original_module.attn, "to_add_out", None), "bias", None)
        )

        # --- Extract parameters for norm2 (RMSNorm) ---
        self.norm2_weight = _get_param_or_dummy_zero(
            original_module.norm2.weight,
            torch.zeros(
                original_module.norm2.weight.shape
                if hasattr(original_module.norm2, "weight") and original_module.norm2.weight is not None
                else original_module.norm2.normalized_shape
            ),
        )
        self.norm2_eps = original_module.norm2.eps if hasattr(original_module.norm2, "eps") else 1e-6
        self.norm2_elementwise_affine = (
            original_module.norm2.elementwise_affine if hasattr(original_module.norm2, "elementwise_affine") else True
        )

        # --- Extract parameters for ff (FeedForward) ---
        # Feedforward input and output dimensions
        self.ff_dim = original_module.ff.net[0].proj.in_features if hasattr(original_module.ff.net[0].proj, "in_features") else None
        self.ff_dim_out = original_module.ff.net[-1].out_features if hasattr(original_module.ff.net[-1], "out_features") else None

        # 'mult' is not a known attribute — fallback or compute manually if needed
        self.ff_mult = getattr(original_module.ff, "mult", None)  # or set to a default like 4 if known

        # Dropout values — check if they exist and are float or nn.Dropout
        self.ff_dropout_ratio = getattr(original_module.ff, "dropout", 0.0)
        self.ff_final_dropout = getattr(original_module.ff, "final_dropout", 0.0)

        # Activation projection weights and bias — check if 'proj' exists inside net[0]
        if hasattr(original_module.ff.net[0], "proj"):
            self.ff_act_fn_proj_weight = original_module.ff.net[0].proj.weight
            self.ff_act_fn_proj_bias = _get_param_or_dummy_zero(
                original_module.ff.net[0].proj.bias,
                torch.zeros(original_module.ff.net[0].proj.out_features)
            )
        else:
            self.ff_act_fn_proj_weight = None
            self.ff_act_fn_proj_bias = None

        self.ff_project_out_weight = original_module.ff.net[2].weight
        self.ff_project_out_bias = _get_param_or_dummy_zero(
            original_module.ff.net[2].bias, torch.zeros(original_module.ff.net[2].out_features)
        )
        # self.ff_activation_fn_type = _get_activation_fn_type_from_module(original_module.ff.net[0])

        # --- Extract parameters for norm2_context (RMSNorm) ---
        
        self.norm2_context_weight = _get_param_or_dummy_zero(
            getattr(getattr(original_module, "norm2_context", None), "weight", None),
            torch.zeros(getattr(getattr(original_module, "norm2_context", None), "weight", None).shape 
                        if getattr(getattr(original_module, "norm2_context", None), "weight", None) is not None 
                        else getattr(getattr(original_module, "norm2_context", None), "normalized_shape", (1,)))
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
        if hasattr(original_module, "ff_context") and original_module.ff_context is not None:
            self.ff_context_dim = original_module.ff_context.net[0].proj.in_features
            self.ff_context_dim_out = original_module.ff_context.net[-1].out_features
            self.ff_context_mult = 4
            self.ff_context_dropout_ratio = original_module.ff_context.net[1].p
            self.ff_context_final_dropout = 0.0
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
        self.attn_upcast_attention = original_module.attn.upcast_attention
        self.attn_upcast_softmax = original_module.attn.upcast_softmax


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        joint_attention_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # --- Handle optional attention_mask for processor if it was None ---
        # JointAttnProcessor2_0Onnx expects attention_mask, even if zero-sized.
        # If it's dynamically generated or None, pass an empty tensor.
        # This wrapper expects attention_mask not to be an input to JTB.
        # So we assume it's always an empty tensor to the processor.
        attention_mask_for_attn = torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype)
        _attn_original_attention_mask_was_none = True

        # JointAttnProcessor2_0 always expects encoder_hidden_states (not None).
        _attn_original_encoder_hidden_states_was_none = False

        return JointTransformerBlockFunc.apply(
            hidden_states,
            encoder_hidden_states,
            temb,
            self.norm1_linear_weight,
            self.norm1_linear_bias,
            self.norm1_epsilon,
            self.norm1_context_linear_weight,
            self.norm1_context_linear_bias,
            self.norm1_context_epsilon,
            self.attn_heads,
            self.attn_head_dim,
            self.attn_scale,
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
            self.attn_norm_k_weight,
            self.attn_norm_k_eps,
            self.attn_add_q_proj_weight,
            self.attn_add_q_proj_bias,
            self.attn_add_k_proj_weight,
            self.attn_add_k_proj_bias,
            self.attn_add_v_proj_weight,
            self.attn_add_v_proj_bias,
            self.attn_norm_added_q_weight,
            self.attn_norm_added_q_eps,
            self.attn_norm_added_k_weight,
            self.attn_norm_added_k_eps,
            self.attn_to_out_0_weight,
            self.attn_to_out_0_bias,
            self.attn_to_out_1_dropout_p,
            self.attn_to_add_out_weight,
            self.attn_to_add_out_bias,
            self.norm2_weight,
            self.norm2_eps,
            self.ff_dim,
            self.ff_dropout_ratio,
            self.ff_act_fn_proj_weight,
            self.ff_act_fn_proj_bias,
            self.ff_project_out_weight,
            self.ff_project_out_bias,
            self.norm2_context_weight,
            self.norm2_context_eps,
            self.ff_context_dim,
            self.ff_context_dropout_ratio,
            self.ff_context_act_fn_proj_weight,
            self.ff_context_act_fn_proj_bias,
            self.ff_context_project_out_weight,
            self.ff_context_project_out_bias,
            self.attn_upcast_attention,
            self.attn_upcast_softmax,
            _attn_original_encoder_hidden_states_was_none,
            _attn_original_attention_mask_was_none,
        )
