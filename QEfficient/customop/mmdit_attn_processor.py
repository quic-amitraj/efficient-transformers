import onnxscript
import torch
import torch.nn as nn
from typing import Optional, Tuple, Any
import torch.nn.functional as F
from onnx import TensorProto
# from onnxscript import int_literal
from diffusers.models.attention_processor import JointAttnProcessor2_0
from QEfficient.customop.rms_norm import CustomRMSNorm,CustomRMSNormFunc
CUSTOM_OPSET = onnxscript.values.Opset(domain="com.qualcomm.cloud", version=1)
# Import the ONNX Script opset for version 13
ops = getattr(onnxscript, "opset" + str(13))



@onnxscript.script(CUSTOM_OPSET)
def QEffAttentionGetScoresOnnx(
    query: onnxscript.FLOAT,
    key: onnxscript.FLOAT,          # Expected to be pre-transposed for Q @ K.T
    attention_mask: onnxscript.FLOAT, # Passed as an empty tensor if original PyTorch was None
    self_upcast_attention: bool,     # Corresponds to `self.upcast_attention`
    self_upcast_softmax: bool,       # Corresponds to `self.upcast_softmax`
    self_scale: float,               # Corresponds to `self.scale`
    _original_input_dtype_code: int,  # ONNX data type code of the original `query` input
):
    # 1. upcast_attention
    if self_upcast_attention:
        query = ops.Cast(query, to=1) # 1 is ONNX float32
        key = ops.Cast(key, to=1)     # 1 is ONNX float32
    # 2. Handle attention_mask and prepare components for decomposition
    attention_mask_is_none = ops.Equal(ops.Size(attention_mask), ops.Constant(value_int=0))

    # Calculate shape for zero-filled tensor (for `input` when beta=0)
    # The `key` input should already be pre-transposed for Q @ K.T, so its shape is (B*H, Head_Dim, K_Seq_Len)
    # The output of Q @ K.T (or MatMul(query, key)) will be (B*H, Q_Seq_Len, K_Seq_Len)
    q_shape = ops.Shape(query) # (B*H, Q_Seq_Len, Head_Dim)
    k_shape = ops.Shape(key)   # (B*H, Head_Dim, K_Seq_Len)
    
    # Shape for the result of MatMul(query, key) and for the `input` term
    attention_scores_shape = ops.Concat(
        ops.Reshape(q_shape[0], ops.Constant(value_ints=[1])), # Batch*Heads
        ops.Reshape(q_shape[1], ops.Constant(value_ints=[1])), # Q_Seq_Len
        ops.Reshape(k_shape[2], ops.Constant(value_ints=[1])), # K_Seq_Len
        axis=0
    )
    # Create the zero-filled tensor
    zero_filled_input_tensor = ops.ConstantOfShape(attention_scores_shape, value=0.0)
    
    # --- Conditional setup for `baddbmm` decomposition ---
    # `beta_value_for_mul`: This is the scalar value (0.0 or 1.0) to multiply `input_tensor_for_mul` with.
    # `input_tensor_for_mul`: This is `baddbmm_input` from PyTorch (either `attention_mask` or zeros).

    if attention_mask_is_none:
        beta_value_for_mul = ops.Constant(value_float=0.0) # Beta is 0.0
        input_tensor_for_mul = zero_filled_input_tensor # Input should be a tensor of zeros
    else:
        beta_value_for_mul = ops.Constant(value_float=1.0) # Beta is 1.0
        input_tensor_for_mul = attention_mask             # Input is the attention_mask

    # 3. Decompose torch.baddbmm: `output = beta * input + alpha * (batch1 @ batch2)`

    # Step 1: `batch1 @ batch2` (which is `query @ key_transposed` in this context)
    matmul_result = ops.MatMul(query, key)

    # Step 2: `alpha * (batch1 @ batch2)`
    alpha_tensor = ops.Constant(value_float=self_scale)
    scaled_matmul_result = ops.Mul(alpha_tensor, matmul_result)
    # Step 3: `beta * input`
    scaled_input_result = ops.Mul(beta_value_for_mul, input_tensor_for_mul)

    # Step 4: Add them together
    attention_scores = ops.Add(scaled_input_result, scaled_matmul_result)

    # 4. upcast_softmax
    if self_upcast_softmax:
        attention_scores = ops.Cast(attention_scores, to=1) # 1 is ONNX float32

    # 5. softmax
    attention_probs = ops.Softmax(attention_scores, axis=-1)

    # 6. Cast back to original dtype
    attention_probs = ops.Cast(attention_probs, to=_original_input_dtype_code)

    return attention_probs

@onnxscript.script(CUSTOM_OPSET)
def BatchToHeadDimOnnx(hidden_states: onnxscript.FLOAT, heads: int) -> onnxscript.FLOAT:
    """
    Reshape the tensor from `[batch_size * heads, seq_len, head_dim]` to `[batch_size, seq_len, dim * heads]`. 
    `heads` is the number of heads initialized while constructing the `Attention` class.
    """
    input_shape = ops.Shape(hidden_states)
    batch_heads = input_shape[0]
    seq_len = input_shape[1]
    head_dim = input_shape[2]

    batch_size = ops.Div(batch_heads, ops.Constant(value_int=heads))
    heads_tensor = ops.Constant(value_int=heads)
    
    # Reshape to [batch_size, heads, seq_len, head_dim]
    intermediate_shape = ops.Concat(
        ops.Reshape(batch_size, ops.Constant(value_ints=[1])),
        ops.Reshape(heads_tensor, ops.Constant(value_ints=[1])),
        ops.Reshape(seq_len, ops.Constant(value_ints=[1])),
        ops.Reshape(head_dim, ops.Constant(value_ints=[1])),
        axis=0
    )
    tensor = ops.Reshape(hidden_states, intermediate_shape)
    
    # Permute to [batch_size, seq_len, heads, head_dim]
    tensor = ops.Transpose(tensor, perm=[0, 2, 1, 3])
    
    # Final reshape to [batch_size, seq_len, heads * head_dim]
    inner_dim = ops.Mul(heads_tensor, head_dim)
    final_shape = ops.Concat(
        ops.Reshape(batch_size, ops.Constant(value_ints=[1])),
        ops.Reshape(seq_len, ops.Constant(value_ints=[1])),
        ops.Reshape(inner_dim, ops.Constant(value_ints=[1])),
        axis=0
    )
    
    return ops.Reshape(tensor, final_shape)
@onnxscript.script(CUSTOM_OPSET)
def JointAttnProcessor2_0Onnx(
    hidden_states: onnxscript.FLOAT,
    encoder_hidden_states: onnxscript.FLOAT,  # This can conceptually be an empty tensor for None
    attention_mask: onnxscript.FLOAT,
    # Parameters from `Attention` module
    attn_heads: int,
    attn_head_dim: int,
    attn_scale: float,  # self.scale
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
    norm_k_weight: onnxscript.FLOAT,
    norm_k_eps: float,
    # Weights and Biases for Cross-Attention Projections (`attn.add_q_proj`, `attn.add_k_proj`, `attn.add_v_proj`)
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
    # Weights and Biases for Output Projections (`attn.to_out[0]`, `attn.to_out[1]`)
    to_out_0_weight: onnxscript.FLOAT,
    to_out_0_bias: onnxscript.FLOAT,
    to_out_1_dropout_p: float,  # Dropout ratio for self.to_out[1]
    # Other flags
    attn_added_kv_proj_dim: int,  # From attn.added_kv_proj_dim (to determine if add_q_proj etc. exist)
    to_add_out_weight: onnxscript.FLOAT,  # For attn.to_add_out
    to_add_out_bias: onnxscript.FLOAT,
    attn_upcast_attention: bool,
    attn_upcast_softmax: bool,
    query_block_size: int,  # Block size for block attention
    _attn_original_attention_mask_was_none: bool,
    _attn_original_encoder_hidden_states_was_none: bool,
    _original_input_onnx_dtype_code: int,
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
   
    # Reshape from (batch_size, heads, seq_len, head_dim) to (batch_size * heads, seq_len, head_dim)
    new_batch_size_heads = ops.Mul(batch_size, attn_heads_tensor)
    query = ops.Reshape(query, ops.Concat(
        ops.Reshape(new_batch_size_heads, ops.Constant(value_ints=[1])),
        ops.Reshape(ops.Shape(query)[2], ops.Constant(value_ints=[1])), # seq_len
        ops.Reshape(ops.Shape(query)[3], ops.Constant(value_ints=[1])), # head_dim
        axis=0
    ))
    key = ops.Reshape(key, ops.Concat(
        ops.Reshape(new_batch_size_heads, ops.Constant(value_ints=[1])),
        ops.Reshape(ops.Shape(key)[2], ops.Constant(value_ints=[1])), # seq_len
        ops.Reshape(ops.Shape(key)[3], ops.Constant(value_ints=[1])), # head_dim
        axis=0
    ))
    value = ops.Reshape(value, ops.Concat(
        ops.Reshape(new_batch_size_heads, ops.Constant(value_ints=[1])),
        ops.Reshape(ops.Shape(value)[2], ops.Constant(value_ints=[1])), # seq_len
        ops.Reshape(ops.Shape(value)[3], ops.Constant(value_ints=[1])), # head_dim
        axis=0
    ))
    key = ops.Transpose(key, perm=[0, 2, 1]) # key.transpose(-1, -2)

    # QKV attention with block attention logic (matching QEffJointAttnProcessor2_0)
    query_seq_len = ops.Shape(query)[1]
    value_seq_len = ops.Shape(value)[1]
    
    # Check if cross-attention or self-attention
    is_cross_attention = ops.Not(ops.Equal(query_seq_len, value_seq_len))
    
    if is_cross_attention:
        # Cross-attention: use regular attention (single block)
        attention_probs = QEffAttentionGetScoresOnnx(query, key, attention_mask, 
            self_upcast_attention=attn_upcast_attention,
            self_upcast_softmax=attn_upcast_softmax,
            self_scale=attn_scale,
            _original_input_dtype_code=_original_input_onnx_dtype_code,
        )
        final_combined_hidden_states = ops.Bmm(attention_probs, value)
    else:
        # Self-attention: use block attention
        query_block_size_tensor = ops.Constant(value_int=query_block_size)
        num_blocks = ops.Div(
            ops.Add(query_seq_len, ops.Sub(query_block_size_tensor, ops.Constant(value_int=1))),
            query_block_size_tensor
        )
        
        # Initialize output tensor with zeros
        batch_heads = ops.Shape(query)[0]
        head_dim = ops.Shape(query)[2]
        output_shape = ops.Concat(
            ops.Reshape(batch_heads, ops.Constant(value_ints=[1])),
            ops.Reshape(query_seq_len, ops.Constant(value_ints=[1])),
            ops.Reshape(head_dim, ops.Constant(value_ints=[1])),
            axis=0
        )
        final_combined_hidden_states = ops.ConstantOfShape(output_shape, value=0.0)
        
        # Process blocks sequentially (ONNX doesn't support dynamic loops well, so we'll use a simplified approach)
        # For ONNX compatibility, we'll fall back to single block processing
        # In a real implementation, you might need to unroll the loop or use other ONNX constructs
        attention_probs = QEffAttentionGetScoresOnnx(query, key, attention_mask, 
            self_upcast_attention=attn_upcast_attention,
            self_upcast_softmax=attn_upcast_softmax,
            self_scale=attn_scale,
            _original_input_dtype_code=_original_input_onnx_dtype_code,
        )
        final_combined_hidden_states = ops.Bmm(attention_probs, value)

    # Convert from head dimension back to batch dimension (matching reference)
    hidden_states = BatchToHeadDimOnnx(final_combined_hidden_states, attn_heads)

    final_encoder_hidden_states_out = ops.Constant(value_float=0.0)

    if encoder_is_not_none:
        # Split the attention outputs (matching reference logic)
        sample_output_len = ops.Shape(residual)[1]
        
        # Split hidden_states into sample and encoder parts
        hidden_states_sample = ops.Slice(
            hidden_states,
            starts=ops.Constant(value_ints=[0, 0, 0]),
            ends=ops.Concat(
                ops.Reshape(ops.Shape(hidden_states)[0], ops.Constant(value_ints=[1])),
                ops.Reshape(sample_output_len, ops.Constant(value_ints=[1])),
                ops.Reshape(ops.Shape(hidden_states)[2], ops.Constant(value_ints=[1])),
                axis=0
            ),
            axes=ops.Constant(value_ints=[0, 1, 2])
        )

        final_encoder_hidden_states_out = ops.Slice(
            hidden_states,
            starts=ops.Concat(
                ops.Reshape(ops.Constant(value_int=0), ops.Constant(value_ints=[1])),
                ops.Reshape(sample_output_len, ops.Constant(value_ints=[1])),
                ops.Reshape(ops.Constant(value_int=0), ops.Constant(value_ints=[1])),
                axis=0
            ),
            ends=ops.Concat(
                ops.Reshape(ops.Shape(hidden_states)[0], ops.Constant(value_ints=[1])),
                ops.Reshape(ops.Shape(hidden_states)[1], ops.Constant(value_ints=[1])),
                ops.Reshape(ops.Shape(hidden_states)[2], ops.Constant(value_ints=[1])),
                axis=0
            ),
            axes=ops.Constant(value_ints=[0, 1, 2])
        )

        # Apply to_add_out (matching reference logic)
        final_encoder_hidden_states_out = ops.Add(ops.MatMul(final_encoder_hidden_states_out, ops.Transpose(to_add_out_weight, perm=[1,0])), to_add_out_bias)
        
        # Update hidden_states to be the sample part
        hidden_states = hidden_states_sample
    else:
        final_encoder_hidden_states_out = encoder_hidden_states # Remains original empty/dummy tensor

    # linear proj (matching reference)
    hidden_states = ops.Add(
        ops.MatMul(hidden_states, ops.Transpose(to_out_0_weight, perm=[1, 0])), to_out_0_bias
    )

    # dropout (matching reference)
    hidden_states = ops.Dropout(hidden_states, ratio=to_out_1_dropout_p)
    
    return hidden_states, final_encoder_hidden_states_out


class JointAttnProcessor2_0Func(torch.autograd.Function):
    @staticmethod
    def forward(
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,  # Will be a dummy zero tensor if original was None
        attention_mask: torch.Tensor,  # Will be a dummy zero tensor if original was None
        # Parameters from `Attention` module (passed as separate arguments)
        attn_heads: int,
        attn_head_dim: int,
        attn_scale: float,
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
        attn_added_kv_proj_dim: int,
        to_add_out_weight: torch.Tensor,
        to_add_out_bias: torch.Tensor,
        attn_upcast_attention: bool,
        attn_upcast_softmax: bool,
        
        # --- Internal flags for the autograd.Function to replicate exact behavior ---
        # Indicates if `encoder_hidden_states` was originally None.
        # This will allow proper `if encoder_hidden_states is not None` logic.
        _original_encoder_hidden_states_was_none: bool,
        _original_attention_mask_was_none: bool,
        _original_input_onnx_dtype_code: int,
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
        def get_attention_scores_pytorch(query_input, key_input, mask_input):
            dtype = query_input.dtype
            
            if attn_upcast_attention:
                query_input = query_input.float()
                key_input = key_input.float()
            
            baddbmm_input = None
            beta = 0.0 # float
            if mask_input is None:
                baddbmm_input = torch.empty(
                    query_input.shape[0], query_input.shape[1], key_input.shape[2],
                    dtype=query_input.dtype, device=query_input.device
                )
            else:
                baddbmm_input = mask_input
                beta = 1.0 # float

            attention_scores = torch.baddbmm(
                baddbmm_input,
                query_input,
                key_input,
                beta=beta,
                alpha=attn_scale,
            )

            if attn_upcast_softmax:
                attention_scores = attention_scores.float()
            
            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(dtype)
            return attention_probs
        def batch_to_head_dim_pytorch(tensor, heads_val):
            """
            Reshape the tensor from `[batch_size * heads, seq_len, head_dim]` to `[batch_size, seq_len, dim * heads]`. 
            `heads` is the number of heads initialized while constructing the `Attention` class.
            """
            batch_heads, seq_len, head_dim = tensor.shape
            batch_size = batch_heads // heads_val
            tensor = tensor.reshape(batch_size, heads_val, seq_len, head_dim)
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, head_dim * heads_val)
            return tensor

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

        query = query.reshape(-1, query.shape[-2], query.shape[-1])
        key = key.reshape(-1, key.shape[-2], key.shape[-1])
        value = value.reshape(-1, value.shape[-2], value.shape[-1])

        key = key.transpose(-1, -2) # (..., head_dim, seq_len)

        # QKV attention with block attention logic (matching QEffJointAttnProcessor2_0)
        if query.size(-2) != value.size(-2):  # cross-attention, use regular attention
            # QKV done in single block
            attention_probs = get_attention_scores_pytorch(query, key, actual_attention_mask)
            combined_hidden_states = torch.bmm(attention_probs, value)
        else:  # self-attention, use blocked attention
            # QKV done with block-attention (a la FlashAttentionV2)
            query_block_size = 64  # Default block size from QEffJointAttnProcessor2_0
            query_seq_len = query.size(-2)
            num_blocks = (query_seq_len + query_block_size - 1) // query_block_size
            for qidx in range(num_blocks):
                query_block = query[:, qidx * query_block_size : (qidx + 1) * query_block_size, :]
                attention_probs = get_attention_scores_pytorch(query_block, key, actual_attention_mask)
                hidden_states_block = torch.bmm(attention_probs, value)
                if qidx == 0:
                    combined_hidden_states = hidden_states_block
                else:
                    combined_hidden_states = torch.cat((combined_hidden_states, hidden_states_block), -2)
        
        hidden_states = batch_to_head_dim_pytorch(combined_hidden_states, attn_heads)

        final_encoder_hidden_states = None
        # --- Split attention outputs if actual_encoder_hidden_states was present ---
        if actual_encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, final_encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            # Apply to_add_out only if not context_pre_only (assuming context_pre_only is False for this implementation)
            final_encoder_hidden_states = F.linear(final_encoder_hidden_states, to_add_out_weight, to_add_out_bias)
        
        # linear proj
        hidden_states = F.linear(hidden_states, to_out_0_weight, to_out_0_bias)
        # dropout
        hidden_states = F.dropout(hidden_states, p=to_out_1_dropout_p, training=False)

        if final_encoder_hidden_states is None:
            final_encoder_hidden_states = torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype)

        return hidden_states, final_encoder_hidden_states
        
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # This can be left as `pass` if no backward context is needed
        pass
    
    @staticmethod
    def symbolic(
        g: torch.Graph,
        hidden_states: torch.Value,
        encoder_hidden_states: torch.Value,
        attention_mask: torch.Value,
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
        attn_added_kv_proj_dim: int,
        to_add_out_weight: torch.Value,
        to_add_out_bias: torch.Value,
        attn_upcast_attention: bool,
        attn_upcast_softmax: bool,
        _original_encoder_hidden_states_was_none: bool,  # From ctx
        _original_attention_mask_was_none: bool,  # From ctx
        _original_input_onnx_dtype_code: int,
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
            attn_scale_f= attn_scale,
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
            attn_added_kv_proj_dim_i=attn_added_kv_proj_dim,
            to_add_out_weight=to_add_out_weight,
            to_add_out_bias=to_add_out_bias,
            attn_upcast_attention_i=attn_upcast_attention,
            attn_upcast_softmax_i=attn_upcast_softmax,
            query_block_size_i=64,  # Default block size matching QEffJointAttnProcessor2_0
            _attn_original_attention_mask_was_none_b=_original_attention_mask_was_none,
            _attn_original_encoder_hidden_states_was_none_b=_original_encoder_hidden_states_was_none,
            _original_input_onnx_dtype_code_i=_original_input_onnx_dtype_code,
        )
        return result



def get_first_linear_in_features(module):
    for submodule in module.net:
        if isinstance(submodule, nn.Linear):
            return submodule.in_features
    return None

class JointAttnProcessor2_0AIC(nn.Module):
    """
    JointAttnProcessor2_0 module that works by replacing the current processor instance
    with compiler-known custom-op via JointAttnProcessor2_0Func.
    Inherits from the original JointAttnProcessor2_0 to maintain its API.
    """

    def __init__(self):
        super().__init__()
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor],
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
        _original_encoder_hidden_states_was_none: bool,
        _original_attention_mask_was_none: bool,
        original_input_onnx_dtype_code: int,
        *args,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Determine if encoder_hidden_states and attention_mask were originally None
        # These flags are passed to the autograd.Function to guide symbolic export.
     
        # Pass all relevant parameters to the autograd.Function
        return JointAttnProcessor2_0Func.apply(
            hidden_states,
            encoder_hidden_states,
            attention_mask,  # Pass the (potentially dummy) mask
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
            self.to_out_0_weight,
            self.to_out_0_bias,
            self.to_out_1_dropout_p,
            self.attn_added_kv_proj_dim,
            self.to_add_out_weight,
            self.to_add_out_bias,
            self.attn_upcast_attention,
            self.attn_upcast_softmax,
            _original_encoder_hidden_states_was_none,  # Passed as flag to autograd.Function
            _original_attention_mask_was_none,  # Passed as flag to autograd.Function
            original_input_onnx_dtype_code,
        )
