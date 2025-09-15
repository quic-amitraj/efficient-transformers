# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
from typing import Tuple

from diffusers.models.attention import JointTransformerBlock
from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0
from diffusers.models.normalization import RMSNorm
from torch import nn

from QEfficient.base.pytorch_transforms import ModuleMappingTransform
from QEfficient.customop.rms_norm import CustomRMSNormAIC
from QEfficient.customop.mmdit_attn_processor import (
    JointAttnProcessor2_0AIC,
    
)
from QEfficient.customop.mmdit_transformer_block import JointTransformerBlockAIC
from QEfficient.customop.mmdit_attn import AttentionAIC
from QEfficient.diffusers.models.attention import QEffJointTransformerBlock
from QEfficient.diffusers.models.attention_processor import (
    QEffAttention,
    QEffJointAttnProcessor2_0,
)

class SD3TransformerBlockTransform:
    
    MODULE_REPLACEMENTS = {
        JointTransformerBlock: JointTransformerBlockAIC,
        Attention: AttentionAIC,
        JointAttnProcessor2_0: JointAttnProcessor2_0AIC,
        # Add more mappings here as needed
    }

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        transformed = False
        
        # Iterate through the named children to allow direct replacement via setattr
        # This will handle the top level of `model`. For nested modules, we recurse.
        # This is essentially what `model.named_modules()` provides in a more controlled way for replacement.
        for name, child_module in model.named_children():
            # Check if the child_module is a JointTransformerBlock
            for original_cls, replacement_cls in cls.MODULE_REPLACEMENTS.items():
                if isinstance(child_module, original_cls):
                    # Initialize the replacement with the original module
                    # breakpoint()
                    new_module = replacement_cls(original_module=child_module)
                    setattr(model, name, new_module)
                    transformed = True
                    break
        
            else:
                # If the child module is not the one we want to replace at this level,
                # recurse into its children to find JointTransformerBlocks deeper in the hierarchy.
                # This handles cases where JTB is inside ModuleList or other containers.
                # breakpoint()
                _, child_transformed = cls.apply(child_module)
                if child_transformed:
                    transformed = True
                    
        return model, transformed
    
class CustomOpsTransform(ModuleMappingTransform):
    _module_mapping = {RMSNorm: CustomRMSNormAIC,
                    #    JointAttnProcessor2_0: JointAttnProcessor2_0AIC,
                    #    JointTransformerBlock: JointTransformerBlockAIC,
                    #    Attention: CustomAttentionAIC,
                    }

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        model, transformed = super().apply(model)
        return model, transformed


class AttentionTransform(ModuleMappingTransform):
    _module_mapping = {
        # Attention: QEffAttention,
        # JointAttnProcessor2_0: QEffJointAttnProcessor2_0,
        # JointTransformerBlock: QEffJointTransformerBlock,
    }

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        model, transformed = super().apply(model)
        return model, transformed
