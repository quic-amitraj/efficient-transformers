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
from QEfficient.customop.mmdit import (
    JointAttnProcessor2_0AIC,
    JointTransformerBlockAIC,
)
from QEfficient.diffusers.models.attention import QEffJointTransformerBlock
from QEfficient.diffusers.models.attention_processor import (
    QEffAttention,
    QEffJointAttnProcessor2_0,
)

class SD3TransformerBlockTransform:
    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        transformed = False
        

        # Iterate through the named children to allow direct replacement via setattr
        # This will handle the top level of `model`. For nested modules, we recurse.
        # This is essentially what `model.named_modules()` provides in a more controlled way for replacement.
        for name, child_module in model.named_children():
            # Check if the child_module is a JointTransformerBlock
            if isinstance(child_module, JointTransformerBlock):
                # Create a NEW instance of JointTransformerBlockAIC
                # This explicitly calls JointTransformerBlockAIC.__init__(original_module=child_module)
                new_block = JointTransformerBlockAIC(original_module=child_module)
                
                # Replace the old instance with the new one in the parent module (`model`)
                setattr(model, name, new_block)
                transformed = True
            else:
                # If the child module is not the one we want to replace at this level,
                # recurse into its children to find JointTransformerBlocks deeper in the hierarchy.
                # This handles cases where JTB is inside ModuleList or other containers.
                _, child_transformed = cls.apply(child_module)
                if child_transformed:
                    transformed = True
                    
        return model, transformed
    
class CustomOpsTransform(ModuleMappingTransform):
    _module_mapping = {RMSNorm: CustomRMSNormAIC,
                       JointAttnProcessor2_0: JointAttnProcessor2_0AIC,
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
