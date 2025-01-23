from typing import Any

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import import_from_string
from torch import Tensor, nn
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from QEfficient import QEFFAutoModel


class QEffEmbedding(HuggingFaceEmbeddings):
    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super(HuggingFaceEmbeddings, self).__init__(**kwargs)
        try:
            import sentence_transformers  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc

        self._client = QEff_sentence_transformers(self.model_name, cache_folder=self.cache_folder, **self.model_kwargs)

from sentence_transformers.models.Transformer import Transformer


class Qeff_transformers(Transformer):
    def _load_model(self, model_name_or_path, config, cache_dir, backend, **model_args) -> None:
        """Loads the transformer model"""
        self.qeff_auto_model = QEFFAutoModel.from_pretrained(model_name_or_path, **model_args)
        self.auto_model = self.qeff_auto_model.model
        self.qeff_auto_model.compile(batch_size=32)

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """Returns token_embeddings, cls_token"""

        trans_features = {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}
        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]
        output_states = self.qeff_auto_model.generate(trans_features)
        output_tokens = torch.tensor(output_states)
        features["token_embeddings"] = output_tokens
        if self.auto_model.config.output_hidden_states and len(output_states) > 2:
            all_layer_idx = 2  # I.e. after last_hidden_states and pooler_output
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features["all_layer_embeddings"] = hidden_states

        return features


class QEff_sentence_transformers(SentenceTransformer):
    def forward(self, input: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        if self.module_kwargs is None:
            return super().forward(input)
        for module_name, module in self.named_children():
            module_kwarg_keys = self.module_kwargs.get(module_name, [])
            module_kwargs = {key: value for key, value in kwargs.items() if key in module_kwarg_keys}
            input = module(input, **module_kwargs)
        return input

    def _load_module_class_from_ref(
        self,
        class_ref: str,
        model_name_or_path: str,
        trust_remote_code: bool,
        revision: str | None,
        model_kwargs: dict[str, Any] | None,
    ) -> nn.Module:
        # If the class is from sentence_transformers, we can directly import it,
        # otherwise, we try to import it dynamically, and if that fails, we fall back to the default import
        if class_ref == "sentence_transformers.models.Transformer":
            return import_from_string("QEfficient.utils.embedding.Qeff_transformers")

        if class_ref.startswith("sentence_transformers."):
            return import_from_string(class_ref)

        if trust_remote_code:
            code_revision = model_kwargs.pop("code_revision", None) if model_kwargs else None
            try:
                return get_class_from_dynamic_module(
                    class_ref,
                    model_name_or_path,
                    revision=revision,
                    code_revision=code_revision,
                )
            except OSError:
                # Ignore the error if the file does not exist, and fall back to the default import
                pass

        return import_from_string(class_ref)
