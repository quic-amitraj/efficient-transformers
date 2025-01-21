# from typing import Any

# import torch
# from langchain_huggingface import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer
# from sentence_transformers.util import import_from_string
# from torch import Tensor, nn
# from transformers.dynamic_module_utils import get_class_from_dynamic_module
# from sentence_transformers.models.Transformer import Transformer

# from QEfficient import QEFFAutoModel


# class QEffEmbedding(HuggingFaceEmbeddings):
#     def __init__(self, **kwargs: Any):
#         """Initialize the sentence_transformer."""
#         super(HuggingFaceEmbeddings,self).__init__(**kwargs)
#         try:
#             import sentence_transformers  # type: ignore[import]
#         except ImportError as exc:
#             raise ImportError(
#                 "Could not import sentence_transformers python package. "
#                 "Please install it with `pip install sentence-transformers`."
#             ) from exc

#         self._client = QEff_sentence_transformers(
#             self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
#         )



# # class QEFFEmbedding:
# #     def __init__(self, model):
# #         self.model = AutoModel.from_pretrained(model)
# #         config = AutoConfig.from_pretrained(model)
# #         self.max_seq_length = config.max_position_embeddings
# #         self.model.compile(seq_len=self.max_seq_length, num_cores=16,batch_size=1)
# #         self.tokenizer = AutoTokenizer.from_pretrained(model)
        
# #     def embed_text(
# #         self,
# #         sentences: str | list[str],
# #         ):
                
# #         for sentence in sentences:
# #             features = self.tokenizer(
# #                 sentence,
# #                 return_tensors="pt",
# #                 )
# #             embeddings=self.model.generate(features, device_ids=[5])
# #         token_embedding=torch.tensor(embeddings)
# #         features['token_embeddings']=token_embedding
# #         pooling=Pooling(word_embedding_dimension=384)
# #         result_after_pooling=pooling.forward(features=features)
# #         normalize=Normalize()
# #         After_normalize=normalize(result_after_pooling)
# #         all_embeddings=After_normalize['sentence_embedding']
# #         import ipdb; ipdb.set_trace()
# #         if not isinstance(all_embeddings, np.ndarray):
# #                 if all_embeddings and all_embeddings[0].dtype == torch.bfloat16:
# #                     all_embeddings = np.asarray([emb.float().numpy() for emb in all_embeddings])
# #                 else:
# #                     all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
# #         import ipdb; ipdb.set_trace()
# #         return token_embedding
# #     def embed_documents(self, texts):
# #         return self.embed_text(sentences=texts)
# #     def __call__(self, text):
# #         # Make the class callable for single queries
# #         return self.embed_text(sentences=text)
            
# # from sentence_transformers.SentenceTransformer import SentenceTransformer

# # class QEFFEmbedding(SentenceTransformer):
    
#     # def __init__(
#     # self,
#     # model_name_or_path: str | None = None,
#     # modules: Iterable[nn.Module] | None = None,
#     # device: str = "cpu",
#     # prompts: dict[str, str] | None = None,
#     # default_prompt_name: str | None = None,
#     # similarity_fn_name: str | SimilarityFunction | None = None,
#     # cache_folder: str | None = None,
#     # trust_remote_code: bool = False,
#     # revision: str | None = None,
#     # local_files_only: bool = False,
#     # token: bool | str | None = None,
#     # use_auth_token: bool | str | None = None,
#     # truncate_dim: int | None = None,
#     # model_kwargs: dict[str, Any] | None = None,
#     # tokenizer_kwargs: dict[str, Any] | None = None,
#     # config_kwargs: dict[str, Any] | None = None,
#     # model_card_data: SentenceTransformerModelCardData | None = None,
#     # backend: Literal["torch", "onnx", "openvino"] = "torch",
#     # ) -> None:
#     #     # Note: self._load_sbert_model can also update `self.prompts` and `self.default_prompt_name`
        
#     #     self.prompts = prompts or {}
#     #     self.default_prompt_name = default_prompt_name
#     #     self.similarity_fn_name = similarity_fn_name
#     #     self.trust_remote_code = trust_remote_code
#     #     self.truncate_dim = truncate_dim
#     #     self.model_card_data = model_card_data or SentenceTransformerModelCardData()
#     #     self.module_kwargs = None
#     #     self._model_card_vars = {}
#     #     self._model_card_text = None
#     #     self._model_config = {}
#     #     self.backend = backend
#     #     self.model = AutoModel.from_pretrained(model_name_or_path)
#     #     self.model.compile(seq_len=512, num_cores=16,batch_size=1)
        
        
#     #     super(SentenceTransformer, self).__init__() 

#     #     self.to(device)
#     #     import ipdb; ipdb.set_trace()
#     #     # Ideally, INSTRUCTOR models should set `include_prompt=False` in their pooling configuration, but
#     #     # that would be a breaking change for users currently using the InstructorEmbedding project.
#     #     # So, instead we hardcode setting it for the main INSTRUCTOR models, and otherwise give a warning if we
#     #     # suspect the user is using an INSTRUCTOR model.
#     #     if model_name_or_path in ("hkunlp/instructor-base", "hkunlp/instructor-large", "hkunlp/instructor-xl"):
#     #         self.set_pooling_include_prompt(include_prompt=False)
#     #     elif (
#     #         model_name_or_path
#     #         and "/" in model_name_or_path
#     #         and "instructor" in model_name_or_path.split("/")[1].lower()
#     #     ):
#     #         if any([module.include_prompt for module in self if isinstance(module, Pooling)]):
#     #             warning(
#     #                 "Instructor models require `include_prompt=False` in the pooling configuration. "
#     #                 "Either update the model configuration or call `model.set_pooling_include_prompt(False)` after loading the model."
#     #             )

#     #     # Pass the model to the model card data for later use in generating a model card upon saving this model
#     #     self.model_card_data.register_model(self)
#     #     import ipdb; ipdb.set_trace()

#     #     @property
#     #     def device(self):
#     #         return s

    
    
    
#     # def __init__(self, model = None, modules = None, device = None, prompts = None, default_prompt_name = None, similarity_fn_name = None, cache_folder = None, trust_remote_code = False, revision = None, local_files_only = False, token = None, use_auth_token = None, truncate_dim = None, model_kwargs = None, tokenizer_kwargs = None, config_kwargs = None, model_card_data = None, backend = "torch"):
        
#     #     self.model = AutoModel.from_pretrained(model)
#     #     self.model.compile(seq_len=512, num_cores=16,batch_size=1)
        
#     # def forward(self, input, **kwargs):
#     #     if self.module_kwargs is None:
#     #         return super().forward(input)
#     #     import ipdb; ipdb.set_trace()
       
#     #     token_embeddings= self.model.generate(input, device_ids=[5])
#     #     input['token_embeddings']=torch.tensor(token_embeddings)
#     #     output=Pooling.forward(features=input)
#     # def enccode(self, input):
        
            
            
            
#     #     return input
    
#     # def embed_documents(self, texts: List[str]) -> List[List[float]]:
#     #     """Compute doc embeddings using a HuggingFace transformer model.

#     #     Args:
#     #         texts: The list of texts to embed.

#     #     Returns:
#     #         List of embeddings, one for each text.
#     #     """
#     #     import sentence_transformers
#     #     import ipdb; ipdb.set_trace()
#     #     texts = list(map(lambda x: x.replace("\n", " "), texts))
#     #     embeddings = self.encode(
#     #         texts
#     #     )

#     #     return embeddings.tolist()
    
#     # def embed_query(self, text: str) -> List[float]:
#     #     """Compute query embeddings using a HuggingFace transformer model.

#     #     Args:
#     #         text: The text to embed.

#     #     Returns:
#     #         Embeddings for the text.
#     #     """
#     #     return self.embed_documents([text])[0]
    
    
# # class qeff_embedding:
# #     def __init__(self, model):
# #         self.model = AutoModel.from_pretrained(model)
        
# #         self.model.compile(seq_len=512, num_cores=16,batch_size=1)
# #     def encode()



# class Qeff_transformers(Transformer):
  
#     def _load_model(self, model_name_or_path, config, cache_dir, backend, **model_args) -> None:
#         """Loads the transformer model"""
#         self.qeff_auto_model = QEFFAutoModel.from_pretrained(
#             model_name_or_path, **model_args
#         )
#         self.auto_model = self.qeff_auto_model.model
#         self.qeff_auto_model.compile()
#         print("hello")
#         print("testing")
    
    
#     def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
#         """Returns token_embeddings, cls_token"""
        
#         trans_features = {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}
#         if "token_type_ids" in features:
#             trans_features["token_type_ids"] = features["token_type_ids"]
#         output_states = self.qeff_auto_model.generate(trans_features)
#         output_tokens = torch.tensor(output_states)
#         features["token_embeddings"] = output_tokens
#         if self.auto_model.config.output_hidden_states and len(output_states) > 2:
#             all_layer_idx = 2  # I.e. after last_hidden_states and pooler_output
#             if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
#                 all_layer_idx = 1

#             hidden_states = output_states[all_layer_idx]
#             features["all_layer_embeddings"] = hidden_states

#         return features

# class QEff_sentence_transformers(SentenceTransformer):

#     def forward(self, input: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
#         if self.module_kwargs is None:
#             return super().forward(input)
#         for module_name, module in self.named_children():
#             module_kwarg_keys = self.module_kwargs.get(module_name, [])
#             module_kwargs = {key: value for key, value in kwargs.items() if key in module_kwarg_keys}
#             input = module(input, **module_kwargs)
#         return input
    
#     def _load_module_class_from_ref(
#         self,
#         class_ref: str,
#         model_name_or_path: str,
#         trust_remote_code: bool,
#         revision: str | None,
#         model_kwargs: dict[str, Any] | None,
#     ) -> nn.Module:
#         # If the class is from sentence_transformers, we can directly import it,
#         # otherwise, we try to import it dynamically, and if that fails, we fall back to the default import
#         if class_ref == 'sentence_transformers.models.Transformer':
#             return import_from_string('QEfficient.utils.embedding.Qeff_transformers')    
        
#         if class_ref.startswith("sentence_transformers."):
#             return import_from_string(class_ref)

#         if trust_remote_code:
#             code_revision = model_kwargs.pop("code_revision", None) if model_kwargs else None
#             try:
#                 return get_class_from_dynamic_module(
#                     class_ref,
#                     model_name_or_path,
#                     revision=revision,
#                     code_revision=code_revision,
#                 )
#             except OSError:
#                 # Ignore the error if the file does not exist, and fall back to the default import
#                 pass

#         return import_from_string(class_ref)