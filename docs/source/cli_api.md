(infer_api)=
# `QEfficient.cloud.infer`
```{eval-rst}
   - Check if compiled qpc for given config already exists, if it does jump to execute, else
   - Check if exported ONNX file already exists, if true, jump to compilation -> execution, else
   - Check if HF model exists in cache, if true, start transform -> export -> compilation -> execution, else,
   - Download HF model -> transform -> export -> compile -> execute

```
```bash
   python -m QEfficient.cloud.infer OPTIONS
```

```{eval-rst}
   :model_name: `str` - Hugging Face Model Card name, Example: `gpt2`
   :num_cores: `int` - Number of cores to compile model on.
   :device_group: `List[int]` - Device Ids to be used for compilation. If len(device_group) > 1, multiple Card setup is enabled.
   :prompt: `str` - Sample prompt for the model text generation.
   :prompts_txt_file_path: `str` - Path to txt file for multiple input prompts.
   :aic_enable_depth_first: `bool` - Enables DFS with default memory size, disabled by default.
   :mos: `int` - Effort level to reduce the on-chip memory.
   :batch_size: `int` - Batch size to compile the model for.
   :prompt_len: `int` - Prompt length for the model to compile.
   :ctx_len: `int` - Maximum context length to compile the model.
   :generation_len: `int` - Number of tokens to be generated.
   :mxfp6: `bool` - Enable compilation for MXFP6 precision.
   :mxint8: `bool` - Compress Present/Past KV to MXINT8 using CustomIO config, default is False.
   :local_model_dir: `str` - Path to custom model weights and config files.
   :cache_dir: `str` - Cache dir where downloaded HuggingFace files are stored.
   :hf_token: `str` - HuggingFace login token to access private repos.
```
(execute_api)=
# `QEfficient.cloud.execute`
```{eval-rst}
   Execute `qpc` on given device and prompt. 
```
```bash
   python -m QEfficient.cloud.execute OPTIONS
```
```{eval-rst}
   :model_name: `str` - Hugging Face Model Card name, Example: `gpt2`
   :qpc_path: `str` - Path to the generated binary after compilation.
   :device_group: `List[int]` - Device Ids to be used for compilation. if len(device_group) > 1. Multiple Card setup is enabled.
   :local_model_dir: `str` - Path to custom model weights and config files.
   :prompt: `str` - Sample prompt for the model text generation
   :prompts_txt_file_path: `str` - Path to txt file for multiple input prompts
   :generation_len: `int` - Number of tokens to be generated.
   :cache_dir: `str` - Cache dir where downloaded HuggingFace files are stored.
   :hf_token: `str` - HuggingFace login token to access private repos.
```
# `QEfficient.cloud.compile`
```{eval-rst}
   CLI app to compiling the Onnx Model on Cloud AI 100 Platform with given config.
```
```bash
   python -m QEfficient.cloud.compile OPTIONS
```
```{eval-rst}
   :onnx_path: `str` - Generated Onnx Model Path.
   :qpc_path: `str` - Path for saving compiled qpc binaries.
   :num_cores: `int` - Number of cores to compile model on.
   :device_group: `List[int]` - Used for finding the number of devices to compile for.
   :aic_enable_depth_first: `bool` - Enables DFS with default memory size, disabled by default.
   :mos: `int` - Effort level to reduce the on-chip memory.
   :batch_size: `int` - Batch size to compile the model for.
   :prompt_len: `int` - Prompt length for the model to compile.
   :ctx_len: `int` - Maximum context length to compile the model.
   :mxfp6: `bool` - Enable compilation for MXFP6 precision.
   :mxint8: `bool` - Compress Present/Past KV to MXINT8 using CustomIO config, default is False.
   :custom_io_file_path: `str` - Path to custom IO file.
```
# `QEfficient.cloud.export`
```{eval-rst}
   CLI app to convert torch Bert style and KV style model to ONNX.
```
```bash
   python -m QEfficient.cloud.export OPTIONS
```
```{eval-rst}
   :model_name: `str` - Hugging Face Model Card name, Example: gpt2
   :cache_dir: `str` - Cache dir to store the downloaded HuggingFace files.
   :hf_token: `str` - HuggingFace login token to access private repos.
   :local_model_dir: `str` - Path to custom model weights and config files.
```