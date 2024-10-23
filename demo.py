import gradio as gr
from transformers import pipeline
from QEfficient.generation.text_generation_inference import cloud_ai_100_exec_kv
from QEfficient.utils.run_utils import ApiRunner
from QEfficient.utils._utils import load_hf_tokenizer
from tests.utils import load_pytorch_model, replace_transformers_quantizers
from QEfficient.utils.constants import Constants
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the GPT-2 model and other models (you can add more here)
gpt2 = pipeline("text-generation", model="gpt2")
model_name="gpt2"
model_config = {"model_name": model_name}
model_hf, _ = load_pytorch_model(model_config)

tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name)
config = model_hf.config
batch_size = len(Constants.INPUT_STR)
api_runner = ApiRunner(
   batch_size,
   tokenizer,
   config,
   Constants.INPUT_STR,
   prompt_len=32,
   ctx_len=128,
)
# generated_ids= api_runner.run_hf_model_on_pytorch(model_hf)
# generated_text2 = tokenizer.decode(generated_ids, skip_special_tokens=True)
# print(generated_text2)

# test_qpcs_path="/local/mnt/workspace/amitraj/.cache/qeff_cache/gpt2/qpc_14cores_1bs_32pl_128cl_1mos_1devices_mxfp6/qpcs"
# # cloud_ai_100_tokens = api_runner.run_kv_model_on_cloud_ai_100(test_qpcs_path)
# # generated_text1 = tokenizer.decode(generated_ids, skip_special_tokens=True)
# cloud_ai_100_exec_kv(qpc_path=test_qpcs_path, prompt="My name is",tokenizer=tokenizer)



def generate_text_stream(prompt, model_type, model_name='gpt2', max_length=50):
    # Load pre-trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text in a streaming fashion
    output_ids = input_ids
    generated_text = ""
    for _ in range(max_length):
        outputs = model(output_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        output_ids = torch.cat([output_ids, next_token_id], dim=-1)

        # Decode and append the latest token
        generated_text += tokenizer.decode(next_token_id[0].tolist(), skip_special_tokens=True)
        yield generated_text, generated_text

# Placeholder function for image generation (can be updated later)
def image_generation(prompt):
   return "This would generate an image based on the prompt."

# Function to generate output from GPT-2
def generate_text(prompt):
   result = gpt2(prompt, max_length=100, num_return_sequences=1)
   generated_text = result[0]['generated_text']
   # Output the same text in two different boxes
   return generated_text, generated_text

# Function to handle the model selection and call the respective function
def handle_model_selection(model_type, prompt):
   if model_type == "Text Generation (GPT-2)":
       return generate_text(prompt)
   elif model_type == "Image Generation":
       return image_generation(prompt), "Image output placeholder"  # Dummy outputs for now

# Create the Gradio interface with customized layout
with gr.Blocks() as demo:
   gr.Markdown(
            """
            <div style="text-align: center;">
                <h1>Qualcomm Efficient-Transformers Demo</h1>
            </div>
            """
        )
   
   # Dropdown to select the model type
   model_type = gr.Dropdown(label="Select Model Type", choices=["Text Generation (GPT-2)", "Text Generation (Llama-3.1)"], value="Text Generation (GPT-2)")
   
   # Input box taking full width, with increased height
   input_box = gr.Textbox(label="Input Prompt", placeholder="Enter your prompt here", lines=4)  # Increased lines
   
   # Button to trigger the generation
   submit_button = gr.Button("Generate")
   
   # Two output boxes in half-width columns, with increased height
   with gr.Row():
       output1 = gr.Textbox(label="PyTorch Output:", lines=10)  # Increased height
       output2 = gr.Textbox(label="AI 100 Output:", lines=10)  # Increased height

   

   # Define the function to call on button click based on the selected model
   submit_button.click(generate_text_stream, inputs= [input_box, model_type], outputs=[output1, output2])

# Launch the demo
demo.launch()

