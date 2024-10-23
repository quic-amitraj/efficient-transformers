import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr

def generate_text_stream(prompt, model_name='gpt2', max_length=50):
    # Load pre-trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

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
        yield generated_text  # Yield the current state of the generated text

# Create a Gradio interface with a manual submit button and live text streaming
with gr.Blocks() as demo:
    prompt = gr.Textbox(label="Enter your prompt")
    submit = gr.Button("Submit")
    output = gr.Textbox(label="Generated Text", lines=10)

    submit.click(fn=generate_text_stream, inputs=prompt, outputs=output)

if __name__ == "__main__":
    demo.launch()
