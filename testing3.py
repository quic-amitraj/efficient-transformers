# from sentence_transformers import SentenceTransformer
# sentences = ["This is an example sentence", "Each sentence is converted"]

# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# import ipdb; ipdb.set_trace()
# embeddings = model.encode(sentences)
# import ipdb; ipdb.set_trace()
# print(embeddings)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
# Step 1: Fetch README and Prepare Chunks
import requests
repo_url = "https://raw.githubusercontent.com/quic/efficient-transformers/main/README.md"
readme_content = requests.get(repo_url).text
chunks = [chunk.strip() for chunk in readme_content.split("\n") if chunk.strip()]
# Step 2: Create Embeddings and Save FAISS Index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



# Define a custom embedding class
class CustomEmbeddings:
   def __init__(self, model, tokenizer, device="cpu"):
       self.model = model.to(device)
       self.tokenizer = tokenizer
       self.device = device
       self.max_length= model.
   def embed_text(self, text):
       # Tokenize the input text
       inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
       inputs = {key: val.to(self.device) for key, val in inputs.items()}
       # Get the embeddings from the model
       with torch.no_grad():
           outputs = self.model(**inputs)
           # Use the CLS token's representation (or another pooling strategy)
           embeddings = outputs.last_hidden_state[:, 0, :]
       return embeddings.squeeze().cpu().numpy()
   def embed_documents(self, texts):
       return [self.embed_text(text) for text in texts]
   def __call__(self, text):
       # Make the class callable for single queries
       return self.embed_text(text)
   
   
# Load your PyTorch model and tokenizer from Hugging Face
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device="cpu"
custom_embeddings = CustomEmbeddings(model, tokenizer, device)


faiss_index_1 = FAISS.from_texts(chunks, embeddings)
faiss_index_2 = FAISS.from_texts(chunks, custom_embeddings)

# Step 3: Define Query and Retrieve Relevant Chunks
user_query = "What is efficient transformers"
retrieved_chunks_2 = faiss_index_2.similarity_search(user_query, k=10)
retrieved_chunks_1 = faiss_index_1.similarity_search(user_query, k=10)

# Step 4: Load LLM and Generate Response
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
import ipdb; ipdb.set_trace()

context_1 = "\n".join([chunk.page_content for chunk in retrieved_chunks_1])
context_2 = "\n".join([chunk.page_content for chunk in retrieved_chunks_2])
input_text_1 = f"Context:\n{context_1}\n\nQuestion: {user_query}\nAnswer:"
input_text_2 = f"Context:\n{context_2}\n\nQuestion: {user_query}\nAnswer:"

inputs_1 = tokenizer.encode(input_text_1, return_tensors="pt")
inputs_2 = tokenizer.encode(input_text_2, return_tensors="pt")

output_1 = model.generate(inputs_1, max_length=600, temperature=0.1,do_sample=True)
output_2 = model.generate(inputs_2, max_length=600, temperature=0.1,do_sample=True)
# Step 5: Print the Response
response = tokenizer.decode(output_1[0], skip_special_tokens=True)
response = tokenizer.decode(output_2[0], skip_special_tokens=True)
print("Generated Response:")
print(response)