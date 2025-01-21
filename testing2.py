# Step 1: Fetch README and Prepare Chunks
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

pdfLoader = PyPDFLoader("The-AI-Act.pdf")
documents = pdfLoader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)# Step 2: Create Embeddings and Save FAISS Index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Define a custom embedding class
class CustomEmbeddings:
   def __init__(self, model, tokenizer, device="cpu"):
       self.model = model.to(device)
       self.tokenizer = tokenizer
       self.device = device
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

import ipdb

ipdb.set_trace()
faiss_index_1 = FAISS.from_documents(docs, embeddings)
# faiss_index_2 = FAISS.from_documents(docs, custom_embeddings)

# Step 3: Define Query and Retrieve Relevant Chunks
user_query = "What is AI-Act"
retrieved_chunks = faiss_index_1.similarity_search(user_query, k=10)
# Step 4: Load LLM and Generate Response
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
context = "\n".join([chunk.page_content for chunk in retrieved_chunks])
input_text = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
inputs = tokenizer(input_text, return_tensors="pt")
import ipdb; ipdb.set_trace()
output = model.generate(inputs, max_length=3000, temperature=0.1,do_sample=True)
# Step 5: Print the Response
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Response:")
print(response)