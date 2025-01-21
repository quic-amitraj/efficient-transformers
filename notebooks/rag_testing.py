
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

pdfLoader = PyPDFLoader("paper.pdf")
documents = pdfLoader.load()

import ipdb; ipdb.set_trace()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

modelPath = "thenlper/gte-small"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings':False}

import ipdb; ipdb.set_trace()
embeddings = HuggingFaceEmbeddings(
  model_name = modelPath,  
  model_kwargs = model_kwargs,
  encode_kwargs=encode_kwargs
)
import ipdb; ipdb.set_trace()
db = FAISS.from_documents(docs, embeddings)
question = "How many weights llm can contain?"
import ipdb; ipdb.set_trace()
searchDocs = db.similarity_search(question)
import ipdb; ipdb.set_trace()
print(searchDocs[0].page_content)