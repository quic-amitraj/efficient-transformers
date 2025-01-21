from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_community.vectorstores import FAISS

from QEfficient.utils.embedding import QEffEmbedding

# Specify the dataset name and the column containing the content
dataset_name = "databricks/databricks-dolly-15k"
page_content_column = "context"  # or any other column you're interested in

# Create a loader instance
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)

# Load the data
data = loader.load()

# Display the first 15 entries
data[:2]# Specify the dataset name and the column containing the content

# Create a loader instanc

# Display the first 15 entries
data[:2]

# Create an instance of the RecursiveCharacterTextSplitter class with specific parameters.
# It splits text into chunks of 1000 characters each with a 150-character overlap.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
# 'data' holds the text you want to split, split the text into documents using the text splitter.
docs = text_splitter.split_documents(data)

print(docs[0])

# Define the path to the pre-trained model you want to use
modelPath = "sentence-transformers/all-MiniLM-l6-v2"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}


text = "This is a test document."

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
# embeddings_1 = HuggingFaceEmbeddings(
#     model_name=modelPath,     # Provide the pre-trained model's path
#     model_kwargs=model_kwargs, # Pass the model configuration options
#     encode_kwargs=encode_kwargs # Pass the encoding options
# )

embeddings_2 = QEffEmbedding(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

# result2= embeddings_1.embed_documents([text])
result1= embeddings_2.embed_documents([text])


# # Load your PyTorch model and tokenizer from Hugging Face
# # query_result = embeddings.embed_query(text)
# custom_embeddings=QEFFEmbedding(
#     modelPath,     # Provide the pre-trained model's path
# )
# import ipdb; ipdb.set_trace()
# result= custom_embeddings.embed_text([text])


# import ipdb; ipdb.set_trace()
db = FAISS.from_documents(docs[:10], embeddings_2)
# db = FAISS.from_documents(docs[:10], custom_embeddings)


# return self.embed_documents([text])[0]
# texts = list(map(lambda x: x.replace("\n", " "), texts))
# embeddings = self.client.encode(
#                 texts, show_progress_bar=self.show_progress, **self.encode_kwargs

#             )
