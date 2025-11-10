from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen-embedding-0.6B")

db = Chroma(
    persist_directory="./chroma_store",
    embedding_function=embeddings,
    collection_name="aetherflow_memories"
)