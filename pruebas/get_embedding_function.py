from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings


def get_embedding_function():
    embeddings = GPT4AllEmbeddings()
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
