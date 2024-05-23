from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings


def get_embedding_function(model_name):

    if model_name == "gpt4":
        embedding_function = GPT4AllEmbeddings()
    elif model_name == "ollama_nomic":
        embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    elif model_name == "ollama_mxbai":
        embedding_function = OllamaEmbeddings(model="mxbai-embed-large")
    elif model_name == "hf_baai":
        embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    return embedding_function
