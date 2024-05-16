from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings


def get_embedding_function(model_name):

    if model_name == "gpt4":
        embedding_function = GPT4AllEmbeddings()
    elif model_name == "ollama":
        embedding_function = OllamaEmbeddings(model="nomic-embed-text")

    return embedding_function
