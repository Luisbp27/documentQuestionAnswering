from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_function(model_name):

    if model_name == "multilingual_large":
        embedding_function = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    elif model_name == "baai_large":
        embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
    elif model_name == "mxbai_large":
        embedding_function = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
    elif model_name == "baai_small":
        embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    return embedding_function
