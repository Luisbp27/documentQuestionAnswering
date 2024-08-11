from langchain_huggingface import HuggingFaceEmbeddings
import torch


def get_embedding_function(model_name):

    # Make use of a GPU or MPS (Apple) if one is available.
    has_mps = torch.backends.mps.is_built()
    device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_kwargs = {
        "device": device
    }

    if model_name == "multilingual_large":
        embedding_function = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs=model_kwargs
        )
    elif model_name == "baai_large":
        embedding_function = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs=model_kwargs)
    elif model_name == "mxbai_large":
        embedding_function = HuggingFaceEmbeddings(
            model_name="mixedbread-ai/mxbai-embed-large-v1",
            model_kwargs=model_kwargs
        )
    elif model_name == "baai_small":
        embedding_function = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs=model_kwargs
        )

    return embedding_function
