from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks: list[Document]):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def get_embedding_function(model_name):

    if model_name == "google":
        embedding_function = HuggingFaceEmbeddings(model_name="google/canine-c")
    elif model_name == "baai_large":
        embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
    elif model_name == "mxbai":
        embedding_function = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
    elif model_name == "baai_small":
        embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    return embedding_function