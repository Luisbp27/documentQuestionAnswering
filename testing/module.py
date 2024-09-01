from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd

def split_documents(documents: list[Document], chunk_size=1024, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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

    model_kwargs = {
        "device": "cpu",
    }

    if model_name == "multilingual_large":
        embedding_function = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs = model_kwargs
        )
    elif model_name == "baai_large":
        embedding_function = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs = model_kwargs
        )
    elif model_name == "mxbai_large":
        embedding_function = HuggingFaceEmbeddings(
            model_name="mixedbread-ai/mxbai-embed-large-v1",
            model_kwargs = model_kwargs
        )
    elif model_name == "baai_small":
        embedding_function = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs = model_kwargs
        )

    return embedding_function

def get_df(df_cat_mean, df_en_mean, df_es_mean):
    df_means = pd.DataFrame({
        "Idioma": ["Catalán", "Inglés", "Español"],
        "Faithfulness": [df_cat_mean["faithfulness"], df_en_mean["faithfulness"], df_es_mean["faithfulness"]],
        "Answer Relevancy": [df_cat_mean["answer_relevancy"], df_en_mean["answer_relevancy"], df_es_mean["answer_relevancy"]]
    }).set_index("Idioma").T

    return df_means

# Función para añadir valores en las barras
def add_values_labels(ax):
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')