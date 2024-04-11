from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_community.document_loaders import PyPDFDirectoryLoader

DATA_PATH = "./data"

def load_data():
    pdf_files = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            pdf_files.append(file)

    loader = PyPDFLoader(file_path=DATA_PATH + "/" + pdf_files[0])

    chunks = split_text(loader.load())

    return chunks

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Extraer solo el texto de cada fragmento
    text_chunks = [chunk.page_content for chunk in chunks]

    # Si quieres imprimir algunos fragmentos para verificar
    for i, text in enumerate(text_chunks[:5]):  # Imprime los primeros 5 para verificación
        print(f"--- Chunk {i+1} ---")
        print(text)
        print("-----------------\n")

    return text_chunks
