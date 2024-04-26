from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
import os

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

def load_files():
    pdf_files = []
    for file_ in os.listdir(DATA_PATH):
        if file_.endswith(".pdf"):
            pdf_files.append(file_)

    files = []
    for file_ in pdf_files:
        loader = PyPDFLoader(file_path=os.path.join(DATA_PATH, file_))
        pages = loader.load()
        files.append(pages)

    combined_pages = []
    for file_ in files:
        file_content = []
        for page in file_:
            file_content.append(page.page_content)

        combined_pages.append(str(file_content))

    return combined_pages

def load_files_chunked():
    pdf_folder_path = "./data"
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)

    return chunked_documents