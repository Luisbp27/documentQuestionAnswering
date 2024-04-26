from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
import chromadb
import os

def load_chunk_persist_pdf(model_name) -> Chroma:
    # Data loading
    pdf_folder_path = "./data"
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    print(f"Loaded {len(documents)} documents")

    # Chunking
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Chunked {len(chunked_documents)} documents")

    # Vectorization
    client = chromadb.Client()
    if client.list_collections():
        consent_collection = client.create_collection("consent_collection")
    else:
        print("Collection already exists")
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=HuggingFaceEmbeddings(model_name = model_name),
        persist_directory="./databases"
    )
    vectordb.persist()

    return vectordb

def create_agent_chain(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    chain = load_qa_chain(llm=model, chain_type="stuff")
    return chain


def get_llm_response(query, emb_model, llm_model):
    vectordb = load_chunk_persist_pdf(emb_model)
    chain = create_agent_chain(llm_model)
    matching_docs = vectordb.similarity_search(query)
    print(f"Found {len(matching_docs)} matching documents")
    answer = chain.run(input_documents=matching_docs, question=query)
    return answer


# Streamlit UI
# ===============
st.set_page_config(
    page_title="Document Question Answering - UIB",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'About': "# This is a header. This is an *extremely* cool app!"
    })
st.header("Query PDF Source")

# Embedding selection model
emb_model_options = ["BAAI/bge-small-en-v1.5"]
emb_model = st.selectbox("Select Embedding Model", emb_model_options)

# LLM selection model
llm_model_options = ["impira/layoutlm-document-qa", "naver-clova-ix/donut-base-finetuned-docvqa"]
llm_model = st.selectbox("Select LLM Model", llm_model_options)

form_input = st.text_input('Enter Query')
submit = st.button("Generate")

if submit:
    st.write(get_llm_response(form_input, emb_model, llm_model))