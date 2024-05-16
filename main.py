import streamlit as st
from langchain.vectorstores.chroma import Chroma
from embeddings import get_embedding_function
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def get_llm_response(query, emb_model, llm_model):
    # Prepare the DB.
    embedding_function = get_embedding_function(emb_model)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    model = Ollama(model=llm_model)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"

    return formatted_response

def main():
    # Streamlit UI
    # ===============
    st.set_page_config(
        page_title="Document Question Answering - UIB",
        page_icon="🔍",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items={"About": "# This is a header. This is an *extremely* cool app!"},
    )
    st.header("Query PDF Source")

    # Upload files in PDF format
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    upload = st.button("Upload")

    if upload:
        # Write the file to /data directory
        with open(f"data/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Add the file to the database


    # Embedding selection model
    emb_model_options = [
        "gpt4",
        "ollama"
    ]
    emb_model = st.selectbox("Select Embedding Model", emb_model_options)

    # LLM selection model
    llm_model_options = [
        "llama3",
        "llama2",
        "mistral",
        "phi3"
    ]
    llm_model = st.selectbox("Select LLM Model", llm_model_options)

    form_input = st.text_input("Enter Query")
    submit = st.button("Generate")

    if submit:
        st.write(get_llm_response(form_input, emb_model, llm_model))



if __name__ == "__main__":
    main()
