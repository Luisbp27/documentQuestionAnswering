import streamlit as st
import os
from import_data import populate_database, clear_database
from langchain_community.vectorstores import Chroma
from embeddings import get_embedding_function
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

CHROMA_PATH = "database"

PROMPT_TEMPLATE_en = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

PROMPT_TEMPLATE_es = """
Contesta a la pregunta basando tu respuesta en el siguiente contexto:

{context}

---

Contesta la siguiente pregunta, basándote en el contexto anterior: {question}
"""

PROMPT_TEMPLATE_cat = """
Respon a la pregunta basant la teva resposta en el següent context:

{context}

---

Respon a la següent pregunta, basant-te en el context anterior: {question}
"""

def get_prompt(lang):
    if lang == "en":
        return PROMPT_TEMPLATE_en
    elif lang == "es":
        return PROMPT_TEMPLATE_es
    elif lang == "cat":
        return PROMPT_TEMPLATE_cat
    else:
        return PROMPT_TEMPLATE_en

def get_llm_response(query, db, llm_model, lang="en"):

    # Search the DB.
    results = db.similarity_search_with_score(query, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(get_prompt(lang))
    prompt = prompt_template.format(context=context_text, question=query)

    model = Ollama(model=llm_model)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"

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

    st.markdown(
        """
        This is a simple app that allows you to query a PDF document using differents LLMs and Embedding models. This is a prototype for a final thesis, so all uses are focused on educational purposes.
        """
    )

    st.sidebar.markdown(
        """
        ## Author: ## Lluis Barca Pons
        ## Tutors: ## Dr. Isaac Lera, Dr. Antoni-Jaume Capó
        """
    )

    st.sidebar.markdown(
        """
        [Linkedin](https://www.linkedin.com/in/luisbarcapons/)
        [Github](https://www.github.com/Luisbp27)
        """
    )

    # Upload files in PDF format
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    upload = st.button("Upload")

    # Embedding selection model
    emb_model_options = [
        "gpt4",
        "baai_large",
        "mxbai",
        "baai_small"
    ]
    emb_model = st.selectbox("Select Embedding Model", emb_model_options)

    # LLM selection model
    llm_model_options = [
        "llama3",
        "mistral",
        "phi3",
        "gemma"
    ]
    llm_model = st.selectbox("Select LLM Model", llm_model_options)

    lang = st.selectbox("Select Language", ["en", "es", "cat"])

    form_input = st.text_input("Enter Query")
    submit = st.button("Generate")

    # Prepare the DB.
    embedding_function = get_embedding_function(emb_model)

    # If the directory of DB doesn't exist, create it.
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Clear the database
    erase_db = st.button("Erase DB")

    if erase_db:
        clear_database()

    if upload:
        # Write the file to /data directory
        with open(f"data/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Add the file to the database
        populate_database(emb_model)

    if submit:
        st.write(get_llm_response(form_input, db, llm_model, lang))

if __name__ == "__main__":
    main()
