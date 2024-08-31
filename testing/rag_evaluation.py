from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from ragas.run_config import RunConfig
import argparse
import tqdm
import json
from module import *

def data_loading(num_test, db):
    # Load the documents
    documents_loader = PyPDFDirectoryLoader("./test{test}/data")
    documents = documents_loader.load()
    print(f"Loaded {len(documents)} documents of test{num_test}")

    # Split the documents into chunks
    chunks = split_documents(documents, chunk_size=512)
    print("Number of chunks: ", len(chunks))

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        with tqdm.tqdm(total=len(new_chunks)) as pbar:
            for chunk in new_chunks:
                db.add_documents([chunk], ids=[chunk.metadata["id"]])
                pbar.update(1)
        print("Documents added correctly ✅")
    else:
        print("✅ No new documents to add")

    print(f"{len(documents)} documents added to the database correctly")

def create_dataset(db):
    # Load the questions
    with open("questions.json", "r", encoding="utf-8") as f:
        questions_json = json.load(f)

    questions = questions_json["test1"]["cat"] + questions_json["test2"]["en"] + questions_json["test3"]["es"]

    # Get the contexts for each question
    contexts = []
    for question in questions:
        # Get the top 5 most relevant documents
        results = db.similarity_search_with_score(question, k=3)

        # Make a list of the contexts
        question_contexts = []
        for doc, _score in results:
            question_contexts.append(doc.page_content)

        # Append the context sub-list to the list of contexts
        contexts.append(question_contexts)

    # Load the answers
    with open("answers.json", "r", encoding="utf-8") as f:
        answers_json = json.load(f)

    answers = answers_json["test1"]["cat"] + answers_json["test2"]["en"] + answers_json["test3"]["es"]

    data_samples = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": answers
    }

    return Dataset.from_dict(data_samples)

def main(emb_model, test_num=None):
    llm_models = [
        "llama3",
        "mistral",
        "phi3",
        "gemma"
    ]

    tests_to_run = range(1, 4) if test_num is None else [test_num]

    for test in tests_to_run:
        # Load the existing database.
        db = Chroma(
            collection_name=f"test{test}",
            persist_directory="../database",
            embedding_function=get_embedding_function(emb_model)
        )

        # Load the data
        data_loading(test, db)

        # Create the dataset
        dataset = create_dataset(db)

        # Evaluate the dataset
        embeddings = get_embedding_function(emb_model)
        llm = Ollama(model=llm_models[test - 1])

        try:
            score = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=llm,
                embeddings=embeddings,
                raise_exceptions=False,
                run_config=RunConfig(
                    max_retries=30, # Default is 10
                    max_wait=180, # Default is 60
                    #max_workers=64 # Default is 16
                )
            )
        except Exception as e:
            print(f"An error ocurred: {e}")

        df_score = score.to_pandas()
        df_score.to_csv(f"./test{test}/llm/test3_{llm_models[test - 1]}_{emb_model}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("emb_model", type=str, help="The name of the embedding model to use.")
    parser.add_argument("--test", type=int, choices=range(1, 4), help="The test number to run (1, 2, or 3). If not provided, all tests will be run.")

    args = parser.parse_args()

    main(args.emb_model, args.test)