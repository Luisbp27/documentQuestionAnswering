from langchain.vectorstores import Chroma
import chromadb
import embeddings


class VectorDB:
    def __init__(self, chunks):
        try:
            self.client = chromadb.Client()
        except Exception as e:
            print(f"Error connecting to database: {e}")
            exit(1)

        if self.client.list_collections():
            self.collection = self.client.create_collection("vectors")
        else:
            print("Collection already exists.")

        self.vectordb = Chroma.from_documents(
            documents = chunks,
            embedding = embeddings.get_embeddings(chunks),
            persist_directory = "./databases"
        )

        self.vectordb.persist()

    def get_db(self):
        return self.vectordb

    def insert_chunks(self, chunks):
        pass

    def get_all_vectors(self):
        return self.collection.get_all()

    def commit(self):
        self.client.commit()

    def end_connexion(self):
        self.client.close()