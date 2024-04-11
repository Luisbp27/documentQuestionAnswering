import chromadb

class VectorDB:
    def __init__(self):
        try:
            self.db = chromadb.PersistentClient(path="./databases")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            exit(1)

        print("Collections in database:")
        print(self.db.list_collections())
        print(len(self.db.list_collections()))

        if "vectors" not in self.db.list_collections():
            self.collection = self.db.create_collection("vectors")
        else:
            self.collection = self.db.get_collection("vectors")

    def insert_vector(self, vector_id, vector):
        # Ensure vector is a list
        self.collection.add(vector_id, vector)

    def get_vector(self, vector_id):
        return self.collection.query(
            query_texts=[vector_id],
            n_results=1,
        )

    def delete_vector(self, vector_id):
        self.collection.delete(vector_id)

    def search_vectors(self, query_vector, k=5):
        return self.collection.search(query_vector, k)

    def get_all_vectors(self):
        return self.collection.get_all()

    def commit(self):
        self.db.commit()

    def end_connexion(self):
        self.db.close()