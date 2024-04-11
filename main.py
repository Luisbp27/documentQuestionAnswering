import import_data
import embeddings
import vector_db

def main():
    data = import_data.load_data()

    # Create embeddings
    embeddings_ = embeddings.get_embeddings(data)

    print(f"Embeddings shape: {embeddings_.shape}")

    # Create vector database
    db = vector_db.VectorDB()
    for i, embedding in enumerate(embeddings_):
        db.insert_vector(i, embedding)

    vectors = db.get_all_vectors()
    print(f"Number of vectors: {len(vectors)}")

    db.commit()
    db.end_connexion()

if __name__ == "__main__":
    main()


