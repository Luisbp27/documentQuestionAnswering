import import_data
import embeddings
import vector_db

def main():
    data_chunks = import_data.load_files_chunked()

    # Create embeddings
    #embeddings_ = embeddings.get_embeddings(data)

    print(len(data_chunks))
    # Create vector database
    db = vector_db.VectorDB(data_chunks)

    vectors = db.get_all_vectors()
    print(f"Number of vectors: {len(vectors)}")

    db.commit()
    db.end_connexion()

if __name__ == "__main__":
    main()


