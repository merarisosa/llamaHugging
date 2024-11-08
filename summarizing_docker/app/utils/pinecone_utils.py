import pinecone

def upsert_to_pinecone(pdf_filename: str, embedding: list):
    """ Subir embeddings generados a Pinecone """
    vector = (pdf_filename, embedding)
    pinecone.upsert(vectors=[vector])
