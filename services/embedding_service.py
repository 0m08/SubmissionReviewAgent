from langchain_cohere import CohereEmbeddings

def get_embedding_model():
    """
    Get the embedding model.
    :return: The embedding model.
    """
    cohere_embeddings = CohereEmbeddings(model="embed-english-v3.0")
    return cohere_embeddings
