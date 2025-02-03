from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
  """
  Get the embedding model.
  Returns:
    HuggingFaceEmbeddings: The embedding model.
  """
  model_name = "BAAI/bge-large-en-v1.5"
  model_kwargs = {"device": "cpu"}
  encode_kwargs = {"normalize_embeddings": True}
  bge_large = HuggingFaceEmbeddings(
      model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
  )
  return bge_large

#bge_large = get_embedding_model()

#test_embed = bge_large.embed_query("This is a test query.")
#print(len(test_embed))