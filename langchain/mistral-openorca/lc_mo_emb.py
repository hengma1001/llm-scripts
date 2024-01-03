import numpy as np
from langchain.embeddings import LlamaCppEmbeddings

llama = LlamaCppEmbeddings(model_path="mistral-7b-openorca.Q4_K_M.gguf")

text = "This is"
query_result = llama.embed_query(text)
doc_result = llama.embed_documents([text])

print(np.array(query_result))
print(np.array(doc_result))
print(np.array(query_result).shape)
print(np.array(doc_result).shape)


text = "This"
query_result = llama.embed_query(text)
print(np.array(query_result))

text = "is"
query_result = llama.embed_query(text)
print(np.array(query_result))
