import numpy as np
from langchain.embeddings import LlamaCppEmbeddings

llama = LlamaCppEmbeddings(model_path="./llama-2-7b.Q4_K_M.gguf")

text = "This is a test document."
query_result = llama.embed_query(text)
doc_result = llama.embed_documents([text])

print(np.array(query_result).shape)
print(np.array(doc_result).shape)
