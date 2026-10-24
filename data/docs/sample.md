# What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation, or RAG, is a technique that lets a large language model answer questions using information from your own documents instead of relying only on what it learned during training.

## How it works

1. Documents are split into small overlapping pieces called chunks.
2. Each chunk is converted into a numeric vector called an embedding.
3. The embeddings are stored in a vector database such as ChromaDB.
4. When a user asks a question, the question is also embedded and the most similar chunks are retrieved.
5. The retrieved chunks are sent to the language model together with the question, and the model writes an answer grounded in that context.

## Why use RAG?

RAG reduces hallucinations because the model is told to answer only from the supplied context. It also keeps answers up to date without retraining the model: you only need to add or replace documents. Finally, it allows answers to cite their sources, which makes them easier to verify.
