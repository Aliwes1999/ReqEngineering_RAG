from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small"

model = SentenceTransformer(MODEL_NAME)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Convert document chunks into embeddings using a local embedding model.
    """
    prefixed_texts = [f"passage: {text}" for text in texts]

    embeddings = model.encode(
        prefixed_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings.tolist()


def embed_query(question: str) -> list[float]:
    """
    Convert a user question into an embedding using the same local model.
    """
    text = f"query: {question}"

    embedding = model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding.tolist()