import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_local")
COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_docs")

qdrant = QdrantClient(path=QDRANT_PATH)


def ensure_collection(vector_size: int = 384):
    """
    Ensure that the local Qdrant collection exists.
    Default vector size is 384 for intfloat/multilingual-e5-small.
    """
    collections = qdrant.get_collections().collections
    names = [c.name for c in collections]

    if COLLECTION not in names:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        print(f"Local collection '{COLLECTION}' created.")
    else:
        print(f"Local collection '{COLLECTION}' already exists.")


def clear_collection(vector_size: int = 384):
    """
    Clear the whole local Qdrant collection.
    Later we will add project-level deletion.
    """
    collections = qdrant.get_collections().collections
    names = [c.name for c in collections]

    if COLLECTION in names:
        qdrant.delete_collection(collection_name=COLLECTION)
        print(f"Local collection '{COLLECTION}' deleted.")

    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

    print(f"Local collection '{COLLECTION}' recreated with vector size {vector_size}.")