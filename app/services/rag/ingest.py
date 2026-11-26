import argparse
import uuid
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime

from qdrant_client.models import PointStruct

from .utils import read_file, chunk_text
from .db import qdrant, ensure_collection, COLLECTION
from .embedding_local import embed_documents, MODEL_NAME


DATA_FOLDER = Path("data")
PROCESSED_FILE = Path("processed_files.json")
INGEST_LOG_FILE = Path("ingest_log.jsonl")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx"}

VECTOR_SIZE = 384


def get_file_hash(path: Path) -> str:
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def load_processed_files() -> dict:
    if not PROCESSED_FILE.exists():
        return {}

    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_processed_files(processed: dict):
    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)


def write_ingest_log(log_item: dict):
    with open(INGEST_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_item, ensure_ascii=False) + "\n")


def get_project_folder(project_name: str) -> Path:
    """
    Return the folder path for a project.
    Example:
    project_name = "rag_test"
    folder = data/rag_test
    """
    return DATA_FOLDER / project_name


def create_project_folder(project_name: str):
    """
    Create a new project folder under data/.
    """
    project_folder = get_project_folder(project_name)
    project_folder.mkdir(parents=True, exist_ok=True)
    print(f"Project folder is ready: {project_folder}")


def get_supported_files(project_folder: Path):
    """
    Get all supported files in the selected project folder.
    This also searches inside subfolders.
    """
    files = []

    for file in project_folder.rglob("*"):
        if not file.is_file():
            continue

        if file.name.startswith("."):
            continue

        if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        files.append(file)

    return files


def should_skip_file(project_name: str, file_path: Path, processed: dict) -> bool:
    file_hash = get_file_hash(file_path)
    file_key = str(file_path)

    project_records = processed.get(project_name, {})
    record = project_records.get(file_key)

    if not record:
        return False

    same_hash = record.get("file_hash") == file_hash
    same_model = record.get("embedding_model") == MODEL_NAME
    same_collection = record.get("collection") == COLLECTION

    return same_hash and same_model and same_collection


def ingest_file(project_name: str, file_path: Path, processed: dict):
    file_start_time = time.perf_counter()
    file_hash = get_file_hash(file_path)

    file_key = str(file_path)

    log_item = {
        "timestamp": datetime.now().isoformat(),
        "project_name": project_name,
        "file_name": file_path.name,
        "file_path": str(file_path),
        "file_hash": file_hash,
        "status": "started",
        "chunk_count": 0,
        "read_time_seconds": None,
        "chunk_time_seconds": None,
        "embedding_time_seconds": None,
        "qdrant_upsert_time_seconds": None,
        "total_time_seconds": None,
        "embedding_model": MODEL_NAME,
        "collection": COLLECTION,
        "error": None,
    }

    print("\n" + "=" * 80)
    print(f"Project: {project_name}")
    print(f"Processing file: {file_path}")
    print("=" * 80)

    try:
        # 1. Read file
        read_start = time.perf_counter()
        text = read_file(str(file_path))
        read_end = time.perf_counter()

        log_item["read_time_seconds"] = round(read_end - read_start, 3)

        print(f"Extracted text length: {len(text)} characters")
        print(f"Read time: {log_item['read_time_seconds']} seconds")

        if not text.strip():
            raise ValueError("No text extracted from file.")

        # 2. Chunk text
        chunk_start = time.perf_counter()
        chunks = chunk_text(text)
        chunk_end = time.perf_counter()

        log_item["chunk_time_seconds"] = round(chunk_end - chunk_start, 3)
        log_item["chunk_count"] = len(chunks)

        print(f"Total chunks: {len(chunks)}")
        print(f"Chunk time: {log_item['chunk_time_seconds']} seconds")

        if not chunks:
            raise ValueError("No chunks generated.")

        # 3. Local embedding
        print("-" * 80)
        print(f"Generating local embeddings with model: {MODEL_NAME}")
        print("-" * 80)

        embedding_start = time.perf_counter()
        vectors = embed_documents(chunks)
        embedding_end = time.perf_counter()

        log_item["embedding_time_seconds"] = round(
            embedding_end - embedding_start, 3
        )

        if not vectors:
            raise ValueError("No vectors generated.")

        vector_size = len(vectors[0])

        print(f"Vector dimension: {vector_size}")
        print(f"Embedding time: {log_item['embedding_time_seconds']} seconds")

        # 4. Ensure local Qdrant collection
        print(f"Checking local Qdrant collection: {COLLECTION}")
        ensure_collection(vector_size)

        # 5. Build Qdrant points
        points = []

        for i, chunk in enumerate(chunks):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i],
                payload={
                    "project_name": project_name,
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "file_hash": file_hash,
                    "file_id": file_path.stem,
                    "chunk_id": i,
                    "content": chunk,
                    "embedding_model": MODEL_NAME,
                    "created_at": datetime.now().isoformat()
                }
            )

            points.append(point)

        # 6. Upsert to local Qdrant
        print("-" * 80)
        print(
            f"Uploading {len(points)} points from file '{file_path.name}' "
            f"to local Qdrant collection '{COLLECTION}'..."
        )
        print("-" * 80)

        upsert_start = time.perf_counter()
        qdrant.upsert(
            collection_name=COLLECTION,
            points=points
        )
        upsert_end = time.perf_counter()

        log_item["qdrant_upsert_time_seconds"] = round(
            upsert_end - upsert_start, 3
        )

        print(f"Qdrant upsert time: {log_item['qdrant_upsert_time_seconds']} seconds")

        # 7. Save processed record by project
        if project_name not in processed:
            processed[project_name] = {}

        processed[project_name][file_key] = {
            "project_name": project_name,
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_hash": file_hash,
            "processed_at": datetime.now().isoformat(),
            "chunk_count": len(chunks),
            "collection": COLLECTION,
            "embedding_model": MODEL_NAME,
            "vector_size": vector_size,
        }

        save_processed_files(processed)

        file_end_time = time.perf_counter()
        log_item["total_time_seconds"] = round(file_end_time - file_start_time, 3)
        log_item["status"] = "success"

        print(f"Ingested successfully: {file_path}")
        print(f"Project name: {project_name}")
        print(f"Total uploaded points: {len(points)}")
        print(f"Total time: {log_item['total_time_seconds']} seconds")

    except Exception as e:
        file_end_time = time.perf_counter()

        log_item["total_time_seconds"] = round(file_end_time - file_start_time, 3)
        log_item["status"] = "failed"
        log_item["error"] = str(e)

        print("\n" + "!" * 80)
        print(f"Error processing {file_path.name}: {e}")
        print("!" * 80)

    finally:
        write_ingest_log(log_item)


def ingest_project(project_name: str, create_if_missing: bool = False):
    """
    Ingest only one selected project folder.
    """
    if not DATA_FOLDER.exists():
        DATA_FOLDER.mkdir(parents=True, exist_ok=True)
        print(f"Created data folder: {DATA_FOLDER}")

    project_folder = get_project_folder(project_name)

    if not project_folder.exists():
        if create_if_missing:
            create_project_folder(project_name)
            print("Project folder created. Put files into it and run ingest again.")
            return
        else:
            print(f"Project folder does not exist: {project_folder}")
            print("Create it manually or run:")
            print(f"python ingest.py --project {project_name} --create")
            return

    processed = load_processed_files()
    files = get_supported_files(project_folder)

    if not files:
        print(f"No supported files found in project folder: {project_folder}")
        print("Supported file types: .pdf, .docx, .xlsx")
        return

    print(f"Selected project: {project_name}")
    print(f"Project folder: {project_folder}")
    print(f"Supported files found: {len(files)}")

    for file in files:
        if should_skip_file(project_name, file, processed):
            print(f"Skipping already processed file: {file.name}")
            continue

        ingest_file(project_name, file, processed)

    print("\n" + "=" * 80)
    print(f"Finished ingesting project: {project_name}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        required=True,
        help="Name of the project folder under data/"
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create the project folder if it does not exist"
    )

    args = parser.parse_args()

    try:
        ingest_project(
            project_name=args.project,
            create_if_missing=args.create
        )
    finally:
        qdrant.close()