import argparse
import json
import shutil
from pathlib import Path

from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

from .db import qdrant, COLLECTION


DATA_FOLDER = Path("data")
PROCESSED_FILE = Path("processed_files.json")


def make_project_filter(project_name: str) -> Filter:
    """
    Create a Qdrant filter for one project.
    Only points with this project_name will be selected.
    """
    return Filter(
        must=[
            FieldCondition(
                key="project_name",
                match=MatchValue(value=project_name)
            )
        ]
    )


def delete_project_vectors(project_name: str):
    """
    Delete only the vectors belonging to one project from Qdrant.
    Other projects remain untouched.
    """
    qdrant.delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(
            filter=make_project_filter(project_name)
        )
    )

    print(f"Deleted vectors for project: {project_name}")


def remove_project_from_processed(project_name: str):
    """
    Remove the project record from processed_files.json.
    """
    if not PROCESSED_FILE.exists():
        print("processed_files.json not found. Nothing to remove.")
        return

    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        processed = json.load(f)

    if project_name in processed:
        del processed[project_name]

        with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

        print(f"Removed project from processed_files.json: {project_name}")
    else:
        print(f"No processed record found for project: {project_name}")


def delete_project_files(project_name: str):
    """
    Delete the project folder under data/.
    """
    project_folder = DATA_FOLDER / project_name

    if project_folder.exists():
        shutil.rmtree(project_folder)
        print(f"Deleted project folder: {project_folder}")
    else:
        print(f"Project folder not found: {project_folder}")


def clear_project(project_name: str, delete_files: bool = False):
    """
    Clear one project:
    1. Delete project vectors from Qdrant
    2. Remove project record from processed_files.json
    3. Optionally delete the project folder
    """
    print("\n" + "=" * 80)
    print(f"Clearing project: {project_name}")
    print("=" * 80)

    delete_project_vectors(project_name)
    remove_project_from_processed(project_name)

    if delete_files:
        delete_project_files(project_name)

    print("=" * 80)
    print(f"Project cleared: {project_name}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        required=True,
        help="Project name to clear"
    )
    parser.add_argument(
        "--delete-files",
        action="store_true",
        help="Also delete files under data/project_name/"
    )

    args = parser.parse_args()

    try:
        clear_project(
            project_name=args.project,
            delete_files=args.delete_files
        )
    finally:
        qdrant.close()
        print("Local Qdrant client closed.")