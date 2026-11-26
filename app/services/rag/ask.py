import os
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from config import OPENAI_MODEL
from qdrant_client.models import Filter, FieldCondition, MatchValue

from .db import qdrant, COLLECTION
from .embedding_local import embed_query, MODEL_NAME

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

LOG_ROOT = Path("logs")


def safe_project_name(project_name: str) -> str:
    """
    Make project name safe for folder paths.
    """
    return (
        project_name
        .replace("/", "_")
        .replace("\\", "_")
        .strip()
    )


def get_project_log_dir(project_name: str) -> Path:
    """
    Return logs/project_name/ and create it if needed.
    """
    project_log_dir = LOG_ROOT / safe_project_name(project_name)
    project_log_dir.mkdir(parents=True, exist_ok=True)
    return project_log_dir


def get_conversation_log_file(project_name: str) -> Path:
    return get_project_log_dir(project_name) / "conversation_log.jsonl"


def get_qa_table_log_file(project_name: str) -> Path:
    return get_project_log_dir(project_name) / "qa_log.csv"


def get_qa_chat_log_file(project_name: str) -> Path:
    return get_project_log_dir(project_name) / "qa_chat_log.html"


def html_escape(text: str) -> str:
    """
    Escape special HTML characters and keep line breaks.
    """
    if text is None:
        return ""

    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


def make_project_filter(project_name: str) -> Filter:
    """
    Create a Qdrant filter so that only chunks from the selected project
    are retrieved.
    """
    return Filter(
        must=[
            FieldCondition(
                key="project_name",
                match=MatchValue(value=project_name)
            )
        ]
    )


def retrieve(question: str, project_name: str, top_k: int = 5):
    """
    Use the local embedding model to embed the user question,
    then retrieve relevant chunks from the selected project only.
    """
    query_vector = embed_query(question)

    result = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        query_filter=make_project_filter(project_name),
        limit=top_k
    )

    return result.points


def build_context(points):
    """
    Build the context string from retrieved Qdrant points.
    """
    parts = []

    for i, p in enumerate(points, start=1):
        payload = p.payload or {}

        content = payload.get("content") or payload.get("text") or ""

        parts.append(
            f"[{i}] project={payload.get('project_name')} "
            f"source={payload.get('source')} "
            f"file_name={payload.get('file_name')} "
            f"chunk_id={payload.get('chunk_id')} "
            f"score={getattr(p, 'score', None)}\n"
            f"{content}"
        )

    return "\n\n".join(parts)


def save_conversation_log(
    project_name: str,
    question: str,
    answer: str,
    context: str,
    points
):
    """
    Save one Q&A interaction as one JSON line.
    This detailed log is useful for debugging and evaluation.
    """
    log_item = {
        "timestamp": datetime.now().isoformat(),
        "project_name": project_name,
        "question": question,
        "answer": answer,
        "context": context,
        "retrieved_chunks": [
            {
                "project_name": (p.payload or {}).get("project_name"),
                "source": (p.payload or {}).get("source"),
                "file_name": (p.payload or {}).get("file_name"),
                "file_id": (p.payload or {}).get("file_id"),
                "chunk_id": (p.payload or {}).get("chunk_id"),
                "score": getattr(p, "score", None),
                "content_preview": (
                    (p.payload or {}).get("content")
                    or (p.payload or {}).get("text")
                    or ""
                )[:500]
            }
            for p in points
        ],
        "qdrant_collection": COLLECTION,
        "embedding_model": MODEL_NAME,
        "generation_model": OPENAI_MODEL
    }

    log_file = get_conversation_log_file(project_name)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_item, ensure_ascii=False) + "\n")


def save_qa_table(project_name: str, question: str, answer: str):
    """
    Save project name, question and answer into a project-specific CSV table.
    """
    log_file = get_qa_table_log_file(project_name)
    file_exists = log_file.exists()

    with open(log_file, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["project_name", "question", "answer"])

        writer.writerow([project_name, question, answer])


def initialize_chat_html(project_name: str):
    """
    Create the project-specific HTML chat log file if it does not exist.
    """
    log_file = get_qa_chat_log_file(project_name)

    if log_file.exists():
        return

    project_title = html_escape(project_name)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>RAG Chat Log - {project_title}</title>
<style>
body {{
    font-family: Arial, sans-serif;
    background-color: #f5f5f5;
    margin: 0;
    padding: 30px;
}}

.container {{
    max-width: 900px;
    margin: 0 auto;
}}

.title {{
    text-align: center;
    margin-bottom: 10px;
    color: #333;
}}

.subtitle {{
    text-align: center;
    margin-bottom: 30px;
    color: #666;
    font-size: 14px;
}}

.message-block {{
    margin-bottom: 28px;
}}

.timestamp {{
    text-align: center;
    color: #888;
    font-size: 12px;
    margin-bottom: 10px;
}}

.project-name {{
    text-align: center;
    color: #555;
    font-size: 13px;
    margin-bottom: 10px;
}}

.user-message {{
    background-color: #d9ecff;
    color: #111;
    padding: 14px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
    margin-left: auto;
    margin-bottom: 10px;
    line-height: 1.5;
    white-space: normal;
}}

.assistant-message {{
    background-color: #ffffff;
    color: #111;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    max-width: 75%;
    margin-right: auto;
    border: 1px solid #ddd;
    line-height: 1.5;
    white-space: normal;
}}

.label {{
    font-weight: bold;
    font-size: 13px;
    margin-bottom: 6px;
    color: #555;
}}

.source-info {{
    color: #777;
    font-size: 12px;
    margin-top: 8px;
    border-top: 1px solid #eee;
    padding-top: 8px;
}}

.footer {{
    text-align: center;
    color: #999;
    font-size: 12px;
    margin-top: 40px;
}}
</style>
</head>
<body>
<div class="container">
<h1 class="title">RAG Chat Log</h1>
<div class="subtitle">Project: {project_title}</div>
""")


def save_qa_chat_html(project_name: str, question: str, answer: str, points):
    """
    Save question and answer into a project-specific HTML chat-style log.
    The file can be opened directly in a browser.
    """
    initialize_chat_html(project_name)

    log_file = get_qa_chat_log_file(project_name)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    project_html = html_escape(project_name)
    question_html = html_escape(question)
    answer_html = html_escape(answer)

    source_lines = []

    for p in points:
        payload = p.payload or {}
        source_lines.append(
            f"project={payload.get('project_name')} | "
            f"file={payload.get('file_name')} | "
            f"chunk={payload.get('chunk_id')} | "
            f"score={getattr(p, 'score', None)}"
        )

    source_html = html_escape("\n".join(source_lines))

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"""
<div class="message-block">
    <div class="timestamp">{timestamp}</div>
    <div class="project-name">Project: {project_html}</div>

    <div class="user-message">
        <div class="label">Question</div>
        {question_html}
    </div>

    <div class="assistant-message">
        <div class="label">Answer</div>
        {answer_html}
        <div class="source-info">
            <strong>Retrieved Chunks</strong><br>
            {source_html}
        </div>
    </div>
</div>
""")


def answer_question(project_name: str, question: str):
    points = retrieve(question, project_name, top_k=5)
    context = build_context(points)

    if not context.strip():
        answer = (
            f"Es wurden keine relevanten Auszüge im Projekt "
            f"'{project_name}' gefunden."
        )

        save_conversation_log(
            project_name=project_name,
            question=question,
            answer=answer,
            context=context,
            points=points
        )

        save_qa_table(project_name, question, answer)
        save_qa_chat_html(project_name, question, answer, points)

        return answer

    prompt = f"""Du bist ein Assistent für technische Dokumente, MBSE und SysML v2.

Bearbeite die Nutzeranfrage ausschließlich auf Basis der bereitgestellten Auszüge aus der Wissensbasis.
Erfinde keine fachlichen Inhalte, die nicht durch die Auszüge gestützt werden.

Regeln:
- Nutze nur die bereitgestellten Auszüge als fachliche Grundlage.
- Du darfst Inhalte aus mehreren Auszügen zusammenführen, strukturieren, verdichten und umformulieren.
- Wenn die Nutzeranfrage eine normale Wissensfrage ist, beantworte sie präzise und direkt.
- Wenn die Nutzeranfrage eine Generierungsaufgabe ist, z. B. Code, SysML v2 textual notation, Modellstruktur, RFLP-Struktur oder technischer Entwurf, dann überführe die Inhalte der Auszüge in eine vollständige, gut strukturierte Lösung.
- Füge keine unbelegten Anforderungen, Funktionen, logischen Elemente oder physischen Komponenten hinzu.
- Wenn Informationen für bestimmte Teile fehlen, nenne diese ausdrücklich unter „Fehlende Informationen“.
- Wenn der Nutzer „decke alles ab“ verlangt, decke alles ab, was in den Auszügen enthalten oder daraus direkt ableitbar ist.
- Antworte immer auf Deutsch.
- Wenn Code oder SysML v2 textual notation verlangt wird, gib einen klar strukturierten Codeblock aus.
- Wenn RFLP verlangt wird, strukturiere die Antwort nach Requirements, Functional, Logical und Physical.

Arbeitsweise:
1. Analysiere alle relevanten Auszüge.
2. Führe zusammengehörige Informationen zusammen.
3. Ordne sie passend zur Nutzeranfrage.
4. Erstelle daraus die gewünschte Antwort oder Generierung.
5. Nenne fehlende Informationen, falls notwendig.

【Aktuelles Projekt】
{project_name}

【Nutzeranfrage】
{question}

【Auszüge aus der Wissensbasis】
{context}

Erstelle jetzt die passende Antwort."""

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=2500,
    )

    answer = response.choices[0].message.content

    save_conversation_log(
        project_name=project_name,
        question=question,
        answer=answer,
        context=context,
        points=points
    )

    save_qa_table(project_name, question, answer)
    save_qa_chat_html(project_name, question, answer, points)

    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        required=True,
        help="Name of the project folder under data/"
    )

    args = parser.parse_args()
    project_name = args.project

    print(f"Selected project: {project_name}")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("Frage: ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        if not question:
            print("Bitte gib eine Frage ein.\n")
            continue

        try:
            answer = answer_question(project_name, question)

            print("\nAntwort:")
            print(answer)
            print("\n" + "-" * 50 + "\n")

            print(
                f"Detailed conversation saved to "
                f"{get_conversation_log_file(project_name)}"
            )
            print(
                f"Q&A table saved to "
                f"{get_qa_table_log_file(project_name)}"
            )
            print(
                f"Chat-style log saved to "
                f"{get_qa_chat_log_file(project_name)}\n"
            )

        except Exception as e:
            print(f"Error: {e}")