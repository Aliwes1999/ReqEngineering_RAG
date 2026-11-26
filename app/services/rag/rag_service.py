from app.services.rag.ask import retrieve, build_context

# Der Projektname, unter dem die Excel-Datei ingestiert wurde:
RAG_KNOWLEDGE_PROJECT = "test_project"


def get_rag_knowledge_context(requirements_summary: str, top_k: int = 8) -> str:
    """
    Sucht passende Referenz-Funktionen und Strukturen aus der vorab ingestierten
    Excel-Wissensbasis basierend auf den aktuellen Projektanforderungen.
    """
    query = (
        f"Relevante Hauptfunktionen, Teilfunktionen, Funktionsstrukturen und "
        f"MBSE-Muster für folgende Anforderungen:\n{requirements_summary}"
    )
    points = retrieve(
        question=query,
        project_name=RAG_KNOWLEDGE_PROJECT,
        top_k=top_k
    )
    return build_context(points)