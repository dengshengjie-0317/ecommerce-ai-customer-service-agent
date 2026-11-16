from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class RetrievalHit:
    document: Document
    score: float
    retriever: str