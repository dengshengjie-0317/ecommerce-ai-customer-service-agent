from collections import defaultdict

from langchain_core.documents import Document

from retrieval.schemas import RetrievalHit


def reciprocal_rank_fusion(
    retrieval_lists: list[list[RetrievalHit]],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[RetrievalHit]:
    """按名次融合不同检索器的结果，不直接混加原始分数。"""

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if rrf_k <= 0:
        raise ValueError("rrf_k 必须大于 0")

    fused_scores: dict[str, float] = defaultdict(float)
    documents: dict[str, Document] = {}
    ranks: dict[str, dict[str, int]] = defaultdict(dict)

    for ranked_hits in retrieval_lists:
        for rank, hit in enumerate(ranked_hits, start=1):
            chunk_id = hit.document.metadata.get("chunk_id")
            if not chunk_id:
                raise ValueError("RRF 的每个结果都必须包含 chunk_id")

            retriever_name = hit.retriever
            fused_scores[chunk_id] += 1.0 / (rrf_k + rank)
            ranks[chunk_id][retriever_name] = rank
            documents[chunk_id] = hit.document

    ordered_chunk_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id),
    )

    fused_hits: list[RetrievalHit] = []

    for chunk_id in ordered_chunk_ids[:top_k]:
        original = documents[chunk_id]
        metadata = {
            **original.metadata,
            "rrf_score": fused_scores[chunk_id],
            "retrieval_ranks": ranks[chunk_id],
        }

        fused_hits.append(
            RetrievalHit(
                document=Document(
                    page_content=original.page_content,
                    metadata=metadata,
                ),
                score=fused_scores[chunk_id],
                retriever="rrf",
            )
        )

    return fused_hits