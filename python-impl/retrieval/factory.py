from contextlib import contextmanager
from typing import Any, Iterator

from core.config import Settings, settings
from retrieval.bm25 import BM25Retriever
from retrieval.documents import (
    load_knowledge_documents,
    split_knowledge_documents,
)


@contextmanager
def open_retriever(
    config: Settings = settings,
) -> Iterator[Any]:
    backend = config.retrieval_backend
    if config.rag_min_rerank_score is not None:
        if backend != "hybrid" or not config.reranker_enabled:
            raise ValueError("启用相关性闸门需要 hybrid 模式并开启精排")
    if backend not in {"bm25", "hybrid"}:
        raise ValueError(
            "RETRIEVAL_BACKEND 只能是 bm25 或 hybrid"
        )

    chunks = split_knowledge_documents(load_knowledge_documents())
    bm25 = BM25Retriever(chunks)

    if backend == "bm25":
        yield bm25
        return

    if not config.milvus_db_path.is_file():
        raise FileNotFoundError(
            f"知识库数据库不存在：{config.milvus_db_path}"
        )

    from retrieval.dense import MilvusDenseRetriever
    from retrieval.embeddings import BGEM3Encoder
    from retrieval.hybrid import HybridRetriever

    encoder = BGEM3Encoder(str(config.bge_model_path))
    dense = MilvusDenseRetriever(
        db_path=str(config.milvus_db_path),
        encoder=encoder,
    )

    try:
        stats = dense.client.get_collection_stats(
            dense.collection_name
        )
        if int(stats.get("row_count", 0)) == 0:
            raise RuntimeError(
                "Dense 知识库为空，请先完成知识块入库"
            )

        retriever = HybridRetriever(
            bm25=bm25,
            dense=dense,
        )

        if config.reranker_enabled:
            from retrieval.reranker import RerankingRetriever

            retriever = RerankingRetriever(
                retriever=retriever,
                model_path=str(config.reranker_model_path),
                candidate_k=10,
            )

        if config.rag_min_rerank_score is not None:
            from retrieval.confidence import ConfidenceGatedRetriever

            retriever = ConfidenceGatedRetriever(
                retriever=retriever,
                min_score=config.rag_min_rerank_score,
            )

        yield retriever

    finally:
        dense.close()
