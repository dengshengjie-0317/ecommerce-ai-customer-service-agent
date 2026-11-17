from pathlib import Path

from retrieval.bm25 import BM25Retriever
from retrieval.dense import MilvusDenseRetriever
from retrieval.documents import (
    load_knowledge_documents,
    split_knowledge_documents,
)
from retrieval.embeddings import BGEM3Encoder
from retrieval.hybrid import HybridRetriever


def main():
    db_path = "/root/.cache/huggingface/milvus/ecommerce.db"
    if not Path(db_path).is_file():
        raise FileNotFoundError("知识库数据库不存在，请先运行 check_dense")

    chunks = split_knowledge_documents(load_knowledge_documents())
    bm25 = BM25Retriever(chunks)

    encoder = BGEM3Encoder(
        "/root/.cache/huggingface/local-model"
    )
    dense = MilvusDenseRetriever(
        db_path=db_path,
        encoder=encoder,
    )
    hybrid = HybridRetriever(bm25=bm25, dense=dense)

    try:
        cases = [
            (
                "签收后几天内可以申请退货？",
                "DEMO-RETURNS-001:v1:0001",
            ),
            (
                "KB-200 键盘是多少键？",
                "DEMO-PRODUCT-001:v1:0002",
            ),
        ]

        for question, expected_chunk in cases:
            hits = hybrid.search(question, top_k=3)
            print("\n问题：", question)

            for rank, hit in enumerate(hits, start=1):
                print(
                    rank,
                    hit.document.metadata["chunk_id"],
                    round(hit.score, 6),
                    hit.document.metadata["retrieval_ranks"],
                )

            chunk_ids = [
                hit.document.metadata["chunk_id"]
                for hit in hits
            ]
            assert expected_chunk in chunk_ids
            assert len(chunk_ids) == len(set(chunk_ids))
            assert all(hit.retriever == "rrf" for hit in hits)
            assert any(
                {"bm25", "dense"}.issubset(
                    hit.document.metadata["retrieval_ranks"]
                )
                for hit in hits
            ), "没有找到两路共同召回的结果，请检查 Dense 数据库"

        print("\nHybrid 检索基本检查通过")
    finally:
        dense.close()


if __name__ == "__main__":
    main()