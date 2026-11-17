from retrieval.dense import MilvusDenseRetriever
from retrieval.documents import (
    load_knowledge_documents,
    split_knowledge_documents,
)
from retrieval.embeddings import BGEM3Encoder


def main():
    chunks = split_knowledge_documents(load_knowledge_documents())
    print("知识块数量：", len(chunks), flush=True)

    encoder = BGEM3Encoder(
        "/root/.cache/huggingface/local-model"
    )
    retriever = MilvusDenseRetriever(
        db_path="/root/.cache/huggingface/milvus/ecommerce.db",
        encoder=encoder,
    )

    try:
        written = retriever.upsert_documents(chunks)
        print("本次写入或更新数量：", written, flush=True)

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
            hits = retriever.search(question, top_k=3)
            print("\n问题：", question)

            for rank, hit in enumerate(hits, start=1):
                print(
                    rank,
                    hit.document.metadata["chunk_id"],
                    round(hit.score, 4),
                )

            assert hits, "Dense 检索没有返回结果"
            assert expected_chunk in {
                hit.document.metadata["chunk_id"]
                for hit in hits
            }, "Top-3 未包含预期知识块"

        print("\nDense 检索基本检查通过")
    finally:
        retriever.close()


if __name__ == "__main__":
    main()