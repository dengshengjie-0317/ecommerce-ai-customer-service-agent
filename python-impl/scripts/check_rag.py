import argparse
import asyncio
import csv
from pathlib import Path

from agents.experts.rag_graph import create_rag_subgraph
from core.llm import create_chat_model
from retrieval.bm25 import BM25Retriever
from retrieval.documents import (
    KNOWLEDGE_DIR,
    load_knowledge_documents,
    split_knowledge_documents,
)


async def main(backend: str):
    dense = None
    llm = None

    try:
        chunks = split_knowledge_documents(load_knowledge_documents())
        retriever = BM25Retriever(chunks)

        if backend == "hybrid":
            from retrieval.dense import MilvusDenseRetriever
            from retrieval.embeddings import BGEM3Encoder
            from retrieval.hybrid import HybridRetriever

            db_path = "/root/.cache/huggingface/milvus/ecommerce.db"
            if not Path(db_path).is_file():
                raise FileNotFoundError("请先运行 check_dense 建立知识库")

            encoder = BGEM3Encoder(
                "/root/.cache/huggingface/local-model"
            )
            dense = MilvusDenseRetriever(
                db_path=db_path,
                encoder=encoder,
            )
            retriever = HybridRetriever(
                bm25=retriever,
                dense=dense,
            )

        print("本次使用的检索方式：", backend, flush=True)

        llm = create_chat_model()
        graph = create_rag_subgraph(retriever=retriever, llm=llm)

        path = KNOWLEDGE_DIR / "evaluation" / "questions.csv"
        with path.open(encoding="utf-8-sig", newline="") as file:
            cases = list(csv.DictReader(file))

        known_case = next(
            case for case in cases
            if case["answerable"].strip().lower() == "true"
        )
        unknown_case = next(
            case for case in cases
            if case["answerable"].strip().lower() == "false"
        )

        for case, expected_status in (
            (known_case, "completed"),
            (unknown_case, "handoff"),
        ):
            result = await graph.ainvoke({"query": case["question"]})
            answer = result["agent_result"]
            citations = answer.get("citations", [])

            print("\n问题：", case["question"])
            print("状态：", answer["status"])
            print("回答：", answer["response"])
            print("引用：", citations)

            assert answer["status"] == expected_status, answer

            if expected_status == "completed":
                assert citations, answer
                assert any(
                    citation["chunk_id"] == case["expected_chunk_id"]
                    for citation in citations
                ), answer
            else:
                assert not citations, answer

        print(f"\n{backend} RAG 检查通过")

    finally:
        if dense is not None:
            dense.close()
        if llm is not None:
            await llm.root_async_client.close()
            llm.root_client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["bm25", "hybrid"],
        default="bm25",
    )
    args = parser.parse_args()
    asyncio.run(main(args.backend))