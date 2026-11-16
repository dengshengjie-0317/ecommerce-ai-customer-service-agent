import asyncio
import csv

from agents.experts.rag_graph import create_rag_subgraph
from core.llm import create_chat_model
from retrieval.bm25 import BM25Retriever
from retrieval.documents import (
    KNOWLEDGE_DIR,
    load_knowledge_documents,
    split_knowledge_documents,
)


async def main():
    chunks = split_knowledge_documents(load_knowledge_documents())
    llm = create_chat_model()

    try:
        graph = create_rag_subgraph(
            BM25Retriever(chunks),
            llm=llm,
        )

        path = KNOWLEDGE_DIR / "evaluation" / "questions.csv"
        with path.open(encoding="utf-8-sig", newline="") as file:
            cases = list(csv.DictReader(file))

        known_case = cases[0]
        unknown_case = next(
            case
            for case in cases
            if case["answerable"].strip().lower() == "false"
        )

        for case in (known_case, unknown_case):
            result = await graph.ainvoke({"query": case["question"]})
            answer = result["agent_result"]

            print("\n问题：", case["question"])
            print("状态：", answer["status"])
            print("回答：", answer["response"])
            print("引用：", answer.get("citations", []))

            if case is known_case:
                assert answer["status"] == "completed", answer
                assert answer["citations"], answer
                assert any(
                    citation["doc_id"] == case["expected_source"]
                    for citation in answer["citations"]
                ), answer
            else:
                assert answer["status"] == "handoff", answer
                assert not answer["citations"], answer

        print("\nRAG checks passed")

    finally:
        await llm.root_async_client.close()
        llm.root_client.close()


if __name__ == "__main__":
    asyncio.run(main())