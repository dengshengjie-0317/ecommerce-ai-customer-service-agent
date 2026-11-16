import csv

from retrieval.bm25 import BM25Retriever
from retrieval.documents import (
    KNOWLEDGE_DIR,
    load_knowledge_documents,
    split_knowledge_documents,
)


def main():
    documents = load_knowledge_documents()
    chunks = split_knowledge_documents(documents)
    retriever = BM25Retriever(chunks)

    evaluation_file = KNOWLEDGE_DIR / "evaluation" / "questions.csv"
    with evaluation_file.open(encoding="utf-8-sig", newline="") as file:
        cases = list(csv.DictReader(file))

    answerable_cases = [
        case
        for case in cases
        if case["answerable"].strip().lower() == "true"
    ]

    if not answerable_cases:
        raise ValueError("评测集里没有可回答的问题")

    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_rank_sum = 0.0

    for case in answerable_cases:
        hits = retriever.search(case["question"], top_k=5)
        expected_chunk_id = case["expected_chunk_id"].strip()
        rank = next(
            (
                index
                for index, hit in enumerate(hits, start=1)
                if hit.document.metadata["chunk_id"] == expected_chunk_id            ),
            None,
        )

        if rank == 1:
            hit_at_1 += 1
        if rank is not None and rank <= 3:
            hit_at_3 += 1
        if rank is not None:
            reciprocal_rank_sum += 1 / rank

        print(
            f"rank={rank or '未命中'} | "
            f"expected={expected_chunk_id} | "
            f"question={case['question']}"
        )

    total = len(answerable_cases)
    print(f"\n可回答问题数: {total}")
    print(f"Hit@1: {hit_at_1}/{total} = {hit_at_1 / total:.1%}")
    print(f"Hit@3: {hit_at_3}/{total} = {hit_at_3 / total:.1%}")
    print(f"MRR@5: {reciprocal_rank_sum / total:.3f}")


if __name__ == "__main__":
    main()