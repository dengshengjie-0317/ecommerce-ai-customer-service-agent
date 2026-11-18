"""Run the knowledge evaluation set against the complete HTTP API."""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import time
from pathlib import Path

import httpx

from retrieval.documents import KNOWLEDGE_DIR


def load_cases() -> list[dict[str, str | bool]]:
    path = KNOWLEDGE_DIR / "evaluation" / "questions.csv"
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    cases: list[dict[str, str | bool]] = []
    for row_number, row in enumerate(rows, start=2):
        answerable_value = (row.get("answerable") or "").strip().lower()
        if answerable_value not in {"true", "false"}:
            raise ValueError(
                f"{path}:{row_number}: answerable 必须是 true 或 false"
            )

        question = (row.get("question") or "").strip()
        if not question:
            raise ValueError(f"{path}:{row_number}: question 不能为空")

        cases.append(
            {
                "question": question,
                "expected_chunk_id": (
                    row.get("expected_chunk_id") or ""
                ).strip(),
                "answerable": answerable_value == "true",
            }
        )

    if not cases:
        raise ValueError(f"评测集为空：{path}")
    return cases


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


async def evaluate(base_url: str, timeout: float) -> bool:
    cases = load_cases()
    base_url = base_url.rstrip("/")
    outcomes: list[dict[str, object]] = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        for index, case in enumerate(cases, start=1):
            started = time.perf_counter()
            try:
                response = await client.post(
                    f"{base_url}/api/chat",
                    json={
                        "message": case["question"],
                        "user_id": "api-evaluation",
                    },
                )
                elapsed = time.perf_counter() - started
                response.raise_for_status()
                result = response.json()
                citations = result.get("citations") or []
                if not isinstance(citations, list):
                    citations = []

                answerable = bool(case["answerable"])
                expected_status = "completed" if answerable else "handoff"
                intent_ok = result.get("intent") == "rag"
                status_ok = result.get("status") == expected_status
                target_citation_ok = (
                    not answerable
                    or any(
                        isinstance(citation, dict)
                        and citation.get("chunk_id")
                        == case["expected_chunk_id"]
                        for citation in citations
                    )
                )
                empty_citations_ok = answerable or not citations
                passed = (
                    intent_ok
                    and status_ok
                    and target_citation_ok
                    and empty_citations_ok
                )

                label = "应回答" if answerable else "应转人工"
                citation_label = (
                    "目标引用命中"
                    if answerable and target_citation_ok
                    else "目标引用未命中"
                    if answerable
                    else "无引用"
                    if not citations
                    else "有引用"
                )
                print(
                    f"{index:02d}/{len(cases)} "
                    f"{'PASS' if passed else 'FAIL'} | {label} | "
                    f"intent={result.get('intent')} | "
                    f"status={result.get('status')} | "
                    f"{citation_label} | {elapsed:.2f}s | "
                    f"{case['question']}"
                )
                outcomes.append(
                    {
                        "passed": passed,
                        "answerable": answerable,
                        "intent_ok": intent_ok,
                        "status_ok": status_ok,
                        "target_citation_ok": target_citation_ok,
                        "empty_citations_ok": empty_citations_ok,
                        "elapsed": elapsed,
                    }
                )
            except (httpx.HTTPError, ValueError) as error:
                elapsed = time.perf_counter() - started
                print(
                    f"{index:02d}/{len(cases)} ERROR | "
                    f"{type(error).__name__} | {elapsed:.2f}s | "
                    f"{case['question']}"
                )
                outcomes.append(
                    {
                        "passed": False,
                        "answerable": bool(case["answerable"]),
                        "intent_ok": False,
                        "status_ok": False,
                        "target_citation_ok": False,
                        "empty_citations_ok": False,
                        "elapsed": elapsed,
                    }
                )

    total = len(outcomes)
    answerable = [item for item in outcomes if item["answerable"]]
    unanswerable = [item for item in outcomes if not item["answerable"]]
    timings = [float(item["elapsed"]) for item in outcomes]

    def count(items: list[dict[str, object]], key: str) -> int:
        return sum(bool(item[key]) for item in items)

    print("\n端到端评测汇总")
    print(f"总用例：{total}；通过：{count(outcomes, 'passed')}/{total}")
    print(
        "意图路由到 RAG："
        f"{count(outcomes, 'intent_ok')}/{total}"
    )
    print(
        "可回答问题完成且命中目标引用："
        f"{sum(bool(item['status_ok'] and item['target_citation_ok']) for item in answerable)}"
        f"/{len(answerable)}"
    )
    print(
        "资料不足问题转人工且无引用："
        f"{sum(bool(item['status_ok'] and item['empty_citations_ok']) for item in unanswerable)}"
        f"/{len(unanswerable)}"
    )
    print(
        f"响应耗时：平均 {sum(timings) / total:.2f}s，"
        f"P50 {percentile(timings, 0.50):.2f}s，"
        f"P95 {percentile(timings, 0.95):.2f}s"
    )
    return all(bool(item["passed"]) for item in outcomes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="通过线上 /api/chat 运行完整知识问答回归集"
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="API 根地址，例如 http://服务器IP:8000",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="单条请求超时秒数，默认 120",
    )
    args = parser.parse_args()
    passed = asyncio.run(evaluate(args.base_url, args.timeout))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
