from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from core.state import Intent, RouteSource


@dataclass(frozen=True)
class IntentDecision:
    intent: Intent
    route_source: RouteSource
    confidence: float
    reason: str
    needs_clarification: bool = False


class IntentClassification(BaseModel):
    intent: Literal["rag", "order", "complaint", "chat"]
    confidence: float = Field(ge=0.0, le=1.0)
    needs_clarification: bool
    reason: str


INTENT_PROMPT = """你是电商客服系统的意图分类器。
根据用户最新消息，只能选择以下一种意图：
- rag：商品信息、退换货规则、配送政策、优惠和其他知识咨询
- order：查询用户订单、订单状态、物流或配送进度
- complaint：投诉、退款纠纷、商品问题申诉、要求人工处理
- chat：问候、闲聊、身份询问

分类规则：
1. 明确投诉或纠纷优先归为 complaint。
2. 查询某个订单或包裹当前状态归为 order。
3. 询问政策、规则、商品说明归为 rag。
4. 无法判断时设置 needs_clarification=true，不要猜测业务操作。
5. 只返回结构化字段，不执行任何业务操作。
"""

ORDER_ID_PATTERN = re.compile(r"\bORD-[A-Z0-9-]+\b", re.IGNORECASE)

COMPLAINT_TERMS = (
    "投诉",
    "维权",
    "举报",
    "客服态度差",
    "退款没到账",
    "退款被拒",
    "一直不退款",
    "要求人工处理",
)

RAG_TERMS = (
    "政策",
    "规则",
    "怎么退",
    "如何退",
    "退货流程",
    "退换货",
    "保修",
    "运费险",
    "尺码",
    "材质",
    "成分",
    "优惠",
    "发票",
    "配送范围",
    "多久能到",
    "几天送达",
    "多久发货",
    "发货时间",
    "配送时效",
    "物流信息长时间没有更新",
    "物流信息没有更新怎么办",
    "长时间没更新怎么办",
    "签收后几天内",
    "签收后多少天",
    "几天内可以申请退货",
    "申请退货",
    "七天无理由退货",
    "商品参数",
    "产品参数",
    "电池容量",
    "充电接口",
    "多少键",
)

ORDER_TERMS = (
    "订单状态",
    "查订单",
    "查询订单",
    "我的订单",
    "物流",
    "快递到哪",
    "包裹到哪",
    "发货了吗",
    "发货没有",
    "订单号",
    "物流单号",
)

CHAT_MESSAGES = {
    "你好",
    "你好呀",
    "您好",
    "在吗",
    "你是谁",
    "介绍一下你自己",
    "谢谢",
    "再见",
}


def classify_by_rules(message: str) -> IntentDecision | None:
    """只对高确定性的规则命中直接分类；不确定时返回 None。"""

    text = message.strip().lower()
    if not text:
        return None

    if any(term in text for term in COMPLAINT_TERMS):
        return IntentDecision(
            intent="complaint",
            route_source="rule",
            confidence=0.95,
            reason="命中明确投诉或退款纠纷表达",
        )

    # 询问退换货规则属于知识咨询，不要仅因出现“退货”而路由到投诉。
    if any(term in text for term in RAG_TERMS):
        return IntentDecision(
            intent="rag",
            route_source="rule",
            confidence=0.9,
            reason="命中商品或政策知识咨询表达",
        )
    if ORDER_ID_PATTERN.search(text):
        return IntentDecision(
            intent="order",
            route_source="rule",
            confidence=0.95,
            reason="消息中包含订单号",
        )
    if any(term in text for term in ORDER_TERMS):
        return IntentDecision(
            intent="order",
            route_source="rule",
            confidence=0.9,
            reason="命中订单或物流状态查询表达",
        )

    if text in CHAT_MESSAGES:
        return IntentDecision(
            intent="chat",
            route_source="rule",
            confidence=0.95,
            reason="命中常见问候或闲聊表达",
        )

    return None


class IntentRouterAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.structured_llm = llm.with_structured_output(IntentClassification)

    async def classify(self, message: str) -> IntentDecision:
        """规则优先；规则无法确定时调用一次 LLM 分类。"""

        rule_decision = classify_by_rules(message)
        if rule_decision is not None:
            return rule_decision

        try:
            result = await self.structured_llm.ainvoke(
                [
                    SystemMessage(content=INTENT_PROMPT),
                    HumanMessage(content=message),
                ]
            )
            return IntentDecision(
                intent=result.intent,
                route_source="llm",
                confidence=result.confidence,
                reason=result.reason,
                needs_clarification=result.needs_clarification,
            )
        except Exception:
            # 分类服务不可用时不触发订单或投诉等业务操作。
            return IntentDecision(
                intent="chat",
                route_source="fallback",
                confidence=0.0,
                reason="意图分类暂不可用，需要向用户澄清",
                needs_clarification=True,
            )