from langchain_openai import ChatOpenAI

from core.config import settings


def create_chat_model(temperature: float = 0) -> ChatOpenAI:
    if not settings.api_key:
        raise RuntimeError(
            "未设置 OPENAI_API_KEY。请在 python-impl/.env 中填写 DeepSeek API Key。"
        )

    return ChatOpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model_name,
        temperature=temperature,
    )