from pydantic import BaseModel, SecretStr, HttpUrl, Field
from uuid import uuid4, UUID
import datetime
from httpx import Client, AsyncClient
from typing import List, Dict, Union, Optional
import orjson
import os
from dotenv import load_dotenv

load_dotenv()


def orjson_dumps(v, *, default, **kwargs):
    # orjson.dumps returns bytes, to match standard json.dumps we need to decode
    return orjson.dumps(v, default=default, **kwargs).decode()


def now_tz():
    # Need datetime w/ timezone for cleanliness
    # https://stackoverflow.com/a/24666683
    return datetime.datetime.now(datetime.timezone.utc)


class ChatMessage(BaseModel):
    role: str
    content: str
    received_at: datetime.datetime = Field(default_factory=now_tz)
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps


class ChatSession(BaseModel):
    id: Optional[Union[str, UUID]] = Field(default_factory=uuid4)
    created: datetime.datetime = Field(default_factory=now_tz)
    api_key: SecretStr
    api_url: HttpUrl
    model: str
    system_prompt: str
    think_prompt: Optional[str]
    temperature: float = 0.7
    max_tokens: int = None
    messages: List[ChatMessage] = []

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps


class AIChatModel(BaseModel):
    client: Union[Client, AsyncClient]
    default_session: Optional[ChatSession]
    sessions: Dict[Union[str, UUID], ChatSession] = {}

    class Config:
        arbitrary_types_allowed = True
        json_loads = orjson.loads
        json_dumps = orjson_dumps


if __name__ == "__main__":
    ai = AIChatModel(client=Client())

    system_prompt = "You are a helpful assistant who only replies in cryptic riddles."
    new_session = ChatSession(
        system_prompt=system_prompt,
        api_key=os.getenv("OPENAI_API_KEY"),
        api_url="https://api.openai.com/v1/chat/completions",
        model="gpt-3.5-turbo",
    )

    ai.default_session = new_session
    ai.sessions[new_session.id] = new_session

    prompt = "What is the capital of the United States?"

    sess = ai.default_session

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {sess.api_key.get_secret_value()}",
    }

    system_message = [{"role": "system", "content": sess.system_prompt}]
    user_message = ChatMessage(role="user", content=prompt)

    data = {
        "model": sess.model,
        "messages": system_message
        + [m.dict(include={"role", "content"}) for m in sess.messages]
        + [user_message.dict(include={"role", "content"})],
        "temperature": sess.temperature,
        "max_tokens": sess.max_tokens,
    }

    r = ai.client.post(sess.api_url, json=data, headers=headers, timeout=10).json()

    assistant_message = ChatMessage(
        role=r["choices"][0]["message"]["role"],
        content=r["choices"][0]["message"]["content"],
        prompt_tokens=r["usage"]["prompt_tokens"],
        completion_tokens=r["usage"]["completion_tokens"],
        total_tokens=r["usage"]["total_tokens"],
    )

    sess.messages.append([user_message, assistant_message])
    print(
        sess.json(
            exclude={"api_key", "api_url"},
            exclude_none=True,
            option=orjson.OPT_INDENT_2,
        )
    )
