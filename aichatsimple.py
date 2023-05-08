import os
import datetime
from uuid import uuid4, UUID

from pydantic import BaseModel, SecretStr, HttpUrl, Field
from httpx import Client, AsyncClient
from typing import List, Dict, Union, Optional
import orjson
from dotenv import load_dotenv

from utils import wikipedia_search_lookup

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
    created_at: datetime.datetime = Field(default_factory=now_tz)
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


class AIChat(BaseModel):
    client: Union[Client, AsyncClient]
    default_session: Optional[ChatSession]
    sessions: Dict[Union[str, UUID], ChatSession] = {}

    class Config:
        arbitrary_types_allowed = True
        json_loads = orjson.loads
        json_dumps = orjson_dumps

    def __init__(self, character: str = None, system_prompt: str = None, **kwargs):

        client = Client()
        system_prompt = self.build_system_prompt(character, system_prompt)

        new_session = ChatSession(
            system_prompt=system_prompt,
            api_key=os.getenv("OPENAI_API_KEY"),
            api_url="https://api.openai.com/v1/chat/completions",
            model="gpt-3.5-turbo",
        )

        sessions = {}
        default_session = new_session
        sessions[new_session.id] = new_session

        super().__init__(
            client=client, default_session=default_session, sessions=sessions
        )

    def __call__(self, prompt: str):
        sess = self.default_session

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

        r = self.client.post(
            sess.api_url, json=data, headers=headers, timeout=10
        ).json()

        assistant_message = ChatMessage(
            role=r["choices"][0]["message"]["role"],
            content=r["choices"][0]["message"]["content"],
            prompt_tokens=r["usage"]["prompt_tokens"],
            completion_tokens=r["usage"]["completion_tokens"],
            total_tokens=r["usage"]["total_tokens"],
        )

        sess.messages.append(user_message)
        sess.messages.append(assistant_message)

        return r["choices"][0]["message"]["content"]

    def __str__(self):
        if self.default_session:
            return self.default_session.json(
                exclude={"api_key", "api_url"},
                exclude_none=True,
                option=orjson.OPT_INDENT_2,
            )

    def build_system_prompt(self, character: str = None, system_prompt: str = None):
        default = "You are a helpful assistant."
        if character:
            character_prompt = """
            You are the following character and should speak as they would: {0}
            """
            prompt = character_prompt.format(wikipedia_search_lookup(character)).strip()
            if system_prompt:
                character_system_prompt = """
                You MUST also follow this rule: {0}
                """
                prompt = (
                    prompt
                    + "\n\n"
                    + character_system_prompt.format(system_prompt).strip()
                )
            return prompt
        elif system_prompt:
            return system_prompt
        else:
            return default


if __name__ == "__main__":
    ai = AIChat("Steve Jobs", "Speak only in emoji.")
    m = ai("What is your favorite product?")
    print(m)
