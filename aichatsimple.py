import os
import datetime
from uuid import uuid4, UUID

from pydantic import BaseModel, SecretStr, Field
from httpx import Client, AsyncClient
from typing import List, Dict, Union, Optional, Set
import orjson
from dotenv import load_dotenv
from rich.console import Console

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
    prompt_length: Optional[int]
    completion_length: Optional[int]
    total_length: Optional[int]

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps

    def __str__(self) -> str:
        return self.content


class ChatSession(BaseModel):
    id: Union[str, UUID] = Field(default_factory=uuid4)
    created_at: datetime.datetime = Field(default_factory=now_tz)
    auth: Dict[str, SecretStr]
    model: str
    system_prompt: str
    think_prompt: Optional[str]
    temperature: float = 0.7
    max_length: int = None
    messages: List[ChatMessage] = []
    input_fields: Set[str] = {}
    recent_messages: Optional[int] = None

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps

    def __str__(self) -> str:
        sess_start_str = self.created_at.strftime("%Y-%m-%d %H:%M:%S")
        last_message_str = self.messages[-1].received_at.strftime("%Y-%m-%d %H:%M:%S")
        return f"""Chat session started at {sess_start_str}:
        - {len(self.messages):,} Messages
        - Last message sent at {last_message_str})"""

    def format_input_messages(
        self, system_message: ChatMessage, user_message: ChatMessage
    ) -> list:
        recent_messages = (
            self.messages[-self.recent_messages :]
            if self.recent_messages
            else self.messages
        )
        return (
            [system_message.dict(include=self.input_fields)]
            + [m.dict(include=self.input_fields) for m in recent_messages]
            + [user_message.dict(include=self.input_fields)]
        )


class ChatGPTSession(ChatSession):
    def __call__(
        self,
        prompt: str,
        client: Union[Client, AsyncClient],
        save_messages: bool = True,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth['api_key'].get_secret_value()}",
        }

        system_message = ChatMessage(role="system", content=self.system_prompt)
        user_message = ChatMessage(role="user", content=prompt)

        data = {
            "model": self.model,
            "messages": self.format_input_messages(system_message, user_message),
            "temperature": self.temperature,
            "max_tokens": self.max_length,
        }

        r = client.post(
            self.auth["api_url"].get_secret_value(),
            json=data,
            headers=headers,
            timeout=20,
        ).json()

        assistant_message = ChatMessage(
            role=r["choices"][0]["message"]["role"],
            content=r["choices"][0]["message"]["content"],
            prompt_length=r["usage"]["prompt_tokens"],
            completion_length=r["usage"]["completion_tokens"],
            total_length=r["usage"]["total_tokens"],
        )

        if save_messages:
            self.messages.append(user_message)
            self.messages.append(assistant_message)

        return r["choices"][0]["message"]["content"]


class AIChat(BaseModel):
    client: Union[Client, AsyncClient]
    default_session: Optional[ChatSession]
    sessions: Dict[Union[str, UUID], ChatSession] = {}

    class Config:
        arbitrary_types_allowed = True
        json_loads = orjson.loads
        json_dumps = orjson_dumps

    def __init__(
        self,
        character: str = None,
        system_prompt: str = None,
        prime: bool = True,
        model: str = "gpt-3.5-turbo",
        is_async: bool = False,
        api_key: str = None,
        session_id: Union[str, UUID] = uuid4(),
        **kwargs,
    ):

        client = Client() if not is_async else AsyncClient()
        system_prompt = self.build_system_prompt(character, system_prompt)

        new_session = self.new_session(
            model, system_prompt, api_key, session_id, return_session=True
        )

        default_session = new_session
        sessions = {new_session.id: new_session}

        super().__init__(
            client=client, default_session=default_session, sessions=sessions
        )

        if character:
            self.interactive_console(character=character, prime=prime)

    def new_session(
        self,
        model,
        system_prompt: str = None,
        api_key: str = None,
        session_id: Union[str, UUID] = uuid4(),
        return_session: bool = False,
    ) -> Optional[ChatGPTSession]:

        # TODO: Add support for more models (PaLM, Claude)
        if "gpt-" in model:
            gpt_api_key = os.getenv("OPENAI_API_KEY") or api_key
            assert gpt_api_key, f"An API key for {model} was not defined."
            sess = ChatGPTSession(
                id=session_id,
                system_prompt=system_prompt,
                auth={
                    "api_key": gpt_api_key,
                    "api_url": "https://api.openai.com/v1/chat/completions",
                },
                model=model,
                input_fields={"role", "content"},
            )

        if return_session:
            return sess
        else:
            self.sessions[sess.id] = sess

    def get_session(self, session_id: Union[str, UUID] = None) -> ChatSession:
        return self.sessions[session_id] if session_id else self.default_session

    def __call__(
        self,
        prompt: str,
        session_id: Union[str, UUID] = None,
        save_messages: bool = True,
    ) -> str:
        sess = self.get_session(session_id)
        return sess(prompt, client=self.client, save_messages=save_messages)

    def build_system_prompt(
        self, character: str = None, system_prompt: str = None
    ) -> str:
        default = "You are a helpful assistant."
        if character:
            character_prompt = """
            You are the following character and should speak as they would: {0}

            Concisely introduce yourself first.
            """
            prompt = character_prompt.format(wikipedia_search_lookup(character)).strip()
            if system_prompt:
                character_system_prompt = """
                You MUST obey the following rule at all times: {0}
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

    def interactive_console(self, character: str = None, prime: bool = True) -> None:
        console = Console(width=40, highlight=False)
        sess = self.default_session

        # prime with a unique starting response to the user
        if prime:
            ai_response = sess("Hello!", self.client)
            console.print(f"[b]{character}[/b]: {ai_response}", style="bright_magenta")

        while True:
            try:
                user_input = console.input("[b]You:[/b] ").strip()
            except KeyboardInterrupt:
                break
            if not user_input:
                break

            with console.status("", spinner="point"):
                ai_response = sess(user_input, self.client)
            console.print(f"[b]{character}[/b]: {ai_response}", style="bright_magenta")

    def __str__(self) -> str:
        if self.default_session:
            return self.default_session.json(
                exclude={"api_key", "api_url"},
                exclude_none=True,
                option=orjson.OPT_INDENT_2,
            )

    def __repr__(self) -> str:
        return ""

    # Tabulators for returning total token counts
    def message_totals(self, attr: str, session_id: Union[str, UUID] = None) -> int:
        sess = self.get_session(session_id)
        return sum([x.dict().get(attr, 0)] for x in sess.messages)

    @property
    def total_prompt_length(self, session_id: Union[str, UUID] = None) -> int:
        return self.message_totals("prompt_length", session_id)

    @property
    def total_completion_length(self, session_id: Union[str, UUID] = None) -> int:
        return self.message_totals("completion_length", session_id)

    @property
    def total_length(self, session_id: Union[str, UUID] = None) -> int:
        return self.message_totals("total_length", session_id)

    # alias total_tokens to total_length for common use
    @property
    def total_tokens(self, session_id: Union[str, UUID] = None) -> int:
        return self.total_length(session_id)
