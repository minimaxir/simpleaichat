import os
import datetime
import dateutil
from uuid import uuid4, UUID
from contextlib import contextmanager, asynccontextmanager
import csv

from pydantic import BaseModel
from httpx import Client, AsyncClient
from typing import List, Dict, Union, Optional, Any
import orjson
from dotenv import load_dotenv
from rich.console import Console

from .utils import wikipedia_search_lookup
from .models import ChatMessage, ChatSession, orjson_dumps
from .chatgpt import ChatGPTSession

load_dotenv()


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
        character_command: str = None,
        system: str = None,
        id: Union[str, UUID] = uuid4(),
        prime: bool = True,
        default_session: bool = True,
        console: bool = True,
        **kwargs,
    ):

        client = Client()
        system_format = self.build_system(character, character_command, system)

        sessions = {}
        new_default_session = None
        if default_session:
            new_session = self.new_session(
                return_session=True, system=system_format, id=id, **kwargs
            )

            new_default_session = new_session
            sessions = {new_session.id: new_session}

        super().__init__(
            client=client, default_session=new_default_session, sessions=sessions
        )

        if not system and console:
            character = character or "ChatGPT"
            new_default_session.title = character
            self.interactive_console(character=character, prime=prime)

    def new_session(
        self,
        return_session: bool = False,
        **kwargs,
    ) -> Optional[ChatGPTSession]:

        if "model" not in kwargs:  # set default
            kwargs["model"] = "gpt-3.5-turbo"
        # TODO: Add support for more models (PaLM, Claude)
        if "gpt-" in kwargs["model"]:
            gpt_api_key = os.getenv("OPENAI_API_KEY") or kwargs.get("api_key")
            assert gpt_api_key, f"An API key for {kwargs['model'] } was not defined."
            sess = ChatGPTSession(
                auth={
                    "api_key": gpt_api_key,
                },
                **kwargs,
            )

        if return_session:
            return sess
        else:
            self.sessions[sess.id] = sess

    def get_session(self, id: Union[str, UUID] = None) -> ChatSession:
        try:
            sess = self.sessions[id] if id else self.default_session
        except KeyError:
            raise KeyError("No session by that key exists.")
        if not sess:
            raise ValueError("No default session exists.")
        return sess

    def reset_session(self, id: Union[str, UUID] = None) -> None:
        sess = self.get_session(id)
        sess.messages = []

    def delete_session(self, id: Union[str, UUID] = None) -> None:
        sess = self.get_session(id)
        if self.default_session and sess.id == self.default_session.id:
            self.default_session = None
        del self.sessions[sess.id]
        del sess

    @contextmanager
    def session(self, **kwargs):
        sess = self.new_session(return_session=True, **kwargs)
        self.sessions[sess.id] = sess
        try:
            yield sess
        finally:
            self.delete_session(sess.id)

    def __call__(
        self,
        prompt: Union[str, Any],
        id: Union[str, UUID] = None,
        system: str = None,
        save_messages: bool = None,
        params: Dict[str, Any] = None,
        tools: List[Any] = None,
        input_schema: Any = None,
        output_schema: Any = None,
    ) -> str:
        sess = self.get_session(id)
        if tools:
            for tool in tools:
                assert tool.__doc__, f"Tool {tool} does not have a docstring."
            assert len(tools) <= 9, "You can only have a maximum of 9 tools."
            return sess.gen_with_tools(
                prompt,
                tools,
                client=self.client,
                system=system,
                save_messages=save_messages,
                params=params,
            )
        else:
            return sess.gen(
                prompt,
                client=self.client,
                system=system,
                save_messages=save_messages,
                params=params,
                input_schema=input_schema,
                output_schema=output_schema,
            )

    def stream(
        self,
        prompt: str,
        id: Union[str, UUID] = None,
        system: str = None,
        save_messages: bool = None,
        params: Dict[str, Any] = None,
        input_schema: Any = None,
    ) -> str:
        sess = self.get_session(id)
        return sess.stream(
            prompt,
            client=self.client,
            system=system,
            save_messages=save_messages,
            params=params,
            input_schema=input_schema,
        )

    def build_system(
        self, character: str = None, character_command: str = None, system: str = None
    ) -> str:
        if character:
            character_prompt = """
            You must follow ALL these rules in all responses:
            - You are the following character and should ALWAYS act as them: {0}
            - NEVER speak in a formal tone.
            - Concisely introduce yourself first in character.
            """
            prompt = character_prompt.format(wikipedia_search_lookup(character)).strip()
            if character_command:
                character_system = """
                - {0}
                """
                prompt = (
                    prompt + "\n" + character_system.format(character_command).strip()
                )
            return prompt
        elif system:
            return system
        else:
            return "You are a helpful assistant."

    def interactive_console(self, character: str = None, prime: bool = True) -> None:
        console = Console(highlight=False)
        sess = self.default_session
        ai_text_color = "bright_magenta"

        # prime with a unique starting response to the user
        if prime:
            console.print(f"[b]{character}[/b]: ", end="", style=ai_text_color)
            for chunk in sess.stream("Hello!", self.client):
                console.print(chunk["delta"], end="", style=ai_text_color)

        while True:
            console.print()
            try:
                user_input = console.input("[b]You:[/b] ").strip()
                if not user_input:
                    break

                console.print(f"[b]{character}[/b]: ", end="", style=ai_text_color)
                for chunk in sess.stream(user_input, self.client):
                    console.print(chunk["delta"], end="", style=ai_text_color)
            except KeyboardInterrupt:
                break

    def __str__(self) -> str:
        if self.default_session:
            return self.default_session.json(
                exclude={"api_key", "api_url"},
                exclude_none=True,
                option=orjson.OPT_INDENT_2,
            )

    def __repr__(self) -> str:
        return ""

    # Save/Load Chats given a session id
    def save_session(
        self,
        output_path: str = None,
        id: Union[str, UUID] = None,
        format: str = "csv",
        minify: bool = False,
    ):
        sess = self.get_session(id)
        sess_dict = sess.dict(
            exclude={"auth", "api_url", "input_fields"},
            exclude_none=True,
        )
        output_path = output_path or f"chat_session.{format}"
        if format == "csv":
            with open(output_path, "w", encoding="utf-8") as f:
                fields = [
                    "role",
                    "content",
                    "received_at",
                    "prompt_length",
                    "completion_length",
                    "total_length",
                ]
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for message in sess_dict["messages"]:
                    # datetime must be in common format to be loaded into spreadsheet
                    # for human-readability, the timezone is set to local machine
                    local_datetime = message["received_at"].astimezone()
                    message["received_at"] = local_datetime.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    w.writerow(message)
        elif format == "json":
            with open(output_path, "wb") as f:
                f.write(
                    orjson.dumps(
                        sess_dict, option=None if minify else orjson.OPT_INDENT_2
                    )
                )

    def load_session(self, input_path: str, id: Union[str, UUID] = uuid4(), **kwargs):

        assert input_path.endswith(".csv") or input_path.endswith(
            ".json"
        ), "Only CSV and JSON imports are accepted."

        if input_path.endswith(".csv"):
            with open(input_path, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                messages = []
                for row in r:
                    # need to convert the datetime back to UTC
                    local_datetime = datetime.datetime.strptime(
                        row["received_at"], "%Y-%m-%d %H:%M:%S"
                    ).replace(tzinfo=dateutil.tz.tzlocal())
                    row["received_at"] = local_datetime.astimezone(
                        datetime.timezone.utc
                    )
                    # https://stackoverflow.com/a/68305271
                    row = {k: (None if v == "" else v) for k, v in row.items()}
                    messages.append(ChatMessage(**row))

            self.new_session(id=id, **kwargs)
            self.sessions[id].messages = messages

        if input_path.endswith(".json"):
            with open(input_path, "rb") as f:
                sess_dict = orjson.loads(f.read())
            # update session with info not loaded, e.g. auth/api_url
            for arg in kwargs:
                sess_dict[arg] = kwargs[arg]
            self.new_session(**sess_dict)

    # Tabulators for returning total token counts
    def message_totals(self, attr: str, id: Union[str, UUID] = None) -> int:
        sess = self.get_session(id)
        return getattr(sess, attr)

    @property
    def total_prompt_length(self, id: Union[str, UUID] = None) -> int:
        return self.message_totals("total_prompt_length", id)

    @property
    def total_completion_length(self, id: Union[str, UUID] = None) -> int:
        return self.message_totals("total_completion_length", id)

    @property
    def total_length(self, id: Union[str, UUID] = None) -> int:
        return self.message_totals("total_length", id)

    # alias total_tokens to total_length for common use
    @property
    def total_tokens(self, id: Union[str, UUID] = None) -> int:
        return self.total_length(id)


class AsyncAIChat(AIChat):
    async def __call__(
        self,
        prompt: str,
        id: Union[str, UUID] = None,
        system: str = None,
        save_messages: bool = None,
        params: Dict[str, Any] = None,
        tools: List[Any] = None,
        input_schema: Any = None,
        output_schema: Any = None,
    ) -> str:
        # TODO: move to a __post_init__ in Pydantic 2.0
        if isinstance(self.client, Client):
            self.client = AsyncClient()
        sess = self.get_session(id)
        if tools:
            for tool in tools:
                assert tool.__doc__, f"Tool {tool} does not have a docstring."
            assert len(tools) <= 9, "You can only have a maximum of 9 tools."
            return await sess.gen_with_tools_async(
                prompt,
                tools,
                client=self.client,
                system=system,
                save_messages=save_messages,
                params=params,
            )
        else:
            return await sess.gen_async(
                prompt,
                client=self.client,
                system=system,
                save_messages=save_messages,
                params=params,
                input_schema=input_schema,
                output_schema=output_schema,
            )

    async def stream(
        self,
        prompt: str,
        id: Union[str, UUID] = None,
        system: str = None,
        save_messages: bool = None,
        params: Dict[str, Any] = None,
        input_schema: Any = None,
    ) -> str:
        # TODO: move to a __post_init__ in Pydantic 2.0
        if isinstance(self.client, Client):
            self.client = AsyncClient()
        sess = self.get_session(id)
        return sess.stream_async(
            prompt,
            client=self.client,
            system=system,
            save_messages=save_messages,
            params=params,
            input_schema=input_schema,
        )

    @asynccontextmanager
    async def session(self, **kwargs):
        sess = self.new_session(return_session=True, **kwargs)
        self.sessions[sess.id] = sess
        try:
            yield sess
        finally:
            self.delete_session(sess.id)
