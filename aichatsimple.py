import os
import datetime
import dateutil
from uuid import uuid4, UUID
import csv

from pydantic import BaseModel, SecretStr, HttpUrl, Field
from httpx import Client, AsyncClient
from typing import List, Dict, Union, Optional, Set, Any
import orjson
from dotenv import load_dotenv
from rich.console import Console

from utils import wikipedia_search_lookup

load_dotenv()

tool_prompt = """From the list of tools below:
- Reply ONLY with the number of the tool appropriate in response to the user's message.
- If no tool is appropriate, ONLY reply with \"0\".

{tools}"""


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
    api_url: HttpUrl
    model: str
    system: str
    params: Dict[str, Any] = {}
    messages: List[ChatMessage] = []
    input_fields: Set[str] = {}
    recent_messages: Optional[int] = None
    save_messages: Optional[bool] = True
    total_prompt_length: int = 0
    total_completion_length: int = 0
    total_length: int = 0
    title: Optional[str] = None

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps

    def __str__(self) -> str:
        sess_start_str = self.created_at.strftime("%Y-%m-%d %H:%M:%S")
        last_message_str = self.messages[-1].received_at.strftime("%Y-%m-%d %H:%M:%S")
        return f"""Chat session started at {sess_start_str}:
        - {len(self.messages):,} Messages
        - Last message sent at {last_message_str}"""

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
    api_url: HttpUrl = "https://api.openai.com/v1/chat/completions"
    input_fields: Set[str] = {"role", "content"}
    system: str = "You are a helpful assistant."
    params: Dict[str, Any] = {"temperature": 0.7}
    tool_logit_bias: Dict[str, int] = {k: 10 for k in range(15, 25)}

    def prepare_request(
        self,
        prompt: str,
        system: str = None,
        params: Dict[str, Any] = None,
        stream: bool = False,
    ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth['api_key'].get_secret_value()}",
        }

        system_message = ChatMessage(role="system", content=system or self.system)
        user_message = ChatMessage(role="user", content=prompt)

        gen_params = params or self.params
        data = {
            "model": self.model,
            "messages": self.format_input_messages(system_message, user_message),
            "stream": stream,
            **gen_params,
        }

        return headers, data, user_message

    def gen(
        self,
        prompt: str,
        client: Union[Client, AsyncClient],
        system: str = None,
        save_messages: bool = None,
        params: Dict[str, Any] = None,
    ):
        headers, data, user_message = self.prepare_request(prompt, system, params)

        r = client.post(
            self.api_url,
            json=data,
            headers=headers,
            timeout=20,
        ).json()

        content = r["choices"][0]["message"]["content"]
        assistant_message = ChatMessage(
            role=r["choices"][0]["message"]["role"],
            content=content,
            prompt_length=r["usage"]["prompt_tokens"],
            completion_length=r["usage"]["completion_tokens"],
            total_length=r["usage"]["total_tokens"],
        )

        self.total_prompt_length += r["usage"]["prompt_tokens"]
        self.total_completion_length += r["usage"]["completion_tokens"]
        self.total_length += r["usage"]["total_tokens"]

        if save_messages or self.save_messages:
            self.messages.append(user_message)
            self.messages.append(assistant_message)

        return content

    def stream(
        self,
        prompt: str,
        client: Union[Client, AsyncClient],
        system: str = None,
        save_messages: bool = None,
        params: Dict[str, Any] = None,
    ):
        headers, data, user_message = self.prepare_request(
            prompt, system, params, stream=True
        )

        with client.stream(
            "POST",
            self.api_url,
            json=data,
            headers=headers,
            timeout=None,
        ) as r:
            content = []
            for chunk in r.iter_lines():
                if len(chunk) > 0:
                    chunk = chunk[6:]  # SSE JSON chunks are prepended with "data: "
                    if chunk != "[DONE]":
                        chunk_dict = orjson.loads(chunk)
                        delta = chunk_dict["choices"][0]["delta"].get("content")
                        if delta:
                            content.append(delta)
                            yield {"delta": delta, "response": "".join(content)}

        # streaming does not currently return token counts
        assistant_message = ChatMessage(
            role="assistant",
            content="".join(content),
        )

        if save_messages or self.save_messages:
            self.messages.append(user_message)
            self.messages.append(assistant_message)
        return assistant_message

    def gen_with_tools(
        self,
        prompt: str,
        tools: List[Any],
        client: Union[Client, AsyncClient],
        system: str = None,
        save_messages: bool = None,
        params: Dict[str, Any] = None,
    ) -> Dict[str, Any]:

        # call 1: select tool and populate context
        tools_list = "\n".join(f"{i+1}: {f.__doc__}" for i, f in enumerate(tools))
        tool_prompt_format = tool_prompt.format(tools=tools_list)

        tool_idx = int(
            self(
                prompt,
                client=client,
                system=tool_prompt_format,
                save_messages=False,
                params={
                    "temperature": 0.0,
                    "max_tokens": 1,
                    "logit_bias": self.tool_logit_bias,
                },
            )
        )
        # if no tool is selected, do a standard generation instead.
        if tool_idx == 0:
            return {
                "response": self(
                    prompt,
                    client=client,
                    system=system,
                    save_messages=save_messages,
                    params=params,
                ),
                "tool": None,
            }
        selected_tool = tools[tool_idx - 1]
        context_dict = selected_tool(prompt)
        if isinstance(context_dict, str):
            context_dict = {"context": context_dict}

        context_dict["tool"] = selected_tool.__name__

        print(context_dict)

        # call 2: generate from the context
        new_system = f"{system or self.system}\n\nYou MUST use information from the context in your response."
        new_prompt = f"Context: {context_dict['context']}\n\nUser: {prompt}"

        context_dict["response"] = self(
            new_prompt,
            client=client,
            system=new_system,
            save_messages=False,
            params=params,
        )

        return context_dict

    async def gen_async(
        self,
        prompt: str,
        client: Union[Client, AsyncClient],
        system: str = None,
        save_messages: bool = None,
        params: Dict[str, Any] = None,
    ):
        headers, data, user_message = self.prepare_request(prompt, system, params)

        r = await client.post(
            self.api_url,
            json=data,
            headers=headers,
            timeout=20,
        )
        r = r.json()

        content = r["choices"][0]["message"]["content"]
        assistant_message = ChatMessage(
            role=r["choices"][0]["message"]["role"],
            content=content,
            prompt_length=r["usage"]["prompt_tokens"],
            completion_length=r["usage"]["completion_tokens"],
            total_length=r["usage"]["total_tokens"],
        )

        self.total_prompt_length += r["usage"]["prompt_tokens"]
        self.total_completion_length += r["usage"]["completion_tokens"]
        self.total_length += r["usage"]["total_tokens"]

        if save_messages or self.save_messages:
            self.messages.append(user_message)
            self.messages.append(assistant_message)

        return content


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
        system: str = None,
        id: Union[str, UUID] = uuid4(),
        prime: bool = True,
        default_session: bool = True,
        console: bool = True,
        **kwargs,
    ):

        client = Client()
        system = self.build_system(character, system)

        sessions = {}
        new_default_session = None
        if default_session:
            new_session = self.new_session(
                return_session=True, system=system, id=id, **kwargs
            )

            new_default_session = new_session
            sessions = {new_session.id: new_session}

        super().__init__(
            client=client, default_session=new_default_session, sessions=sessions
        )

        if character and console:
            default_session.title = character
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
        if sess.id == self.default_session.id:
            self.default_session = None
        del self.sessions[sess.id]
        del sess

    def __call__(
        self,
        prompt: str,
        id: Union[str, UUID] = None,
        system: str = None,
        save_messages: bool = None,
        params: Dict[str, Any] = None,
        tools: List[Any] = None,
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
            )

    def stream(
        self,
        prompt: str,
        id: Union[str, UUID] = None,
        system: str = None,
        save_messages: bool = None,
        params: Dict[str, Any] = None,
    ) -> str:
        sess = self.get_session(id)
        return sess.stream(
            prompt,
            client=self.client,
            system=system,
            save_messages=save_messages,
            params=params,
        )

    def build_system(self, character: str = None, system: str = None) -> str:
        default = "You are a helpful assistant."
        if character:
            character_prompt = """
            You are the following character and should act as they would: {0}

            CONCISELY introduce yourself first.
            """
            prompt = character_prompt.format(wikipedia_search_lookup(character)).strip()
            if system:
                character_system = """
                You MUST obey the following rule at all times: {0}
                """
                prompt = prompt + "\n\n" + character_system.format(system).strip()
            return prompt
        elif system:
            return system
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

    # Save/Load Chats given a session id
    def save_session(
        self, id: Union[str, UUID] = None, format: str = "csv", minify: bool = False
    ):
        sess = self.get_session(id)
        sess_dict = sess.dict(
            exclude={"auth", "api_url", "input_fields"},
            exclude_none=True,
        )
        out_path = f"test.{format}"
        if format == "csv":
            with open(out_path, "w", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(ChatMessage.__fields__.keys()))
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
            with open(out_path, "wb") as f:
                f.write(
                    orjson.dumps(
                        sess_dict, option=orjson.OPT_INDENT_2 if not minify else None
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
            )
