from pydantic import HttpUrl
from httpx import Client, AsyncClient
from typing import List, Dict, Union, Set, Any
import orjson

from models import ChatMessage, ChatSession

tool_prompt = """From the list of tools below:
- Reply ONLY with the number of the tool appropriate in response to the user's message.
- If no tool is appropriate, ONLY reply with \"0\".

{tools}"""


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

    async def stream_async(
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

        async with client.stream(
            "POST",
            self.api_url,
            json=data,
            headers=headers,
            timeout=None,
        ) as r:
            content = []
            async for chunk in r.aiter_lines():
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
