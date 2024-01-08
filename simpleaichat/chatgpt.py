from typing import Any, Dict, List, Set, Union

import orjson
from httpx import AsyncClient, Client
from pydantic import HttpUrl

from .models import ChatMessage, ChatSession
from .utils import remove_a_key

tool_prompt = """From the list of tools below:
- Reply ONLY with the number of the tool appropriate in response to the user's last message.
- If no tool is appropriate, ONLY reply with \"0\".

{tools}"""


class ChatGPTSession(ChatSession):
    api_url: HttpUrl = "https://api.openai.com/v1/chat/completions"
    input_fields: Set[str] = {"role", "content", "name"}
    system: str = "You are a helpful assistant."
    params: Dict[str, Any] = {"temperature": 0.7}

    def prepare_request(
        self,
        prompt: str,
        system: str = None,
        params: Dict[str, Any] = None,
        stream: bool = False,
        input_schema: Any = None,
        output_schema: Any = None,
        is_function_calling_required: bool = True,
    ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth['api_key'].get_secret_value()}",
        }

        system_message = ChatMessage(role="system", content=system or self.system)
        if not input_schema:
            user_message = ChatMessage(role="user", content=prompt)
        else:
            assert isinstance(
                prompt, input_schema
            ), f"prompt must be an instance of {input_schema.__name__}"
            user_message = ChatMessage(
                role="function",
                content=prompt.model_dump_json(),
                name=input_schema.__name__,
            )

        gen_params = params or self.params
        data = {
            "model": self.model,
            "messages": self.format_input_messages(system_message, user_message),
            "stream": stream,
            **gen_params,
        }

        # Add function calling parameters if a schema is provided
        if input_schema or output_schema:
            functions = []
            if input_schema:
                input_function = self.schema_to_function(input_schema)
                functions.append(input_function)
            if output_schema:
                output_function = self.schema_to_function(output_schema)
                functions.append(
                    output_function
                ) if output_function not in functions else None
                if is_function_calling_required:
                    data["function_call"] = {"name": output_schema.__name__}
            data["functions"] = functions

        return headers, data, user_message

    def schema_to_function(self, schema: Any):
        assert schema.__doc__, f"{schema.__name__} is missing a docstring."
        assert (
            "title" not in schema.model_fields.keys()
        ), "`title` is a reserved keyword and cannot be used as a field name."
        schema_dict = schema.model_json_schema()
        remove_a_key(schema_dict, "title")

        return {
            "name": schema.__name__,
            "description": schema.__doc__,
            "parameters": schema_dict,
        }

    def gen(
        self,
        prompt: str,
        client: Union[Client, AsyncClient],
        system: str = None,
        save_messages: bool = None,
        params: Dict[str, Any] = None,
        input_schema: Any = None,
        output_schema: Any = None,
    ):
        headers, data, user_message = self.prepare_request(
            prompt, system, params, False, input_schema, output_schema
        )

        r = client.post(
            str(self.api_url),
            json=data,
            headers=headers,
            timeout=None,
        )
        r = r.json()

        try:
            if not output_schema:
                content = r["choices"][0]["message"]["content"]
                assistant_message = ChatMessage(
                    role=r["choices"][0]["message"]["role"],
                    content=content,
                    finish_reason=r["choices"][0]["finish_reason"],
                    prompt_length=r["usage"]["prompt_tokens"],
                    completion_length=r["usage"]["completion_tokens"],
                    total_length=r["usage"]["total_tokens"],
                )
                self.add_messages(user_message, assistant_message, save_messages)
            else:
                content = r["choices"][0]["message"]["function_call"]["arguments"]
                content = orjson.loads(content)

            self.total_prompt_length += r["usage"]["prompt_tokens"]
            self.total_completion_length += r["usage"]["completion_tokens"]
            self.total_length += r["usage"]["total_tokens"]
        except KeyError:
            raise KeyError(f"No AI generation: {r}")

        return content

    def stream(
        self,
        prompt: str,
        client: Union[Client, AsyncClient],
        system: str = None,
        save_messages: bool = None,
        params: Dict[str, Any] = None,
        input_schema: Any = None,
    ):
        headers, data, user_message = self.prepare_request(
            prompt, system, params, True, input_schema
        )

        with client.stream(
            "POST",
            str(self.api_url),
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

        self.add_messages(user_message, assistant_message, save_messages)

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

        logit_bias_weight = 100
        logit_bias = {str(k): logit_bias_weight for k in range(15, 15 + len(tools) + 1)}

        tool_idx = int(
            self.gen(
                prompt,
                client=client,
                system=tool_prompt_format,
                save_messages=False,
                params={
                    "temperature": 0.0,
                    "max_tokens": 1,
                    "logit_bias": logit_bias,
                },
            )
        )

        # if no tool is selected, do a standard generation instead.
        if tool_idx == 0:
            return {
                "response": self.gen(
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

        # call 2: generate from the context
        new_system = f"{system or self.system}\n\nYou MUST use information from the context in your response."
        new_prompt = f"Context: {context_dict['context']}\n\nUser: {prompt}"

        context_dict["response"] = self.gen(
            new_prompt,
            client=client,
            system=new_system,
            save_messages=False,
            params=params,
        )

        # manually append the nonmodified user message + normal AI response
        user_message = ChatMessage(role="user", content=prompt)
        assistant_message = ChatMessage(
            role="assistant", content=context_dict["response"]
        )
        self.add_messages(user_message, assistant_message, save_messages)

        return context_dict

    async def gen_async(
        self,
        prompt: str,
        client: Union[Client, AsyncClient],
        system: str = None,
        save_messages: bool = None,
        params: Dict[str, Any] = None,
        input_schema: Any = None,
        output_schema: Any = None,
    ):
        headers, data, user_message = self.prepare_request(
            prompt, system, params, False, input_schema, output_schema
        )

        r = await client.post(
            str(self.api_url),
            json=data,
            headers=headers,
            timeout=None,
        )
        r = r.json()

        try:
            if not output_schema:
                content = r["choices"][0]["message"]["content"]
                assistant_message = ChatMessage(
                    role=r["choices"][0]["message"]["role"],
                    content=content,
                    finish_reason=r["choices"][0]["finish_reason"],
                    prompt_length=r["usage"]["prompt_tokens"],
                    completion_length=r["usage"]["completion_tokens"],
                    total_length=r["usage"]["total_tokens"],
                )
                self.add_messages(user_message, assistant_message, save_messages)
            else:
                content = r["choices"][0]["message"]["function_call"]["arguments"]
                content = orjson.loads(content)

            self.total_prompt_length += r["usage"]["prompt_tokens"]
            self.total_completion_length += r["usage"]["completion_tokens"]
            self.total_length += r["usage"]["total_tokens"]
        except KeyError:
            raise KeyError(f"No AI generation: {r}")

        return content

    async def stream_async(
        self,
        prompt: str,
        client: Union[Client, AsyncClient],
        system: str = None,
        save_messages: bool = None,
        params: Dict[str, Any] = None,
        input_schema: Any = None,
    ):
        headers, data, user_message = self.prepare_request(
            prompt, system, params, True, input_schema
        )

        async with client.stream(
            "POST",
            str(self.api_url),
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

        self.add_messages(user_message, assistant_message, save_messages)

    async def gen_with_tools_async(
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

        logit_bias_weight = 100
        logit_bias = {str(k): logit_bias_weight for k in range(15, 15 + len(tools) + 1)}

        tool_idx = int(
            await self.gen_async(
                prompt,
                client=client,
                system=tool_prompt_format,
                save_messages=False,
                params={
                    "temperature": 0.0,
                    "max_tokens": 1,
                    "logit_bias": logit_bias,
                },
            )
        )

        # if no tool is selected, do a standard generation instead.
        if tool_idx == 0:
            return {
                "response": await self.gen_async(
                    prompt,
                    client=client,
                    system=system,
                    save_messages=save_messages,
                    params=params,
                ),
                "tool": None,
            }
        selected_tool = tools[tool_idx - 1]
        context_dict = await selected_tool(prompt)
        if isinstance(context_dict, str):
            context_dict = {"context": context_dict}

        context_dict["tool"] = selected_tool.__name__

        # call 2: generate from the context
        new_system = f"{system or self.system}\n\nYou MUST use information from the context in your response."
        new_prompt = f"Context: {context_dict['context']}\n\nUser: {prompt}"

        context_dict["response"] = await self.gen_async(
            new_prompt,
            client=client,
            system=new_system,
            save_messages=False,
            params=params,
        )

        # manually append the nonmodified user message + normal AI response
        user_message = ChatMessage(role="user", content=prompt)
        assistant_message = ChatMessage(
            role="assistant", content=context_dict["response"]
        )
        self.add_messages(user_message, assistant_message, save_messages)

        return context_dict
