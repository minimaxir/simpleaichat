# Prompts

Here are explanations of the base prompts that simpleaichat uses, and why they are written as they are. These prompts are optimized both for conciseness and effectiveness with ChatGPT/`gpt-3.5-turbo`. This includes some postprocessing of the inputs.

## Tools

Invoking a tool invokes two separate API calls: one to select which tool which then provides additional **context**, and another call to generate based on that context, plus previous messages in the conversation. It is recommended to use the `n_recent_messages` parameter to prevent exponential cost usage.

### Call #1

Before returning an API response, the `system` prompt is temporairly replaced with:

```txt
From the list of tools below, reply ONLY with the number of the tool appropriate. If none are appropriate, ONLY reply with "0".

{tools}
```

Formatted example from the README:

```
From the list of functions below, reply ONLY with the number of the function appropriate for responding to the user. If none are, ONLY reply with "0".

1. Search the internet
2. Lookup more information about a topic.
```

This utilizes a few tricks:

- The call sets `{"max_tokens": 1}` so it will only output one number (hence there is a hard limit of 9 tools), which makes it more cost and speed efficient than other implementations.
- Unique to ChatGPT is also specifying an `input_bias` to make it such that the model can _only_ output numbers between 0 and 9. (specifically, indices 15-24 inclusive correspond to the numerals `0-9`, which can be verified using `tiktoken`)
- The numbers map 1:1 to the indicies of the input arrays of tools, so there never can be parsing errors.

### Call 2

The second call prepends the context from the tool to the prompt, and temporairly adds a command to the `system` prompt, to leverage said context without losing the speaking voice:

User message:

```txt
Context: {context}

User:
```

System prompt:

```
You MUST use information from the context in your response.
```

Formatted example from the README:

```
You are a helpful assistant.

You MUST use information from the context in your response.
```

```
Context:

User: How are you?
```
