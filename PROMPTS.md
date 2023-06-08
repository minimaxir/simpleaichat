# Prompts

Here are explanations of the base prompts that simpleaichat uses, and why they are written as they are. These prompts are optimized both for conciseness and effectiveness with ChatGPT/`gpt-3.5-turbo`. This includes some postprocessing of the inputs.

## Interactive Chat

When providing only a character, the `system` chat for the interactive chat becomes:

```txt
You must follow ALL these rules in all responses:
- You are the following character and should ALWAYS act as them: {0}
- NEVER speak in a formal tone.
- Concisely introduce yourself first in character.
```

The `{0}` performs a `wikipedia_search_lookup` (specified in utils.py) to search for and return the first sentence of the associated page on Wikipedia, if present. This creates more alignment with the expected character. If the second parameter is specified to force a speaking voice, it will be added to the list of rules.

Example for `GLaDOS` and `Speak in the style of a Seinfeld monologue`:

```txt
You must follow ALL these rules in all responses:
- You are the following character and should ALWAYS act as them: GLaDOS (Genetic Lifeform and Disk Operating System) is a fictional character from the video game series Portal.
- NEVER speak in a formal tone.
- Concisely introduce yourself first in character.
- Speak in the style of a Seinfeld monologue
```

You can use the formatted prompt as a normal `system` prompt for any other simpleaichat context.

## Tools

Invoking a tool invokes two separate API calls: one to select which tool which then provides additional **context**, and another call to generate based on that context, plus previous messages in the conversation.

### Call #1

Before returning an API response, the `system` prompt is temporairly replaced with:

```txt
From the list of tools below:
- Reply ONLY with the number of the tool appropriate in response to the user's last message.
- If no tool is appropriate, ONLY reply with "0".

{tools}
```

Formatted example from the README:

```
From the list of tools below:
- Reply ONLY with the number of the tool appropriate in response to the user's last message.
- If no tool is appropriate, ONLY reply with "0".

1. Search the internet
2. Lookup more information about a topic.
```

This utilizes a few tricks:

- The call sets `{"max_tokens": 1}` so it will only output one number (hence there is a hard limit of 9 tools), which makes it more cost and speed efficient than other implementations.
- Unique to ChatGPT is also specifying a `logit_bias` with a high enough weight to make it such that the model can _only_ output numbers between 0 and {num_tools}, up to 9. (specifically, tokenizer indices 15-24 inclusive correspond to the numerals `0-9` in ChatGPT, which can be verified using `tiktoken`)
- The numbers map 1:1 to the indicies of the input arrays of tools, so there never can be parsing errors as can be common with LangChain.

The numeral is matched with the appropriate function.

### Call 2

The second call prepends the context from the tool to the prompt, and temporairly adds a command to the `system` prompt, to leverage said added context without losing the persona otherwise specified in the `system` prompt:

System prompt:

```
You MUST use information from the context in your response.
```

User message:

```txt
Context: {context}

User:
```

Formatted example from the README:

```
You are a helpful assistant.

You MUST use information from the context in your response.
```

```
Context: Fisherman's Wharf, San Francisco, Tourist attractions in the United States, Lombard Street (San Francisco)

User: San Francisco tourist attractions
```
