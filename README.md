# simpleaichat

## Installation

simpleaichat can be installed [from PyPI](https://pypi.org/project/simpleaichat/):

```sh
pip3 install simpleaichat
```

## Quick, Fun Demo

You can demo chat-apps very quickly with simpleaichat! First, you will need to get an OpenAI API key, and then with one line of code:

```py3
from simpleaichat import AIChat

AIChat(api_key="sk-...")
```

And with that, you'll be thrust directly into a chat!

This AI chat will mimic the behavior of OpenAI's webapp, but on your local computer!

You can also pass the API key by storing it in an `.env` file with a `OPEN_AI_KEY` field in the working directory (recommended), or by setting the environment variable of `OPEN_AI_KEY` directly to the API key.

But what about creating your own custom conversations? That's where things get fun. Just input whatever person, place or thing, fictional or nonfictional, that you want to chat with!

```py3
AIChat("GLaDOS")  # assuming API key loaded via methods above
```

But that's not all! You can customize exactly how they behave too with additional commands!

```py3
AIChat("GLaDOS", "Speak in the style of a Seinfeld monologue")
```

```py3
AIChat("New York City", "Speak using only emoji")
```

## Building AI-based Apps

The trick with working with new chat-based apps that wasn't readily available with earlier iterations of GPT-3 is the addition of the system prompt: a different class. In fact, the chat demos above are actually using system prompt tricks behind the scenes! You can see how those system prompts are constructed here.

For developers, you can instantiate a programmatic instance of `AIChat` by explicitly specifying a system prompt, or by disabling the console app.

```py3
ai = AIChat(system_prompt="You are a helpful assistant")
ai = AIChat(console=False)  # same as above
```

You can then feed the new `ai` class with user input, and it will return and save the response from ChatGPT:

```py3
response = ai("What is the capital of California?")
print(response)
```

Future inputs to the `ai` object by default

In actuality, the `AIChat` class is a manager of chat _sessions_, which means you can have multiple independent chats happening! The examples above use a default session, but you can create new ones by specifying a `session_key` when calling `ai`.

```py3
ai.new_session(session_key="conv1")
```

You can also save chat sessions (as JSON) and load them later.

### Functions

A large number of ChatGPT-based apps don't actually use the "chat" part of the model. Instead, they just use the system prompt/first user prompt as a form of natural language programming. simpleaichat has a special mode which lets you create such functions without any overhead. And even better, you can create multiple functions within a single AIChat!

```py3
func1 = "Format the user-provided JSON as YAML."
func2 = "Write a 5/7/5 haiku based on the user-provided JSON."
func3 = "Translate the values in the user-provided JSON from English to French."

json = "{}"

ai = AIChat(functions=[func1, func2, func3])
ai(json, function=func1)
ai(json, function=func2)
ai(json, function=func3)
```

## Tools

One of the most recent aspects of interacting with ChatGPT is the ability for the model to use "tools." As defined from the ReAct paper, tools allow the model to decide when to use custom functions, which can extend beyond just the chat app.

Using tools typically requires a number of shennanigans, but simpleaichat uses a neat trick to make it fast and easy!

You will need to specify functions

## Roadmap

- PaLM Chat (Bard) and Anthropic Claude support

## Miscellaneous Notes

- simpleaichat very intentionally avoids coupling features with common use cases where possible (e.g. Tools) in order to avoid software lock-in due to complexity.

## Maintainer/Creator

Max Woolf ([@minimaxir](https://minimaxir.com))

_Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir) and [GitHub Sponsors](https://github.com/sponsors/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use._

## License

MIT
