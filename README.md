# simpleaichat

```py3
from simpleaichat import AIChat

ai = AIChat(system="Write a professional GitHub README based on the user-provided coding project description.")
ai("simpleaichat — a Python package for easily interfacing with ChatGPT with robust features and minimal code complexity.")
```

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
ai = AIChat(system="You are a helpful assistant")
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
ai.new_session(id="conv1")
```

You can also save chat sessions (as JSON) and load them later.

### Functions

A large number of popular venture-capital-funded ChatGPT apps don't actually use the "chat" part of the model. Instead, they just use the system prompt/first user prompt as a form of natural language programming. You can emulate this behavior by passing a new system prompt when generating text, and not saving the resulting messages.

```py3
json = '{"title": "An array of integers.", "array": [-1, 0, 1]}'
functions = [
             "Format the user-provided JSON as YAML.",
             "Write a limerick based on the user-provided JSON.",
             "Translate the user-provided JSON from English to French."
            ]
params = {"temperature": 0.0, "max_tokens": 100}  # a temperature of 0.0 is deterministic

# We namespace the function by `id` so it doesn't affect other chats.
# Settings set during session creation will apply to all generations from the session,
# but you can change them per-generation, as is the case with the `system` prompt here.
ai = AIChat(id="function", params=params, save_messages=False)
for function in functions:
    output = ai(json, id="function", system=function)
    print(output)
```

Results:

```txt
title: "An array of integers."
array:
  - -1
  - 0
  - 1
```

```txt
An array of integers so neat,
With values that can't be beat,
From negative to positive one,
It's a range that's quite fun,
This JSON is really quite sweet!
```

```txt
{"titre": "Un tableau d'entiers.", "tableau": [-1, 0, 1]}
```

## Tools

One of the most recent aspects of interacting with ChatGPT is the ability for the model to use "tools." As defined from the ReAct paper, tools allow the model to decide when to use custom functions, which can extend beyond just the chat app. This workflow is analogous to ChatGPT Plugins.

Using tools typically requires a number of shennanigans, but simpleaichat uses a neat trick to make it fast and easy!

You will need to specify functions

## Roadmap

- PaLM Chat (Bard) and Anthropic Claude support

## Miscellaneous Notes

- Like gpt-2-simple before it, the primary motivation behind releasing simpleaichat is to both democratize access to ChatGPT even more without extolling complexity as a virtue, and also offer more transparency for non-engineers into how Chat AI-based apps work under the hood given the disproportionate amount of media misinformation about their capabilities. This is inspired by real-world experience from [my work with BuzzFeed](https://tech.buzzfeed.com/the-right-tools-for-the-job-c05de96e949e) in the domain, where after spending a long time working with the popular LangChain, a more-simple implementation was both much easier to maintain and resulted in much better generations.
  - simpleaichat very intentionally avoids coupling features with common use cases where possible (e.g. Tools) in order to avoid software lock-in due to the difficulty implementing anything not explicitly mentioned in the project's documentation. The philosophy behind simpleaichat is to provide good demos, and let the user's creativity and business needs take priority instead of having to fit a round peg into a square hole like with LangChain.
  - simpleaichat makes it easier to interface with Chat AIs, but it does not attempt to solve common technical and ethical problems inherent to large language models trained on the internet, including prompt injection and unintended plagiarism. The use should exercise good judgment when implementing simpleaichat.
  - simpleaichat does not use the "Agent" logical metaphor for its actions. If needed be, you can emulate the Agent workflow with a `while` loop without much additional code, plus with the additional benefit of much more flexibility such as debugging.
- The session manager implements some sensible security defaults, such as using UUIDs as session ids by default and storing authentication information in a way to minimize unintentional leakage. Your end-user application should still be aware of potential security issues, however.
- Outside of the explicit examples, none of this README uses AI-generated text. The introduction code example is just a joke, but it was too good of a real-world use case!

## Maintainer/Creator

Max Woolf ([@minimaxir](https://minimaxir.com))

_Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir) and [GitHub Sponsors](https://github.com/sponsors/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use._

## License

MIT
