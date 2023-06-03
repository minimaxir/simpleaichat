import fire
import os
from getpass import getpass
from aichatsimple import AIChat
from dotenv import load_dotenv

load_dotenv()


def interactive_chat(character=None, system=None, prime=True):
    gpt_api_key = os.getenv("OPENAI_API_KEY")
    if not gpt_api_key:
        gpt_api_key = getpass("Input your OpenAI key here: ")
    assert gpt_api_key, "An API key was not defined."
    chat = AIChat(character=character, system=system)
    chat.interactive_console(character=character or "ChatGPT", prime=prime)
    return


if __name__ == "__main__":
    fire.Fire(interactive_chat)
