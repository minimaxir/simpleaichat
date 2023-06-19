import fire
import os
from getpass import getpass
from .simpleaichat import AIChat
from dotenv import load_dotenv

load_dotenv()


def interactive_chat(character=None, character_command=None, prime=True):
    gpt_api_key = os.getenv("OPENAI_API_KEY") or getpass("Input your OpenAI key here: ")
    assert gpt_api_key, "An API key was not defined."
    _ = AIChat(character=character, character_command=character_command, prime=prime)
    return


if __name__ == "__main__":
    fire.Fire(interactive_chat)
