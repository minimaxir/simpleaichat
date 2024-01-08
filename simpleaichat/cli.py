import argparse
import os
from getpass import getpass

import fire
from dotenv import load_dotenv

from .simpleaichat import AIChat

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("character", help="Specify the character", default=None, nargs="?")
parser.add_argument(
    "character_command", help="Specify the character command", default=None, nargs="?"
)
parser.add_argument("--prime", action="store_true", help="Enable priming")

ARGS = parser.parse_args()


def interactive_chat():
    gpt_api_key = os.getenv("OPENAI_API_KEY")
    if not gpt_api_key:
        gpt_api_key = getpass("Input your OpenAI key here: ")
    assert gpt_api_key, "An API key was not defined."
    _ = AIChat(ARGS.character, ARGS.character_command, ARGS.prime)


if __name__ == "__main__":
    fire.Fire(interactive_chat)
