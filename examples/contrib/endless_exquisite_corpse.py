import uuid
import time
import argparse
from simpleaichat import AIChat

SYSTEM_DEFAULT = "Write a short single line that continues the line."

ai = AIChat(console=False)


class ExquisiteCorpse:
    """
    Endless exquisite corpse generator
    """
    def __init__(self, seed, system=SYSTEM_DEFAULT, temp=2, last_n_words=None):
        self.last_seed = seed
        self.last_n_words = last_n_words
        self.system = system
        self.temp = temp
        self.total_tokens = 0

    def __iter__(self):
        return self

    def __next__(self):
        _id = uuid.uuid4()
        ai.new_session(id=_id, system=self.system, params={"temperature": self.temp})
        seed = " ".join(self.last_seed.split(" ")[-self.last_n_words:]) if self.last_n_words else self.last_seed
        response = ai(seed, id=_id)
        self.total_tokens += ai.message_totals("total_length", id=_id)
        ai.delete_session(id=_id)
        self.last_seed = response
        return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Endless exquisite corpse generator")
    parser.add_argument("seed", help="Initial seed")
    parser.add_argument("--system", default=SYSTEM_DEFAULT, help="System name")
    parser.add_argument("--temp", type=float, default=1, help="Temperature")
    parser.add_argument("--delay", type=float, default=15, help="Delay between lines in seconds")
    parser.add_argument("--last_n_words", type=int, help="Number of words to use from last line")
    args = parser.parse_args()
    
    corpse = ExquisiteCorpse(
        args.seed,
        system=args.system,
        temp=args.temp,
        last_n_words=args.last_n_words,
    )

    try:
        print(args.seed)
        for line in corpse:
            print(line)
            time.sleep(args.delay)
    except KeyboardInterrupt:
        print()
        print(f"Total tokens used: {corpse.total_tokens}")
