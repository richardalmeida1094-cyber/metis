import random


class RandomAgent:
    def act(self, observation: str) -> str:
        return random.choice(("left", "right"))
