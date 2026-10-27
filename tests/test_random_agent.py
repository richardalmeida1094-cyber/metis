from metis.agents.random_agent import RandomAgent
from metis.seed_manager import set_seed


def test_random_agent_returns_only_valid_actions():
    set_seed(42)
    agent = RandomAgent()

    actions = [agent.act("start") for _ in range(20)]

    assert all(action in ("left", "right") for action in actions)


def test_random_agent_is_reproducible_with_same_seed():
    set_seed(42)
    first_agent = RandomAgent()
    first_sequence = [first_agent.act("start") for _ in range(20)]

    set_seed(42)
    second_agent = RandomAgent()
    second_sequence = [second_agent.act("start") for _ in range(20)]

    assert first_sequence == second_sequence
