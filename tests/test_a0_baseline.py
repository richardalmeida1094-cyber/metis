from metis.agents.random_agent import RandomAgent
from metis.learning.tabular import update
from metis.seed_manager import set_seed
from metis.worlds.family_a import A0Environment


def test_a0_random_agent_baseline():
    set_seed(42)

    agent = RandomAgent()
    environment = A0Environment()

    counts = {
        "right": {
            "success": 0,
            "failure": 0,
        },
        "left": {
            "success": 0,
            "failure": 0,
        },
    }

    total_steps = 50

    for _ in range(total_steps):
        observation = environment.reset()
        action = agent.act(observation)
        outcome = environment.step(action)

        update(counts, action, outcome)

    successes = counts["right"]["success"] + counts["left"]["success"]

    failures = counts["right"]["failure"] + counts["left"]["failure"]

    success_rate = successes / total_steps

    assert successes + failures == total_steps
    assert counts["right"]["failure"] == 0
    assert counts["left"]["success"] == 0
    assert 0.0 <= success_rate <= 1.0

    print()
    print("seed: 42")
    print(f"budget: {total_steps}")
    print(f"successes: {successes}")
    print(f"failures: {failures}")
    print(f"success_rate: {success_rate:.2f}")
