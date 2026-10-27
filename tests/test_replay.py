from metis.replay import replay_hash


def test_same_replay_same_hash():
    replay_a = "action=left;observation=wall"
    replay_b = "action=left;observation=wall"

    hash_a = replay_hash(replay_a)
    hash_b = replay_hash(replay_b)

    assert hash_a == hash_b
