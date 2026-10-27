import hashlib


def replay_hash(replay: str) -> str:
    return hashlib.sha256(replay.encode()).hexdigest()
