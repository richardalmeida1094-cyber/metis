def update(counts, action, outcome):
    """Increment the success/failure count of `action`."""
    counts[action][outcome] += 1
