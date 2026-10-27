from metis.learning.tabular import update


def test_update_increments_only_target_count():
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

    update(counts, "right", "success")

    assert counts["right"]["success"] == 1
    assert counts["right"]["failure"] == 0
    assert counts["left"]["success"] == 0
    assert counts["left"]["failure"] == 0


def test_update_increments_left_failure_only():
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

    update(counts, "left", "failure")

    assert counts["right"]["success"] == 0
    assert counts["right"]["failure"] == 0
    assert counts["left"]["success"] == 0
    assert counts["left"]["failure"] == 1


def test_update_accumulates_repeated_experiences():
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

    update(counts, "right", "success")
    update(counts, "right", "success")

    assert counts["right"]["success"] == 2
    assert counts["right"]["failure"] == 0
    assert counts["left"]["success"] == 0
    assert counts["left"]["failure"] == 0
