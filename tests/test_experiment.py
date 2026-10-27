import pytest

from metis.experiment import Experiment


def test_experiment_stores_values():
    experiment = Experiment(
        experiment_id="experiment_001",
        objective="to understand if an odd seed is better than an even seed",
        action="Next",
        repetitions=50,
    )

    assert experiment.experiment_id == "experiment_001"
    assert experiment.objective == "to understand if an odd seed is better than an even seed"
    assert experiment.action == "Next"
    assert experiment.repetitions == 50


@pytest.mark.parametrize("invalid_repetitions", [0, -1, -50])
def test_experiment_rejects_invalid_repetitions(invalid_repetitions):
    with pytest.raises(ValueError):
        Experiment(
            experiment_id="experiment_001",
            objective="to understand if an odd seed is better than an even seed",
            action="Next",
            repetitions=invalid_repetitions,
        )
