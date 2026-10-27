from metis.environment_capabilities import EnvironmentCapabilities


def test_environment_capabilities_stores_values():
    capabilities = EnvironmentCapabilities(
        supports_actions=True, supports_experiments=True, supports_snapshot=False, supports_restore=False
    )

    assert capabilities.supports_actions
    assert capabilities.supports_experiments
    assert not capabilities.supports_snapshot
    assert not capabilities.supports_restore
