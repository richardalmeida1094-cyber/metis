from dataclasses import dataclass


@dataclass
class EnvironmentCapabilities:
    supports_actions: bool
    supports_experiments: bool
    supports_snapshot: bool
    supports_restore: bool
