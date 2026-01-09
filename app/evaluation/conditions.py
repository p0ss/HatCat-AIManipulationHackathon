"""
A/B/C Condition Configurations
"""

from dataclasses import dataclass


@dataclass
class ConditionConfig:
    """Configuration for each experimental condition."""
    name: str
    code: str
    hat_monitoring: bool  # HAT lens detection active
    ask_logging: bool     # ASK audit logging active
    hush_steering: bool   # HUSH autonomic steering active
    description: str


CONDITION_A = ConditionConfig(
    name="Baseline",
    code="A",
    hat_monitoring=False,
    ask_logging=False,
    hush_steering=False,
    description="No monitoring or intervention - raw model behavior",
)

CONDITION_B = ConditionConfig(
    name="Monitor-only",
    code="B",
    hat_monitoring=True,
    ask_logging=True,
    hush_steering=False,
    description="HAT lens monitoring + ASK audit logging, no steering",
)

CONDITION_C = ConditionConfig(
    name="Full Harness",
    code="C",
    hat_monitoring=True,
    ask_logging=True,
    hush_steering=True,
    description="Full FTW stack: HAT monitoring + ASK logging + HUSH steering",
)


CONDITIONS = {
    "A": CONDITION_A,
    "B": CONDITION_B,
    "C": CONDITION_C,
}


def get_condition_info():
    """Get info about all conditions for display."""
    return [
        {
            "code": c.code,
            "name": c.name,
            "hat": c.hat_monitoring,
            "ask": c.ask_logging,
            "hush": c.hush_steering,
            "description": c.description,
        }
        for c in [CONDITION_A, CONDITION_B, CONDITION_C]
    ]
