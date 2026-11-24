"""Quality gate: compare measured metrics against minimum thresholds."""

from __future__ import annotations

from collections.abc import Mapping


def check_thresholds(
    metrics: Mapping[str, float], minimums: Mapping[str, float]
) -> list[str]:
    """Return a failure message for every metric below its minimum."""
    failures: list[str] = []
    for name, minimum in minimums.items():
        if name not in metrics:
            failures.append(f"{name}: metric missing")
        elif metrics[name] + 1e-9 < minimum:
            failures.append(
                f"{name}: {metrics[name]:.3f} is below the minimum {minimum:.3f}"
            )
    return failures
