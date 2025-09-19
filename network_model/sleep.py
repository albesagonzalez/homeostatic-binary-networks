"""
Standalone sleep pass placeholder.

Import this in your notebook and assign to `network_model["sleep"]`.
"""

from typing import Any


def sleep(self: Any) -> None:
    """
    Minimal sleep routine using the network's replay helper.

    Expects the instance to define:
      - `sleep_duration` (int): number of timesteps to replay
      - `replay(post, pre)`: method that performs one replay step
    """

    # Iterate over sleep timesteps
    for timestep in range(int(self.sleep_duration)):
        self.replay("output", "hidden")

        self.record()
        self.time_index += 1
        self.sleep_indices.append(self.time_index)
