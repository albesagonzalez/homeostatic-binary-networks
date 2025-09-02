"""
Standalone forward pass placeholder.

Import this in your notebook and assign to `network_model["forward"]`.
"""

from typing import Any

import torch
import torch.nn.functional as F


def forward(self: Any, x: torch.Tensor, debug: bool = False) -> None:
    """
    Minimal forward pass using the network's activation routine.

    Args:
        self: SSCNetwork (or compatible) instance (bound as a method)
        x: Input tensor for a "day"; shape [T, ...]
        debug: Verbose prints if True
    """

    # Iterate over timesteps
    T = x.shape[0]
    for timestep in range(T):
        self.hidden, _ = self.activation(x[timestep], 'hidden')
        
        self.output_hat = F.linear(self.output_hidden, self.hidden) + self.output_b*self.output_IM
        self.output, _ = self.activation(self.output_hat, 'output')

        self.hebbian('hidden', 'hidden')
        self.homeostasis('hidden', 'hidden')

        self.record()
        self.time_index += 1
        self.awake_indices.append(self.time_index)


    self.day += 1
