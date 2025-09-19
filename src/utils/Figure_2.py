import random
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.model import SSCNetwork
from src.utils.general import make_input, get_selectivity, get_latent_accuracy, LatentSpace



from src.model import SSCNetwork
from network_model.forward import forward
from params.default import network_parameters
from src.utils.general import make_input, LatentSpace, get_ordered_indices, test_network, get_latent_accuracy


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def figure2_accuracy(
    network_parameters: Dict[str, Any],
    model: Dict[str, Any],
    recording_parameters: Dict[str, Any],
    input_parameters: Dict[str, Any],
    latent_specs: Optional[Dict[str, Any]] = None,
    noise_level: Optional[int] = None,
    seed: Optional[int] = None,
    eval_region: str = "output",
    get_aux_results : bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Run the network without sleep and compute per-latent accuracy (Figure 2).

    Args:
        network_parameters: dict passed to SSCNetwork
        model: dict of methods to bind (e.g., {'forward': forward_fn})
        recording_parameters: must include 'regions' with eval_region and 'rate_activity' == 1
        input_parameters: parameters for make_input; will be copied and possibly updated
        latent_specs: optional specs to build LatentSpace if not already provided in input_parameters
        noise_level: if provided, overrides input_parameters['num_swaps']
        seed: random seed for reproducibility
        eval_region: region name to evaluate (default 'ctx')

    Returns:
        (accuracy_vector, aux):
          - accuracy_vector: (L,) per-latent accuracies (torch.float32)
          - aux: dict with intermediate tensors for inspection (X, labels, selectivity)
    """


    print("Starting Figure 2 recall - seed {}, noise {}".format(seed, noise_level))

    if seed is not None:
        seed_everything(seed)

    # Prepare input parameters (copy to avoid mutating caller dict)
    in_params = dict(input_parameters)
    if noise_level is not None:
        in_params["num_swaps"] = noise_level

    # Build latent space if needed
    if "latent_space" not in in_params:
        if latent_specs is None:
            raise ValueError("latent_specs must be provided when input_parameters lacks 'latent_space'.")
        in_params["latent_space"] = LatentSpace(**latent_specs)

    # Ensure we record eval_region at every step
    rec_params = dict(recording_parameters)
    regions = set(rec_params.get("regions", []))
    regions.add(eval_region)
    rec_params["regions"] = list(regions)
    rec_params["rate_activity"] = rec_params.get("rate_activity", 1)
    if rec_params["rate_activity"] != 1:
        raise ValueError("Figure 2 accuracy requires recording rate_activity == 1 for alignment.")

    # Instantiate network and run without sleep
    net = SSCNetwork(network_parameters, rec_params, model)

    # Generate input
    input_tensor, input_episodes, input_latents = make_input(**in_params)

    # Run days
    with torch.no_grad():
        num_days = in_params["num_days"]
        for day in range(num_days):
            net(input_tensor[day], debug=False)

    # Extract recordings for eval_region (only awake timesteps were executed)
    rec_list = net.activity_recordings.get(eval_region, None)
    rec_times = net.activity_recordings_time
    if rec_list is None:
        raise ValueError(f"Recording for region '{eval_region}' not found. Ensure it's in recording_parameters['regions'].")

    # Drop the initial seed entry added during init_recordings
    rec_list = rec_list[1:]

    # Stack to (T, N)
    X_output = torch.stack(net.activity_recordings["output"], dim=0)[net.awake_indices][-100*in_params["day_length"]:]
    X_episodes = F.one_hot(input_episodes[-100:].long(), num_classes=np.prod(latent_specs["dims"]))

    # Selectivity from data, then latent accuracy
    selectivity = get_selectivity(X_output, X_episodes)
    acc = get_latent_accuracy(X_output, selectivity, X_episodes.flatten(end_dim=1))

    aux = {"X": X_output, "labels": X_episodes, "selectivity": selectivity}
    
    return (acc, selectivity, aux) if get_aux_results else (acc, selectivity)


__all__ = [
    "figure2_accuracy",
]
