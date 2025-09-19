import random
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.model import SSCNetwork
from src.utils.general import make_input, get_selectivity, get_latent_accuracy, LatentSpace, get_sample_from_num_swaps, get_cos_sim_torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def figure3_recall(
    network_parameters: Dict[str, Any],
    model: Dict[str, Any],
    recording_parameters: Dict[str, Any],
    input_parameters: Dict[str, Any],
    latent_specs: Optional[Dict[str, Any]] = None,
    num_episodes: Optional[int] = None,
    noise_level_input: Optional[int] = None,
    noise_level_recall: Optional[int] = None,
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
    print("Starting Figure 3 recall - seed{}, noise_in{}, noise_rec{}".format(seed, noise_level_input, noise_level_recall))
    if seed is not None:
        seed_everything(seed)

    # Prepare input parameters (copy to avoid mutating caller dict)
    in_params = dict(input_parameters)
    if noise_level_input is not None:
        in_params["num_swaps"] = noise_level_input
    in_params["num_days"] = 1
    in_params["day_length"] = num_episodes*in_params["mean_duration"]

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
    input_tensor, input_episodes, input_latents = make_input(**in_params, regions=net.hidden_subregions)

    # Run days
    with torch.no_grad():
        num_days = in_params["num_days"]
        for day in range(num_days):
            net(input_tensor[day], debug=False)

    in_params["num_days"] = 1
    in_params["day_length"] = 100
    in_params["num_swaps"] = 0

    input_tensor, input_episodes, input_latents = make_input(**in_params, regions=net.hidden_subregions)
    X_input = input_tensor.flatten(end_dim=1)
    recalls = torch.zeros(X_input.shape[1])

    for t, X_input_t in enumerate(X_input):
        X_input_t_corrupted = get_sample_from_num_swaps(X_input_t, noise_level_recall)
        X_recalled = net.pattern_complete('hidden', X_input_t_corrupted)
        recalls[t] = get_cos_sim_torch(X_input_t, X_recalled)

    return recalls.mean(), recalls.std()

__all__ = [
    "figure3_recall",
]
