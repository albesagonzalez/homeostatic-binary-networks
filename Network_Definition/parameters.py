import numpy as np
import torch


network_parameters = {}

###################
#general paramerers
###################

#network regions
network_parameters["regions"] = ["hidden", "output"]

#number of pattern complete operations
network_parameters["hidden_pattern_complete_iterations"] = 10

##################
#region parameters
##################

network_parameters["hidden_num_subregions"] = 1
network_parameters["hidden_size_subregions"] = torch.tensor([100])
network_parameters["hidden_sparsity"] = torch.tensor([0.2])
network_parameters["hidden_sparsity_sleep"] = torch.tensor([0.2])
network_parameters["hidden_immature"] = False
network_parameters["hidden_b"] = 0

network_parameters["output_num_subregions"] = 1
network_parameters["output_size_subregions"] = torch.tensor([100])
network_parameters["output_sparsity"] = torch.tensor([4/100])
network_parameters["output_sparsity_sleep"] = torch.tensor([4/100])
network_parameters["output_immature"] = True
network_parameters["output_b"] = 0.7


####################
#synaptic parameters
####################

network_parameters["connectivity"] = ["hidden_hidden", "output_hidden"]
network_parameters["frozen"] = True

#hidden_hidden
#init
network_parameters["hidden_hidden_init"] = "random"
network_parameters["hidden_hidden_init_random_max"] = "pre"
network_parameters["hidden_hidden_init_max_pre"] = 1
network_parameters["hidden_hidden_init_rf_size"] = 30
network_parameters["hidden_hidden_init_std"] = 0
#learning
network_parameters["hidden_hidden_lmbda"] = 5e-4
network_parameters["max_pre_hidden_hidden"] = 1
network_parameters["max_post_hidden_hidden"] = np.inf

#output_hidden
#init
network_parameters["output_hidden_init"] = "random"
network_parameters["output_hidden_init_random_max"] = "post"
network_parameters["output_hidden_init_max_post"] = 0
network_parameters["output_hidden_init_rf_size"] = 10
network_parameters["output_hidden_init_std"] = 0.005
#learning
network_parameters["output_hidden_lmbda"] = 5e-4
network_parameters["max_pre_output_hidden"] = np.inf
network_parameters["max_post_output_hidden"] = 1