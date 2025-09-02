import torch
import types
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy


class SSCNetwork(nn.Module):
    """
    Abstract scaffold for a sparse, subregional cortical network.

    This class preserves the general structure (lifecycle, utilities, and
    generic plasticity/recording helpers) while leaving the concrete
    architecture unspecified. Implement the forward/sleep passes and define
    region activities/weights in a subclass or by extending this file.
    """

    def __init__(self, net_params, rec_params, net_model):
        super().__init__()
        self.init_network(net_params, net_model)
        self.init_recordings(rec_params)

    # --------------------
    # Generic utilities
    # --------------------
    def activation(self, x, region, x_conditioned=None, subregion_index=None, sleep=False, sparsity=None):
        """
        Winner-take-most activation per subregion with optional conditioning.
        Relies on attributes:
          - `<region>_sparsity`, `<region>_sparsity_sleep`
          - `<region>_subregions`
        """
        # Add tiny noise to break ties deterministically
        x = x + (1e-10 + torch.max(x) - torch.min(x)) / 100000 * torch.randn_like(x)

        if x_conditioned is not None:
            x = x.clone()
            x[x_conditioned == 1] = torch.max(x) + 1

        x_prime = torch.zeros_like(x)
        x_sparsity = getattr(self, region + '_sparsity') if not sleep else getattr(self, region + '_sparsity_sleep')
        x_sparsity = x_sparsity if sparsity is None else sparsity
        x_subregions = getattr(self, region + '_subregions')

        if sleep:
            subregional_input = [x[subregion].sum() for subregion in x_subregions]
            subregion_index = (
                torch.topk(torch.tensor(subregional_input), 1).indices.int()
                if subregion_index is None else subregion_index
            )
            subregion = x_subregions[subregion_index]
            x_subregion = torch.zeros_like(subregion).float()
            top_indices = torch.topk(x[subregion], int(len(subregion) * x_sparsity[subregion_index])).indices
            x_subregion[top_indices] = 1
            x_prime[subregion] = x_subregion
        else:
            for idx, subregion in enumerate(x_subregions):
                x_subregion = torch.zeros_like(subregion).float()
                top_indices = torch.topk(x[subregion], int(len(subregion) * x_sparsity[idx])).indices
                x_subregion[top_indices] = 1
                x_prime[subregion] = x_subregion

        return x_prime, subregion_index

    def pattern_complete(self, region, h_0=None, h_conditioned=None, subregion_index=None, sleep=False, num_iterations=None, sparsity=None):
        """
        Generic iterative pattern completion: h <- activation(W h).
        Relies on attribute `<region>_<region>` as the recurrent weight matrix.
        """
        num_iterations = num_iterations if num_iterations is not None else getattr(self, region + '_pattern_complete_iterations')
        h = h_0 if h_0 is not None else getattr(self, region)
        w = getattr(self, region + '_' + region)
        for _ in range(num_iterations):
            h, subregion_index = self.activation(F.linear(h, w), region, h_conditioned, subregion_index, sleep=sleep, sparsity=sparsity)
        return h

    def hebbian(self, post_region, pre_region):
        """
        Generic Hebbian update: W += lambda * (post ⊗ pre)
        Expects attributes:
          - `<post>_<pre>` weight matrix
          - `<post>_<pre>_lmbda` learning rate (optional; defaults to 1.0)
        """
        w_name = f"{post_region}_{pre_region}"
        if self._is_frozen(w_name):
            return
        w = getattr(self, w_name)

        lmbda = getattr(self, w_name + '_lmbda')

        if post_region != pre_region:
            IM = getattr(self, post_region + '_IM')
            IM_lmbda = getattr(self, 'max_post_' + w_name)/ torch.sum(getattr(self, pre_region))
            lmbda = lmbda*(1 - IM) + IM_lmbda*IM
            lmbda = lmbda[:, None]

        delta_w = torch.outer(getattr(self, post_region), getattr(self, pre_region))
        w = w + lmbda * delta_w
        setattr(self, w_name, w)

    def homeostasis(self, post_region, pre_region):
        """
        Caps total outgoing (row-wise) and incoming (col-wise) strength.
        Expects optional attributes:
          - `max_post_<post>_<pre>` and `max_pre_<post>_<pre>`
        If not present, the corresponding constraint is skipped.
        """
        w_name = f"{post_region}_{pre_region}"
        if self._is_frozen(w_name):
            return
        w = getattr(self, w_name)

        max_post = getattr(self, 'max_post_' + w_name, None)
        if max_post is not None and torch.isfinite(torch.tensor(max_post)):
            total_post = torch.sum(w, dim=1)
            post_exceeding_mask = total_post > max_post
            post_scaling = torch.where(
                post_exceeding_mask,
                max_post / total_post,
                torch.ones_like(total_post)
            )
            w = w * post_scaling.unsqueeze(1)

        max_pre = getattr(self, 'max_pre_' + w_name, None)
        if max_pre is not None and torch.isfinite(torch.tensor(max_pre)):
            total_pre = torch.sum(w, dim=0)
            pre_exceeding_mask = total_pre > max_pre
            pre_scaling = torch.where(
                pre_exceeding_mask,
                max_pre / total_pre,
                torch.ones_like(total_pre)
            )
            w = w * pre_scaling

        setattr(self, w_name, w)


    def replay(self, post_region, pre_region):
        #obtain presynaptic pattern from pattern completing noise
        pre_size = getattr(self, f"{pre_region}_size")
        X_pre_0 = torch.randn(pre_size)**2
        X_pre = self.pattern_complete(pre_region, h_0=X_pre_0)
        #forard presyaptic pattern to post
        IM = getattr(self, f"{post_region}_IM")
        b = getattr(self, f"{post_region}_b")
        X_post_hat = F.linear(X_pre, getattr(self, f"{post_region}_{pre_region}")) + b*IM
        X_post = self.activation(X_post_hat, post_region)[0]
        setattr(self, f"{post_region}_hat", X_post_hat)
        setattr(self, pre_region, X_pre)
        setattr(self, post_region, X_post)

        self.hebbian(post_region, pre_region)
        self.homeostasis(post_region, pre_region)

        IM = getattr(self, post_region + '_IM')
        IM[X_post==1] = 0
        IM = setattr(self, post_region + '_IM', IM)

    
    def _is_frozen(self, conn_name: str) -> bool:
        """Return True if the given connection is frozen.

        Supports legacy booleans:
          - True: all connections frozen
          - False/None: none frozen
          - list/tuple/set: only listed names are frozen
        """
        fr = getattr(self, 'frozen', [])
        if isinstance(fr, bool):
            return fr
        if fr is None:
            return False
        if isinstance(fr, (list, tuple, set)):
            return conn_name in fr
        return False

    # --------------------
    # Recording utilities
    # --------------------
    def init_recordings(self, rec_params):
        self.activity_recordings = {}
        for region in rec_params.get('regions', []):
            if hasattr(self, region):
                self.activity_recordings[region] = [getattr(self, region)]
        self.activity_recordings_rate = rec_params.get('rate_activity', 1)
        self.activity_recordings_time = []

        self.connectivity_recordings = {}
        for connection in rec_params.get('connections', []):
            if hasattr(self, connection):
                self.connectivity_recordings[connection] = [getattr(self, connection)]
        self.connectivity_recordings_rate = rec_params.get('rate_connectivity', 1)
        self.connectivity_recordings_time = []

        self.time_index = 0
        self.awake_indices = []
        self.sleep_indices = []

    def record(self):
        if self.time_index % self.activity_recordings_rate == 0:
            for region in list(self.activity_recordings.keys()):
                if hasattr(self, region):
                    layer_activity = getattr(self, region)
                    self.activity_recordings[region].append(deepcopy(layer_activity.detach().clone()))
                    self.activity_recordings_time.append(self.time_index)
        if self.time_index % self.connectivity_recordings_rate == 0:
            for connection in list(self.connectivity_recordings.keys()):
                if hasattr(self, connection):
                    connection_state = getattr(self, connection)
                    self.connectivity_recordings[connection].append(deepcopy(connection_state.detach().clone()))
                    self.connectivity_recordings_time.append(self.time_index)

    # --------------------
    # Network init (generic)
    # --------------------
    def init_network(self, net_params, net_model):
        # Keep a reference to raw params for region-specific lookups
        self.net_params = net_params

        # Set provided hyperparameters/attributes verbatim
        for key, value in net_params.items():
            setattr(self, key, value)

        # Derive per-region sizes and index partitions if `regions` is provided
        if hasattr(self, 'regions'):
            for region in self.regions:
                num_subregions = getattr(self, region + '_num_subregions')
                size_subregions = getattr(self, region + '_size_subregions')
                region_size = torch.sum(size_subregions)
                setattr(self, region + '_size', region_size)
                setattr(self, region , torch.zeros(region_size))
                setattr(self, region + '_hat', torch.zeros(region_size))

                subregions = []
                for subregion_index in range(num_subregions):
                    start = int(torch.sum(size_subregions[:subregion_index]).item())
                    end = int(torch.sum(size_subregions[:subregion_index + 1]).item())
                    subregions.append(torch.arange(start, end))
                setattr(self, region + '_subregions', subregions)

                # Initialize per-region immaturity mask (IM) and excitability (b)
                # IM: 1 if region is immature, else 0 (vector of region size)
                # b:  constant vector filled with provided bias value
                imm_key = f"{region}_immature"
                b_key = f"{region}_b"

                is_immature = bool(self.net_params.get(imm_key, False))
                b_value = float(self.net_params.get(b_key, 0.0))

                region_vec = getattr(self, region)
                im_tensor = torch.ones_like(region_vec) if is_immature else torch.zeros_like(region_vec)
                b_tensor = torch.full_like(region_vec, b_value)

                setattr(self, region + '_IM', im_tensor)
                setattr(self, region + '_b', b_tensor)

        # Global state flags
        # `frozen` may be provided as:
        # - False/None: no connections frozen
        # - True: all connections frozen (legacy behavior)
        # - list/tuple/set of connection names (e.g., ["ctx_ctx", "mtl_mtl"]) to freeze selectively
        frozen_val = getattr(self, 'frozen', [])
        if frozen_val is None:
            frozen_val = []
        self.frozen = frozen_val
        self.day = getattr(self, 'day', 0)

        # Attach/override methods from optional `network_model` mapping
        self.bind_network_model(net_model)

        # Connectivity initialization by spec (delegated)
        connectivity = getattr(self, 'connectivity', [])
        if connectivity:
            for conn in connectivity:
                if not isinstance(conn, str):
                    continue
                try:
                    post_region, pre_region = conn.rsplit('_', 1)
                except ValueError:
                    raise ValueError(f"Connectivity name '{conn}' must be 'post_pre'.")
                self.init_connectivity(post_region, pre_region)

    def bind_network_model(self, model_map=None):
        """Bind/refresh instance methods from a mapping of name → function.

        If `model_map` is None, tries `self.network_model`.
        Each callable is bound as a method on the instance under its key.
        """
        if model_map is None:
            model_map = getattr(self, 'network_model', None)
        if isinstance(model_map, dict):
            for name, fn in model_map.items():
                if callable(fn):
                    bound = types.MethodType(fn, self)
                    setattr(self, name, bound)
                    # Keep an internal alias to avoid recursion and allow delegation
                    setattr(self, f"_{name}_fn", bound)

    def init_connectivity(self, post_region: str, pre_region: str):
        """
        Initialize a connectivity matrix for a given (post, pre) pair using
        per-connection attributes:
          - f"{post}_{pre}_init"            → initialization type (supports 'recurrent')
          - f"{post}_{pre}_init_random"     → bool, enable random RF init
          - f"{post}_{pre}_init_random_max" → 'post' or 'pre'
          - f"{post}_{pre}_init_rf_size"    → int, number of RF connections
          - f"{post}_{pre}_init_std"        → float, Gaussian noise std
          - f"{post}_{pre}_init_max_post"   → float, total per-post cap (if random_max == 'post')
          - f"{post}_{pre}_init_max_pre"    → float, total per-pre cap (if random_max == 'pre')
        Stores tensor in attribute f"{post}_{pre}".
        """
        conn = f"{post_region}_{pre_region}"

        # Resolve sizes
        post_size_val = getattr(self, f"{post_region}_size")
        pre_size_val = getattr(self, f"{pre_region}_size")
        post_size = int(post_size_val.item()) if torch.is_tensor(post_size_val) else int(post_size_val)
        pre_size = int(pre_size_val.item()) if torch.is_tensor(pre_size_val) else int(pre_size_val)

        # Base matrix (zeros)
        w = torch.zeros((post_size, pre_size))

        # Required init type per-connection: '{post}_{pre}_init' in {'zero', 'random'}
        init_key = f"{conn}_init"
        init_type = getattr(self, init_key, None)
        if init_type is None:
            raise ValueError(f"Missing required key {init_key} for connectivity initialization.")

        if init_type == 'zero':
            # Keep as zeros
            pass
        elif init_type == 'random':
            # Required keys for random init
            mode_key = f"{conn}_init_random_max"
            rf_key = f"{conn}_init_rf_size"
            std_key = f"{conn}_init_std"

            missing = []
            mode = getattr(self, mode_key, None)
            if mode is None:
                missing.append(mode_key)
            rf_size = getattr(self, rf_key, None)
            if rf_size is None:
                missing.append(rf_key)
            std = getattr(self, std_key, None)
            if std is None:
                missing.append(std_key)

            if missing:
                raise ValueError(f"Missing keys for {conn} random init: {', '.join(missing)}")

            if mode not in {"post", "pre"}:
                raise ValueError(f"{mode_key} must be 'post' or 'pre', got {mode!r} for {conn}.")

            rf_size = int(rf_size)
            std = float(std)

            if mode == 'post':
                max_key = f"{conn}_init_max_post"
                max_post = getattr(self, max_key, None)
                if max_post is None:
                    raise ValueError(f"Missing key for {conn} random init (post): {max_key}")
                strength = float(max_post) / max(1, rf_size)
                k = min(rf_size, pre_size)
                for i in range(post_size):
                    idx = torch.randperm(pre_size)[:k]
                    w[i, idx] = strength
            else:  # mode == 'pre'
                max_key = f"{conn}_init_max_pre"
                max_pre = getattr(self, max_key, None)
                if max_pre is None:
                    raise ValueError(f"Missing key for {conn} random init (pre): {max_key}")
                strength = float(max_pre) / max(1, rf_size)
                k = min(rf_size, post_size)
                for j in range(pre_size):
                    idx = torch.randperm(post_size)[:k]
                    w[idx, j] = strength

            if std > 0:
                w = w + torch.randn_like(w) * std
        else:
            raise NotImplementedError(f"Unsupported {init_key}={init_type!r} for {conn}; use 'zero' or 'random'.")

        setattr(self, conn, w)
