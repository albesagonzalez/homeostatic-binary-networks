import numpy as np
import torch


def _relative_error(observed: float, expected: float, eps: float = 1e-8) -> float:
    if expected == 0:
        return abs(observed - expected)
    return abs(observed - expected) / (abs(expected) + eps)


def test_connectivity_init(
    net,
    plot: bool = False,
    show_stats: bool = False,
    tolerance: float = 0.05,
    max_heatmap: int = 200,
    crop: int = 50,
    seed: int | None = None,
):
    """
    Validate random receptive-field connectivity initialization for each connection
    listed in `net.connectivity`.

    Pass conditions (within `tolerance`):
      - If `<conn>_init_random_max == 'post'`: mean(row sums) ≈ `<conn>_init_max_post`
        and mean(row degree) ≈ `<conn>_init_rf_size`.
      - If `'pre'`: mean(col sums) ≈ `<conn>_init_max_pre` and
        mean(col degree) ≈ `<conn>_init_rf_size`.

    Args:
      net: Network instance with per-connection attributes as defined in src/model.py
      plot: If True, draw simple histograms and heatmaps per connection
      show_stats: If True, print summary statistics and deviations
      tolerance: Relative tolerance for expected value checks (default 5%)
      max_heatmap: Max dim for full heatmap (else cropped)
      crop: Crop size for large matrices
      seed: Optional seed for reproducible crop selection

    Returns:
      bool: True if all applicable checks pass; False otherwise.
    """

    rng = np.random.default_rng(seed)

    conns = getattr(net, "connectivity", [])
    if not conns:
        if show_stats:
            print("No connectivity specified on network.")
        return True

    # Deferred imports for plotting to avoid heavy deps when not needed
    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

    all_pass = True

    for conn in conns:
        if not hasattr(net, conn):
            if show_stats:
                print(f"Skipping {conn}: no matrix attribute present.")
            continue

        W = getattr(net, conn)
        if isinstance(W, torch.Tensor):
            W = W.detach().cpu().numpy()

        post_size, pre_size = W.shape
        nnz = int((W != 0).sum())
        row_sums = W.sum(axis=1)
        col_sums = W.sum(axis=0)
        row_deg = (W != 0).sum(axis=1)
        col_deg = (W != 0).sum(axis=0)

        rand_flag = bool(getattr(net, f"{conn}_init_random", False))
        if not rand_flag:
            # Nothing to validate for this connection
            if show_stats:
                print(f"{conn}: random init not enabled; skipping checks.")
            # Still optionally plot/basic stats
        else:
            mode = getattr(net, f"{conn}_init_random_max", None)
            rf_size = getattr(net, f"{conn}_init_rf_size", None)
            max_post = getattr(net, f"{conn}_init_max_post", None)
            max_pre = getattr(net, f"{conn}_init_max_pre", None)

            conn_pass = True

            if mode == 'post':
                # Check row sums against max_post
                if max_post is not None:
                    err_sum = _relative_error(float(row_sums.mean()), float(max_post))
                    conn_pass &= (err_sum <= tolerance)
                    if show_stats:
                        print(f"{conn} row sums: mean={row_sums.mean():.6f}, exp={float(max_post):.6f}, rel.err={err_sum:.3%}")
                # Check row degree against rf_size
                if rf_size is not None and rf_size > 0:
                    err_deg = _relative_error(float(row_deg.mean()), float(rf_size))
                    conn_pass &= (err_deg <= tolerance)
                    if show_stats:
                        print(f"{conn} row degree: mean={row_deg.mean():.3f}, exp={int(rf_size)}, rel.err={err_deg:.3%}")

            elif mode == 'pre':
                # Check col sums against max_pre
                if max_pre is not None:
                    err_sum = _relative_error(float(col_sums.mean()), float(max_pre))
                    conn_pass &= (err_sum <= tolerance)
                    if show_stats:
                        print(f"{conn} col sums: mean={col_sums.mean():.6f}, exp={float(max_pre):.6f}, rel.err={err_sum:.3%}")
                # Check col degree against rf_size
                if rf_size is not None and rf_size > 0:
                    err_deg = _relative_error(float(col_deg.mean()), float(rf_size))
                    conn_pass &= (err_deg <= tolerance)
                    if show_stats:
                        print(f"{conn} col degree: mean={col_deg.mean():.3f}, exp={int(rf_size)}, rel.err={err_deg:.3%}")
            else:
                if show_stats:
                    print(f"{conn}: invalid or missing {conn}_init_random_max (got {mode}); skipping checks.")
                conn_pass = False

            all_pass &= conn_pass

        if show_stats:
            print(
                f"{conn} shape={post_size}x{pre_size}, nnz={nnz}, "
                f"w_mean={W.mean():.6f}, w_std={W.std():.6f}, w_min={W.min():.6f}, w_max={W.max():.6f}"
            )

        if plot:
            # Basic visuals
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.ravel()

            sns.histplot(W.ravel(), bins=50, ax=axes[0], color="#0073B7")
            axes[0].set_title(f"{conn} weights")

            sns.histplot(row_sums, bins=40, ax=axes[1], color="#00A651")
            axes[1].set_title("Row sums")

            sns.histplot(col_sums, bins=40, ax=axes[2], color="#F7941D")
            axes[2].set_title("Col sums")

            if max(post_size, pre_size) <= max_heatmap:
                sns.heatmap(W, cmap="magma", ax=axes[3])
                axes[3].set_title("Heatmap")
            else:
                r0 = rng.integers(0, max(1, post_size - crop + 1))
                c0 = rng.integers(0, max(1, pre_size - crop + 1))
                sns.heatmap(W[r0:r0+crop, c0:c0+crop], cmap="magma", ax=axes[3])
                axes[3].set_title(f"Heatmap crop {r0}:{r0+crop},{c0}:{c0+crop}")

            plt.tight_layout()
            plt.show()

    return bool(all_pass)


__all__ = [
    "test_connectivity_init",
]

