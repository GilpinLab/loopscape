import numpy as np
from scipy.spatial import cKDTree
from numpy.lib.stride_tricks import sliding_window_view

def estimate_knn_ordinal_complexity(
    distances,
    values,
    k=25,
    p=1.0,
    use_ranks=False,
    value_scale=None,
):
    """
    Estimate local outcome complexity using metric-aware pairwise dispersion.

    The local statistic is the mean pairwise distance between outcome values
    within each k-nearest-neighbor neighborhood,

        D_i = mean(|y_a - y_b|**p).

    This is a sample estimator of Rao's quadratic entropy generalized to a
    distance raised to the power p. Unlike Shannon entropy, it respects the
    ordering and spacing of outcome values.

    Args:
        distances (array-like): Distance of each sample from the center.
        values (array-like): Ordinal or continuous outcome values.
        k (int, optional): Number of radial neighbors, excluding the sample
            itself.
        p (float, optional): Power applied to outcome differences. Use p=1
            for mean absolute differences or p=2 for variance-like dispersion.
        use_ranks (bool, optional): Replace values by equally spaced ranks.
            This is appropriate when values are ordinal but their numerical
            spacing is not meaningful.
        value_scale (float, optional): Scale used to normalize outcome
            differences. Defaults to the full observed range.

    Returns:
        (dict): Global complexity, local complexity, and neighborhood scales.
    """
    distances = np.asarray(distances, dtype=float)
    values = np.asarray(values)

    if distances.ndim != 1 or values.ndim != 1:
        raise ValueError("distances and values must be one-dimensional.")
    if distances.size != values.size:
        raise ValueError("distances and values must have equal length.")

    numeric_values = values.astype(float)
    valid = np.isfinite(distances) & np.isfinite(numeric_values)
    distances = distances[valid]
    numeric_values = numeric_values[valid]

    if k >= distances.size:
        raise ValueError("k must be smaller than the number of samples.")

    if use_ranks:
        _, numeric_values = np.unique(
            numeric_values,
            return_inverse=True,
        )
        numeric_values = numeric_values.astype(float)

    if value_scale is None:
        value_scale = np.ptp(numeric_values)

    if value_scale == 0:
        return {
            "complexity": 0.0,
            "local_complexity": np.zeros(distances.size),
            "neighborhood_radius": np.zeros(distances.size),
            "effective_epsilon": 0.0,
            "k": k,
            "p": p,
        }

    tree = cKDTree(distances[:, None])
    neighbor_distances, neighbor_indices = tree.query(
        distances[:, None],
        k=k + 1,
    )

    # Include the center sample and its k neighbors.
    neighborhood_values = numeric_values[neighbor_indices]
    neighborhood_radius = neighbor_distances[:, -1]

    n_local = k + 1
    row, col = np.triu_indices(n_local, k=1)

    differences = np.abs(
        neighborhood_values[:, row]
        - neighborhood_values[:, col]
    )
    local_complexity = np.mean(
        (differences / value_scale) ** p,
        axis=1,
    )

    return {
        "complexity": local_complexity.mean(),
        "local_complexity": local_complexity,
        "neighborhood_radius": neighborhood_radius,
        "effective_epsilon": np.median(neighborhood_radius),
        "k": k,
        "p": p,
        "use_ranks": use_ranks,
        "value_scale": value_scale,
    }


def ordinal_complexity_curve(
    distances,
    values,
    k_values,
    p=1.0,
    use_ranks=False,
    value_scale=None,
):
    """
    Evaluate ordinal or continuous outcome complexity across spatial scales.

    Args:
        distances (array-like): Distance of each sample from the center.
        values (array-like): Ordinal or continuous outcome values.
        k_values (sequence of int): Neighborhood sizes.
        p (float, optional): Power applied to outcome differences.
        use_ranks (bool, optional): Convert ordinal values to equally spaced
            ranks.
        value_scale (float, optional): Common normalization scale.

    Returns:
        (dict): Complexity and effective resolution for every neighborhood.
    """
    values_array = np.asarray(values, dtype=float)

    if value_scale is None:
        if use_ranks:
            n_unique = np.unique(values_array[np.isfinite(values_array)]).size
            value_scale = max(n_unique - 1, 1)
        else:
            value_scale = np.ptp(values_array[np.isfinite(values_array)])

    results = [
        estimate_knn_ordinal_complexity(
            distances,
            values,
            k=k,
            p=p,
            use_ranks=use_ranks,
            value_scale=value_scale,
        )
        for k in k_values
    ]

    return {
        "k": np.asarray(k_values),
        "epsilon": np.asarray(
            [result["effective_epsilon"] for result in results]
        ),
        "complexity": np.asarray(
            [result["complexity"] for result in results]
        ),
        "results": results,
    }


import numpy as np


def estimate_shell_ordinal_complexity(
    values,
    radii,
    n_radii,
    n_directions,
    p=1.0,
    use_ranks=False,
    value_scale=None,
    shell_weights=None,
):
    """
    Estimate angular outcome complexity on prescribed radial shells.

    Args:
        values (array-like): Ordinal or continuous values, ordered consistently
            with the shell sampler.
        radii (array-like): Radius corresponding to each value.
        n_radii (int): Number of radial shells.
        n_directions (int): Number of angular samples per shell.
        p (float, optional): Power applied to pairwise value differences.
        use_ranks (bool, optional): Replace unique values with equally spaced
            ordinal ranks.
        value_scale (float, optional): Scale used to normalize differences.
        shell_weights (array-like, optional): Weights for forming a global
            average over shells.

    Returns:
        (dict): Per-shell and global complexity estimates.
    """
    values = np.asarray(values, dtype=float)
    radii = np.asarray(radii, dtype=float)

    if use_ranks:
        _, values = np.unique(values, return_inverse=True)
        values = values.astype(float)

    values = values.reshape(n_radii, n_directions)
    radii = radii.reshape(n_radii, n_directions)[:, 0]

    if value_scale is None:
        value_scale = np.ptp(values)

    if value_scale == 0:
        shell_complexity = np.zeros(n_radii)
    else:
        i, j = np.triu_indices(n_directions, k=1)

        differences = np.abs(
            values[:, i] - values[:, j]
        )

        shell_complexity = np.mean(
            (differences / value_scale) ** p,
            axis=1,
        )

    if shell_weights is None:
        global_complexity = shell_complexity.mean()
    else:
        shell_weights = np.asarray(shell_weights, dtype=float)
        shell_weights /= shell_weights.sum()
        global_complexity = np.sum(
            shell_weights * shell_complexity
        )

    return {
        "complexity": global_complexity,
        "shell_complexity": shell_complexity,
        "shell_radii": radii,
        "value_scale": value_scale,
        "p": p,
        "use_ranks": use_ranks,
    }


def equal_count_bins(array: np.ndarray, k: int) -> np.ndarray:
    """
    Bin array values into approximately equal-population bins.

    Equal values are always assigned to the same bin. Consequently, bins may
    have unequal populations when values occur repeatedly.

    Args:
        array (np.ndarray): Input array.
        k (int): Maximum number of bins.

    Returns:
        np.ndarray: Integer bin labels with the same shape as the input.
    """
    values, inverse, counts = np.unique(
        array, return_inverse=True, return_counts=True
    )

    n_bins = min(k, values.size)
    cumulative = np.cumsum(counts)

    boundaries = []
    previous = 0

    for j in range(1, n_bins):
        target = j * array.size / n_bins

        # Each remaining bin must contain at least one unique value.
        lower = previous + 1
        upper = values.size - (n_bins - j)

        candidates = np.arange(lower, upper + 1)
        boundary = candidates[
            np.argmin(np.abs(cumulative[candidates - 1] - target))
        ]

        boundaries.append(boundary)
        previous = boundary

    labels_for_unique_values = np.searchsorted(boundaries, np.arange(values.size))
    return labels_for_unique_values[inverse].reshape(array.shape)

def basin_entropy(labels, box_size=5):
    """
    Calculate basin entropy and boundary basin entropy from a 2D label array.

    The array is partitioned into non-overlapping square boxes. Within each
    box, basin-label frequencies define the probabilities used to calculate
    the local Gibbs entropy.

    Args:
        labels (array): Two-dimensional array of basin IDs.
        box_size (int, optional): Width and height of each coarse-graining box.

    Returns:
        (dict): Dictionary containing:
            - basin_entropy: Mean entropy over all boxes.
            - boundary_basin_entropy: Mean entropy over boxes containing
              more than one basin ID.
            - total_entropy: Sum of all box entropies.
            - boundary_box_fraction: Fraction of boxes containing multiple
              basin IDs.
            - n_boxes: Total number of complete boxes.
            - n_boundary_boxes: Number of boxes containing multiple basin IDs.
            - log2_criterion: Whether boundary basin entropy exceeds log(2).
            - box_entropies: Two-dimensional array of local box entropies.

    References:
        Daza, M., Wagemakers, A., Georgeot, B., Guéry-Odelin, D., & Sanjuán, M. A. F. (2016). Basin entropy: a new tool to analyze uncertainty in dynamical systems. Scientific reports, 6
    """
    labels = np.asarray(labels)

    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array.")

    n_rows = labels.shape[0] // box_size
    n_cols = labels.shape[1] // box_size

    if n_rows == 0 or n_cols == 0:
        raise ValueError("box_size is larger than the input array.")

    # Ignore incomplete boxes along the lower and right edges.
    labels = labels[: n_rows * box_size, : n_cols * box_size]

    boxes = labels.reshape(
        n_rows,
        box_size,
        n_cols,
        box_size,
    ).transpose(0, 2, 1, 3)

    box_entropies = np.empty((n_rows, n_cols), dtype=float)
    boundary_mask = np.zeros((n_rows, n_cols), dtype=bool)

    for i in range(n_rows):
        for j in range(n_cols):
            box = boxes[i, j].ravel()
            _, counts = np.unique(box, return_counts=True)

            probabilities = counts / counts.sum()
            box_entropies[i, j] = -np.sum(
                probabilities * np.log(probabilities)
            )
            boundary_mask[i, j] = counts.size > 1

    total_entropy = box_entropies.sum()
    n_boxes = box_entropies.size
    n_boundary_boxes = boundary_mask.sum()

    basin_entropy_value = total_entropy / n_boxes

    if n_boundary_boxes:
        boundary_basin_entropy = (
            box_entropies[boundary_mask].sum() / n_boundary_boxes
        )
    else:
        boundary_basin_entropy = 0.0

    return {
        "basin_entropy": basin_entropy_value,
        "boundary_basin_entropy": boundary_basin_entropy,
        "total_entropy": total_entropy,
        "boundary_box_fraction": n_boundary_boxes / n_boxes,
        "n_boxes": n_boxes,
        "n_boundary_boxes": int(n_boundary_boxes),
        "log2_criterion": boundary_basin_entropy > np.log(2),
        "box_entropies": box_entropies,
    }

def uncertainty_exponent(labels, box_size=5):
    """
    Estimate the uncertainty exponent from a 2D basin-label array.

    For perturbation distances epsilon = 1, ..., box_size pixels, the
    uncertainty fraction is estimated as the fraction of pairs of points
    separated by epsilon that belong to different basins. Horizontal,
    vertical, and diagonal perturbations are included.

    The uncertainty exponent alpha is obtained from

        f(epsilon) ~ epsilon**alpha,

    using a linear fit in log-log space.

    Args:
        labels (array): Two-dimensional array of basin IDs.
        box_size (int, optional): Maximum perturbation distance, in pixels,
            included in the scaling fit.

    Returns:
        (dict): Dictionary containing:
            - uncertainty_exponent: Estimated uncertainty exponent alpha.
            - boundary_dimension: Estimated basin-boundary dimension,
              2 - alpha.
            - scales: Perturbation distances used in the calculation.
            - uncertainty_fractions: Fraction of uncertain pairs at each scale.
            - n_pairs: Number of point pairs tested at each scale.
            - fit_intercept: Intercept of the log-log linear fit.
            - r_squared: Coefficient of determination of the log-log fit.

    References:
        McDonald, S. W., Grebogi, C., Ott, E., & Yorke, J. A. (1985).
        Fractal basin boundaries. Physica D, 17, 125-153.

        Grebogi, C., McDonald, S. W., Ott, E., & Yorke, J. A. (1983).
        Final state sensitivity: An obstruction to predictability.
        Physics Letters A, 99, 415-418.
    """
    labels = np.asarray(labels)

    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array.")

    max_scale = min(box_size, labels.shape[0] - 1, labels.shape[1] - 1)
    scales = np.arange(1, max_scale + 1)

    uncertainty_fractions = np.empty(scales.size, dtype=float)
    n_pairs = np.empty(scales.size, dtype=int)

    for k, scale in enumerate(scales):
        comparisons = (
            (labels[:, :-scale] != labels[:, scale:]),
            (labels[:-scale, :] != labels[scale:, :]),
            (labels[:-scale, :-scale] != labels[scale:, scale:]),
            (labels[scale:, :-scale] != labels[:-scale, scale:]),
        )

        n_uncertain = sum(comparison.sum() for comparison in comparisons)
        n_total = sum(comparison.size for comparison in comparisons)

        uncertainty_fractions[k] = n_uncertain / n_total
        n_pairs[k] = n_total

    # Only positive fractions can be used in log space.
    fit_mask = uncertainty_fractions > 0

    if fit_mask.sum() < 2:
        uncertainty_exponent_value = np.nan
        fit_intercept = np.nan
        r_squared = np.nan
    else:
        x = np.log(scales[fit_mask])
        y = np.log(uncertainty_fractions[fit_mask])

        uncertainty_exponent_value, fit_intercept = np.polyfit(x, y, 1)

        y_fit = uncertainty_exponent_value * x + fit_intercept
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot else np.nan

    return {
        "uncertainty_exponent": uncertainty_exponent_value,
        "boundary_dimension": 2 - uncertainty_exponent_value,
        "scales": scales,
        "uncertainty_fractions": uncertainty_fractions,
        "n_pairs": n_pairs,
        "fit_intercept": fit_intercept,
        "r_squared": r_squared,
    }

def basin_entropy_sliding(labels, box_size=5, stride=1):
    """
    Calculate local basin entropy using sliding windows.

    Args:
        labels (array): Two-dimensional array of basin IDs.
        box_size (int, optional): Width and height of each window.
        stride (int, optional): Step between adjacent windows.

    Returns:
        (dict): Basin entropy metrics and the local entropy map.
    """
    labels = np.asarray(labels)

    windows = sliding_window_view(
        labels,
        (box_size, box_size),
    )[::stride, ::stride]

    entropies = np.empty(windows.shape[:2], dtype=float)
    boundary_mask = np.zeros(windows.shape[:2], dtype=bool)

    for index in np.ndindex(windows.shape[:2]):
        _, counts = np.unique(windows[index], return_counts=True)
        probabilities = counts / counts.sum()

        entropies[index] = -np.sum(
            probabilities * np.log(probabilities)
        )
        boundary_mask[index] = counts.size > 1

    n_boundary = boundary_mask.sum()

    return {
        "basin_entropy": entropies.mean(),
        "boundary_basin_entropy": (
            entropies[boundary_mask].mean() if n_boundary else 0.0
        ),
        "boundary_window_fraction": boundary_mask.mean(),
        "n_windows": entropies.size,
        "n_boundary_windows": int(n_boundary),
        "log2_criterion": (
            entropies[boundary_mask].mean() > np.log(2)
            if n_boundary
            else False
        ),
        "window_entropies": entropies,
    }

from scipy.ndimage import convolve

def fli_discrete(traj):
    """
    Compute the Fast Lyapunov Indicator (FLI) for a given trajectory. This function 
    estimates the FLI by convolving the trajectory with a radius-one differencing kernel
      and then computing the norm of the result.

    Args:
        traj (np.ndarray): A 3D array of shape (width, height, n_timesteps, ...) 
            representing the trajectory.
    """
    ## Flatten any extra dimensions beyond the first three
    if traj.ndim > 3:
        traj = traj.reshape(traj.shape[0], traj.shape[1], traj.shape[2], -1)

    ## Kernel for differencing
    kernel = -np.ones((3, 3), dtype=float) / 8.0
    kernel[1, 1] = 1.0

    difference = np.linalg.norm(convolve(
        traj,
        weights=kernel[:, :, None, None],
        mode="reflect",
    ), axis=-1)
    difference = np.max(difference, axis=-1)
    return difference