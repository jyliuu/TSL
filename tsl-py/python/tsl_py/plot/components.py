"""Plot one-dimensional components of GridTensors (raw, pre-aggregation)."""

from typing import Optional

from ._common import PALETTE, PALETTE_CYCLE, _require_matplotlib


def plot_grid_tensor_components(grid_tensor, individual_plots: bool = False, axis: Optional[int] = None):
    """Plot one-dimensional components of a GridTensor.

    Parameters
    ----------
    grid_tensor : GridTensor
        A fitted GridTensor instance.
    individual_plots : bool, default=False
        If True, each component gets its own figure.
    axis : int, optional
        If provided, only plot the component for this feature index.
    """
    plt = _require_matplotlib()

    n_components = len(grid_tensor.intervals)
    colors = [PALETTE_CYCLE[i % len(PALETTE_CYCLE)] for i in range(n_components)]

    if axis is not None:
        if not 0 <= axis < n_components:
            raise ValueError(f"axis must be between 0 and {n_components - 1}")
        axes_to_plot = [(axis, (grid_tensor.intervals[axis], grid_tensor.mean_factor[axis]))]
    else:
        axes_to_plot = enumerate(zip(grid_tensor.intervals, grid_tensor.mean_factor))

    if not individual_plots:
        plt.figure(figsize=(10, 6))

    for axis_idx, (intervals, values) in axes_to_plot:
        if individual_plots:
            plt.figure(figsize=(10, 6))

        color = colors[axis_idx]
        x_points = []
        y_points = []

        for (x_start, x_end), y in zip(intervals, values):
            if x_start == float("-inf") or x_end == float("inf"):
                continue
            x_points.extend([x_start, x_end])
            y_points.extend([y, y])

        if x_points:
            plt.step(x_points, y_points, where="post", lw=1.6, color=color, label=f"Axis {axis_idx}")

        if individual_plots:
            plt.xlabel("X-axis")
            plt.ylabel("Value")
            plt.title(f"GridTensor Component for Axis {axis_idx}, Scaling: {grid_tensor.scaling}",
                      color=PALETTE["neutral_dark"], fontweight="semibold")
            plt.grid(True, alpha=0.25)
            plt.legend()

    if not individual_plots:
        plt.xlabel("X-axis")
        plt.ylabel("Value")
        plt.title(f"GridTensor One-Dimensional Components, Scaling: {grid_tensor.scaling}",
                  color=PALETTE["neutral_dark"], fontweight="semibold")
        plt.grid(True, alpha=0.25)
        plt.legend()


def plot_combined_grid_tensors(model, individual_plots: bool = True, axis: Optional[int] = None):
    """Plot combined grid-tensor components for each stage of a TSL model."""
    for tgf in model.stage_predictors:
        plot_grid_tensor_components(tgf.combined_grid_tensor, individual_plots=individual_plots, axis=axis)


def plot_epoch_components(model, epoch: int) -> None:
    """Plot all per-tree grid components for a given stage.

    Parameters
    ----------
    model : TSL
        A fitted TSL instance.
    epoch : int
        Zero-based stage index to visualize.
    """
    plt = _require_matplotlib()

    families = model.stage_predictors
    total_epochs = len(families)

    if epoch < 0 or epoch >= total_epochs:
        raise ValueError(f"epoch must be between 0 and {total_epochs - 1}")

    epoch_grid_tensors = families[epoch].grid_tensors

    if len(epoch_grid_tensors) == 0:
        raise ValueError(f"No tree grids found for epoch {epoch}")

    num_components = len(epoch_grid_tensors[0].intervals)
    colors = PALETTE_CYCLE

    for component_index in range(num_components):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        for grid_index, grid in enumerate(epoch_grid_tensors):
            intervals = grid.intervals[component_index]
            values = grid.mean_factor[component_index]

            x_points = []
            y_points = []
            for (x_start, x_end), y in zip(intervals, values):
                if x_start == float("-inf") or x_end == float("inf"):
                    continue
                x_points.extend([x_start, x_end])
                y_points.extend([y, y])

            if x_points:
                ax.step(
                    x_points,
                    y_points,
                    where="post",
                    lw=1.4,
                    color=colors[grid_index % len(colors)],
                    label=f"Grid {grid_index}",
                )

        ax.set_xlabel("X")
        ax.set_ylabel("Value")
        ax.set_title(f"Stage {epoch} — Component {component_index}",
                     color=PALETTE["neutral_dark"], fontweight="semibold")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
