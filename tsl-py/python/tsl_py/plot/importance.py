"""Feature importance summary plot for fitted TSL models."""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ._common import (
    PALETTE,
    _as_array_and_names,
    _require_matplotlib,
    tsl_sequential_cmap,
    tsl_sequential_pink_cmap,
)


class FeatureImportanceResult(NamedTuple):
    fig: object
    axes: np.ndarray
    feature_names: list
    backbone_per_stage: np.ndarray    # (n_stages, n_features)
    tilt_per_stage: np.ndarray        # (n_stages, n_features)
    global_backbone: np.ndarray       # (n_features,)
    global_tilt: np.ndarray           # (n_features,)
    combined: np.ndarray              # (n_features,)
    combined_backbone: np.ndarray     # (n_features,)
    combined_tilt: np.ndarray         # (n_features,)
    stage_weights: np.ndarray         # (n_stages,)


def plot_feature_importance(
    model,
    X,
    feature_names: Optional[Sequence[str]] = None,
    gamma: float = 1.0,
    figsize: Tuple[float, float] = (14, 10),
) -> FeatureImportanceResult:
    """Summary of per-stage and global feature importance for a TSL model.

    Renders a 6-panel figure:
      1. Per-stage backbone importance (heatmap)
      2. Per-stage tilt importance (heatmap)
      3. Global backbone importance (bar)
      4. Global tilt importance (bar)
      5. Combined importance I_j = I_j^b + gamma * I_j^d (bar)
      6. Energy-based stage weights (bar)

    Parameters
    ----------
    model : TSL
    X : np.ndarray or pandas.DataFrame
        Training data used for variance estimation.
    feature_names : sequence of str, optional
    gamma : float
        Weight for tilt importance in the combined score.
    """
    plt = _require_matplotlib()
    X_arr, names = _as_array_and_names(X, feature_names)

    backbone_per_stage, tilt_per_stage = model.compute_per_stage_feature_importance(X_arr)
    global_backbone, global_tilt, stage_weights = model.compute_aggregated_feature_importance(X_arr)
    combined, combined_backbone, combined_tilt = model.compute_combined_feature_importance(
        X_arr, gamma=gamma
    )

    n_stages, n_features = backbone_per_stage.shape
    fig = plt.figure(figsize=figsize)

    title_kw = dict(
        color=PALETTE["neutral_dark"], fontweight="semibold", fontsize=11,
    )

    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(
        backbone_per_stage.T, aspect="auto",
        cmap=tsl_sequential_cmap(), interpolation="nearest",
    )
    ax1.set_xlabel("Stage")
    ax1.set_ylabel("Feature")
    ax1.set_title("Per-Stage Backbone Importance", **title_kw)
    ax1.set_xticks(range(n_stages))
    ax1.set_xticklabels([str(i + 1) for i in range(n_stages)])
    ax1.set_yticks(range(n_features))
    ax1.set_yticklabels(names, fontsize=8)
    fig.colorbar(im1, ax=ax1, label="Backbone Variance")

    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(
        tilt_per_stage.T, aspect="auto",
        cmap=tsl_sequential_pink_cmap(), interpolation="nearest",
    )
    ax2.set_xlabel("Stage")
    ax2.set_ylabel("Feature")
    ax2.set_title("Per-Stage Tilt Importance", **title_kw)
    ax2.set_xticks(range(n_stages))
    ax2.set_xticklabels([str(i + 1) for i in range(n_stages)])
    ax2.set_yticks(range(n_features))
    ax2.set_yticklabels(names, fontsize=8)
    fig.colorbar(im2, ax=ax2, label="Tilt Variance")

    bar_kw = dict(alpha=0.92, edgecolor="white", linewidth=0.6)

    ax3 = fig.add_subplot(2, 3, 3)
    order = np.argsort(global_backbone)[::-1]
    ax3.barh(range(n_features), global_backbone[order],
             color=PALETTE["backbone"], **bar_kw)
    ax3.set_yticks(range(n_features))
    ax3.set_yticklabels([names[i] for i in order], fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlabel("Global Backbone Importance")
    ax3.set_title("Aggregated Backbone Importance", **title_kw)
    ax3.grid(True, alpha=0.25, axis="x")

    ax4 = fig.add_subplot(2, 3, 4)
    order = np.argsort(global_tilt)[::-1]
    ax4.barh(range(n_features), global_tilt[order],
             color=PALETTE["neg"], **bar_kw)
    ax4.set_yticks(range(n_features))
    ax4.set_yticklabels([names[i] for i in order], fontsize=8)
    ax4.invert_yaxis()
    ax4.set_xlabel("Global Tilt Importance")
    ax4.set_title("Aggregated Tilt Importance", **title_kw)
    ax4.grid(True, alpha=0.25, axis="x")

    ax5 = fig.add_subplot(2, 3, 5)
    order = np.argsort(combined)[::-1]
    ax5.barh(range(n_features), combined[order],
             color=PALETTE["pos"], **bar_kw)
    ax5.set_yticks(range(n_features))
    ax5.set_yticklabels([names[i] for i in order], fontsize=8)
    ax5.invert_yaxis()
    ax5.set_xlabel(f"Combined Importance (γ={gamma})")
    ax5.set_title("Combined Feature Importance", **title_kw)
    ax5.grid(True, alpha=0.25, axis="x")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.bar(range(n_stages), stage_weights,
            color=PALETTE["neutral_dark"], **bar_kw)
    ax6.set_xlabel("Stage")
    ax6.set_ylabel("Weight")
    ax6.set_title("Stage Weights (Energy-based)", **title_kw)
    ax6.set_xticks(range(n_stages))
    ax6.set_xticklabels([str(i + 1) for i in range(n_stages)])
    ax6.grid(True, alpha=0.25, axis="y")

    for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    axes = np.array([ax1, ax2, ax3, ax4, ax5, ax6])
    return FeatureImportanceResult(
        fig=fig,
        axes=axes,
        feature_names=list(names),
        backbone_per_stage=backbone_per_stage,
        tilt_per_stage=tilt_per_stage,
        global_backbone=global_backbone,
        global_tilt=global_tilt,
        combined=combined,
        combined_backbone=combined_backbone,
        combined_tilt=combined_tilt,
        stage_weights=stage_weights,
    )
