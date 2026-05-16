"""Local interpretation plot with intercept-absorbed backbone and tilt.

Replicates the "Backbone x Tilt" three-column panel from the paper's
illustrative redesign: per-stage rows showing (net contribution / unsigned
backbone share / signed tilt per axis), with the constant intercept axis
(b_0, d_0) treated as a zeroth "feature" so it appears in both the backbone
composition and the tilt-direction columns.

Math (intercept absorption):
    lam_+ = b_0 * exp(+d_0)        b_0 = sqrt(lam_+ * lam_-)
    lam_- = b_0 * exp(-d_0)        d_0 = 0.5 * log(lam_+ / lam_-)
The "effective" lam_+- absorb the per-stage OLS scaling coefficients:
    eff_lam_+ = scaling_plus  * lambda_plus
    eff_lam_- = scaling_minus * lambda_minus
The full stage prediction is then
    m^(l)(x) = 2 * b^(l)(x) * sinh(D^(l)(x))
with b(x) = b_0 * prod_j b_j(x_j) and D(x) = d_0 + sum_j d_j(x_j).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional, Sequence

import numpy as np

from ._common import PALETTE, _require_matplotlib

_logger = logging.getLogger(__name__)

INTERCEPT_LABEL = "Intercept"


class LocalExplanation(NamedTuple):
    """Per-stage decomposition including the constant intercept axis (j=0)."""

    stage_contributions: np.ndarray         # (n_stages,)
    f_plus_contributions: np.ndarray        # (n_stages,)  scaling_plus  * f+
    f_minus_contributions: np.ndarray       # (n_stages,) -scaling_minus * f-
    backbone_magnitudes: np.ndarray         # (n_stages,)  prod_j b_j(x_j) over j=1..p
    tilt_sums: np.ndarray                   # (n_stages,)  sum_j d_j(x_j) over j=1..p
    feature_backbone: np.ndarray            # (n_stages, n_features)
    feature_tilt: np.ndarray                # (n_stages, n_features)
    intercept_backbone: np.ndarray          # (n_stages,)  b_0 = sqrt(eff_lam_+ * eff_lam_-)
    intercept_tilt: np.ndarray              # (n_stages,)  d_0 = 0.5 * log(eff_lam_+ / eff_lam_-)
    total_prediction: float


def compute_local_explanation(
    model, x: np.ndarray
) -> LocalExplanation:
    """Per-stage decomposition of a TSL prediction for a single point `x`.

    For each stage, returns the f+/f- contributions, the per-feature
    backbone and tilt values, and the intercept (b_0, d_0) that absorbs
    scaling_plus * lambda_plus and scaling_minus * lambda_minus.  With the
    intercept treated as axis j=0, every stage satisfies
    m^(l)(x) = 2 * b^(l)(x) * sinh(D^(l)(x))  where
    b(x) = prod_{j=0..p} b_j and D(x) = sum_{j=0..p} d_j.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n_features = x.size
    stage_predictors = model.stage_predictors
    n_stages = len(stage_predictors)

    stage_contrib = np.zeros(n_stages)
    f_plus_contrib = np.zeros(n_stages)
    f_minus_contrib = np.zeros(n_stages)
    backbone_mag = np.zeros(n_stages)
    tilt_sum_arr = np.zeros(n_stages)
    feat_b = np.zeros((n_stages, n_features))
    feat_d = np.zeros((n_stages, n_features))
    intercept_b = np.zeros(n_stages)
    intercept_d = np.zeros(n_stages)

    for s, sp in enumerate(stage_predictors):
        gt = sp.combined_grid_tensor
        lam_plus = float(gt.lambda_plus)
        lam_minus = float(gt.lambda_minus)
        scaling_plus = sp.scaling_plus if sp.scaling_plus is not None else 1.0
        scaling_minus = sp.scaling_minus if sp.scaling_minus is not None else 0.0

        backbone_per_feature = np.zeros(n_features)
        tilt_per_feature = np.zeros(n_features)
        for j in range(n_features):
            bvals = np.asarray(gt.backbone_values[j], dtype=np.float64)
            dvals = np.asarray(gt.tilt_values[j], dtype=np.float64)
            splits = np.asarray(gt.splits[j], dtype=np.float64)
            if splits.size == 0:
                bin_idx = 0
            else:
                bin_idx = int(np.searchsorted(splits, x[j], side="right"))
                bin_idx = min(bin_idx, bvals.size - 1)
            backbone_per_feature[j] = bvals[bin_idx]
            tilt_per_feature[j] = dvals[bin_idx]

        b_mag = float(np.prod(backbone_per_feature))
        d_sum = float(np.sum(tilt_per_feature))
        f_plus = lam_plus * b_mag * np.exp(d_sum)
        f_minus = lam_minus * b_mag * np.exp(-d_sum)
        fp = scaling_plus * f_plus
        fm = -scaling_minus * f_minus

        eff_lam_plus = scaling_plus * lam_plus
        eff_lam_minus = scaling_minus * lam_minus
        product = eff_lam_plus * eff_lam_minus
        if product > 0 and eff_lam_minus > 0:
            b0 = float(np.sqrt(product))
            d0 = 0.5 * float(np.log(eff_lam_plus / eff_lam_minus))
        else:
            b0 = float(np.sqrt(abs(product)))
            d0 = 0.0

        stage_contrib[s] = fp + fm
        f_plus_contrib[s] = fp
        f_minus_contrib[s] = fm
        backbone_mag[s] = b_mag
        tilt_sum_arr[s] = d_sum
        feat_b[s] = backbone_per_feature
        feat_d[s] = tilt_per_feature
        intercept_b[s] = b0
        intercept_d[s] = d0

    return LocalExplanation(
        stage_contributions=stage_contrib,
        f_plus_contributions=f_plus_contrib,
        f_minus_contributions=f_minus_contrib,
        backbone_magnitudes=backbone_mag,
        tilt_sums=tilt_sum_arr,
        feature_backbone=feat_b,
        feature_tilt=feat_d,
        intercept_backbone=intercept_b,
        intercept_tilt=intercept_d,
        total_prediction=float(stage_contrib.sum()),
    )


def _format_money(value: float) -> str:
    sign = "-" if value < 0 else "+"
    return f"{sign}{abs(value):,.0f}"


def _axes_backbone_share(bb_axis: np.ndarray) -> tuple:
    """Return (sorted_indices, percentages) using |log b_j| over all axes."""
    logs = np.zeros_like(bb_axis, dtype=np.float64)
    for j, bv in enumerate(bb_axis):
        if bv > 1e-15:
            v = abs(np.log(bv))
            if v > 1e-4:
                logs[j] = v
    total = logs.sum()
    if total <= 0:
        return [], []
    order = np.argsort(-logs)
    order = [int(i) for i in order if logs[i] > 0]
    return order, [logs[i] / total for i in order]


def plot_local_interpretation(
    explanations: List[LocalExplanation],
    points: List[np.ndarray],
    titles: List[str],
    feature_names: Sequence[str],
    save_path: Path,
    top_k_features: int = 3,
    point_value_formatter: Optional[Callable[[Sequence[str], np.ndarray], str]] = None,
    units_label: str = "Contribution to prediction",
    prediction_format: Callable[[float], str] = lambda v: f"{v:,.0f}",
    header: bool = True,
) -> object:
    """Three-column "Backbone x Tilt" local-interpretation plot.

    Each panel renders one local point with rows = stages (sorted by absolute
    net contribution, descending). The three columns are:

        1. Net stage contribution    — signed bar
        2. Backbone / activation     — stacked unsigned segments (top-k + Other)
        3. Tilt / signed local effect — signed bars per top-k axis

    The constant intercept axis (b_0, d_0) is treated as axis j=0 and is
    eligible to appear in both the backbone and tilt columns.

    When `header=False`, the per-panel header (title block, local-point
    values box, and the inset $\\sinh$ sparkline) is suppressed entirely.
    """
    plt = _require_matplotlib()
    import matplotlib.gridspec as gridspec
    from matplotlib.offsetbox import (
        AnchoredOffsetbox,
        TextArea,
        VPacker,
    )

    feat_names = list(feature_names)
    axis_labels = [INTERCEPT_LABEL] + feat_names

    color_pos = PALETTE["pos"]
    color_neg = PALETTE["neg"]
    color_bb = PALETTE["backbone"]
    color_other = PALETTE["other"]
    color_title_main = PALETTE["neutral_dark"]
    color_title_bb = PALETTE["backbone_dark"]
    color_title_pred = PALETTE["pos_dark"]

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.labelweight": "regular",
        "axes.edgecolor": "#666",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.color": "#444",
        "ytick.color": "#444",
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#bbb",
        "legend.fontsize": 8,
        "mathtext.fontset": "cm",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.prop_cycle": plt.cycler(color=[color_pos, color_neg, color_bb]),
    }
    _saved_rc = {k: plt.rcParams[k] for k in rc}
    plt.rcParams.update(rc)
    n_panels = len(explanations)
    max_stages = max(len(e.stage_contributions) for e in explanations)
    row_h = 0.85
    # Reserve fixed inches for the header (title + values box) so the
    # values box doesn't get squeezed when n_stages is small.
    header_inches = 1.30 if header else 0.0
    sub_inches = 0.35
    data_inches = max_stages * row_h
    panel_h = header_inches + sub_inches + data_inches
    gap_inches = 0.15
    fig_h = panel_h * n_panels + gap_inches * max(n_panels - 1, 0)
    fig = plt.figure(figsize=(18, fig_h))
    # hspace is a fraction of the average panel height — convert from a fixed
    # inch target so the gap stays small regardless of n_stages per panel.
    outer = gridspec.GridSpec(
        n_panels, 1, figure=fig,
        hspace=gap_inches / panel_h,
        top=0.97, bottom=0.03, left=0.04, right=0.98,
    )

    sinh_anchors: list = []  # (values_box, color) for post-draw sinh placement

    for panel_idx, (expl, point, title) in enumerate(zip(explanations, points, titles)):
        stage_contribs = np.asarray(expl.stage_contributions)
        n_stages = len(stage_contribs)
        order = np.argsort(-np.abs(stage_contribs))

        # Per-panel layout (rows):
        #   0: header — title (cols 0..1) + local point values (col 2)
        #   1: column subtitles
        #   2: data axes (3 columns)
        # The panel splits into 3 rows: header (fixed inches), column
        # subtitles (fixed inches), data axes (variable).  Sizing this way
        # keeps the header content-sized regardless of n_stages.
        panel_total = header_inches + sub_inches + data_inches
        if header:
            inner = gridspec.GridSpecFromSubplotSpec(
                3, 3, subplot_spec=outer[panel_idx],
                width_ratios=[1.0, 1.55, 1.4],
                height_ratios=[header_inches, sub_inches, data_inches],
                wspace=0.18, hspace=0.0,
            )
            ax_header = fig.add_subplot(inner[0, :])
            ax_header.axis("off")
            sub_row, data_row = 1, 2
        else:
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 3, subplot_spec=outer[panel_idx],
                width_ratios=[1.0, 1.55, 1.4],
                height_ratios=[sub_inches, data_inches],
                wspace=0.18, hspace=0.0,
            )
            ax_header = None
            sub_row, data_row = 0, 1

        ax_sub_net = fig.add_subplot(inner[sub_row, 0])
        ax_sub_bb = fig.add_subplot(inner[sub_row, 1])
        ax_sub_tilt = fig.add_subplot(inner[sub_row, 2])
        for ax in (ax_sub_net, ax_sub_bb, ax_sub_tilt):
            ax.axis("off")
        ax_net = fig.add_subplot(inner[data_row, 0])
        ax_bb = fig.add_subplot(inner[data_row, 1], sharey=ax_net)
        ax_tilt = fig.add_subplot(inner[data_row, 2], sharey=ax_net)

        bar_h = 0.65

        # ---- Header (flex-style via offsetbox) ---------------------------
        if header:
            # Left side: title block (title + subtitle stacked tight via VPacker)
            title_pack = VPacker(
                children=[
                    TextArea(
                        f"{title} — Local Interpretation with Backbone and Tilt",
                        textprops=dict(fontsize=14, fontweight="bold", color="#111"),
                    ),
                    TextArea(
                        "Stage contribution $= 2 \\cdot b(x) \\cdot \\sinh(D(x))$",
                        textprops=dict(fontsize=10, style="italic", color="#555"),
                    ),
                    TextArea(
                        "Backbone  $b(x) = b_0 \\prod_j b_j(x_j)$  —  "
                        "$b_j$ is the per-feature activation,  $b_0$ the intercept scale",
                        textprops=dict(fontsize=9, color="#666"),
                    ),
                    TextArea(
                        "Tilt sum  $D(x) = d_0 + \\sum_j d_j(x_j)$  —  "
                        "$d_j$ is the per-feature signed local effect,  $d_0$ the intercept tilt",
                        textprops=dict(fontsize=9, color="#666"),
                    ),
                ],
                pad=0, sep=3, align="left",
            )
            title_box = AnchoredOffsetbox(
                loc="lower left", child=title_pack,
                pad=0.0, borderpad=0.0, frameon=False,
                bbox_to_anchor=(0.0, 0.22), bbox_transform=ax_header.transAxes,
            )
            ax_header.add_artist(title_box)

            # Right side: local point values, label + values stacked tight
            if point_value_formatter is not None:
                values_text = point_value_formatter(feat_names, point)
            else:
                values_text = "\n".join(
                    f"{n}: {v:.2f}" for n, v in zip(feat_names, point)
                )
            values_label = TextArea(
                "Local point values",
                textprops=dict(fontsize=10, fontweight="bold",
                               color=color_title_main),
            )
            values_body = TextArea(
                values_text,
                textprops=dict(fontsize=10, family="monospace", color="#222"),
            )
            values_pack = VPacker(
                children=[values_label, values_body],
                pad=0, sep=4, align="left",
            )
            values_box = AnchoredOffsetbox(
                loc="lower right", child=values_pack,
                pad=0.4, borderpad=0.0, frameon=True,
                bbox_to_anchor=(1.0, 0.22), bbox_transform=ax_header.transAxes,
            )
            values_box.patch.set_boxstyle("round,pad=0.4")
            values_box.patch.set_facecolor("#f1f5f9")
            values_box.patch.set_edgecolor(color_title_main)
            values_box.patch.set_linewidth(1.0)
            ax_header.add_artist(values_box)
            sinh_anchors.append(values_box)

        # ---- Column subtitles (single line per column) -------------------
        ax_sub_net.text(
            0.5, 0.5,
            "Stage contribution (each bar is one stage's signed prediction)",
            ha="center", va="center", fontsize=11, color=color_title_main,
            fontweight="bold", transform=ax_sub_net.transAxes,
        )
        ax_sub_bb.text(
            0.5, 0.5,
            "Relative backbone contribution (each segment represents relative impact on prediction)",
            ha="center", va="center", fontsize=11, color=color_title_bb,
            fontweight="bold", transform=ax_sub_bb.transAxes,
        )
        ax_sub_tilt.text(
            0.5, 0.5,
            "Signed tilt (each bar is one feature's signed local effect)",
            ha="center", va="center", fontsize=11, color=color_title_main,
            fontweight="bold", transform=ax_sub_tilt.transAxes,
        )

        # ---- Column 1: stage contribution as a waterfall ----------------
        # Bars are cumulative: stage i extends from cumulative[i] to
        # cumulative[i+1]. Sum lands on total_prediction; show that final
        # x-value as a green-boxed tick.
        ordered_contribs = stage_contribs[order]
        cumulative = np.zeros(n_stages + 1)
        cumulative[1:] = np.cumsum(ordered_contribs)
        total = float(cumulative[-1])

        x_min = float(np.min(cumulative))
        x_max = float(np.max(cumulative))
        # Always include zero in the visible range so the baseline is shown.
        x_min = min(x_min, 0.0)
        x_max = max(x_max, 0.0)
        x_span = max(x_max - x_min, 1.0)
        x_lo = x_min - 0.05 * x_span
        x_hi = x_max + 0.18 * x_span  # extra room on the right for the final tick

        for row in range(n_stages):
            start = cumulative[row]
            width = ordered_contribs[row]
            color = color_pos if width >= 0 else color_neg
            ax_net.barh(row, width, height=bar_h, left=start,
                        color=color, alpha=0.88, edgecolor="white",
                        linewidth=0.6, zorder=2)
            # Annotate the stage's signed value near the bar tip.
            end = cumulative[row + 1]
            if width >= 0:
                tx, ha = end + 0.01 * x_span, "left"
            else:
                tx, ha = end - 0.01 * x_span, "right"
            ax_net.text(tx, row, _format_money(width),
                        ha=ha, va="center", fontsize=9.5, color=color,
                        fontweight="bold", zorder=3)
            # Dashed waterfall connector to the next stage's start.
            if row < n_stages - 1:
                ax_net.plot(
                    [end, end], [row + bar_h / 2, row + 1 - bar_h / 2],
                    color="#888", linestyle="--", linewidth=0.8,
                    alpha=0.7, zorder=1,
                )

        ax_net.axvline(0, color="#444", lw=0.9, alpha=0.6, zorder=1)
        ax_net.set_xlim(x_lo, x_hi)
        ax_net.set_ylim(n_stages - 0.5, -0.5)
        ax_net.set_yticks(np.arange(n_stages))
        ax_net.set_yticklabels([f"Stage {int(i) + 1}" for i in order], fontsize=9)
        ax_net.set_xlabel(units_label, fontsize=9)
        ax_net.grid(True, axis="x", alpha=0.2, zorder=0)
        for spine in ("top", "right"):
            ax_net.spines[spine].set_visible(False)

        # Green-boxed final-prediction tick at x = total.  Drop any default
        # tick that would visually collide with the boxed label.
        default_ticks = list(ax_net.get_xticks())
        view_span = x_hi - x_lo
        # Boxed currency label (e.g. "$149,491") is roughly ~14% of x-range
        # wide; reserve a margin slightly larger than half-width on each side.
        collision_margin = view_span * 0.14
        ticks = [t for t in default_ticks if abs(t - total) > collision_margin]
        ticks = sorted(ticks + [total])
        labels = [
            (prediction_format(total) if abs(t - total) < 1e-9 else f"{t:,.0f}")
            for t in ticks
        ]
        ax_net.set_xticks(ticks)
        ax_net.set_xticklabels(labels)
        for tick, lab in zip(ax_net.get_xticks(), ax_net.get_xticklabels()):
            if abs(tick - total) < 1e-9:
                lab.set_color("white")
                lab.set_fontweight("bold")
                lab.set_bbox(dict(
                    boxstyle="round,pad=0.3", facecolor=color_title_pred,
                    edgecolor=color_title_pred, linewidth=1.4,
                ))
                lab.set_clip_on(False)
        # Vertical guide at the final cumulative value.
        ax_net.axvline(total, color=color_title_pred, lw=1.0,
                       alpha=0.6, linestyle=":", zorder=1)

        # ---- Column 2: unsigned backbone share ---------------------------
        ax_bb.set_xlim(0, 1)
        ax_bb.tick_params(axis="y", length=0, labelleft=False)
        ax_bb.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_bb.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
        for spine in ("top", "right"):
            ax_bb.spines[spine].set_visible(False)

        for row, s_idx in enumerate(order):
            # Backbone share uses only the per-feature backbones (j=1..p).
            # The intercept b_0 carries the absolute scale of the prediction
            # and would dominate |log b_j| if included, so it is shown only
            # in the tilt column.
            bb_features = expl.feature_backbone[s_idx]
            order_feat, pcts = _axes_backbone_share(bb_features)
            # Grow the explicit-segment count until the residual "Other" is
            # under 10% (or until every contributing axis is included).
            k = 0
            cum = 0.0
            for k, p in enumerate(pcts, start=1):
                cum += p
                if 1.0 - cum < 0.10 - 1e-9:
                    break
            top_idx = [j + 1 for j in order_feat[:k]]
            top_pct = pcts[:k]
            tail_pct = max(0.0, 1.0 - sum(top_pct))

            left = 0.0
            for seg_i, (j, pct) in enumerate(zip(top_idx, top_pct)):
                alpha = max(0.45, 0.95 - seg_i * 0.18)
                ax_bb.barh(row, pct, height=bar_h, left=left,
                           color=color_bb, alpha=alpha,
                           edgecolor="white", linewidth=0.8, zorder=2)
                label = axis_labels[j]
                if pct >= 0.06:
                    ax_bb.text(
                        left + pct / 2, row - 0.10, label,
                        ha="center", va="center", fontsize=8.5,
                        color="white", fontweight="bold", zorder=3,
                    )
                    ax_bb.text(
                        left + pct / 2, row + 0.13, f"{pct * 100:.0f}%",
                        ha="center", va="center", fontsize=8.5,
                        color="white", zorder=3,
                    )
                left += pct
            if tail_pct > 1e-6:
                ax_bb.barh(row, tail_pct, height=bar_h, left=left,
                           color=color_other, alpha=0.95,
                           edgecolor="white", linewidth=0.8, zorder=2)
                if tail_pct >= 0.06:
                    ax_bb.text(
                        left + tail_pct / 2, row - 0.10, "Other",
                        ha="center", va="center", fontsize=8.5,
                        color="#404040", fontweight="bold", zorder=3,
                    )
                    ax_bb.text(
                        left + tail_pct / 2, row + 0.13, f"{tail_pct * 100:.0f}%",
                        ha="center", va="center", fontsize=8.5,
                        color="#404040", zorder=3,
                    )

        # ---- Column 3: signed tilt per axis ------------------------------
        # Build per-row top-k tilt selections, then choose a global scale
        # from feature tilts (excluding the intercept) so a one-sided stage
        # (where d_0 absorbs all of log(lam_+/lam_-)) doesn't dominate.
        per_row_tilts: list = []
        feature_only_mags: list = []
        for s_idx in order:
            tilt_axis = np.concatenate(
                [[expl.intercept_tilt[s_idx]], expl.feature_tilt[s_idx]]
            )
            mag = np.abs(tilt_axis)
            top = [int(j) for j in np.argsort(-mag) if mag[j] > 1e-12][:top_k_features]
            per_row_tilts.append(top)
            for j in top:
                if j != 0:
                    feature_only_mags.append(mag[j])

        global_scale = max(feature_only_mags) if feature_only_mags else 1.0
        tilt_pad = max(global_scale * 1.30, 1e-6)

        ax_tilt.set_xlim(-tilt_pad, tilt_pad)
        ax_tilt.tick_params(axis="y", length=0, labelleft=False)
        ax_tilt.axvline(0, color="black", lw=0.9, alpha=0.5)
        ax_tilt.set_xlabel("Signed local effect (tilt $d_j$)", fontsize=9)
        ax_tilt.grid(True, axis="x", alpha=0.2, zorder=0)
        for spine in ("top", "right"):
            ax_tilt.spines[spine].set_visible(False)

        for row, s_idx in enumerate(order):
            tilt_axis = np.concatenate(
                [[expl.intercept_tilt[s_idx]], expl.feature_tilt[s_idx]]
            )
            top = per_row_tilts[row]
            # Fix the sub-bar thickness at top_k_features so stages with
            # fewer active tilts render thin bars (matching the per-feature
            # height in fully-populated rows), not one wide bar.
            n_sub = max(top_k_features, 1)
            sub_height = bar_h / n_sub
            for k, j in enumerate(top):
                y = row - bar_h / 2 + sub_height * (k + 0.5)
                raw = float(tilt_axis[j])
                color = color_pos if raw >= 0 else color_neg
                clipped = max(min(raw, tilt_pad * 0.96), -tilt_pad * 0.96)
                ax_tilt.barh(y, clipped, height=sub_height * 0.85, color=color,
                             alpha=0.88, edgecolor="white", linewidth=0.4,
                             zorder=2)
                # Axis label on the far left of the panel column.
                ax_tilt.text(
                    -tilt_pad * 0.98, y, axis_labels[j],
                    ha="left", va="center", fontsize=8.5, color="#222",
                    zorder=3,
                )
                # Raw numeric annotation at the bar tip.
                tx = clipped + (0.02 * tilt_pad if raw >= 0 else -0.02 * tilt_pad)
                ha = "left" if raw >= 0 else "right"
                # Off-scale values (typical for a one-sided stage where d_0
                # is large) are emphasised by rendering the numeric label in
                # bold instead of using a chevron prefix.  The numeric label
                # gets a small opaque white pill so it stays readable when
                # the bar tip lands over the feature-name label on the left.
                ax_tilt.text(
                    tx, y, f"{raw:+.2f}",
                    ha=ha, va="center", fontsize=8.5, color=color,
                    fontweight="bold" if abs(raw) > tilt_pad else "normal",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white", edgecolor="none", alpha=0.9,
                    ),
                    zorder=4,
                )

        # (No bottom footer: local-point values live in the top-right
        # header slot, and per-column legends are intentionally omitted —
        # colour semantics are stated in the column subtitles.)

    # Sinh sparklines are placed AFTER a first draw so we can anchor them
    # immediately to the left of each (content-sized) values box.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w_in, fig_h_in = fig.get_size_inches()
    for values_box in sinh_anchors:
        ext = values_box.get_window_extent(renderer)
        inv = fig.transFigure.inverted()
        (fbx0, fby0) = inv.transform((ext.x0, ext.y0))
        (_, fby1) = inv.transform((ext.x1, ext.y1))
        box_h_fig = fby1 - fby0
        sinh_h_in = box_h_fig * fig_h_in
        sinh_w_in = sinh_h_in  # square in inches
        sinh_w_fig = sinh_w_in / fig_w_in
        sinh_h_fig = box_h_fig
        gap_fig_x = 0.20 / fig_w_in
        sinh_x = fbx0 - sinh_w_fig - gap_fig_x
        sinh_y = fby0
        ax_sinh = fig.add_axes([sinh_x, sinh_y, sinh_w_fig, sinh_h_fig])
        sinh_xs = np.linspace(-1.0, 1.0, 100)
        ax_sinh.plot(sinh_xs, np.sinh(sinh_xs),
                     color=color_title_main, lw=1.2)
        ax_sinh.axhline(0, color="#aaa", lw=0.4)
        ax_sinh.axvline(0, color="#aaa", lw=0.4)
        ax_sinh.set_xlim(-1.05, 1.05)
        ax_sinh.set_ylim(-1.25, 1.25)
        ax_sinh.set_xticks([-1, 0, 1])
        ax_sinh.set_yticks([-1, 0, 1])
        ax_sinh.tick_params(axis="both", labelsize=6, length=2, pad=1)
        for spine in ("top", "right"):
            ax_sinh.spines[spine].set_visible(False)
        ax_sinh.spines["left"].set_linewidth(0.5)
        ax_sinh.spines["bottom"].set_linewidth(0.5)
        ax_sinh.set_title("$\\sinh$", fontsize=8,
                          color=color_title_main, pad=2)

    fig.savefig(save_path, bbox_inches="tight")
    _logger.info("wrote %s", save_path)
    plt.rcParams.update(_saved_rc)
    return fig
