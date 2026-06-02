"""Flat visual theme for the tsl_py.plot diagnostics.

Solid material colours, hairline borders, a faint dot-grid, monospace labels,
and a ``plot_…()`` tag on each card. The whole look is driven by the ``TOKENS``
dict so it re-skins from one place. Public plot functions opt into it; nothing
here is part of the package's public API.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np

from ._common import _require_matplotlib

# --------------------------------------------------------------- TOKENS -----
TOKENS = dict(
    bg="#FFFFFF", card="#FFFFFF", border="#E4E4E7", divider="#EFEFF1",
    dot="#E9E9EE", grid="#F1F1F3",
    ink="#18181B", muted="#71717A", faint="#B4B4BB",
    accent="#4F46E5",                  # solid indigo, primary
    neg="#2563EB", pos="#F97316",      # solid blue / orange for signed
    greys=["#C7C7CE", "#A8A8B0", "#8A8A93"],
    radius_px=7, card_lw=0.9, line_w=2.2, base_w=1.4,
)

FONT_FILES = ("Geist-Light.ttf", "Geist-SemiBold.ttf", "GeistMono-Light.ttf")
_DISPLAY, _MONO = "Geist", "Geist Mono"


# ----------------------------------------------------------------- colour ---
def _rgb(hex_c: str) -> np.ndarray:
    hex_c = hex_c.lstrip("#")
    return np.array([int(hex_c[i:i + 2], 16) for i in (0, 2, 4)], float) / 255


def mix(hex_c: str, w: float = 0.0):
    """Solid blend of ``hex_c`` toward white by ``w`` (stays fully opaque)."""
    c = _rgb(hex_c) * (1 - w) + w
    return tuple(c)


def flat_diverging_cmap(name: str = "flat_div"):
    """Blue → pale → orange. For signed PD / tilt surfaces, anchored at zero."""
    _require_matplotlib()
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        name, ["#2563EB", "#9DBDFB", "#F4F4F5", "#FDC089", "#F97316"]
    )


def flat_backbone_cmap(name: str = "flat_backbone"):
    """Pale → indigo. For unsigned backbone-magnitude heatmaps."""
    _require_matplotlib()
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        name, ["#F4F4F5", "#A5A0F0", "#4F46E5", "#312E81"]
    )


def flat_tilt_cmap(name: str = "flat_tilt"):
    """Pale → orange. For tilt-magnitude heatmaps."""
    _require_matplotlib()
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        name, ["#F4F4F5", "#FBC99A", "#F97316", "#9A3412"]
    )


def _text_on(rgba) -> str:
    """Readable label colour (white or ink) for text drawn on a cell of `rgba`."""
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#FFFFFF" if lum < 0.6 else TOKENS["ink"]


# ------------------------------------------------------------------ fonts ---
def setup_fonts() -> Tuple[str, str]:
    """Return ``(display_family, mono_family)`` for the flat theme.

    Registers any Geist / Geist Mono TTFs found under ``$TSL_PLOT_FONTS``, a
    ``fonts/`` directory beside the current working directory, or a
    package-bundled ``_assets/fonts`` — then resolves to those families, falling
    back to DejaVu when they are unavailable.
    """
    _require_matplotlib()
    from matplotlib import font_manager as fm

    candidates = []
    env = os.environ.get("TSL_PLOT_FONTS")
    if env:
        candidates.append(env)
    candidates.append(os.path.join(os.getcwd(), "fonts"))
    candidates.append(os.path.join(os.path.dirname(__file__), "_assets", "fonts"))

    for d in candidates:
        if d and os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.lower().endswith((".ttf", ".otf")):
                    try:
                        fm.fontManager.addfont(os.path.join(d, fn))
                    except Exception:
                        pass

    have = {f.name for f in fm.fontManager.ttflist}
    disp = _DISPLAY if _DISPLAY in have else "DejaVu Sans"
    mono = _MONO if _MONO in have else "DejaVu Sans Mono"
    return disp, mono


# ---------------------------------------------------- background + cards ----
def flat_background(fig, cards: Dict[str, Tuple[float, float, float, float]]):
    """Paint the flat canvas: white ground, faint dot-grid, hairline cards.

    ``cards`` maps a key to a ``(x0, y0, w, h)`` rectangle in figure-fraction
    coordinates. Returns the background axes (used for header dividers).
    """
    _require_matplotlib()
    from matplotlib.patches import FancyBboxPatch

    fw, fh = fig.get_size_inches()
    dpi = fig.dpi
    T = TOKENS
    fig.patch.set_facecolor(T["bg"])

    bgax = fig.add_axes([0, 0, 1, 1], zorder=0)
    bgax.set_xlim(0, 1)
    bgax.set_ylim(0, 1)
    bgax.axis("off")

    sx = 0.018
    sy = sx * fw / fh
    gx, gy = np.meshgrid(np.arange(0.01, 1.0, sx), np.arange(0.01, 1.0, sy))
    bgax.scatter(gx, gy, s=0.7, c=T["dot"], marker=".", linewidths=0, zorder=0)

    rounding = T["radius_px"] / dpi / fw
    for (x0, y0, w, h) in cards.values():
        bgax.add_patch(FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0,rounding_size=%.5f" % rounding,
            mutation_aspect=fw / fh, fc=T["card"], ec=T["border"],
            lw=T["card_lw"], zorder=1))
    return bgax


def card_axes(fig, cards, key, pad_l=0.030, pad_r=0.024, pad_t_in=1.05, pad_b_in=0.50):
    """Inset plotting axes within a card.

    Horizontal pads (``pad_l``/``pad_r``) are figure-fraction; vertical pads
    (``pad_t_in``/``pad_b_in``) are **inches**, so the header band and bottom
    margin keep a constant physical size as the figure grows — only the plot
    area between them stretches.
    """
    fh = fig.get_size_inches()[1]
    x0, y0, w, h = cards[key]
    pad_t, pad_b = pad_t_in / fh, pad_b_in / fh
    ax = fig.add_axes([x0 + pad_l, y0 + pad_b,
                       w - pad_l - pad_r, h - pad_t - pad_b], zorder=3)
    ax.set_facecolor("none")
    return ax


# Card header geometry, in inches — fixed regardless of figure size so the
# kicker → title → divider block stays tight as the canvas grows. The header
# text never changes, so its spacing must not scale with the figure.
_HDR_KICK_IN = 0.38     # card top → kicker baseline
_HDR_TITLE_IN = 0.31    # kicker → title baseline
_HDR_DIV_IN = 0.18      # title → divider


def header(fig, bgax, cards, key, kicker, title, fn, disp, mono):
    """Card header: mono kicker (left), accent ``plot_…()`` tag (right),
    display title, and a hairline divider underneath. Vertical spacing is fixed
    in inches so it does not stretch when the figure grows."""
    T = TOKENS
    fh = fig.get_size_inches()[1]
    x0, y0, w, h = cards[key]
    top = y0 + h
    y_kick = top - _HDR_KICK_IN / fh
    y_title = top - (_HDR_KICK_IN + _HDR_TITLE_IN) / fh
    y_div = top - (_HDR_KICK_IN + _HDR_TITLE_IN + _HDR_DIV_IN) / fh
    fig.text(x0 + 0.028, y_kick, kicker.upper(), family=mono,
             fontsize=7.5, color=T["muted"])
    if fn:
        fig.text(x0 + w - 0.028, y_kick, fn, family=mono, fontsize=7.5,
                 color=T["accent"], ha="right")
    fig.text(x0 + 0.028, y_title, title, family=disp, fontsize=12.5,
             color=T["ink"], weight="semibold")
    bgax.plot([x0 + 0.028, x0 + w - 0.028], [y_div] * 2,
              color=T["divider"], lw=0.9, zorder=2)


# ------------------------------------------------- generic card-grid layout ---
# The card dashboards (feature importance, and every detail plot) lay panels
# out on a row×col grid of equal-or-ratioed cards. Margins and inter-card gaps
# are fixed inches so a larger figure grows the cards, never the spacing.
def grid_card_layout(fw, fh, n_rows, n_cols, *, margin_x_in=0.62,
                     margin_top_in=1.15, margin_bot_in=0.55, gap_in=0.45,
                     width_ratios=None, height_ratios=None):
    """Cards on an ``n_rows × n_cols`` grid, keyed by ``(row, col)``.

    Row 0 is the **top** row. ``width_ratios`` / ``height_ratios`` (defaulting
    to equal) split the available span; everything else is a fixed number of
    inches, returned as figure-fraction ``(x0, y0, w, h)`` rectangles.
    """
    wr = list(width_ratios) if width_ratios is not None else [1.0] * n_cols
    hr = list(height_ratios) if height_ratios is not None else [1.0] * n_rows
    avail_w = fw - 2 * margin_x_in - (n_cols - 1) * gap_in
    avail_h = fh - margin_top_in - margin_bot_in - (n_rows - 1) * gap_in
    col_w = [avail_w * r / sum(wr) for r in wr]
    row_h = [avail_h * r / sum(hr) for r in hr]

    x_lefts, x_in = [], margin_x_in
    for c in range(n_cols):
        x_lefts.append(x_in)
        x_in += col_w[c] + gap_in
    y_bottoms, y_in = [0.0] * n_rows, margin_bot_in
    for r in range(n_rows - 1, -1, -1):          # bottom row sits at the margin
        y_bottoms[r] = y_in
        y_in += row_h[r] + gap_in

    cards = {}
    for r in range(n_rows):
        for c in range(n_cols):
            cards[(r, c)] = (x_lefts[c] / fw, y_bottoms[r] / fh,
                             col_w[c] / fw, row_h[r] / fh)
    return cards


def grid_figsize(n_rows, n_cols, *, cell_w_in, cell_h_in, margin_x_in=0.62,
                 margin_top_in=1.15, margin_bot_in=0.55, gap_in=0.45,
                 max_w=72.0, max_h=200.0):
    """Figure size that gives each card a ``cell_w_in × cell_h_in`` target,
    with the same fixed margins/gaps :func:`grid_card_layout` assumes.

    ``max_w`` / ``max_h`` are safety rails for pathologically large grids; they
    sit far above what any realistic stage×feature grid needs, so the figure
    grows with the content and each card keeps a size that :func:`card_inset`'s
    fixed-inch pads fit inside.
    """
    w = 2 * margin_x_in + n_cols * cell_w_in + (n_cols - 1) * gap_in
    h = margin_top_in + margin_bot_in + n_rows * cell_h_in + (n_rows - 1) * gap_in
    return (min(w, max_w), min(h, max_h))


def card_inset(fig, cards, key, *, pad_l_in=0.78, pad_r_in=0.30,
               pad_t_in=1.00, pad_b_in=0.64):
    """Inset plotting axes within a card, with **all four pads in inches** so
    the header band, axis-label margin, and tick gutter keep a constant
    physical size as the figure grows. The fixed top pad clears the card
    header drawn by :func:`header`."""
    fw, fh = fig.get_size_inches()
    x0, y0, w, h = cards[key]
    ax = fig.add_axes([x0 + pad_l_in / fw, y0 + pad_b_in / fh,
                       w - (pad_l_in + pad_r_in) / fw,
                       h - (pad_t_in + pad_b_in) / fh], zorder=3)
    ax.set_facecolor("none")
    return ax


def card_colorbar(fig, cards, key, mappable, mono, label=None,
                  cb_w_in=0.15, cb_right_in=0.92, pad_t_in=1.00, pad_b_in=0.64):
    """Slim colorbar in a card's right gutter via a dedicated ``cax`` — so it
    never steals space from (or repositions) the surface axes. Pair with a
    surface :func:`card_inset` whose ``pad_r_in`` clears ``cb_right_in``."""
    fw, fh = fig.get_size_inches()
    x0, y0, w, h = cards[key]
    cax = fig.add_axes([x0 + w - cb_right_in / fw, y0 + pad_b_in / fh,
                        cb_w_in / fw, h - (pad_t_in + pad_b_in) / fh], zorder=3)
    cb = fig.colorbar(mappable, cax=cax)
    cb.outline.set_edgecolor(TOKENS["border"])
    cb.outline.set_linewidth(0.9)
    cb.ax.tick_params(length=0, labelsize=7.5, colors=TOKENS["muted"])
    for lab in cb.ax.get_yticklabels():
        lab.set_family(mono)
    if label:
        cb.set_label(label, family=mono, fontsize=8, color=TOKENS["muted"])
    return cb


def figure_title(fig, kicker, title, badge=None, badge_color=None, x=0.035):
    """Figure title block, pinned a fixed distance below the top edge so it
    stays tight on tall canvases."""
    disp, mono = setup_fonts()
    T = TOKENS
    fh = fig.get_size_inches()[1]
    fig.text(x, 1 - 0.52 / fh, kicker.upper(), family=mono, fontsize=10,
             color=T["muted"])
    fig.text(x, 1 - 0.90 / fh, title, family=disp, fontsize=20, color=T["ink"],
             weight="semibold")
    if badge:
        fig.text(0.965, 1 - 0.665 / fh, badge, family=mono, fontsize=9.5,
                 color=badge_color or T["muted"], ha="right")


def airy(ax, mono, grid=True, grid_axis="y"):
    T = TOKENS
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(T["faint"])
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(length=0, colors=T["muted"], labelsize=8)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_family(mono)
    if grid:
        ax.grid(axis=grid_axis, color=T["grid"], lw=0.9)
    ax.set_axisbelow(True)


# ------------------------------------------------------------ rounded marks ---
def _ppx(ax):
    """Data units per display pixel, on each axis (no renderer needed)."""
    fig = ax.figure
    fw, fh = fig.get_size_inches()
    dpi = fig.dpi
    pos = ax.get_position()
    return (np.ptp(ax.get_xlim()) / (pos.width * fw * dpi),
            np.ptp(ax.get_ylim()) / (pos.height * fh * dpi))


# Inches given to each row of a bar/heatmap panel. Row spacing is held constant
# at this pitch so a bigger figure adds rows (or whitespace), never gaps.
ROW_PITCH_IN = 0.32


def row_capacity(ax, n_rows, pitch=ROW_PITCH_IN):
    """How many rows of height ``pitch`` fit in this axes — at least ``n_rows``.

    Used to lay rows out top-aligned at a fixed pitch: a panel with fewer rows
    than the figure could hold leaves whitespace below, keeping the row pitch
    independent of figure size.
    """
    fh = ax.figure.get_size_inches()[1]
    return max(ax.get_position().height * fh / pitch, float(n_rows))


def rbar_h(ax, x_left, width, y_center, thick, color, r_disp=3, z=3):
    """Solid horizontal bar with crisp ``r_disp``-pixel corners."""
    from matplotlib.patches import FancyBboxPatch
    xpp, ypp = _ppx(ax)
    wpx = abs(width) / xpp if width else 0.0
    r = min(r_disp, wpx / 2, thick / ypp / 2)
    R = max(r * xpp, 1e-9)
    ax.add_patch(FancyBboxPatch(
        (min(x_left, x_left + width), y_center - thick / 2), abs(width), thick,
        boxstyle="round,pad=0,rounding_size=%.6f" % R, mutation_aspect=ypp / xpp,
        fc=color, ec="none", zorder=z, clip_on=False))


def rbar_v(ax, x_center, height, y_base, thick, color, r_disp=3, z=3):
    """Solid vertical bar with crisp ``r_disp``-pixel corners (a histogram column)."""
    from matplotlib.patches import FancyBboxPatch
    xpp, ypp = _ppx(ax)
    hpx = abs(height) / ypp if height else 0.0
    r = min(r_disp, hpx / 2, thick / xpp / 2)
    R = max(r * xpp, 1e-9)
    ax.add_patch(FancyBboxPatch(
        (x_center - thick / 2, min(y_base, y_base + height)), thick, abs(height),
        boxstyle="round,pad=0,rounding_size=%.6f" % R, mutation_aspect=ypp / xpp,
        fc=color, ec="none", zorder=z, clip_on=False))


# ----------------------------------------------------- text-measuring fit ---
def _renderer(fig):
    """A renderer for text metrics, regardless of the active backend.

    Uses the canvas' own renderer when available (Agg), else borrows a temporary
    Agg renderer for measurement only and restores the original canvas.
    """
    get = getattr(fig.canvas, "get_renderer", None)
    if get is not None:
        try:
            return get()
        except Exception:
            pass
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    old = fig.canvas
    try:
        return FigureCanvasAgg(fig).get_renderer()
    finally:
        fig.canvas = old


def _text_width_px(renderer, s, family, fontsize):
    from matplotlib.font_manager import FontProperties
    fp = FontProperties(family=family, size=fontsize)
    return renderer.get_text_width_height_descent(str(s), fp, False)[0]


def _ellipsize(renderer, s, family, fontsize, max_px):
    """Trim ``s`` with an ellipsis until it fits within ``max_px`` pixels."""
    s = str(s)
    if _text_width_px(renderer, s, family, fontsize) <= max_px:
        return s
    ell = "…"
    while s and _text_width_px(renderer, s + ell, family, fontsize) > max_px:
        s = s[:-1]
    return (s + ell) if s else ell


def fit_row_labels(fig, ax, x0, w, labels, family, fontsize, color,
                   min_pad_frac=0.014, gap_frac=0.010, max_left_frac=0.46):
    """Place y-tick labels so they cannot cross the card's left border.

    Measures the widest label with the renderer, sets the axes' left edge to
    just clear it (capped at ``max_left_frac`` of the card width), and ellipsises
    any label that still wouldn't fit. Right edge stays put, so only the plotting
    width gives. Falls back to a plain label set if measurement is unavailable.
    """
    ax.set_yticklabels(labels, family=family, fontsize=fontsize, color=color)
    try:
        fw, _ = fig.get_size_inches()
        fig_w_px = fw * fig.dpi
        renderer = _renderer(fig)
        widths = [_text_width_px(renderer, s, family, fontsize) for s in labels]
        max_w_frac = (max(widths) if widths else 0.0) / fig_w_px
        cap_left = x0 + max_left_frac * w
        needed_left = x0 + min_pad_frac + gap_frac + max_w_frac
        final = list(labels)
        if needed_left > cap_left:
            allowed_px = (max_left_frac * w - min_pad_frac - gap_frac) * fig_w_px
            final = [_ellipsize(renderer, s, family, fontsize, allowed_px)
                     for s in labels]
            new_left = cap_left
        else:
            new_left = needed_left
        pos = ax.get_position()
        right = pos.x0 + pos.width
        new_left = min(max(new_left, x0 + min_pad_frac), right - 0.02)
        ax.set_position([new_left, pos.y0, right - new_left, pos.height])
        if final != list(labels):
            ax.set_yticklabels(final, family=family, fontsize=fontsize, color=color)
    except Exception:
        pass


def tile_grid(ax, M, cmap, row_labels, col_labels, disp, mono, card=None,
              vmin=None, vmax=None, gap=0.06, radius_px=3,
              show_values=None, value_fmt="{:.2f}"):
    """Heatmap drawn as solid hairline-bordered cells — a crisp, modular grid.
    ``M`` is ``(n_rows, n_cols)`` laid out top→bottom; rows are features,
    columns stages by convention. When ``card=(x0, y0, w, h)`` is given, the row
    labels are measured and fitted so they stay inside the card."""
    from matplotlib.patches import FancyBboxPatch
    import matplotlib as mpl

    T = TOKENS
    M = np.asarray(M, dtype=float)
    nr, nc = M.shape
    lo = float(np.min(M)) if vmin is None else vmin
    hi = float(np.max(M)) if vmax is None else vmax
    if hi <= lo:
        hi = lo + 1e-9
    norm = mpl.colors.Normalize(lo, hi)

    ax.set_xlim(0, nc)
    ax.set_ylim(0, row_capacity(ax, nr))
    ax.invert_yaxis()
    if show_values is None:
        show_values = nr * nc <= 56 and hi >= 5e-3

    ax.set_xticks(np.arange(nc) + 0.5)
    ax.set_xticklabels(col_labels, family=mono, fontsize=8, color=T["muted"])
    ax.set_yticks(np.arange(nr) + 0.5)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if card is not None:
        fit_row_labels(ax.figure, ax, card[0], card[2], row_labels, disp, 8.5,
                       T["ink"])
    else:
        ax.set_yticklabels(row_labels, family=disp, fontsize=8.5, color=T["ink"])

    # thin column labels so they never collide: show every k-th when crowded
    try:
        fig = ax.figure
        fw, _ = fig.get_size_inches()
        pos = ax.get_position()
        col_w_px = pos.width * fw * fig.dpi / nc
        renderer = _renderer(fig)
        max_lab_px = max(_text_width_px(renderer, s, mono, 8) for s in col_labels)
        step = max(1, int(np.ceil((max_lab_px + 6) / col_w_px)))
        if step > 1:
            shown = [s if (i % step == 0) else "" for i, s in enumerate(col_labels)]
            ax.set_xticklabels(shown, family=mono, fontsize=8, color=T["muted"])
    except Exception:
        pass

    xpp, ypp = _ppx(ax)
    R = max(radius_px * xpp, 1e-9)
    for r in range(nr):
        for c in range(nc):
            rgba = cmap(norm(M[r, c]))
            x, y, w, h = c + gap / 2, r + gap / 2, 1 - gap, 1 - gap
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0,rounding_size=%.6f" % R,
                mutation_aspect=ypp / xpp, fc=rgba, ec=T["border"],
                lw=T["card_lw"], zorder=3))
            if show_values:
                ax.text(c + 0.5, r + 0.5, value_fmt.format(M[r, c]),
                        ha="center", va="center", fontsize=7.5,
                        family=mono, color=_text_on(rgba), zorder=4)


# ----------------------------------------------- multi-panel grid layouts ---
# The detail plots (PD / tilt / surfaces / components) lay panels out with
# ``subplots`` instead of cards. They share the flat ground, figure title, and
# panel styling below so they read as the same family as the card dashboards.
def flat_canvas(fig, dot_grid=True):
    """White ground (optionally a faint dot-grid) behind a grid of panels.

    The companion to :func:`flat_background` for subplot grids: panels keep
    their white facecolor, so they read as clean rectangles over the dotted
    ground. Returns the background axes.
    """
    _require_matplotlib()
    T = TOKENS
    fig.patch.set_facecolor(T["bg"])
    bgax = fig.add_axes([0, 0, 1, 1], zorder=0)
    bgax.set_xlim(0, 1)
    bgax.set_ylim(0, 1)
    bgax.axis("off")
    if dot_grid:
        fw, fh = fig.get_size_inches()
        sx = 0.018
        sy = sx * fw / fh
        gx, gy = np.meshgrid(np.arange(0.01, 1.0, sx), np.arange(0.01, 1.0, sy))
        bgax.scatter(gx, gy, s=0.7, c=T["dot"], marker=".", linewidths=0, zorder=0)
    return bgax


def reserve_title_band(fig, band_in=1.15):
    """Top figure-fraction to keep clear for :func:`figure_title`, given a
    fixed ``band_in`` inches. Pass as the ``top`` of a ``tight_layout`` rect so
    the title band stays a constant physical size as the figure grows."""
    return 1.0 - band_in / fig.get_size_inches()[1]


def panel_title(ax, title, disp, fontsize=10.5, pad=6):
    """Left-aligned display-font panel title in ink."""
    ax.set_title(title, family=disp, fontsize=fontsize, color=TOKENS["ink"],
                 weight="semibold", loc="left", pad=pad)


def panel_note(ax, text, mono, x=0.98, y=0.96, ha="right", va="top",
               color=None, fontsize=7.5):
    """Small mono annotation pinned in axes-fraction coords (e.g. constants)."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, family=mono,
            fontsize=fontsize, color=color or TOKENS["muted"], zorder=5)


def axis_label(ax, mono, xlabel=None, ylabel=None, fontsize=8.5):
    """Set mono, muted axis labels in the flat style."""
    T = TOKENS
    if xlabel is not None:
        ax.set_xlabel(xlabel, family=mono, fontsize=fontsize, color=T["muted"])
    if ylabel is not None:
        ax.set_ylabel(ylabel, family=mono, fontsize=fontsize, color=T["muted"])


def zero_ref(ax, axis="y", lw=0.8):
    """Faint dashed reference line at zero on the given axis."""
    T = TOKENS
    if axis == "y":
        ax.axhline(0, color=T["faint"], ls=(0, (3, 3)), lw=lw, zorder=1)
    else:
        ax.axvline(0, color=T["faint"], ls=(0, (3, 3)), lw=lw, zorder=1)


def signed_fill(ax, x, lo, hi, step=False, pos=None, neg=None, w=0.82, zorder=1):
    """Solid pale fill between ``lo`` and ``hi``: the positive token where
    ``hi ≥ lo``, the negative token elsewhere. Pale = blended toward white by
    ``w``, so the band stays fully opaque."""
    pos_c = mix(pos or TOKENS["pos"], w)
    neg_c = mix(neg or TOKENS["neg"], w)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    diff = hi - lo
    kw = dict(step="post") if step else {}
    ax.fill_between(x, lo, hi, where=(diff >= 0), color=pos_c, zorder=zorder, **kw)
    ax.fill_between(x, lo, hi, where=(diff < 0), color=neg_c, zorder=zorder, **kw)


# ------------------------------------------------------ surfaces + colorbar --
def flat_surface_axes(ax, mono, xlabel=None, ylabel=None):
    """Frame a contour/imshow panel: hairline border box, mono ticks, muted
    axis labels. A surface reads as a framed tile, so all four spines show."""
    T = TOKENS
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(T["border"])
        s.set_linewidth(0.9)
    ax.tick_params(length=0, colors=T["muted"], labelsize=8)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_family(mono)
    axis_label(ax, mono, xlabel, ylabel)


def flat_colorbar(fig, ax, mappable, mono, label=None):
    """Slim colorbar with a hairline border and mono ticks, beside ``ax``."""
    T = TOKENS
    cb = fig.colorbar(mappable, ax=ax, shrink=0.82, pad=0.03, aspect=26)
    cb.outline.set_edgecolor(T["border"])
    cb.outline.set_linewidth(0.9)
    cb.ax.tick_params(length=0, labelsize=7.5, colors=T["muted"])
    for lab in cb.ax.get_yticklabels():
        lab.set_family(mono)
    if label:
        cb.set_label(label, family=mono, fontsize=8, color=T["muted"])
    return cb
