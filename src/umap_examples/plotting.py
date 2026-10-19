"""Plots for the blog figures: no axis ticks, since UMAP axes have no meaning."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.text import Text

from .config import FIGURES_DIR, SEED


def plot_scatter(points, labels=None, ax=None, s=2, cmap="tab10", title=None, legend=True, other="other",
                 legend_loc="best"):
    """Scatter of a 2D embedding, colored by labels.

    Integer labels use `cmap`. String labels get one color each, in sorted order,
    and points labeled `other` are drawn first, in light grey, so the labeled points stay on top.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    labels = None if labels is None else np.asarray(labels)
    if labels is None:
        ax.scatter(points[:, 0], points[:, 1], s=s, c="0.3", alpha=0.8, linewidths=0)
    elif np.issubdtype(labels.dtype, np.number):
        sc = ax.scatter(points[:, 0], points[:, 1], c=labels, s=s, cmap=cmap, alpha=0.8, linewidths=0)
        if legend and labels is not None:
            ax.legend(*sc.legend_elements(), loc=legend_loc, fontsize=8, markerscale=1.5, frameon=False)
    else:
        rest = labels == other
        ax.scatter(points[rest, 0], points[rest, 1], s=s, c="lightgrey", linewidths=0, label=other)
        categories = sorted(set(labels[~rest]))
        # tab20 without its greys (reserved for `other`): 9 strong colors, then their 9 light versions
        tab20 = plt.get_cmap("tab20").colors
        colors = [c for i, c in enumerate(tab20[0::2] + tab20[1::2]) if i not in (7, 17)]
        for i, c in enumerate(categories):
            mask = labels == c
            ax.scatter(points[mask, 0], points[mask, 1], s=s, color=colors[i % len(colors)], linewidths=0, label=c)
        if legend:
            ax.legend(loc=legend_loc, fontsize=8, markerscale=4, frameon=False)

    # Square box, equal aspect kept by widening the data limits, so side-by-side panels get the same size
    ax.set(xticks=[], yticks=[], aspect="equal", adjustable="datalim", box_aspect=1)
    if title:
        ax.set_title(title)
    return ax


def plot_maps(points, labelings, s=0.5, panel_size=6, legend_loc="upper right", title=None):
    """The same 2-D map side by side, one panel per {panel title: labels (None for no labels)}."""
    fig, axes = plt.subplots(1, len(labelings), figsize=(panel_size * len(labelings), panel_size + 0.4),
                             layout="constrained", squeeze=False)
    for ax, (panel, labels) in zip(axes.flat, labelings.items()):
        plot_scatter(points, labels, ax=ax, s=s, title=panel, legend_loc=legend_loc)
    if title:
        fig.suptitle(title, fontsize=16)
    return fig


def draw_boxes(ax, boxes, color="black"):
    """Draw named boxes {name: (x0, x1, y0, y1)} on a 2D plot, with the name above each box."""
    from matplotlib.patches import Rectangle

    for name, (x0, x1, y0, y1) in boxes.items():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1))
        ax.text(x0, y1, name, color=color, fontsize=9, va="bottom")
    return ax


def plot_image_grid(filepaths, ncols=6, size=2, title=None):
    """Grid of image thumbnails, `size` inches per cell."""
    from PIL import Image

    nrows = -(-len(filepaths) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, path in zip(axes.flat, filepaths):
        image = Image.open(path)
        image.thumbnail((400, 400))
        ax.imshow(image)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_3d(points, labels, hover=None, size=2, height=800, other="other"):
    """Interactive Plotly scatter of a 3D embedding, one trace per label (click the legend to toggle).

    Points labeled `other` are light grey. `hover` is an optional list of strings (one per point), e.g. descriptions.
    Scatter3d needs WebGL: show it with fig.show(renderer="browser") or save it with fig.write_html.
    """
    import plotly.colors
    import plotly.graph_objects as go

    labels = np.asarray(labels)
    hover = np.asarray(hover if hover is not None else [""] * len(points), dtype=object)
    palette = plotly.colors.qualitative.Dark24

    fig = go.Figure()
    categories = sorted(set(labels) - {other}) + ([other] if other in labels else [])
    for i, c in enumerate(categories):
        mask = labels == c
        fig.add_trace(go.Scatter3d(
            x=points[mask, 0], y=points[mask, 1], z=points[mask, 2],
            mode="markers", name=str(c),
            marker=dict(size=size, color="lightgrey" if c == other else palette[i % len(palette)], opacity=0.8),
            text=hover[mask], hovertemplate=f"<b>{c}</b><br>%{{text}}<extra></extra>",
        ))

    no_axis = dict(showticklabels=False, showgrid=False, zeroline=False, showbackground=False, title="")
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(xaxis=no_axis, yaxis=no_axis, zaxis=no_axis, aspectmode="data"),
        legend=dict(itemsizing="constant", font=dict(size=10)),
    )
    return fig


def plot_3d_vs_2d(X, points, labels, view=(8, -10), s=1.5, figsize=(12, 5.6)):
    """A 3-D point cloud (left) next to its 2-D UMAP (right), colored by int labels: the post's intro figure.

    Colors are tab10 plus a light blue, enough for the mammoth's 11 body parts. `view` is (elevation, azimuth).
    """
    palette = np.array(list(plt.get_cmap("tab10").colors) + [plt.get_cmap("tab20").colors[1]])
    colors = palette[np.asarray(labels) % len(palette)]

    fig = plt.figure(figsize=figsize, layout="constrained")
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.scatter(*X.T, c=colors, s=s, linewidths=0, depthshade=False)
    ax3.view_init(elev=view[0], azim=view[1])
    ax3.set_box_aspect(np.ptp(X, axis=0))
    ax3.set_axis_off()
    ax3.set_title("3-D")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(*points.T, c=colors, s=s, linewidths=0)
    ax2.set(xticks=[], yticks=[], aspect="equal", adjustable="datalim", box_aspect=1)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_title("2-D UMAP")
    return fig


def plot_digit_mosaic(points, X, labels, size=(2400, 1260), digit=60, spacing=0.7, margin=0.04, seed=SEED):
    """MNIST digits drawn as images at their 2-D UMAP positions, tinted by label: the post's cover. Returns a PIL image.

    `size` is the canvas in px (2x the 1200x630 of a social card), `digit` the size of each digit in px. Digits are
    taken in random order and kept only if their center is at least `spacing` digits away from every digit already
    kept, so they don't overlap and stay readable when the image is shrunk.
    """
    from PIL import Image

    W, H = size
    # Fit the layout in the canvas, keeping its aspect ratio and leaving room for half a digit at the edges
    lo, hi = points.min(0), points.max(0)
    scale = min((W * (1 - 2 * margin) - digit) / (hi[0] - lo[0]), (H * (1 - 2 * margin) - digit) / (hi[1] - lo[1]))
    px = (points - (lo + hi) / 2) * scale + np.array([W / 2, H / 2])
    px[:, 1] = H - px[:, 1]  # image y goes down

    rng = np.random.default_rng(seed)
    kept = []
    for i in rng.permutation(len(X)):
        if not kept or np.min(np.linalg.norm(px[kept] - px[i], axis=1)) >= spacing * digit:
            kept.append(i)

    colors = np.array(plt.get_cmap("tab10").colors)
    canvas = np.ones((H, W, 3))
    for i in kept:
        tile = Image.fromarray((X[i].reshape(28, 28) * 255).astype(np.uint8)).resize((digit, digit), Image.LANCZOS)
        a = np.clip(np.array(tile) / 255, 0, 1)[..., None]
        x0, y0 = int(round(px[i, 0] - digit / 2)), int(round(px[i, 1] - digit / 2))
        patch = canvas[y0:y0 + digit, x0:x0 + digit]
        # Darkest wins where digits touch, so strokes stay solid on white
        canvas[y0:y0 + digit, x0:x0 + digit] = np.minimum(patch, a * colors[labels[i]] + (1 - a) * patch)
    return Image.fromarray((canvas * 255).astype(np.uint8))


def save_figure(fig, name, dpi=200, out_dir=FIGURES_DIR, dark=False):
    """Save `fig` as <name>.png. With `dark`, also save <name>_dark.png for the blog's dark mode.

    The dark version has a transparent background and light grey text and frames. Grey points are
    inverted (dark grey becomes light grey and the light grey of `other` becomes dark), other colors are kept.
    It is drawn from a copy, so `fig` keeps its colors.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if dark:
        dark_fig = pickle.loads(pickle.dumps(fig))
        _to_dark(dark_fig)
        dark_fig.savefig(out_dir / f"{name}_dark.png", dpi=dpi, bbox_inches="tight", transparent=True)
        plt.close(dark_fig)
    return path


def _to_dark(fig, color="0.85"):
    for text in fig.findobj(Text):
        text.set_color(color)
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
    # Single-color grey scatters (unlabeled points, `other`), legend handles included
    for coll in fig.findobj(PathCollection):
        fc = coll.get_facecolor()
        if len(fc) == 1 and np.allclose(fc[0, 0], fc[0, :3]):
            grey = np.clip(1 - fc[0, 0], 0.3, 0.8)
            coll.set_facecolor((grey, grey, grey, fc[0, 3]))
