"""Data for the interactive "3-D shape → UMAP" blog figure (`blog/_includes/umap/umap_explorer.html`).

For every shape, UMAP is precomputed on a grid of (n_neighbors, min_dist), so the page only looks layouts up. Layouts
are aligned to a reference one (orthogonal Procrustes, reflections allowed), otherwise moving a slider could rotate or
mirror the whole picture, since UMAP's orientation is arbitrary. Output in `out_dir`:

- `index.json`: the grid and the list of shapes (key, title).
- `<shape>.json`: `points3d` (flat x,y,z in [-1, 1]), `colors` (hex per point) and `layouts` ({"k15_md0.1": flat x,y})
  with 2-D coordinates quantized to ints 0..999 (the page maps them back to [-1, 1]) to keep the files small.
- `thumbs/<shape>.png`: small 3-D view for the shape buttons.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from scipy.linalg import orthogonal_procrustes

from .config import CACHE_DIR, EXPLORER_DIR, SEED
from .reduction import compute_umap, normalize
from .shapes import SHAPES

KS = [5, 10, 15, 30, 50, 100, 200]
MIN_DISTS = [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]
REFERENCE = (15, 0.1)  # layout the others are aligned to (umap-learn's defaults)

TITLES = {"mammoth": "Mammoth", "swiss_roll": "Swiss roll", "linked_rings": "Linked rings", "torus": "Torus",
          "blobs": "Blobs"}
# Categorical palette that reads on both light and dark backgrounds; continuous shapes use turbo (cyclic ones hsv)
CATEGORICAL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf",
               "#aec7e8", "#ffbb78"]
CYCLIC = {"linked_rings", "torus"}


def layout_key(k, min_dist):
    return f"k{k}_md{min_dist}"


def point_colors(shape, color):
    if np.issubdtype(np.asarray(color).dtype, np.integer):
        return [CATEGORICAL[c % len(CATEGORICAL)] for c in color]
    cmap = plt.get_cmap("hsv" if shape in CYCLIC else "turbo")
    # Trim turbo's very dark ends, which vanish on a dark background
    values = 0.92 * np.asarray(color) if shape in CYCLIC else 0.08 + 0.84 * np.asarray(color)
    return [to_hex(c) for c in cmap(values)]


def align(P, reference):
    """Rotate/reflect P onto the reference layout (both centered); returns P normalized to [-1, 1]."""
    A, B = P - P.mean(0), reference - reference.mean(0)
    R, _ = orthogonal_procrustes(A / np.linalg.norm(A), B / np.linalg.norm(B))
    return normalize(A @ R)


def quantize(P):
    """[-1, 1] floats → flat list of ints 0..999."""
    return np.rint((P + 1) / 2 * 999).astype(int).ravel().tolist()


def save_thumbnail(X, colors, path, size=160):
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, s=2, linewidths=0, depthshade=False)
    ax.set_box_aspect(np.ptp(X, axis=0))
    ax.set_axis_off()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def export_shape(shape, out_dir, n_points=2500, ks=KS, min_dists=MIN_DISTS, seed=SEED, cache_dir=CACHE_DIR):
    X, color = SHAPES[shape](n=n_points, seed=seed)
    X = normalize(X)
    colors = point_colors(shape, color)

    def run(k, md):
        points, _ = compute_umap(X, n_neighbors=k, min_dist=md, metric="euclidean", seed=seed, cache_dir=cache_dir)
        return points

    reference = run(*REFERENCE)
    layouts = {}
    for k in ks:
        for md in min_dists:
            print(f"{shape}: k={k}, min_dist={md}")
            layouts[layout_key(k, md)] = quantize(align(run(k, md), reference))

    data = {"points3d": np.round(X, 3).ravel().tolist(), "colors": colors, "layouts": layouts}
    (out_dir / f"{shape}.json").write_text(json.dumps(data, separators=(",", ":")))
    save_thumbnail(X, colors, out_dir / "thumbs" / f"{shape}.png")


def export_widget_data(shapes=tuple(SHAPES), out_dir=EXPLORER_DIR, n_points=2500, ks=KS, min_dists=MIN_DISTS,
                       seed=SEED, cache_dir=CACHE_DIR):
    (out_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    for shape in shapes:
        export_shape(shape, out_dir, n_points, ks, min_dists, seed, cache_dir)
    # min_dists as strings, the same spelling as in the layout keys ("0.0", not JavaScript's "0")
    index = {"ks": list(ks), "min_dists": [str(md) for md in min_dists],
             "default": {"k": REFERENCE[0], "min_dist": str(REFERENCE[1])},
             "shapes": [{"key": s, "title": TITLES.get(s, s)} for s in shapes]}
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
