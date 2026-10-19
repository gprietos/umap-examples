"""3-D toy shapes for the interactive "shape → UMAP" figure. Each generator returns (X[n, 3], color[n]).

`color` is either an int label (categorical) or a float in [0, 1] along the shape's generating parameter, so the 3-D
view and the UMAP can be painted the same way. The mammoth is the Smithsonian scan as used by PAIR's
"Understanding UMAP" (`raw_data/mammoth_umap.json` in github.com/PAIR-code/understanding-umap, Apache-2.0).
"""

import json

import numpy as np
import requests
from sklearn.datasets import make_swiss_roll

from .config import SEED, SHAPES_DIR

MAMMOTH_URL = "https://raw.githubusercontent.com/PAIR-code/understanding-umap/master/raw_data/mammoth_umap.json"


def _noise(rng, X, scale):
    return X + rng.normal(scale=scale, size=X.shape)


def mammoth(n=2500, seed=SEED, data_dir=SHAPES_DIR):
    """Subsample of the 10k-point mammoth, colored by its 11 body-part labels. Downloaded once into `data_dir`."""
    path = data_dir / "mammoth_umap.json"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        r = requests.get(MAMMOTH_URL, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    data = json.loads(path.read_text())
    X, labels = np.asarray(data["3d"], dtype=float), np.asarray(data["labels"])
    idx = np.random.default_rng(seed).choice(len(X), size=min(n, len(X)), replace=False)
    return X[idx], labels[idx]  # axes: width, length, height (z up, as Plotly expects)


def swiss_roll(n=2500, seed=SEED):
    X, t = make_swiss_roll(n, noise=0.3, random_state=seed)  # spiral in the x-z plane, roll axis along y
    return X, (t - t.min()) / (t.max() - t.min())


def linked_rings(n=2500, seed=SEED):
    """Two interlocking circles. Color runs around each ring; the second ring uses the upper half of the scale."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    ring = np.arange(n) % 2
    a = np.stack([np.cos(t), np.sin(t), np.zeros(n)], 1)
    b = np.stack([1 + np.cos(t), np.zeros(n), np.sin(t)], 1)
    X = _noise(rng, np.where(ring[:, None] == 0, a, b), 0.04)
    return X, (ring + t / (2 * np.pi)) / 2


def torus(n=2500, seed=SEED, R=1.0, r=0.4):
    """Surface of a torus, colored by the angle around the big circle."""
    rng = np.random.default_rng(seed)
    u, v = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
    X = np.stack([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)], 1)
    return _noise(rng, X, 0.01), u / (2 * np.pi)


def blobs(n=2500, seed=SEED):
    """Three Gaussian clusters with very different sizes, densities and gaps between them."""
    rng = np.random.default_rng(seed)
    sizes = [n // 2, n // 4, n - n // 2 - n // 4]
    centers = np.array([[0, 0, 0], [3, 0, 0], [3.8, 0.4, 0.3]])
    spreads = [1.0, 0.15, 0.3]
    X = np.concatenate([rng.normal(c, s, size=(m, 3)) for c, s, m in zip(centers, spreads, sizes)])
    labels = np.repeat(np.arange(3), sizes)
    return X, labels


SHAPES = {
    "mammoth": mammoth,
    "swiss_roll": swiss_roll,
    "linked_rings": linked_rings,
    "torus": torus,
    "blobs": blobs,
}
