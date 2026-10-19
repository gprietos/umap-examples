"""UMAP results are saved, so re-runs (e.g. after a kernel restart) only load them:

- `umap_run` for FiftyOne datasets: stored as brain runs (points + parameters), also shown in the App's embeddings panel.
- `compute_umap` for plain arrays (e.g. MNIST pixels, supervised UMAP): cached on disk with joblib, fitted model included.
  `store_points` adds its points to a FiftyOne dataset as a brain run.
"""

import joblib
import numpy as np
import umap

from .config import CACHE_DIR, SEED


def umap_brain_key(embeddings_field, num_dims=2, n_neighbors=15, min_dist=0.1):
    """Default brain key, e.g. umap_emb_dinov2_vitb14_2d_k15_md01."""
    return f"umap_{embeddings_field}_{num_dims}d_k{n_neighbors}_md{str(min_dist).replace('.', '')}"


def umap_run(dataset, embeddings_field, num_dims=2, n_neighbors=15, min_dist=0.1, metric="cosine", seed=SEED,
             brain_key=None, overwrite=False):
    """UMAP of the embeddings in `embeddings_field`, stored in the dataset as a brain run. Returns the (N, num_dims) points.

    If the run already exists it is loaded instead of recomputed. A stored run with other parameters raises,
    unless overwrite=True; one computed before samples were added is recomputed. Points are in the order of `dataset`.
    """
    import fiftyone.brain as fob

    brain_key = brain_key or umap_brain_key(embeddings_field, num_dims, n_neighbors, min_dist)
    params = dict(embeddings_field=embeddings_field, num_dims=num_dims, num_neighbors=n_neighbors,
                  min_dist=min_dist, metric=metric, seed=seed)

    if brain_key in dataset.list_brain_runs():
        if not overwrite:
            config = dataset.get_brain_info(brain_key).config
            stored = {k: getattr(config, k) for k in params}
            if stored != params:
                raise ValueError(f"Brain run {brain_key!r} has {stored}, not {params}. Pass overwrite=True to replace it.")
            points = dataset.load_brain_results(brain_key).points
            if len(points) == len(dataset):
                return points
            print(f"{brain_key}: {len(points)} points for {len(dataset)} samples, recomputing")
        dataset.delete_brain_run(brain_key)

    results = fob.compute_visualization(
        dataset, embeddings=embeddings_field, method="umap", num_dims=num_dims, num_neighbors=n_neighbors,
        min_dist=min_dist, metric=metric, seed=seed, brain_key=brain_key,
    )
    return results.points


def store_points(samples, points, brain_key, overwrite=False):
    """Store points computed outside FiftyOne (e.g. supervised UMAP, or `transform` of new data) as a brain run.

    `points` must be in the order of `samples` (a dataset or view). The run shows up in the App's embeddings panel.
    An existing run with as many points is kept, unless overwrite=True.
    """
    import fiftyone.brain as fob

    dataset = samples._dataset
    if brain_key in dataset.list_brain_runs():
        if not overwrite and len(dataset.load_brain_results(brain_key).points) == len(points):
            return
        dataset.delete_brain_run(brain_key)
    fob.compute_visualization(samples, points=points, brain_key=brain_key)


def points_in_box(points, box):
    """Indices of the 2D points inside box = (x0, x1, y0, y1)."""
    x0, x1, y0, y1 = box
    inside = (points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)
    return np.flatnonzero(inside)


def normalize(P):
    """Center and scale so the largest |coordinate| is 1."""
    P = P - P.mean(0)
    return P / np.abs(P).max()


def _fit_umap(X, y, params):
    model = umap.UMAP(**params, verbose=True)
    points = model.fit_transform(X, y=y)
    return points, model


def compute_umap(X, n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", y=None, seed=SEED,
                 cache_dir=CACHE_DIR, **kwargs):
    """Fit UMAP on an array and return (points, fitted model).

    Pass `y` for supervised UMAP (use -1 for unknown labels in the semi-supervised case).
    Results are cached in `cache_dir` by input and parameters (None disables the cache).
    """
    params = dict(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                  random_state=seed, **kwargs)
    fit = joblib.Memory(cache_dir, verbose=0).cache(_fit_umap) if cache_dir else _fit_umap
    return fit(X, y, params)
