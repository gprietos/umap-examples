"""MNIST (raw pixels), its FiftyOne dataset, and the data for the 3-D MNIST blog figure
(`blog/_includes/umap/mnist_3d.html`).

The figure draws every test digit as its own image (scatter-gl sprites).
Output in `out_dir`:

- `mnist_3d.json`: `points` (flat x,y,z in [-1, 1], 3-D UMAP of the 10k test digits) and `labels`.
- `sprites.png`: the digits in a 100-per-row grid of 28×28 tiles, in point order. White strokes with alpha = ink on
  a transparent background: scatter-gl multiplies each sprite by its point color, so digits take their class color.
"""

import json

import numpy as np
from PIL import Image

from .config import CACHE_DIR, MNIST3D_DIR, MNIST_DIR, SEED
from .reduction import compute_umap, normalize


def load_mnist(train, data_dir=MNIST_DIR):
    """(X[n, 784] floats in [0, 1], y[n]) for the train or test split, downloaded into `data_dir` if needed."""
    from torchvision.datasets import MNIST

    ds = MNIST(data_dir, train=train, download=True)
    X = ds.data.numpy().reshape(len(ds), -1).astype(np.float32) / 255.0
    return X, ds.targets.numpy()


def load_dataset(name="mnist", data_dir=MNIST_DIR):
    """Load the persistent FiftyOne dataset, creating it if needed: one sample per digit (60k train + 10k test).

    Each digit is written once as a PNG to `data_dir/images/<split>/<index>.png`. Samples have `ground_truth`
    ("0".."9"), `index` (row in `load_mnist`) and a `train`/`test` tag. Pixels are not stored: use `load_mnist`, whose
    rows match `split_view`.
    """
    import fiftyone as fo

    if fo.dataset_exists(name):
        return fo.load_dataset(name)

    samples = []
    for split, train in (("train", True), ("test", False)):
        X, y = load_mnist(train=train, data_dir=data_dir)
        images_dir = data_dir / "images" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        for i, (x, label) in enumerate(zip(X, y)):
            path = images_dir / f"{i:05d}.png"
            if not path.exists():
                Image.fromarray(np.rint(x.reshape(28, 28) * 255).astype(np.uint8)).save(path)
            samples.append(fo.Sample(filepath=str(path), tags=[split], index=i,
                                     ground_truth=fo.Classification(label=str(label))))

    dataset = fo.Dataset(name, persistent=True)
    dataset.add_samples(samples)
    return dataset


def split_view(dataset, split):
    """The "train" or "test" samples, in the row order of `load_mnist`."""
    return dataset.match_tags(split).sort_by("index")


def knn_predict(A_train, y_train, A_test, k=5):
    """Labels of the test points by majority vote of their k nearest training points in A (pixels or a UMAP layout)."""
    from sklearn.neighbors import KNeighborsClassifier

    return KNeighborsClassifier(n_neighbors=k).fit(A_train, y_train).predict(A_test)


def store_predictions(samples, field, preds, gt_field="ground_truth"):
    """Save predicted digits as a Classification field and evaluate them (eval_key `eval_<field>`), replacing any
    previous evaluation. In the App, filter `eval_<field>` = False to see the mistakes."""
    import fiftyone as fo

    samples.set_values(field, [fo.Classification(label=str(p)) for p in preds])
    eval_key = f"eval_{field}"
    dataset = samples._dataset
    if eval_key in dataset.list_evaluations():
        dataset.delete_evaluation(eval_key)
    samples.evaluate_classifications(field, gt_field=gt_field, eval_key=eval_key)


def sprite_sheet(X, per_row=100, size=28):
    n_rows = -(-len(X) // per_row)
    tiles = np.zeros((n_rows * per_row, size, size), dtype=np.uint8)
    tiles[: len(X)] = np.rint(X.reshape(-1, size, size) * 255)
    alpha = tiles.reshape(n_rows, per_row, size, size).transpose(0, 2, 1, 3).reshape(n_rows * size, per_row * size)
    white = np.full_like(alpha, 255)
    return Image.fromarray(np.stack([white, alpha], -1), "LA")  # grayscale + alpha: smaller than RGBA


def export_mnist_3d(out_dir=MNIST3D_DIR, data_dir=MNIST_DIR, seed=SEED, cache_dir=CACHE_DIR):
    X, y = load_mnist(train=False, data_dir=data_dir)
    points, _ = compute_umap(X, n_components=3, metric="euclidean", seed=seed, cache_dir=cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {"points": np.round(normalize(points), 3).ravel().tolist(), "labels": y.tolist()}
    (out_dir / "mnist_3d.json").write_text(json.dumps(data, separators=(",", ":")))
    sprite_sheet(X).save(out_dir / "sprites.png", optimize=True)
