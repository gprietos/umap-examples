"""Precompute the data for the 3-D MNIST blog figure (default: blog/assets/data/umap/mnist_3d/).

    uv run python scripts/export_mnist_3d.py [--out-dir DIR]

The blog/ folder mirrors the blog repo: once copied there, the `umap/mnist_3d` include reads the files from it.
"""

import argparse
from pathlib import Path

from umap_examples.config import MNIST3D_DIR
from umap_examples.mnist import export_mnist_3d

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", type=Path, default=MNIST3D_DIR, help=f"default: {MNIST3D_DIR}")
args = parser.parse_args()

export_mnist_3d(args.out_dir)
