"""Precompute the data for the interactive "3-D shape → UMAP" blog figure (default: blog/assets/data/umap/umap_explorer/).

    uv run python scripts/export_umap_explorer.py [--out-dir DIR] [--n-points 2500] [--shapes mammoth torus ...]

The blog/ folder mirrors the blog repo: once copied there, the `umap/umap_explorer` include reads `index.json` from it.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # thumbnails only, no window

from umap_examples.config import EXPLORER_DIR
from umap_examples.umap_explorer import export_widget_data
from umap_examples.shapes import SHAPES

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", type=Path, default=EXPLORER_DIR, help=f"default: {EXPLORER_DIR}")
parser.add_argument("--n-points", type=int, default=2500)
parser.add_argument("--shapes", nargs="+", default=list(SHAPES), choices=list(SHAPES))
args = parser.parse_args()

export_widget_data(args.shapes, out_dir=args.out_dir, n_points=args.n_points)
