"""Default paths and constants. Every function that reads or writes files takes its path as an argument."""

from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_DIR / "data"
UNSPLASH_DIR = DATA_DIR / "unsplash"  # Unsplash Lite metadata (photos.tsv000, ...) and images/
ZOO_DIR = DATA_DIR / "zoo"  # FiftyOne zoo downloads (open-images-v7/)
MNIST_DIR = DATA_DIR / "mnist"
SHAPES_DIR = DATA_DIR / "shapes"  # downloaded 3-D point clouds (mammoth)
CACHE_DIR = DATA_DIR / "cache"  # joblib cache for UMAPs of plain arrays
FIGURES_DIR = REPO_DIR / "outputs" / "figures"
EXPLORER_DIR = REPO_DIR / "blog" / "assets" / "data" / "umap" / "umap_explorer"  # data for the UMAP explorer figure
MNIST3D_DIR = REPO_DIR / "blog" / "assets" / "data" / "umap" / "mnist_3d"  # data for the 3-D MNIST blog figure

SEED = 51
